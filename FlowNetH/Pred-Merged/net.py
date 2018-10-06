#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import netdef_slim as nd
from netdef_slim.networks.base_network import BaseNetwork
from netdef_slim.networks.flownet.flownet_2f_env import FlowNet2f_Environment
from netdef_slim.architectures import Architecture_C
from netdef_slim.architectures import Architecture_S

class FlowNetHMSS_Network(BaseNetwork):

    def __init__(self, output_type='iul', num_hypotheses=8, dist=1, **kwargs):
        super().__init__(**kwargs)
        
        self._output_type = output_type
        self._num_hypotheses = num_hypotheses
        self._log_sigmoid = self.default_log_sigmoid
        
        
        
    def default_log_sigmoid(self, value):
        val = 10
        log_limit_min = -val
        log_limit_max = val

        adjusted = nd.ops.adjusted_sigmoid(value, min=log_limit_min, max=log_limit_max)

        return adjusted

    def make_graph(self, data, include_losses=False):
        
        # hypNet
        pred_config = nd.PredConfig()
        pred_config.add(nd.PredConfigId(type='flow_hyp', dir='fwd', offset=0, channels=2, scale=self._scale, array_length=self._num_hypotheses))
        pred_config.add(nd.PredConfigId(type='iul_b_hyp_log', dir='fwd', offset=0, channels=2, scale=self._scale, array_length=self._num_hypotheses, mod_func=self._log_sigmoid))
            
        nd.log('pred_config:')
        nd.log(pred_config)

        with nd.Scope('hypNet', shared_batchnorm=False, correlation_leaky_relu=True, **self.scope_args()):
            arch = Architecture_C(
                num_outputs=pred_config.total_channels(),
                disassembling_function=pred_config.disassemble,
                conv_upsample=True,
                loss_function= None,
                channel_factor=self._channel_factor,
                feature_channels=self._feature_channels
            )

            out_hyp = arch.make_graph(data.img[0], data.img[1])
        
        # mergeNet
        pred_config = nd.PredConfig()
        pred_config.add(nd.PredConfigId(type='flow', dir='fwd', offset=0, channels=2, scale=self._scale, dist=1))
        pred_config.add(nd.PredConfigId(type='iul_b_log', dir='fwd', offset=0, channels=2, scale=self._scale, dist=1, mod_func=self.iul_b_log_sigmoid))
        nd.log('pred_config:')
        nd.log(pred_config)
        hyps = [nd.ops.resample(hyp, reference=data.img[0], antialias=False, type='LINEAR') for hyp in [out_hyp.final.flow_hyp[0].fwd[i] for i in range(self._num_hypotheses)]]
        uncertainties = [nd.ops.resample(unc, reference=data.img[0], antialias=False, type='LINEAR') for unc in [out_hyp.final.iul_b_hyp_log[0].fwd[i] for i in range(self._num_hypotheses)]]
        img_warped = [nd.ops.warp(data.img[1], hyp) for hyp in hyps]
        with nd.Scope('mergeNet', shared_batchnorm=False, **self.scope_args()):            
            input = nd.ops.concat([data.img[0]] + [data.img[1]] + hyps + uncertainties + img_warped)
            arch = Architecture_S(
                num_outputs=pred_config.total_channels(),
                disassembling_function=pred_config.disassemble,
                conv_upsample=True,
                loss_function= None,
                channel_factor=self._channel_factor
                )
            out_merge = arch.make_graph(input)
            
        return out_merge
            
    def iul_b_log_sigmoid(self, value):
        sx, sy = nd.ops.slice(value, 1)

        adjusted_sx = nd.ops.adjusted_sigmoid(sx, min=-3, max=3)
        adjusted_sy = nd.ops.adjusted_sigmoid(sy, min=-3, max=3)
        adjusted = nd.ops.concat(adjusted_sx, adjusted_sy, axis=1)

        return adjusted
    
net = FlowNetHMSS_Network(scale=0.05, batch_norm=True)

def get_env():
    return FlowNet2f_Environment(net)


