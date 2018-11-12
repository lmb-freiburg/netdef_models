#-*- coding: utf-8 -*-
import netdef_slim as nd
from netdef_slim.networks.dispnet.dispnet_2f_env import DispNet2f_Environment
from netdef_slim.networks.base_network import BaseNetwork as DispNet_BaseNetwork
from netdef_slim.architectures.architecture_c import  Architecture_C
from netdef_slim.architectures.architecture_s import  Architecture_S
from scene_flow_env import SceneFlowEnv


class Network(DispNet_BaseNetwork):


    def interpolator(self, pred, prev_disp):
        prev_disp = nd.ops.resample(prev_disp, reference=pred, type='LINEAR', antialias=False)
        _sum = nd.ops.add(prev_disp, pred)
        return nd.ops.neg_relu(_sum)


    def make_graph(self, data, include_losses=True):
        pred_config = nd.PredConfig()
        pred_config.add(nd.PredConfigId(type='disp',
                                        perspective='L',
                                        channels=1,
                                        scale=self._scale,
                                        ))


        pred_dispL_t_1 = data.disp.L
        pred_flow_fwd  = data.flow[0].fwd
        pred_occ_fwd = data.occ[0].fwd

        pred_dispL_t1_warped = nd.ops.warp(pred_dispL_t_1, pred_flow_fwd)

        pred_config[0].mod_func = lambda x: self.interpolator(pred=x, prev_disp=pred_dispL_t1_warped)
        inp = nd.ops.concat(data.img.L, nd.ops.scale(pred_dispL_t1_warped, 0.05), pred_occ_fwd)

        with nd.Scope('refine_disp', learn=True, **self.scope_args()):
            arch = Architecture_S(
                num_outputs=pred_config.total_channels(),
                disassembling_function=pred_config.disassemble,
                loss_function=None,
                conv_upsample=self._conv_upsample,
                exit_after=0
            )
            out = arch.make_graph(inp, edge_features=data.img.L)
        return out



net = Network(
    conv_upsample=False,
    scale=1.0,
)

def get_env():
    env = SceneFlowEnv(net,)
    return env



