#! /usr/bin/python3

from netdef_slim.tensorflow.controller.base_controller import BaseTFController
from netdef_slim.tensorflow.controller.net_actions import *
import os
import netdef as nd
from scipy.misc import imread
import numpy as np
import importlib.util
import tensorflow as tf


class SFNetActions(NetActions):

    def _get_dn_controller(self, dn_path):
        dn_controller_spec = importlib.util.spec_from_file_location('dn_controller', os.path.join(dn_path, 'controller.py'))
        dn_controller = importlib.util.module_from_spec(dn_controller_spec)
        dn_controller_spec.loader.exec_module(dn_controller)
        return dn_controller

    def _get_fn_controller(self, fn_path):
        fn_controller_spec = importlib.util.spec_from_file_location('fn_controller', os.path.join(fn_path, 'controller.py'))
        fn_controller = importlib.util.module_from_spec(fn_controller_spec)
        fn_controller_spec.loader.exec_module(fn_controller)
        return fn_controller



    def eval(self, images_t0, images_t1, dn_path, fn_path, state=None):
        im0_t0, im1_t0 = images_t0
        im0_t1, im1_t1 = images_t1

        dn_controller = self._get_dn_controller(dn_path)
        fn_controller = self._get_fn_controller(fn_path)

        if isinstance(im0_t0, str): im0_t0=imread(im0_t0).transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
        if isinstance(im1_t0, str): im1_t0=imread(im1_t0).transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

        if isinstance(im0_t1, str): im0_t1=imread(im0_t1).transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)
        if isinstance(im1_t1, str): im1_t1=imread(im1_t1).transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

        disp_controller = dn_controller.Controller(path=dn_path)
        print("processing images at t=0 for disp:")
        out_disp_t0 = disp_controller.net_actions(net_dir=dn_path).eval(image_0=im0_t0, image_1=im1_t0, state=state)
        print("processing images at t=1 for disp:")
        out_disp_t1 = disp_controller.net_actions(net_dir=dn_path).eval(image_0=im0_t1, image_1=im1_t1, state=state)
        print("processing images at t=0 for flow:")

        flow_controller = fn_controller.Controller(path=fn_path)
        out_flow_t0 = flow_controller.net_actions(net_dir=fn_path).eval(image_0=im0_t0, image_1=im0_t1, state=state)
        print("filling disp at t=1:")
        out = self._eval_sf(im0_t0, out_disp_t1['disp.L'], out_flow_t0['flow[0].fwd'], out_flow_t0['occ[0].fwd'])
        return out


    def _eval_sf(self, image_0, dispL_t1, flow_0, occ_0, state=None):
        nd.evo_manager.clear()
        nd.load_module('config.py')
        nd.load_module('net.py')

        nd.phase = 'test'
        tf.reset_default_graph()
        height = image_0.shape[2]
        width = image_0.shape[3]
        last_evo, current_evo = nd.evo_manager.get_status()
        env = self.net.get_env()
        print('Evolution: ' + last_evo.path())
        eval_out = env.make_eval_graph(
                                        width = width,
                                        height = height,
                                        )
        session = self._create_session()
        trainer = SimpleTrainer(session=session, train_dir=last_evo.path())
        session.run(tf.global_variables_initializer())
        ignore_vars = []
        if len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="copy")) > 0:
            ignore_vars = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="copy")[0]]
        if state is None:
            state = last_evo.last_state()
            trainer.load_checkpoint(state.path(), ignore_vars=ignore_vars)
        else:
            state = nd.evo_manager.get_state(state)
            trainer.load_checkpoint(state.path(), ignore_vars=ignore_vars)

        placeholders = tf.get_collection('placeholders')
        _img0 = placeholders[0]
        _dispL = placeholders[1]
        _flow_0 = placeholders[2]
        _occ_0 = placeholders[3]
        out = session.run(eval_out.get_list(), feed_dict={ _img0: image_0,
                                                           _dispL: dispL_t1,
                                                           _flow_0 : flow_0,
                                                           _occ_0 : occ_0
                                                           })

        return out




class Controller(BaseTFController):
    base_path = os.path.dirname(os.path.realpath(__file__))
    def _configure_subparsers(self):
        super()._configure_subparsers()
        del(self._command_hooks['eval'])
        # eval
        subparser = self._subparsers.add_parser('eval', help='run network on images')
        subparser.add_argument('--imgs_t0', nargs='+', help='L,R images at t0')
        subparser.add_argument('--imgs_t1', nargs='+', help='L,R images at t1')
        subparser.add_argument('--dn_path',        help='path to dispnet')
        subparser.add_argument('--fn_path',        help='path to flownet')
        subparser.add_argument('--out_dir',        help='path to output dir')
        subparser.add_argument('--state',     help='state of the snapshot', default=None)
        def eval():
            self.eval(images_t0=self._args.imgs_t0,
                      images_t1=self._args.imgs_t1,
                      dn_path = self._args.dn_path,
                      fn_path = self._args.fn_path,
                      out_dir=self._args.out_dir,
                      state=self._args.state)
            sys.exit(nd.status.SUCCESS)
        self._command_hooks['eval'] = eval


if __name__ == '__main__':
	controller = Controller(net_actions=SFNetActions)
	controller.run()

