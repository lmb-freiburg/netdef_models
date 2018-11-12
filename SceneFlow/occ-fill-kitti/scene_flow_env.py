import netdef as nd
from netdef_slim.ops import image_to_range_01, const_like, disp_to_flow
from netdef_slim.networks.dispnet.dispnet_2f_env import DispNet2f_Environment

from netdef_slim.deploy import StandardDeployment

class SceneFlowEnv(DispNet2f_Environment):

    def make_eval_graph(self, width, height, scale=1.0):
        data = nd.Struct()
        data.width = width
        data.height = height

        data.img = nd.Struct()
        data.img.L = nd.placeholder('data.img.L', (1, 3, height, width))

        data.disp = nd.Struct()
        data.disp.L = nd.placeholder('data.disp.L', (1, 1, height, width))

        data.flow = nd.Struct()
        data.flow[0] = nd.Struct()
        data.flow[0].fwd  = nd.placeholder('data.flow_0.fwd', (1, 2, height, width))

        data.occ = nd.Struct()
        data.occ[0] = nd.Struct()
        data.occ[0].fwd   = nd.placeholder('data.occ_0.fwd', (1, 1, height, width))

        output = StandardDeployment().make_graph(
            data=data,
            net_graph_constructor=lambda data: self.make_net_graph(data, include_losses=False),
            divisor=self._deploy_divisor,
            scale = scale)

        return output
