import os
import torch
import numpy as np

from thop import profile
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        input = torch.randn(1,args.seq_len ,args.enc_in ).to(self.device)

        macs, params = profile(self.model, inputs=(input, ))
        print(f"MACs: {macs}")
        print(f"Params: {params}")
        if macs >= 1e9:
            print( f"{macs / 1e9:.2f}G MACs")
        elif macs >= 1e6:
            print( f"{macs / 1e6:.2f}M MACs")
        else:
            print( f"{macs} MACs")

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
