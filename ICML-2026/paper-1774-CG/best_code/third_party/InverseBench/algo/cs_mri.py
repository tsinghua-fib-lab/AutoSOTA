import torch
import numpy as np
from .base import Algo
from sigpy.mri import app


class CompressedSensingMRI(Algo):
    def __init__(self, net, forward_op, mode, lamda):
        super(CompressedSensingMRI, self).__init__(net, forward_op)
        self.mode = mode
        self.lamda = lamda

    @torch.no_grad()
    def inference(self, observation: torch.Tensor, **kwargs) -> torch.Tensor:
        observation = torch.view_as_complex(observation)
        recon = torch.zeros(observation.shape[0], observation.shape[2], observation.shape[3], dtype=torch.complex128).to(self.forward_op.device)
        for i in range(len(observation)):
            masked_kspace = observation[i].cpu().numpy()
            s_maps = self.forward_op.maps[i].cpu().numpy()
            if self.mode == 'Sense':
                recon_app = app.SenseRecon(masked_kspace, s_maps, self.lamda)
            elif self.mode == 'L1Wavelet':
                recon_app = app.L1WaveletRecon(masked_kspace, s_maps, self.lamda)
            elif self.mode == 'TV':
                recon_app = app.TotalVariationRecon(masked_kspace, s_maps, self.lamda)
            else:
                raise ValueError(f'Invalid mode: {self.mode}. Choose from Sense, L1Wavelet, TV')
            recon[i] = torch.from_numpy(recon_app.run())
        return self.forward_op.normalize(torch.view_as_real(recon).permute(0, 3, 1, 2).contiguous())