import torch
import numpy as np
from typing import Dict
from .base import BaseOperator
from training.dataset import MultiCoilMRIData

class MultiCoilMRI(BaseOperator):
    def __init__(self, total_lines=128, acceleration_ratio=8, pattern='random', orientation='vertical', mask_fname=None, mask_seed=0, device='cuda', sigma_noise=0.0):
        '''
        MRI forward operator
        Args:
            - mask: sampling mask
            - device: device to run the operator
            - dtype: data type
        '''
        super(MultiCoilMRI, self).__init__(sigma_noise=sigma_noise)
        if mask_fname is None:
            if 1 < acceleration_ratio <= 6:
                # Keep 8% of center samples
                acs_lines = np.floor(0.08 * total_lines).astype(int)
            else:
                # Keep 4% of center samples
                acs_lines = np.floor(0.04 * total_lines).astype(int)
            mask = self.get_mask(acs_lines, total_lines, acceleration_ratio, pattern, seed=mask_seed)
        else:
            mask = np.load(mask_fname)
        if orientation == 'vertical':
            mask = mask[None, None, None, :].astype(bool)
        elif orientation == 'horizontal':
            mask = mask[None, None, :, None].astype(bool)
        else:
            raise NotImplementedError
        self.mask = torch.from_numpy(mask).to(device)
        self.device = device

    @staticmethod
    # Phase encode random mask generator
    def get_mask(acs_lines=30, total_lines=384, R=1, pattern='random', seed=0):
        np.random.seed(seed)

        # Overall sampling budget
        num_sampled_lines = np.floor(total_lines / R)

        # Get locations of ACS lines
        # !!! Assumes k-space is even sized and centered, true for fastMRI
        center_line_idx = np.arange((total_lines - acs_lines) // 2,
                             (total_lines + acs_lines) // 2)

        # Find remaining candidates
        outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
        if pattern == 'random':
            # Sample remaining lines from outside the ACS at random
            random_line_idx = np.random.choice(outer_line_idx,
                       size=int(num_sampled_lines - acs_lines), replace=False)
        elif pattern == 'equispaced':
            # Sample equispaced lines
            # !!! Only supports integer for now
            random_line_idx = outer_line_idx[::int(R)]
        else:
            raise NotImplementedError('Mask pattern not implemented')

        # Create a mask and place ones at the right locations
        mask = np.zeros((total_lines))
        mask[center_line_idx] = 1.
        mask[random_line_idx] = 1.

        return mask

    def unnormalize(self, gen_img):
        return gen_img
        # scaling = np.quantile(np.abs(self.estimated_mvue), 0.99)
        # return gen_img * torch.tensor(scaling)

    def normalize(self, gen_img):
        return gen_img
        # scaling = np.quantile(np.abs(self.estimated_mvue), 0.99)
        # return gen_img / torch.tensor(scaling)

    @staticmethod
    def ifft(x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x

    @staticmethod
    def fft(x: torch.Tensor) -> torch.Tensor:
        x = torch.fft.fftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        return x

    def __call__(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.maps = data['maps']
        self.masked_kspace = self.mask * data['kspace']
        self.estimated_mvue = MultiCoilMRIData.get_mvue(
            self.masked_kspace.cpu().numpy(),
            self.maps.cpu().numpy()
        )
        return torch.view_as_real(self.masked_kspace)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.unnormalize(image).to(torch.float64)
        coils = self.maps * torch.view_as_complex(image.permute(0, 2, 3, 1).contiguous()).unsqueeze(1)
        return torch.view_as_real(self.mask * self.fft(coils))