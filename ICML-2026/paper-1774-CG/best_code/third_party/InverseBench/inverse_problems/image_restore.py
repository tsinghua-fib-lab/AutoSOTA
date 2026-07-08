from abc import ABC, abstractmethod
import numpy as np
import scipy

import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from typing import List, Optional
from .base import BaseOperator

# helper functions for implementing the operators
class Blurkernel(torch.nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(self.kernel_size//2),
            torch.nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k

def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.
    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.
    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.
    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.
    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x

def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.
    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.
    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.
    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data

def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t + h, l:l + w] = 0

    return mask, t, t + h, l, l + w


class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(32, 32)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                                            mask_shape=(mask_h, mask_w),
                                            image_size=self.image_size,
                                            margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask


class Denoise(BaseOperator):
    def __init__(self, **kwargs):
        super(Denoise, self).__init__(**kwargs)
    
    def forward(self, data):
        return data


class SuperResolution(BaseOperator):
    def __init__(self, in_resolution: int, scale_factor: int, **kwargs):
        super(SuperResolution, self).__init__(**kwargs)
        self.out_resolution = in_resolution // scale_factor

    def forward(self, data, # data: (batch_size, channel, height, width)
                **kwargs):
        down_sampled = TF.resize(data, size=self.out_resolution, interpolation=InterpolationMode.BICUBIC, antialias=True)
        return down_sampled


class GaussialBlur(BaseOperator):
    def __init__(self, kernel_size, intensity, **kwargs):
        super(GaussialBlur, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=self.device).to(self.device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)


class Inpainting(BaseOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None, resolution=256, **kwargs):
        super(Inpainting, self).__init__(**kwargs)
        margin = resolution // 8
        self.mask_gen = mask_generator(mask_type, mask_len_range, mask_prob_range, resolution, margin=(margin, margin))
        self.mask = self.mask_gen(torch.zeros(1, 3, resolution, resolution, device=self.device))
        self.mask = self.mask[0:1, 0:1, :, :]
    
    def forward(self, data, **kwargs):
        if self.mask.device != data.device:
            self.mask = self.mask.to(data.device)
        return data * self.mask
    
    # Utils for linear operators
    # A = UMSV^T, where M is a 0/1 mask matrix and S is a diagonal matrix
    def Vt(self, data):
        # input: x, return: V^T x
        return data

    @property
    def M(self):
        # return M
        return self.mask[0].repeat(3,1,1)
    
    @property
    def S(self):
        # return S
        return torch.ones_like(self.M)
    
    def Ut(self, y):
        # input: y, return: U^T y
        return y
    
    def V(self, x):
        # input: x, return: V x
        return x
    
    def pseudo_inverse(self, y):
        # input: y, return: A^{-1} y
        return self.V(self.M * self.Ut(y))
    
# ----------------- Nonlinear Operators ----------------- #

class PhaseRetrieval(BaseOperator):
    def __init__(self, oversample=0.0, resolution=64, **kwargs):
        super(PhaseRetrieval, self).__init__(**kwargs)
        self.pad = int((oversample / 8.0) * resolution)
        
    def forward(self, data):
        x = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        if not torch.is_complex(x):
            x = x.type(torch.complex64)
        fft2_m = torch.view_as_complex(fft2c_new(torch.view_as_real(x)))
        amplitude = fft2_m.abs()
        return amplitude


