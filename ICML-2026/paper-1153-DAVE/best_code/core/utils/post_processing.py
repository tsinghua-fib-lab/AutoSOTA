import math
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional


@dataclass
class GaussianCFG:
    kernel_size: int
    sgm: float


@dataclass
class BilateralCFG:
    kernel_size: int
    sgm_spatial: float
    sgm_range: float


class GaussianFilter2D(nn.Module):
    def __init__(
        self, 
        kernel_size: int = 11, 
        sgm: float = 5.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1

        self.kernel_size = kernel_size
        self.sgm = sgm

        kernel = gaussian_kernel_2d(kernel_size, sgm)
        kernel /= kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        self.register_buffer('kernel', kernel)

    def forward(self, x: Tensor) -> Tensor:
        in_shape = x.shape
        assert len(in_shape) == 4
        assert in_shape[1] == 1
        return F.conv2d(x, self.kernel, padding=self.kernel_size//2)


class BilateralFilter2D(nn.Module):
    def __init__(
        self,
        kernel_size: int = 3,
        sgm_spatial: float = 0.5,
        sgm_range: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert kernel_size % 2 == 1

        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.sgm_spatial = sgm_spatial
        self.sgm_range = sgm_range
        self.eps = eps

        k_spatial = gaussian_kernel_2d(kernel_size, sgm_spatial).reshape(1, -1, 1)
        self.register_buffer("kernel_spatial", k_spatial)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        assert C == 1

        x_p = F.unfold(x, kernel_size=self.kernel_size, padding=self.pad)
        x_c = x.view(N, 1, H * W)

        kernel_range = gaussian(x_p - x_c, self.sgm_range)
        weights = kernel_range * self.kernel_spatial

        out = (weights * x_p).sum(dim=1)
        norm = weights.sum(dim=1)
        out = out / (norm + self.eps)

        return out.view(N, 1, H, W)


class PostProcessing(nn.Module):
    def __init__(
        self, 
        gaussian_cfg: Optional[GaussianCFG] = None, 
        bilateral_cfg: Optional[BilateralCFG] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps

        if gaussian_cfg is None:
            self.gaussian = nn.Identity()
        else:
            self.gaussian = GaussianFilter2D(
                kernel_size=gaussian_cfg.kernel_size,
                sgm=gaussian_cfg.sgm,
            )

        if bilateral_cfg is None:
            self.bilateral = nn.Identity()
        else:
            self.bilateral = BilateralFilter2D(
                kernel_size=bilateral_cfg.kernel_size,
                sgm_spatial=bilateral_cfg.sgm_spatial,
                sgm_range=bilateral_cfg.sgm_range,
                eps=eps,
            )

    def lerp(
        self, x0: Tensor, x1: Tensor, alpha: float = 0.5,
    ) -> Tensor:
        return (1-alpha) * x0 + alpha * x1

    def normalize_imgs(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4
        N = x.shape[0]
        norm = x.reshape(N, -1).max(dim=-1, keepdim=True).values
        norm = norm.reshape(N, 1, 1, 1)
        return x / norm
        
    def forward(self, x: Tensor) -> Tensor:
        assert len(x.shape) == 4
        x = x.sum(axis=1, keepdims=True)
        x = F.relu(x)

        # Adaptive edge-preserving smoothing:
        # Compute per-pixel "edgeness" from local gradient magnitude
        # Use it to modulate smoothing strength
        x_smoothed = self.gaussian(x)

        # Edge detection: Sobel-like gradient via center difference
        grad_x = torch.abs(x[:, :, :, 2:] - x[:, :, :, :-2])
        grad_x = F.pad(grad_x, (1, 1, 0, 0))
        grad_y = torch.abs(x[:, :, 2:, :] - x[:, :, :-2, :])
        grad_y = F.pad(grad_y, (0, 0, 1, 1))
        edge_strength = (grad_x + grad_y) / 2.0

        # Normalize edge strength to [0, 1]
        edge_max = edge_strength.amax(dim=(2,3), keepdim=True).clamp(min=self.eps)
        edge_norm = (edge_strength / edge_max).clamp(0, 1)

        # At edges: keep original (alpha=0). In flat regions: more smoothing (alpha=0.6)
        alpha = 0.6 * (1.0 - edge_norm)

        x = (1.0 - alpha) * x + alpha * x_smoothed

        x = self.bilateral(x)
        x = self.normalize_imgs(x)
        x = x.sum(dim=1, keepdim=True)
        return x


def gaussian(x: Tensor, sgm: float) -> Tensor:
    return torch.exp(-(x**2)/(2*(sgm**2)))


def gaussian_kernel_2d(kernel_size: int, sgm: float):
    coords = torch.arange(kernel_size) - kernel_size // 2
    x_grid, y_grid = torch.meshgrid(coords, coords, indexing='ij')
    kernel = gaussian(x_grid, sgm) * gaussian(y_grid, sgm)
    return kernel

