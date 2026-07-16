"""
HermitePINN: thin high-level wrapper combining Hermite hash encoding + SIREN MLP.

The training scripts in `examples/` import the CUDA MLP extensions
(`hermite_mlp_cuda_v2`, `hermite_mlp_cuda_3d_v2`) directly for their custom
backward pass. This module is provided as a convenience for users who want a
single nn.Module they can plug into their own training loops.
"""

import torch
import torch.nn as nn
import numpy as np

from hermite_ngp.encoding import HermiteHashEncoding2D, HermiteHashEncoding3D


def _siren_init(layer: nn.Linear, omega: float, is_first: bool = False):
    """SIREN initialization (Sitzmann et al., NeurIPS 2020)."""
    with torch.no_grad():
        n = layer.in_features
        bound = 1.0 / n if is_first else np.sqrt(6.0 / n) / omega
        layer.weight.uniform_(-bound, bound)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound)


class _SIRENLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, omega: float, is_first: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.omega = omega
        _siren_init(self.linear, omega, is_first)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class HermitePINN2D(nn.Module):
    """Hermite-NGP 2D network: hash encoding + SIREN MLP -> scalar u(x, y).

    Note: this wrapper uses PyTorch autograd for the Laplacian. For maximum
    speed, use the CUDA-accelerated training scripts in `examples/` which call
    `hermite_mlp_cuda_v2` for analytic gradients and Laplacians.
    """

    def __init__(
        self,
        n_levels: int = 8,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 14,
        base_resolution: int = 16,
        per_level_scale: float = 2.0,
        hidden: int = 128,
        n_layers: int = 2,
        omega: float = 30.0,
    ):
        super().__init__()
        self.encoding = HermiteHashEncoding2D(
            n_input_dims=2,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size_1=log2_hashmap_size,
            log2_hashmap_size_2=log2_hashmap_size,
            log2_hashmap_size_3=log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=per_level_scale,
        )
        encoding_dim = n_levels * n_features_per_level
        layers = [_SIRENLayer(encoding_dim, hidden, omega, is_first=True)]
        for _ in range(n_layers - 1):
            layers.append(_SIRENLayer(hidden, hidden, omega))
        layers.append(nn.Linear(hidden, 1))
        with torch.no_grad():
            _siren_init(layers[-1], omega)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoding(x)).squeeze(-1)


class HermitePINN3D(nn.Module):
    """Hermite-NGP 3D network: hash encoding + SIREN MLP -> scalar u(x, y, z).

    Same caveat as HermitePINN2D: for maximum speed use the CUDA training
    scripts in `examples/` instead of this PyTorch-autograd wrapper.
    """

    def __init__(
        self,
        n_levels: int = 8,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 14,
        base_resolution: int = 16,
        per_level_scale: float = 2.0,
        hidden: int = 128,
        n_layers: int = 2,
        omega: float = 30.0,
    ):
        super().__init__()
        self.encoding = HermiteHashEncoding3D(
            n_input_dims=3,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size_1=log2_hashmap_size,
            log2_hashmap_size_2=log2_hashmap_size,
            log2_hashmap_size_3=log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=per_level_scale,
        )
        encoding_dim = n_levels * n_features_per_level
        layers = [_SIRENLayer(encoding_dim, hidden, omega, is_first=True)]
        for _ in range(n_layers - 1):
            layers.append(_SIRENLayer(hidden, hidden, omega))
        layers.append(nn.Linear(hidden, 1))
        with torch.no_grad():
            _siren_init(layers[-1], omega)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.encoding(x)).squeeze(-1)
