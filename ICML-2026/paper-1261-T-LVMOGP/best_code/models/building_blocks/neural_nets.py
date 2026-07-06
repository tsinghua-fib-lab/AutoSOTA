from typing import Iterable, Optional, Union, Callable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize as P
from torch.nn.utils.parametrizations import _SpectralNorm
import torch.linalg as linalg


__all__ = ["SpectralNormToConstant", "SimpleResidualBlock", "ResidualNetMLP", "Identity"]


class SpectralNormToConstant(_SpectralNorm):
    # override
    def __init__(
        self,
        weight: torch.Tensor,
        sn_ub: float = 1.,   # spectral norm upper bound
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:
        self.sn_ub = sn_ub
        super().__init__(weight, n_power_iterations, dim, eps)

    # override
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            # See above on why we need to clone
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            # The proper way of computing this should be through F.bilinear, but
            # it seems to have some efficiency issues:
            # https://github.com/pytorch/pytorch/issues/58093
            sigma = torch.vdot(u, torch.mv(weight_mat, v))   # original code
            # sigma = torch.dot(u, torch.matmul(weight_mat, v))   # for MPS
            if sigma < self.sn_ub:
                return weight
            return (self.sn_ub * weight) / sigma


class SimpleResidualBlock(nn.Module):
    """
    f(x) = a(Wx + b) + x
    """
    def __init__(self, feature_dim: int, spectral_norm: bool = True, sn_ub: float = 1.):
        super().__init__()
        self.linear = nn.Linear(feature_dim, feature_dim)
        if spectral_norm:   # if apply spectral normalisation for weight params.
            weight = getattr(self.linear, "weight", None)
            P.register_parametrization(
                self.linear, "weight", SpectralNormToConstant(weight, sn_ub=sn_ub)
            )
        self.activation = nn.ReLU()

    def forward(self, x):
        return x + self.activation(self.linear(x))


class ResidualNetMLP(nn.Module):
    def __init__(self, feature_dim: int, num_blocks: int, spectral_norm: bool = True, sn_ub: float = 1.):
        super().__init__()
        self.blocks = nn.Sequential(
            *[SimpleResidualBlock(feature_dim, spectral_norm, sn_ub) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)


class Identity(nn.Module):
    """A no-op layer that returns its input unchanged.

    Example
    -------
    >>> layer = Identity()
    >>> x = torch.randn(2, 3)
    >>> torch.equal(layer(x), x)
    True
    """
    __slots__ = ()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FCNetMLP(nn.Module):
    """Fully connected neural network (MLP)"""
    def __init__(
        self, in_dim: int, out_dim: int, hidden_dims: Iterable[int], spectral_norm: bool = True, sn_ub_per_layer: float = 1.
    ):
        super().__init__()
        hidden_dims = list(hidden_dims)
        self.activation = nn.ReLU()
        self.layers = nn.ModuleList()

        dims = [in_dim] + hidden_dims + [out_dim]
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i + 1])
            self.layers.append(layer)

        if spectral_norm:
            for layer in self.layers:
                weight = getattr(layer, "weight", None)
                P.register_parametrization(
                    layer, "weight", SpectralNormToConstant(weight, sn_ub=sn_ub_per_layer)
                )

        self._num_linear = len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self._num_linear - 1:
                x = self.activation(x)
        return x


if __name__ == "__main__":
    # from torch.nn.utils.parametrizations import spectral_norm
    feature_dim, num_blocks = 5, 1

    x1 = torch.randn(32, 128, feature_dim)
    x2 = torch.randn(128, feature_dim)

    resnet = ResidualNetMLP(feature_dim, num_blocks, spectral_norm=True, sn_ub=1.0)
    resnet.eval()

    with P.cached(), torch.no_grad():
        y1 = resnet(x1)
        y2 = resnet(x2)

    print(y1.shape, y2.shape)

    print(torch.linalg.matrix_norm(resnet.blocks[0].linear.weight, 2))

    fcnet = FCNetMLP(
        in_dim=feature_dim, out_dim=feature_dim, hidden_dims=[16, 32, 16], spectral_norm=True, sn_ub_per_layer=1.0
    )
    fcnet.eval()

    with P.cached(), torch.no_grad():
        y1 = fcnet(x1)
        y2 = fcnet(x2)

    print(y1.shape, y2.shape)

    print(torch.linalg.matrix_norm(fcnet.layers[1].weight, 2))
