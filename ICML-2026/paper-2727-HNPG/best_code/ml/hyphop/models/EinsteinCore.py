import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import geoopt.manifolds.lorentz.math as g


def to_tangent0(x: torch.Tensor) -> torch.Tensor:
    """
    x: (..., d) in Euclidean
    returns: (..., d+1) in T_0 H via [0,x]
    """
    zeros = torch.zeros(x.shape[:-1] + (1,), device=x.device, dtype=x.dtype)
    x = torch.cat([zeros, x], dim=-1)
    
    norm_x = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    factor = torch.clamp(3.5 / (norm_x + 1e-7), max=1.0)
    return x * factor


def expmap0(x_euc: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    x_euc: (..., d) in Euclidean
    returns: (..., d+1) on hyperboloid via Exp_0([0,x])
    """
    return g.expmap0(to_tangent0(x_euc), k=k)


def logmap0(x_man: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    x_man: (..., d+1) on hyperboloid
    returns: (..., d) in Euclidean tangent at 0
    """
    return g.logmap0(x_man, k=k)[..., 1:]


def _hyperboloid_to_klein(x: torch.Tensor) -> torch.Tensor:
    """
    (..., d+1) on hyperboloid -> (..., d) in Klein ball.
    Klein coords are simply the spatial components divided by the time component.
    """
    return x[..., 1:] / x[..., :1].clamp_min(1e-12)


def _klein_to_hyperboloid(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    (..., d) in Klein ball -> (..., d+1) on hyperboloid with curvature k.
    Hyperboloid constraint: -x0^2 + ||x||^2 = -1/k  =>  x0 = 1/sqrt(k*(1-||u||^2))
    """
    x_norm2 = (x * x).sum(dim=-1, keepdim=True).clamp(max=1.0 - 1e-6)
    x0 = 1.0 / torch.sqrt(k * (1.0 - x_norm2).clamp_min(1e-12))
    return torch.cat([x0, x0 * x], dim=-1)


class EinsteinCore(nn.Module):
    def __init__(
        self,
        state_dim: int,
        memory_dim: int,
        hopfield_dim: int,
        out_dim: int,
        beta: Optional[float] = None,
        init_curvature: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_h = hopfield_dim
        self.d_out = out_dim
        self.beta = beta if beta is not None else 1.0 / math.sqrt(hopfield_dim)
        self.dropout = dropout

        self.register_buffer("k", torch.tensor(float(init_curvature)))

        self.W_Q = nn.Linear(state_dim, hopfield_dim, bias=True)
        self.W_K = nn.Linear(memory_dim, hopfield_dim, bias=True)
        self.W_V = nn.Linear(memory_dim, out_dim, bias=True)

    def _einstein_midpoint(self, weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        weights: (B, S, M)
        values:  (B, M, d_out+1)
        returns: (B, S, d_out+1)
        """
        v_klein = _hyperboloid_to_klein(values)                                          # (B, M, d_out)
        v_norm2 = (v_klein * v_klein).sum(dim=-1, keepdim=True).clamp(max=1.0 - 1e-6)
        lorentz_gamma = 1.0 / torch.sqrt((1.0 - v_norm2).clamp_min(1e-12))               # (B, M, 1)

        w_tilde = weights.unsqueeze(-1) * lorentz_gamma.unsqueeze(1)                     # (B, S, M, 1)
        denom = w_tilde.sum(dim=2).clamp_min(1e-12)                                      # (B, S, 1)
        mid_klein = (w_tilde * v_klein.unsqueeze(1)).sum(dim=2) / denom                  # (B, S, d_out)

        mid_norm = torch.linalg.vector_norm(mid_klein, dim=-1, keepdim=True).clamp_min(1e-12)
        mid_klein = torch.where(
            mid_norm >= 1.0,
            mid_klein / mid_norm * (1.0 - 1e-6),
            mid_klein,
        )
        return _klein_to_hyperboloid(mid_klein, self.k)  # (B, S, d_out+1)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        queries: (B, S, state_dim)
        keys:    (B, M, memory_dim)
        values:  (B, M, memory_dim)
        """
        Qe = self.W_Q(queries)   # (B, S, d_h)
        Ke = self.W_K(keys)      # (B, M, d_h)
        Ve = self.W_V(values)    # (B, M, d_out)

        Q = expmap0(Qe, k=self.k)   # (B, S, d_h+1)
        K = expmap0(Ke, k=self.k)   # (B, M, d_h+1)
        V = expmap0(Ve, k=self.k)   # (B, M, d_out+1)

        dists = g.dist(Q.unsqueeze(2), K.unsqueeze(1), k=self.k)    # (B, S, M)
        alpha = F.softmax(-self.beta * dists, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        Z = self._einstein_midpoint(alpha, V)       # (B, S, out_dim+1)
        return logmap0(Z, k=self.k)                 # (B, S, out_dim)

