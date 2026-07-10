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


class KFCore(nn.Module):
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
        self.W_V = nn.Linear(hopfield_dim, out_dim, bias=True)

    def _karcher_flow(
        self, weights: torch.Tensor, queries: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        weights: (B, S, M)
        queries: (B, S, d_out+1)
        values:  (B, M, d_out+1)
        returns: (B, S, d_out+1)
        """
        log_values = g.logmap(queries.unsqueeze(2), values.unsqueeze(1), k=self.k)  # (B,S,M,d_out+1)
        tangent_mean = torch.sum(weights.unsqueeze(-1) * log_values, dim=2)         # (B,S,d_out+1)
        z = g.expmap(queries, tangent_mean, k=self.k)
        return z

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        queries: (B, S, state_dim)
        keys:    (B, M, memory_dim)
        values:  (B, M, memory_dim)
        """
        Qe = self.W_Q(queries)           # (B,S,d_h)
        Ke = self.W_K(keys)              # (B,M,d_h)
        Ve = self.W_V(self.W_K(values))  # (B,M,d_out)

        Q = expmap0(Qe, k=self.k)  # (B,S,d_h+1)
        K = expmap0(Ke, k=self.k)  # (B,M,d_h+1)
        V = expmap0(Ve, k=self.k)  # (B,M,d_out+1)

        sims = g.inner(Q.unsqueeze(2), K.unsqueeze(1))  # (B,S,M)
        alpha = F.softmax(-self.beta * sims, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        Z_man = self._karcher_flow(alpha, Q, V)  # (B,S,d_out+1)
        return logmap0(Z_man, k=self.k)          # (B,S,d_out)

