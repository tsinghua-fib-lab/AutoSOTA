# -*- coding: utf-8 -*-
"""
UltraDLIF spatial spiking neurons.

Derived from ultradiscretization of a 1D diffusion PDE discretized on a
spatial grid. The membrane update applies 3-term LSE over the left, center,
and right neighbors:

    V_i(t+1) = LSE_eps(V_{i-1}(t), V_i(t), V_{i+1}(t)) + I_i(t)

where neighbors are accessed via torch.roll (periodic boundary).

Classes:
    UltraDLIF:  Fixed tau.  (paper: UltraDLIF)
    UltraDPLIF: Learnable tau.  (paper: UltraDPLIF)

Paper: "UltraLIF: Fully Differentiable Spiking Neural Networks via
Ultradiscretization" (ICML 2026). arXiv:2602.11206
"""

import torch
import torch.nn as nn


class UltraDLIF(nn.Module):
    """
    Ultradiscretized spatial (diffusion) LIF neuron — fixed tau.

    Membrane update via 3-term spatial LSE:
        V(t+1) = tau * V(t) + I(t)                   [linear step]
        V_i'   = LSE_eps(V_{i-1}, V_i, V_{i+1})      [spatial smoothing]
        s(t)   = sigmoid(V'(t) / eps)
        V(t+1) <- V(t+1) * (1 - s(t))                [soft reset]

    The 3-term LSE implements spatial diffusion: each neuron's membrane
    potential is influenced by its neighbors, derived from the max-plus
    discretization of the diffusion PDE.

    Args:
        dim: Number of neurons.
        tau: Membrane time constant (fixed).
        init_eps: Initial eps value.

    Example:
        >>> neuron = UltraDLIF(dim=256)
        >>> neuron.reset(batch_size=32, device='cpu')
        >>> spike = neuron(torch.randn(32, 256))
    """

    def __init__(self, dim: int, tau: float = 0.9, init_eps: float = 1.0):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self._log_eps = nn.Parameter(torch.tensor(float(init_eps)).log())
        self.v = None

    @property
    def eps(self) -> torch.Tensor:
        return self._log_eps.exp().clamp(0.1, 20.0)

    def reset(self, b: int, d):
        self.v = torch.zeros(b, self.dim, device=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v = self.tau * self.v + x
        # 3-term spatial LSE over (left, center, right) neighbors
        v_l = torch.roll(self.v, 1, -1)
        v_r = torch.roll(self.v, -1, -1)
        stack = torch.stack([v_l, self.v, v_r], dim=-1)
        eps = self.eps
        m = stack.max(-1, keepdim=True).values
        v_max = m.squeeze(-1) + eps * torch.logsumexp((stack - m) / eps, dim=-1)
        spike = torch.sigmoid(v_max / eps)
        self.v = self.v * (1 - spike)
        return spike


class UltraDPLIF(nn.Module):
    """
    Ultradiscretized spatial (diffusion) LIF neuron with learnable tau.

    Same as UltraDLIF but tau is a learnable parameter constrained to (0, 1).

    Args:
        dim: Number of neurons.
        init_tau: Initial membrane time constant.
        init_eps: Initial eps value.

    Example:
        >>> neuron = UltraDPLIF(dim=256)
        >>> neuron.reset(32, 'cpu')
        >>> spike = neuron(torch.randn(32, 256))
        >>> print(f"tau={neuron.tau.item():.3f}, eps={neuron.eps.item():.3f}")
    """

    def __init__(self, dim: int, init_tau: float = 0.9, init_eps: float = 1.0):
        super().__init__()
        self.dim = dim
        self._tau = nn.Parameter(torch.tensor(init_tau))
        self._log_eps = nn.Parameter(torch.tensor(float(init_eps)).log())
        self.v = None

    @property
    def tau(self) -> torch.Tensor:
        return torch.sigmoid(self._tau)

    @property
    def eps(self) -> torch.Tensor:
        return self._log_eps.exp().clamp(0.1, 20.0)

    def reset(self, b: int, d):
        self.v = torch.zeros(b, self.dim, device=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v = self.tau * self.v + x
        v_l = torch.roll(self.v, 1, -1)
        v_r = torch.roll(self.v, -1, -1)
        stack = torch.stack([v_l, self.v, v_r], dim=-1)
        eps = self.eps
        m = stack.max(-1, keepdim=True).values
        v_max = m.squeeze(-1) + eps * torch.logsumexp((stack - m) / eps, dim=-1)
        spike = torch.sigmoid(v_max / eps)
        self.v = self.v * (1 - spike)
        return spike
