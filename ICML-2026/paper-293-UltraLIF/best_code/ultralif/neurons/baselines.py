# -*- coding: utf-8 -*-
"""
Differentiable baseline spiking neurons for comparison.

Classes:
    DSpike:     Differentiable Spike (Li et al. NeurIPS 2021).
    DSpikePlus: DSpike with learnable tau (fair comparison to UltraPLIF).
    SigmaLIF:   Sigmoid-only baseline (ablation — isolates LSE contribution).
"""

import torch
import torch.nn as nn


def dspike_fn(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    DSpike surrogate (Li et al. NeurIPS 2021, Eq. 12).

    DSpike(x, b) = [tanh(b*(x-0.5)) + tanh(b/2)] / [2*tanh(b/2)]
    Clipped to [0, 1]. Normalized so that DSpike(0.5, b) = 0.5.

    Args:
        x: Normalized membrane potential (ideally in [0, 1]).
        b: Temperature/sharpness parameter (b > 0).

    Returns:
        Spike probability in [0, 1].
    """
    tanh_b2 = torch.tanh(b / 2.0)
    spike = (torch.tanh(b * (x - 0.5)) + tanh_b2) / (2.0 * tanh_b2 + 1e-8)
    return spike.clamp(0.0, 1.0)


class DSpike(nn.Module):
    """
    Differentiable Spike neuron (Li et al. NeurIPS 2021).

    Uses a tanh-based smooth spike function with learnable temperature b.
    The membrane potential is normalized to [0, 1] before applying DSpike
    so that the threshold corresponds to x=0.5.

    Args:
        dim: Number of neurons.
        tau: Membrane time constant.
        init_b: Initial temperature parameter.
        thresh: Spike threshold (used for normalization).

    Example:
        >>> neuron = DSpike(dim=256)
        >>> neuron.reset(32, 'cpu')
        >>> spike = neuron(torch.randn(32, 256))
        >>> print(f"k={neuron.k.item():.2f}")
    """

    def __init__(self, dim: int, tau: float = 0.9, init_b: float = 4.0, thresh: float = 0.5):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.thresh = thresh
        self._b = nn.Parameter(torch.tensor(float(init_b)))
        self.v = None

    @property
    def k(self) -> torch.Tensor:
        """Temperature b, kept positive via softplus. Named k for compatibility."""
        return torch.nn.functional.softplus(self._b).clamp(0.5, 50.0)

    def reset(self, b: int, d):
        self.v = torch.zeros(b, self.dim, device=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v = self.tau * self.v + x
        x_norm = self.v / (2.0 * self.thresh)
        spike = dspike_fn(x_norm, self.k)
        self.v = self.v * (1 - spike)
        return spike


class DSpikePlus(nn.Module):
    """
    DSpike with learnable tau (fair comparison to UltraPLIF/UltraDPLIF).

    Args:
        dim: Number of neurons.
        init_tau: Initial membrane time constant.
        init_b: Initial temperature parameter.
        thresh: Spike threshold.

    Example:
        >>> neuron = DSpikePlus(dim=256)
        >>> neuron.reset(32, 'cpu')
        >>> spike = neuron(torch.randn(32, 256))
    """

    def __init__(
        self,
        dim: int,
        init_tau: float = 0.9,
        init_b: float = 4.0,
        thresh: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.thresh = thresh
        self._tau = nn.Parameter(torch.tensor(init_tau))
        self._b = nn.Parameter(torch.tensor(float(init_b)))
        self.v = None

    @property
    def tau(self) -> torch.Tensor:
        return torch.sigmoid(self._tau)

    @property
    def k(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._b).clamp(0.5, 50.0)

    def reset(self, b: int, d):
        self.v = torch.zeros(b, self.dim, device=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v = self.tau * self.v + x
        x_norm = self.v / (2.0 * self.thresh)
        spike = dspike_fn(x_norm, self.k)
        self.v = self.v * (1 - spike)
        return spike


class SigmaLIF(nn.Module):
    """
    Sigmoid-only LIF baseline (ablation control).

    Uses a standard linear LIF membrane update with a learnable sigmoid
    spike function identical to UltraLIF's:
        V(t+1) = tau * V(t) + I(t)          [linear, NOT LSE]
        s(t)   = sigmoid(V(t+1) / eps)

    If UltraLIF outperforms SigmaLIF, the gain comes from the LSE membrane
    dynamics (max-plus structure), not from using a soft sigmoid spike function.

    Args:
        dim: Number of neurons.
        tau: Fixed membrane time constant.
        init_eps: Initial eps (controls sigmoid sharpness).

    Example:
        >>> neuron = SigmaLIF(dim=256)
        >>> neuron.reset(32, 'cpu')
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
        # Standard linear membrane (NOT LSE — key difference from UltraLIF)
        self.v = self.tau * self.v + x
        spike = torch.sigmoid(self.v / self.eps)
        self.v = self.v * (1 - spike)
        return spike
