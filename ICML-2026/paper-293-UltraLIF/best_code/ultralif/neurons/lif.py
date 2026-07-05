# -*- coding: utf-8 -*-
"""
Standard LIF neuron variants (baselines).

Classes:
    LIF: Leaky Integrate-and-Fire with fixed parameters.
    PLIF: Parametric LIF with learnable membrane time constant.
    AdaLIF: Adaptive-threshold LIF (Bellec et al. NeurIPS 2018).
    FullPLIF: Parametric LIF with learnable tau and threshold.
"""

import torch
import torch.nn as nn


class LIF(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with sigmoid surrogate gradient.

    Membrane dynamics:
        V(t) = tau * V(t-1) + I(t)
        s(t) = sigmoid(beta * (V(t) - thresh))
        V(t) <- V(t) * (1 - s(t))   [soft reset]

    Args:
        dim: Number of neurons.
        tau: Membrane time constant (fixed).
        thresh: Spike threshold.
        beta: Sigmoid sharpness (surrogate gradient temperature).

    Example:
        >>> neuron = LIF(dim=256)
        >>> neuron.reset(batch_size=32, device='cpu')
        >>> spike = neuron(torch.randn(32, 256))
    """

    def __init__(self, dim: int, tau: float = 0.9, thresh: float = 0.5, beta: float = 10.0):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.thresh = thresh
        self.beta = beta
        self.v = None

    def reset(self, b: int, d):
        self.v = torch.zeros(b, self.dim, device=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v = self.tau * self.v + x
        spike = torch.sigmoid(self.beta * (self.v - self.thresh))
        self.v = self.v * (1 - spike)
        return spike


class PLIF(nn.Module):
    """
    Parametric LIF with learnable membrane time constant.

    Same as LIF but tau is a learned parameter constrained to (0, 1) via sigmoid.

    Args:
        dim: Number of neurons.
        init_tau: Initial value for tau.
        thresh: Spike threshold.
        beta: Sigmoid sharpness.

    Example:
        >>> neuron = PLIF(dim=256)
        >>> neuron.reset(32, 'cpu')
        >>> spike = neuron(torch.randn(32, 256))
        >>> print(neuron.tau.item())   # learned tau
    """

    def __init__(self, dim: int, init_tau: float = 0.9, thresh: float = 0.5, beta: float = 10.0):
        super().__init__()
        self.dim = dim
        self.thresh = thresh
        self.beta = beta
        self._tau = nn.Parameter(torch.tensor(init_tau))
        self.v = None

    @property
    def tau(self) -> torch.Tensor:
        return torch.sigmoid(self._tau)

    def reset(self, b: int, d):
        self.v = torch.zeros(b, self.dim, device=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v = self.tau * self.v + x
        spike = torch.sigmoid(self.beta * (self.v - self.thresh))
        self.v = self.v * (1 - spike)
        return spike


class AdaLIF(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire (LSNN, Bellec et al. NeurIPS 2018).

    The firing threshold adapts dynamically: it rises after each spike and
    decays back to the baseline with time constant tau_adapt.

        B(t) = b0 + beta_adapt * b(t)
        b(t+1) = rho * b(t) + (1 - rho) * s(t)

    Args:
        dim: Number of neurons.
        tau: Membrane time constant.
        base_thresh: Baseline threshold b0.
        beta_adapt: Adaptation strength (beta in paper).
        tau_adapt: Adaptation decay (rho = exp(-dt/tau_a)).
        surrogate_beta: Sigmoid sharpness.

    Example:
        >>> neuron = AdaLIF(dim=256)
        >>> neuron.reset(32, 'cpu')
        >>> spike = neuron(torch.randn(32, 256))
    """

    def __init__(
        self,
        dim: int,
        tau: float = 0.9,
        base_thresh: float = 0.5,
        beta_adapt: float = 0.1,
        tau_adapt: float = 0.9,
        surrogate_beta: float = 10.0,
    ):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.base_thresh = base_thresh
        self.beta_adapt = beta_adapt
        self.rho = tau_adapt
        self.surrogate_beta = surrogate_beta
        self.v = None
        self.b = None

    def reset(self, batch: int, device):
        self.v = torch.zeros(batch, self.dim, device=device)
        self.b = torch.zeros(batch, self.dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v = self.tau * self.v + x
        thresh = self.base_thresh + self.beta_adapt * self.b
        spike = torch.sigmoid(self.surrogate_beta * (self.v - thresh))
        self.b = self.rho * self.b + (1 - self.rho) * spike
        self.v = self.v * (1 - spike)
        return spike


class FullPLIF(nn.Module):
    """
    Parametric LIF with both learnable tau and learnable threshold.

    Args:
        dim: Number of neurons.
        init_tau: Initial membrane time constant.
        init_thresh: Initial spike threshold.
        beta: Sigmoid sharpness.

    Example:
        >>> neuron = FullPLIF(dim=256)
        >>> neuron.reset(32, 'cpu')
        >>> spike = neuron(torch.randn(32, 256))
    """

    def __init__(
        self,
        dim: int,
        init_tau: float = 0.9,
        init_thresh: float = 0.5,
        beta: float = 10.0,
    ):
        super().__init__()
        self.dim = dim
        self.beta = beta
        self._tau = nn.Parameter(torch.tensor(init_tau))
        self._thresh = nn.Parameter(torch.tensor(init_thresh))
        self.v = None

    @property
    def tau(self) -> torch.Tensor:
        return torch.sigmoid(self._tau)

    @property
    def thresh(self) -> torch.Tensor:
        return torch.sigmoid(self._thresh)

    def reset(self, b: int, d):
        self.v = torch.zeros(b, self.dim, device=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v = self.tau * self.v + x
        spike = torch.sigmoid(self.beta * (self.v - self.thresh))
        self.v = self.v * (1 - spike)
        return spike
