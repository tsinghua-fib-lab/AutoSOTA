# -*- coding: utf-8 -*-
"""
UltraLIF temporal spiking neurons — the main paper models.

Derived from ultradiscretization of the Euler-discretized LIF ODE:
    V(t+1) = tau * V(t) + I(t)

Ultradiscretization (replacing addition with LSE) gives:
    V(t+1) = LSE_eps(V(t) + log(tau), I(t))

where LSE_eps(a, b) = eps * log(exp(a/eps) + exp(b/eps)) is the smooth
max-plus (tropical) operator with temperature eps.

Classes:
    UltraLIF:    Fixed tau.  (paper: UltraLIF)
    UltraPLIF:   Learnable tau.  (paper: UltraPLIF)
    UltraLIF_DS: Fixed tau + disentangled spike scale.
    UltraPLIF_DS: Learnable tau + disentangled spike scale.

Paper: "UltraLIF: Fully Differentiable Spiking Neural Networks via
Ultradiscretization" (ICML 2026). arXiv:2602.11206
"""

import torch
import torch.nn as nn


class UltraLIF(nn.Module):
    """
    Ultradiscretized temporal LIF neuron (fixed tau).

    Membrane update via 2-term max-plus LSE:
        V(t+1) = LSE_eps(V(t) + log(tau), I(t))
        s(t)   = sigmoid(V(t+1) / eps)
        V(t+1) <- V(t+1) * (1 - s(t))   [soft reset]

    The learnable parameter eps controls the softness of the max-plus
    operator. As eps -> 0, LSE -> hard max (tropical limit). As eps -> inf,
    LSE -> softmax-weighted average.

    Args:
        dim: Number of neurons.
        tau: Membrane time constant in (0, 1).
        init_eps: Initial value for the learnable eps (>0).

    Example:
        >>> neuron = UltraLIF(dim=256)
        >>> neuron.reset(batch_size=32, device='cpu')
        >>> spike = neuron(torch.randn(32, 256))
        >>> print(f"eps={neuron.eps.item():.3f}")
    """

    def __init__(self, dim: int, tau: float = 0.9, init_eps: float = 1.0):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.log_tau = float(torch.tensor(tau).log())
        self._log_eps = nn.Parameter(torch.tensor(float(init_eps)).log())
        self.v = None

    @property
    def eps(self) -> torch.Tensor:
        return self._log_eps.exp().clamp(0.1, 20.0)

    def reset(self, b: int, d):
        self.v = torch.zeros(b, self.dim, device=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        # 2-term LSE: (V + log(tau), I)
        term1 = self.v + self.log_tau
        term2 = x
        stack = torch.stack([term1, term2], dim=-1)
        m = stack.max(-1, keepdim=True).values
        self.v = m.squeeze(-1) + eps * torch.logsumexp((stack - m) / eps, dim=-1)
        spike = torch.sigmoid(self.v / eps)
        self.v = self.v * (1 - spike)
        return spike


class UltraPLIF(nn.Module):
    """
    Ultradiscretized temporal LIF neuron with learnable tau.

    Same as UltraLIF but tau is a learnable parameter constrained to (0.01, 0.99).

    Args:
        dim: Number of neurons.
        init_tau: Initial membrane time constant.
        init_eps: Initial eps value.

    Example:
        >>> neuron = UltraPLIF(dim=256)
        >>> neuron.reset(32, 'cpu')
        >>> spike = neuron(torch.randn(32, 256))
        >>> print(f"tau={neuron.tau.item():.3f}, eps={neuron.eps.item():.3f}")
    """

    def __init__(self, dim: int, init_tau: float = 0.9, init_eps: float = 1.0):
        super().__init__()
        self.dim = dim
        self._log_tau = nn.Parameter(torch.tensor(float(init_tau)).log())
        self._log_eps = nn.Parameter(torch.tensor(float(init_eps)).log())
        self.v = None

    @property
    def tau(self) -> torch.Tensor:
        return self._log_tau.exp().clamp(0.01, 0.99)

    @property
    def eps(self) -> torch.Tensor:
        return self._log_eps.exp().clamp(0.1, 20.0)

    def reset(self, b: int, d):
        self.v = torch.zeros(b, self.dim, device=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        log_tau = self._log_tau
        term1 = self.v + log_tau
        term2 = x
        stack = torch.stack([term1, term2], dim=-1)
        m = stack.max(-1, keepdim=True).values
        self.v = m.squeeze(-1) + eps * torch.logsumexp((stack - m) / eps, dim=-1)
        spike = torch.sigmoid(self.v / eps)
        self.v = self.v * (1 - spike)
        return spike


class UltraLIF_DS(nn.Module):
    """
    UltraLIF with disentangled spike scale (ablation variant).

    The LSE membrane update uses eps (integration softness), while the
    spike function uses an independent learnable spike_scale:
        s(t) = sigmoid(spike_scale * V(t))

    Motivation: in standard UltraLIF, spike = sigmoid(V/eps) couples
    spike sharpness to LSE temperature. Here they are physically decoupled:
    eps = membrane integration noise, spike_scale = 1/sigma_theta
    (inverse spike threshold noise).

    Args:
        dim: Number of neurons.
        tau: Fixed membrane time constant.
        init_eps: Initial eps.
        init_spike_scale: Initial spike sharpness.
    """

    def __init__(
        self,
        dim: int,
        tau: float = 0.9,
        init_eps: float = 1.0,
        init_spike_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.log_tau = float(torch.tensor(tau).log())
        self._log_eps = nn.Parameter(torch.tensor(float(init_eps)).log())
        self._log_spike_scale = nn.Parameter(torch.tensor(float(init_spike_scale)).log())
        self.v = None

    @property
    def eps(self) -> torch.Tensor:
        return self._log_eps.exp().clamp(0.1, 20.0)

    @property
    def spike_scale(self) -> torch.Tensor:
        return self._log_spike_scale.exp().clamp(0.1, 50.0)

    def reset(self, b: int, d):
        self.v = torch.zeros(b, self.dim, device=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        term1 = self.v + self.log_tau
        term2 = x
        stack = torch.stack([term1, term2], dim=-1)
        m = stack.max(-1, keepdim=True).values
        self.v = m.squeeze(-1) + eps * torch.logsumexp((stack - m) / eps, dim=-1)
        spike = torch.sigmoid(self.spike_scale * self.v)
        self.v = self.v * (1 - spike)
        return spike


class UltraPLIF_DS(nn.Module):
    """
    UltraPLIF with disentangled spike scale (learnable tau + disentangled ablation variant).

    Combines learnable tau from UltraPLIF with the disentangled spike scale
    from UltraLIF_DS.

    Args:
        dim: Number of neurons.
        init_tau: Initial membrane time constant.
        init_eps: Initial eps.
        init_spike_scale: Initial spike sharpness.
    """

    def __init__(
        self,
        dim: int,
        init_tau: float = 0.9,
        init_eps: float = 1.0,
        init_spike_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self._log_tau = nn.Parameter(torch.tensor(float(init_tau)).log())
        self._log_eps = nn.Parameter(torch.tensor(float(init_eps)).log())
        self._log_spike_scale = nn.Parameter(torch.tensor(float(init_spike_scale)).log())
        self.v = None

    @property
    def tau(self) -> torch.Tensor:
        return self._log_tau.exp().clamp(0.01, 0.99)

    @property
    def eps(self) -> torch.Tensor:
        return self._log_eps.exp().clamp(0.1, 20.0)

    @property
    def spike_scale(self) -> torch.Tensor:
        return self._log_spike_scale.exp().clamp(0.1, 50.0)

    def reset(self, b: int, d):
        self.v = torch.zeros(b, self.dim, device=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        log_tau = self._log_tau
        term1 = self.v + log_tau
        term2 = x
        stack = torch.stack([term1, term2], dim=-1)
        m = stack.max(-1, keepdim=True).values
        self.v = m.squeeze(-1) + eps * torch.logsumexp((stack - m) / eps, dim=-1)
        spike = torch.sigmoid(self.spike_scale * self.v)
        self.v = self.v * (1 - spike)
        return spike
