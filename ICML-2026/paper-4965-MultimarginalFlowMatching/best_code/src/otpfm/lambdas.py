"""Classes for time-localized lambda functions λ_k(t) for OT potentials."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class LambdaFunction(ABC):
    """Abstract base class for time-localized lambda functions λ_k(t) (Eq. 10 of :cite:t:`kansal2026otpfm`).

    Args:
        tk (float): central time.
        width (float): half-width of the function in time.
    """

    def __init__(self, tk: float, width: float | str | None):
        self.tk = tk
        self.width = width

        if width != "auto":
            self._precompute_special_values()

    def _precompute_special_values(self):
        """Precompute integrated and double_integrated at t=1 for efficiency."""
        pass

    def set_width(self, width: float | str):
        self.width = width
        if width != "auto":
            self._precompute_special_values()

    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor:
        """λ_k(t)"""

    @abstractmethod
    def integrated(self, t: Tensor) -> Tensor:
        """Λ(t) = ∫_0^t λ_k(s) ds"""

    @abstractmethod
    def double_integrated(self, t: Tensor) -> Tensor:
        """∫_0^t Λ(s) ds = ∫_0^t (∫_0^s λ_k(u) du) ds"""

    @abstractmethod
    def integrated_at_1(self) -> float:
        """Λ(1) - precomputed for efficiency"""

    @abstractmethod
    def double_integrated_at_1(self) -> float:
        """∫_0^1 Λ(s) ds - precomputed for efficiency"""


class DeltaLambda(LambdaFunction):
    """
    Implements Dirac delta λ(t) = δ(t - tk).

    Note: __call__ returns 0 since delta cannot be represented numerically,
    but integrated and double_integrated are well-defined.
    """

    def __call__(self, t: Tensor) -> Tensor:
        # Delta function cannot be represented numerically; return 0
        return torch.zeros_like(t)

    def integrated(self, t: Tensor) -> Tensor:
        """Λ(t) = H(t - tk) (Heaviside step function)"""
        return (t > self.tk).float()

    def double_integrated(self, t: Tensor) -> Tensor:
        """∫_0^t H(s - tk) ds = max(0, t - tk) (ramp function)"""
        return torch.clamp(t - self.tk, min=0.0)

    def integrated_at_1(self) -> float:
        """Λ(1) = H(1 - tk) = 1 for tk < 1"""
        return 1.0

    def double_integrated_at_1(self) -> float:
        """∫_0^1 H(s - tk) ds = 1 - tk"""
        return 1.0 - self.tk


class GaussianLambda(LambdaFunction):
    """
    Implements Gaussian lambda `N(t; tk, width^2)`.
    """

    def _precompute_special_values(self):
        """Precompute values at t=1 using scipy for accuracy."""
        import math

        sigma = self.width
        z0 = -self.tk / sigma
        z1 = (1.0 - self.tk) / sigma

        # Standard normal CDF and PDF (scalar versions)
        def Phi(z):
            return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        def phi(z):
            return math.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)

        # Integrated values: Λ(t) = Φ(z_t) - Φ(z_0)
        self._integrated_at_1 = Phi(z1) - Phi(z0)

        # Double integrated: (t - tk)·Λ(t) + σ·(φ(z_t) - φ(z_0))
        self._double_integrated_at_1 = (1.0 - self.tk) * self._integrated_at_1 + sigma * (
            phi(z1) - phi(z0)
        )

    def __call__(self, t: Tensor) -> Tensor:
        return torch.exp(-0.5 * ((t - self.tk) / self.width) ** 2) / (
            torch.sqrt(torch.tensor(2 * torch.pi, device=t.device, dtype=t.dtype)) * self.width
        )

    def _Phi(self, z: Tensor) -> Tensor:
        """Standard normal CDF"""
        return 0.5 * (
            1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, device=z.device, dtype=z.dtype)))
        )

    def _phi(self, z: Tensor) -> Tensor:
        """Standard normal PDF"""
        return torch.exp(-0.5 * z**2) / torch.sqrt(
            torch.tensor(2 * torch.pi, device=z.device, dtype=z.dtype)
        )

    def integrated(self, t: Tensor) -> Tensor:
        """Λ(t) = Φ(z_t) - Φ(z_0) where z_t = (t - tk) / σ"""
        zt = (t - self.tk) / self.width
        z0 = torch.tensor(-self.tk / self.width, device=t.device, dtype=t.dtype)
        return self._Phi(zt) - self._Phi(z0)

    def double_integrated(self, t: Tensor) -> Tensor:
        """
        ∫_0^t Λ(s) ds = (t - tk) · Λ(t) + σ · (φ(z_t) - φ(z_0))

        Derived using integration by parts.
        """
        sigma = self.width
        zt = (t - self.tk) / sigma
        z0 = torch.tensor(-self.tk / sigma, device=t.device, dtype=t.dtype)

        Lambda_t = self._Phi(zt) - self._Phi(z0)
        return (t - self.tk) * Lambda_t + sigma * (self._phi(zt) - self._phi(z0))

    def integrated_at_1(self) -> float:
        return self._integrated_at_1

    def double_integrated_at_1(self) -> float:
        return self._double_integrated_at_1


class TriangleLambda(LambdaFunction):
    """
    Implements Triangle lambda centered at ``tk`` with half-width ``width``.
    ``λ(t) = (1 - |t - tk| / h) / h`` for ``|t - tk| < h``, else 0.
    """

    def _precompute_special_values(self):
        """Precompute values at t=1."""
        h = self.width
        a = self.tk - h
        b = self.tk + h

        # At t = 1: check which region
        if 1.0 <= a:
            self._integrated_at_1 = 0.0
        elif 1.0 <= self.tk:
            u = 1.0 - a
            self._integrated_at_1 = (u * u) / (2.0 * h * h)
        elif 1.0 < b:
            u = 1.0 - self.tk
            self._integrated_at_1 = 0.5 + (u / h) - (u * u) / (2.0 * h * h)
        else:  # 1.0 >= b
            self._integrated_at_1 = 1.0

        # Double integrated at t = 1: check which region
        if 1.0 <= a:
            self._double_integrated_at_1 = 0.0
        elif 1.0 <= self.tk:
            u = 1.0 - a
            self._double_integrated_at_1 = (u**3) / (6.0 * h * h)
        elif 1.0 < b:
            u = 1.0 - self.tk
            self._double_integrated_at_1 = (
                h / 6.0 + 0.5 * u + (u**2) / (2.0 * h) - (u**3) / (6.0 * h * h)
            )
        else:  # 1.0 >= b
            self._double_integrated_at_1 = h + (1.0 - b)

    def __call__(self, t: Tensor) -> Tensor:
        val = 1.0 - (t - self.tk).abs() / self.width
        return torch.clamp(val, min=0.0) / self.width

    def integrated(self, t: Tensor) -> Tensor:
        """Λ(t) = ∫_0^t λ(s) ds"""
        h = self.width
        a = self.tk - h
        b = self.tk + h

        result = torch.zeros_like(t)

        # Region 1: a < t <= tk
        mask1 = (t > a) & (t <= self.tk)
        if mask1.any():
            u = t[mask1] - a
            result[mask1] = (u * u) / (2.0 * h * h)

        # Region 2: tk < t < b
        mask2 = (t > self.tk) & (t < b)
        if mask2.any():
            u = t[mask2] - self.tk
            result[mask2] = 0.5 + (u / h) - (u * u) / (2.0 * h * h)

        # Region 3: t >= b
        mask3 = t >= b
        result[mask3] = 1.0

        return torch.clamp(result, 0.0, 1.0)

    def double_integrated(self, t: Tensor) -> Tensor:
        """
        ∫_0^t Λ(s) ds computed piecewise.

        Let a = tk - h, b = tk + h.
        Region 0 (t <= a): 0
        Region 1 (a < t <= tk): (t-a)³ / (6h²)
        Region 2 (tk < t < b): h/6 + 0.5(t-tk) + (t-tk)²/(2h) - (t-tk)³/(6h²)
        Region 3 (t >= b): h/6 + 5h/6 + (t - b) = h + (t - b)
        """
        h = self.width
        a = self.tk - h
        b = self.tk + h

        result = torch.zeros_like(t)

        # Region 1: a < t <= tk
        mask1 = (t > a) & (t <= self.tk)
        if mask1.any():
            u = t[mask1] - a
            result[mask1] = (u**3) / (6.0 * h * h)

        # Region 2: tk < t < b
        mask2 = (t > self.tk) & (t < b)
        if mask2.any():
            u = t[mask2] - self.tk
            result[mask2] = h / 6.0 + 0.5 * u + (u**2) / (2.0 * h) - (u**3) / (6.0 * h * h)

        # Region 3: t >= b
        mask3 = t >= b
        result[mask3] = h + (t[mask3] - b)

        return result

    def integrated_at_1(self) -> float:
        return self._integrated_at_1

    def double_integrated_at_1(self) -> float:
        return self._double_integrated_at_1


class BoxLambda(LambdaFunction):
    """
    Implements Box function: constant within ``[tk - width, tk + width]``, 0 outside.
    ``λ(t) = 1/(2h)`` for ``|t - tk| <= h``, else 0.
    """

    def _precompute_special_values(self):
        """Precompute values at t=1."""
        h = self.width
        a = self.tk - h
        b = self.tk + h

        # At t = 1: overlap = min(1, b) - max(0, a)
        left_1 = max(0.0, a)
        right_1 = min(1.0, b)
        overlap_1 = max(0.0, right_1 - left_1)
        self._integrated_at_1 = overlap_1 / (2.0 * h)

        # Double integrated at t = 1: check region
        if 1.0 <= a:
            self._double_integrated_at_1 = 0.0
        elif 1.0 <= b:
            u = 1.0 - a
            self._double_integrated_at_1 = (u**2) / (4.0 * h)
        else:
            self._double_integrated_at_1 = h + (1.0 - b)

    def __call__(self, t: Tensor) -> Tensor:
        inside = (t - self.tk).abs() <= self.width
        norm = 1.0 / (2 * self.width)
        return norm * inside.float()

    def integrated(self, t: Tensor) -> Tensor:
        """Λ(t) = ∫_0^t λ(s) ds"""
        h = self.width
        a = self.tk - h
        b = self.tk + h

        # Compute overlap of [max(0, a), min(t, b)] with the box support
        left = torch.clamp(torch.tensor(a, device=t.device, dtype=t.dtype), min=0.0)
        right = torch.clamp(t, max=b)
        overlap = torch.clamp(right - left, min=0.0)
        return overlap / (2.0 * h)

    def double_integrated(self, t: Tensor) -> Tensor:
        """
        ∫_0^t Λ(s) ds computed piecewise.

        Let a = tk - h, b = tk + h.
        Region 0 (t <= a): 0
        Region 1 (a < t <= b): (t - a)² / (4h)
        Region 2 (t > b): h + (t - b)
        """
        h = self.width
        a = self.tk - h
        b = self.tk + h

        result = torch.zeros_like(t)

        # Region 1: a < t <= b
        mask1 = (t > a) & (t <= b)
        if mask1.any():
            u = t[mask1] - a
            result[mask1] = (u**2) / (4.0 * h)

        # Region 2: t > b
        mask2 = t > b
        if mask2.any():
            result[mask2] = h + (t[mask2] - b)

        return result

    def integrated_at_1(self) -> float:
        return self._integrated_at_1

    def double_integrated_at_1(self) -> float:
        return self._double_integrated_at_1


__all__ = [
    "LambdaFunction",
    "DeltaLambda",
    "GaussianLambda",
    "TriangleLambda",
    "BoxLambda",
]
