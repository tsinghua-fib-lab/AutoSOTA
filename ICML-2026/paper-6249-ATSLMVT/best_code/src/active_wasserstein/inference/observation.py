"""Observation models for tangent coefficients inferred from samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


NoiseInitializer = Callable[[np.ndarray, int], float]


def variance_scaled_noise_initializer(
    scale: float,
    *,
    ddof: int = 1,
    min_variance: float = 1.0e-12,
) -> NoiseInitializer:
    """Return a per-basis noise initializer based on scaled residual variance."""

    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be positive")
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    if min_variance <= 0:
        raise ValueError("min_variance must be positive")

    def _initializer(residuals: np.ndarray, basis_index: int) -> float:
        arr = np.asarray(residuals, dtype=float).reshape(-1)
        if arr.size == 0:
            variance = 0.0
        elif arr.size == 1 or arr.size <= ddof:
            variance = float(arr[0] ** 2)
        else:
            variance = float(np.var(arr, ddof=ddof))
        variance = max(variance, float(min_variance))
        return float(scale * variance)

    return _initializer


@dataclass
class TangentObservation:
    """Basis coefficients describing the empirical tangent vector at time t."""

    time: float
    coefficients: np.ndarray

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=float)
        if self.coefficients.ndim != 1:
            raise ValueError("coefficients must be 1-D")


@dataclass
class TangentObservationModel:
    """Homoskedastic noise surrogate for tangent coefficients."""

    base_variance: float

    def __post_init__(self) -> None:
        self.base_variance = float(self.base_variance)
        if self.base_variance <= 0:
            raise ValueError("base_variance must be positive")

    def noise_from_samples(self, n: int) -> float:
        if n <= 0:
            raise ValueError("sample size must be positive")
        return float(self.base_variance)

    def build_observation(
        self, time: float, coefficients: Sequence[float], sample_size: int
    ) -> TangentObservation:
        coeffs = np.asarray(coefficients, dtype=float)
        _ = self.noise_from_samples(sample_size)
        return TangentObservation(time=time, coefficients=coeffs)
