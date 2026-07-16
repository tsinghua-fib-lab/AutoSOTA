"""Discrete measures with explicit atom weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .base import ProbabilityMeasure


@dataclass
class WeightedEmpiricalMeasure(ProbabilityMeasure):
    """Atomic measure with fixed support and non-uniform weights."""

    support: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 1)))
    weights: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0,)))
    dimension: int = field(init=False, default=0)
    name: str = field(init=False, default="weighted_empirical")

    def __post_init__(self) -> None:
        support = np.asarray(self.support, dtype=float)
        weights = np.asarray(self.weights, dtype=float).reshape(-1)
        if support.ndim != 2:
            raise ValueError("support must have shape (n, d)")
        if support.shape[0] == 0:
            raise ValueError("support must contain at least one point")
        if weights.shape[0] != support.shape[0]:
            raise ValueError("weights must match support length")
        if np.any(weights < 0):
            raise ValueError("weights must be nonnegative")
        total = float(np.sum(weights))
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "weights", weights / total)
        object.__setattr__(self, "dimension", support.shape[1])

    def sample(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive")
        rng = rng or np.random.default_rng()
        idx = rng.choice(self.support.shape[0], size=int(n), replace=True, p=self.weights)
        return self.support[idx]
