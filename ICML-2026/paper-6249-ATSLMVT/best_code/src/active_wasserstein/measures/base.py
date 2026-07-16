"""Abstract base classes for probability measures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


ArrayLike = np.ndarray


@dataclass
class ProbabilityMeasure(ABC):
    """Lightweight abstraction for a probability measure on R^d."""

    dimension: int = field(init=False, default=0)
    name: str = field(init=False, default="measure")

    @abstractmethod
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> ArrayLike:
        """Draw n samples from the measure."""


@dataclass
class EmpiricalMeasure(ProbabilityMeasure):
    """Empirical (atomic) measure backed by a point cloud."""

    support: ArrayLike = field(repr=False, default_factory=lambda: np.zeros((0, 1)))
    dimension: int = field(init=False, default=0)
    name: str = field(init=False, default="empirical")

    def __post_init__(self) -> None:
        if self.support.ndim != 2:
            raise ValueError("support must have shape (n, d)")
        object.__setattr__(self, "dimension", self.support.shape[1])

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> ArrayLike:
        if self.support.size == 0:
            return np.zeros((0, self.dimension))
        rng = rng or np.random.default_rng()
        idx = rng.choice(self.support.shape[0], size=n, replace=True)
        return self.support[idx]
