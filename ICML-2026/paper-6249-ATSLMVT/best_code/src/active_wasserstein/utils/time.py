"""Time-grid helper for integration and acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class TimeGrid:
    """Uniform time grid used for quadrature."""

    start: float
    end: float
    num: int

    def __post_init__(self) -> None:
        if not self.end > self.start:
            raise ValueError("end must be larger than start")
        if self.num <= 1:
            raise ValueError("num must be greater than 1")

    def values(self) -> np.ndarray:
        return np.linspace(self.start, self.end, self.num)

    def refine(self, factor: int) -> "TimeGrid":
        if factor <= 1:
            raise ValueError("factor must be greater than 1")
        return TimeGrid(self.start, self.end, factor * (self.num - 1) + 1)
