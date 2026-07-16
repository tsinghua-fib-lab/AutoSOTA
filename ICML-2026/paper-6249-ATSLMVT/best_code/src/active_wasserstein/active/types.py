"""Shared dataclasses and type aliases for active learning experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from active_wasserstein.inference import PredictiveProcess
from active_wasserstein.measures import ProbabilityMeasure


@dataclass
class AcquiredMeasurement:
    """Container describing a destructive acquisition at a given time."""

    time: float
    measure: ProbabilityMeasure
    sample_size: int
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.time = float(self.time)
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")


MeasurementOracle = Callable[[float], AcquiredMeasurement]


class AcquisitionFunction(Protocol):
    """Acquisition policy returning a selected time and candidate scores."""

    def optimize(
        self, posterior: PredictiveProcess, candidates: np.ndarray
    ) -> tuple[float, np.ndarray]: ...


@dataclass
class AcquisitionRecord:
    """Stores metadata for one iteration of the loop."""

    iteration: int
    selected_time: float
    score: float
    posterior: PredictiveProcess
    remaining_candidates: np.ndarray = field(repr=False)
