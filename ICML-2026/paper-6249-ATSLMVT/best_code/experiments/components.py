"""Reusable experiment components for Hydra-driven scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np

from active_wasserstein import (
    AcquiredMeasurement,
    UncertaintySampler,
    wasserstein_barycenter,
)
from active_wasserstein.acquisition.velocity_uncertainty import (
    VelocityWeightedUncertaintySampler,
)
from active_wasserstein.measures import ProbabilityMeasure

MeasurementOracle = Callable[[float], AcquiredMeasurement]


class AcquisitionFunction(Protocol):
    def optimize(
        self, posterior: object, candidates: Iterable[float]
    ) -> tuple[float, np.ndarray]: ...


def make_measurement_oracle(
    trajectory: object,
    sample_size: int | None,
    sample_method: str = "sample",
    rng: np.random.Generator | None = None,
) -> MeasurementOracle:
    """Build a measurement oracle from a trajectory sampler."""
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be positive")
    rng = rng or np.random.default_rng()
    if not hasattr(trajectory, sample_method):
        raise AttributeError(f"trajectory has no method '{sample_method}'")
    sampler = getattr(trajectory, sample_method)

    def oracle(time: float) -> AcquiredMeasurement:
        measure = sampler(float(time), sample_size, rng=rng)
        resolved_sample_size = sample_size
        if resolved_sample_size is None:
            if not hasattr(measure, "support"):
                raise ValueError("sample_size is None but measure has no support")
            resolved_sample_size = int(np.asarray(measure.support).shape[0])
        if resolved_sample_size <= 0:
            raise ValueError("resolved sample_size must be positive")
        return AcquiredMeasurement(
            time=float(time),
            measure=measure,
            sample_size=int(resolved_sample_size),
        )

    return oracle


def build_reference_from_measurements(
    measurements: Sequence[AcquiredMeasurement],
    barycenter_size: int = 256,
    num_iter: int = 150,
    reg: float = 0.0,
    rng: np.random.Generator | None = None,
    backend: str = "pot",
) -> ProbabilityMeasure:
    """Build a Wasserstein barycenter reference from measurements."""
    if not measurements:
        raise ValueError("must provide at least one measurement")
    rng = rng or np.random.default_rng()
    measures = [m.measure for m in measurements]
    return wasserstein_barycenter(
        measures,
        barycenter_size=barycenter_size,
        num_iter=num_iter,
        reg=float(reg),
        rng=rng,
        backend=backend,
    )


def make_uncertainty_acquisition(
    velocity_weights=None,
    velocity_times=None,
    velocity_power=1.0,
) -> AcquisitionFunction:
    """Return an acquisition function based on posterior uncertainty.
    
    When velocity_weights is provided, uses velocity-weighted sampling.
    """
    if velocity_weights is not None:
        return VelocityWeightedUncertaintySampler(
            velocity_weights=velocity_weights,
            velocity_times=velocity_times,
            velocity_power=velocity_power,
        )
    return UncertaintySampler()


@dataclass
class RandomAcquisition:
    """Select candidates uniformly at random."""

    rng: np.random.Generator

    def optimize(self, _: object, candidates: Iterable[float]) -> tuple[float, np.ndarray]:
        candidates_arr = np.asarray(list(candidates), dtype=float)
        if candidates_arr.size == 0:
            raise ValueError("must provide at least one candidate time")
        scores = self.rng.random(candidates_arr.shape[0])
        idx = int(np.argmax(scores))
        return float(candidates_arr[idx]), scores


def make_velocity_weighted_acquisition(
    velocity_weights=None,
    velocity_times=None,
    velocity_power=1.0,
):
    return VelocityWeightedUncertaintySampler(
        velocity_weights=velocity_weights,
        velocity_times=velocity_times,
        velocity_power=velocity_power,
    )


def make_random_acquisition(
    rng: np.random.Generator | None = None,
) -> AcquisitionFunction:
    """Return an acquisition function that selects candidates uniformly at random."""
    rng = rng or np.random.default_rng()
    return RandomAcquisition(rng=rng)


@dataclass
class UniformScheduleAcquisition:
    """Return candidates in a fixed time schedule."""

    schedule: Sequence[float]
    atol: float = 1e-12
    _idx: int = 0

    def optimize(self, _: object, candidates: Iterable[float]) -> tuple[float, np.ndarray]:
        candidates_arr = np.asarray(list(candidates), dtype=float)
        if candidates_arr.size == 0:
            raise ValueError("must provide at least one candidate time")
        if self._idx >= len(self.schedule):
            scores = np.zeros_like(candidates_arr)
            return float(candidates_arr[0]), scores
        target = float(self.schedule[self._idx])
        deltas = np.abs(candidates_arr - target)
        selected = int(np.argmin(deltas))
        scores = np.zeros_like(candidates_arr)
        scores[selected] = 1.0
        self._idx += 1
        return float(candidates_arr[selected]), scores


def make_uniform_schedule_acquisition(
    schedule: Sequence[float],
    atol: float = 1e-12,
) -> AcquisitionFunction:
    """Return an acquisition function that follows a fixed time schedule."""
    return UniformScheduleAcquisition(
        schedule=[float(t) for t in schedule],
        atol=atol,
    )
