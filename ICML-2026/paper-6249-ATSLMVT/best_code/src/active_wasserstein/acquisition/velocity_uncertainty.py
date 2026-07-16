"""Velocity-aware uncertainty-sampling acquisition for IDEA-01.

Create file: src/active_wasserstein/acquisition/velocity_uncertainty.py
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from active_wasserstein.inference import PredictiveProcess


def _as_numpy_1d(values: Iterable[float] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arr = values
    else:
        arr = np.asarray(list(values), dtype=float)
    arr = np.asarray(arr, dtype=float)
    return arr.reshape(-1)


class VelocityWeightedUncertaintySampler:
    """Rank time points by total posterior Wasserstein variance, weighted by velocity.

    When velocity_weights is None, behaves identically to UncertaintySampler.
    When provided, multiplies uncertainty score by normalized velocity weights
    to bias acquisition toward high-dynamic regions (branching events).
    """

    def __init__(
        self,
        velocity_weights: np.ndarray | None = None,
        velocity_times: np.ndarray | None = None,
        velocity_power: float = 1.0,
    ) -> None:
        self.velocity_weights = (
            np.asarray(velocity_weights, dtype=float).reshape(-1)
            if velocity_weights is not None
            else None
        )
        self.velocity_times = (
            np.asarray(velocity_times, dtype=float).reshape(-1)
            if velocity_times is not None
            else None
        )
        self.velocity_power = float(velocity_power)
        if self.velocity_weights is not None and self.velocity_times is None:
            raise ValueError("velocity_times required when velocity_weights provided")
        if self.velocity_weights is not None and self.velocity_times is not None:
            if self.velocity_weights.shape[0] != self.velocity_times.shape[0]:
                raise ValueError(
                    f"velocity_weights ({self.velocity_weights.shape[0]}) and "
                    f"velocity_times ({self.velocity_times.shape[0]}) must match"
                )

    def _velocity_weight_at(self, t: float) -> float:
        """Return velocity weight for time t via nearest-neighbor lookup."""
        if self.velocity_weights is None or self.velocity_times is None:
            return 1.0
        idx = int(np.argmin(np.abs(self.velocity_times - float(t))))
        weight = float(self.velocity_weights[idx])
        return max(weight, 1e-8) ** self.velocity_power

    def score(self, posterior: PredictiveProcess, times: Iterable[float]) -> np.ndarray:
        times_arr = _as_numpy_1d(times)
        base_scores = np.asarray(
            [posterior.trace_uncertainty(float(t)) for t in times_arr],
            dtype=float,
        )
        if self.velocity_weights is not None:
            v_weights = np.array(
                [self._velocity_weight_at(float(t)) for t in times_arr],
                dtype=float,
            )
            # Normalize velocity weights to have mean 1.0 to avoid changing score scale
            v_mean = float(np.mean(v_weights))
            if v_mean > 1e-12:
                v_weights = v_weights / v_mean
            return base_scores * v_weights
        return base_scores

    def optimize(
        self, posterior: PredictiveProcess, candidates: Iterable[float]
    ) -> tuple[float, np.ndarray]:
        candidates_arr = _as_numpy_1d(candidates)
        if candidates_arr.size == 0:
            raise ValueError("must provide at least one candidate time")
        scores = self.score(posterior, candidates_arr)
        idx = int(np.argmax(scores))
        return float(candidates_arr[idx]), scores
