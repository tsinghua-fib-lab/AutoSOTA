"""Uncertainty-sampling acquisition."""

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


class UncertaintySampler:
    """Rank time points by total posterior Wasserstein variance."""

    def score(self, posterior: PredictiveProcess, times: Iterable[float]) -> np.ndarray:
        times_arr = _as_numpy_1d(times)
        return np.asarray(
            [posterior.trace_uncertainty(float(t)) for t in times_arr],
            dtype=float,
        )

    def optimize(
        self, posterior: PredictiveProcess, candidates: Iterable[float]
    ) -> tuple[float, np.ndarray]:
        candidates_arr = _as_numpy_1d(candidates)
        if candidates_arr.size == 0:
            raise ValueError("must provide at least one candidate time")
        scores = self.score(posterior, candidates_arr)
        idx = int(np.argmax(scores))
        return float(candidates_arr[idx]), scores
