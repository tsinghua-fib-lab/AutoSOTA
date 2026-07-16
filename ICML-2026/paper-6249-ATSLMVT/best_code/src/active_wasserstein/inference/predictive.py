"""Protocols describing predictive GP interfaces."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PredictiveProcess(Protocol):
    """Minimal interface required by acquisition and metric layers."""

    def mean(self, t: float) -> np.ndarray: ...

    def marginal_variance(self, t: float) -> np.ndarray: ...

    def trace_uncertainty(self, t: float) -> float: ...
