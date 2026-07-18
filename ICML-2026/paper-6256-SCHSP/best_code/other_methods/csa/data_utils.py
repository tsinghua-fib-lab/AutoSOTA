"""Minimal data helpers required by the CCA implementation."""

from __future__ import annotations

import numpy as np


def origin_centered(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-mean data and the mean used for centering."""
    mean = data.mean(axis=0)
    return data - mean, mean
