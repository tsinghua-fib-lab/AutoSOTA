"""Sliding-window segmentation (Section 3.2 of the paper)."""

from __future__ import annotations

import numpy as np

__all__ = ["sliding_window_segments"]


def sliding_window_segments(
    X: np.ndarray, L: int, S: int
) -> tuple[np.ndarray, np.ndarray]:
    """Partition a (T, d) series into overlapping segments of length ``L``.

    Parameters
    ----------
    X : (T, d) or (T,) array
        The input time series. 1-D input is promoted to a single channel.
    L : int
        Window length.
    S : int
        Window stride.

    Returns
    -------
    segments : (n, L, d) array
        Stacked segments, ``n = floor((T - L) / S) + 1``.
    starts : (n,) int array
        Start index of each segment in the original series (used by the
        point-wise aggregation, Algorithm 2).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    T = X.shape[0]
    if T < L:
        raise ValueError(f"Series length {T} is shorter than window L={L}.")
    starts = np.arange(0, T - L + 1, S, dtype=int)
    segments = np.stack([X[s : s + L] for s in starts], axis=0)
    return segments, starts
