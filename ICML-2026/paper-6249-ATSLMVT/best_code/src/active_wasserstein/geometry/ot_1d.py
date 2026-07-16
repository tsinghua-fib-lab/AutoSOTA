"""Fast exact 1D optimal transport helpers."""

from __future__ import annotations

import os

import numpy as np

_TRUE_VALUES = {"1", "true", "yes", "on"}


def fast_w2_1d_enabled() -> bool:
    """Return whether the opt-in fast 1D OT path is enabled."""
    value = os.getenv("FAST_W2_1D", "0")
    return str(value).strip().lower() in _TRUE_VALUES


def _sorted_points_and_weights(
    points: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(points, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("points must be non-empty")
    if x.shape[0] != w.shape[0]:
        raise ValueError("points and weights must have matching lengths")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    total = float(np.sum(w))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    order = np.argsort(x, kind="mergesort")
    return x[order], (w[order] / total), order


def _solve_sorted_monotone_coupling(
    x: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Solve 1D monotone coupling on sorted atoms.

    Returns W2^2 and the transported first moment for each source atom.
    """
    rem_a = a.copy()
    rem_b = b.copy()
    pushed_moment = np.zeros_like(x)
    w2_sq = 0.0
    i = 0
    j = 0
    eps = 1e-15
    while i < rem_a.size and j < rem_b.size:
        if rem_a[i] <= eps:
            i += 1
            continue
        if rem_b[j] <= eps:
            j += 1
            continue
        mass = rem_a[i] if rem_a[i] < rem_b[j] else rem_b[j]
        d = x[i] - y[j]
        w2_sq += float(mass * d * d)
        pushed_moment[i] += float(mass * y[j])
        rem_a[i] -= mass
        rem_b[j] -= mass
    return float(max(0.0, w2_sq)), pushed_moment


def weighted_w2_squared_1d(
    points1: np.ndarray,
    weights1: np.ndarray,
    points2: np.ndarray,
    weights2: np.ndarray,
) -> float:
    """Compute exact weighted W2^2 between 1D empirical measures."""
    x, a, _ = _sorted_points_and_weights(points1, weights1)
    y, b, _ = _sorted_points_and_weights(points2, weights2)
    w2_sq, _ = _solve_sorted_monotone_coupling(x, a, y, b)
    return float(w2_sq)


def barycentric_displacement_1d(
    source_points: np.ndarray,
    source_weights: np.ndarray,
    target_points: np.ndarray,
    target_weights: np.ndarray,
) -> np.ndarray:
    """Compute exact 1D OT barycentric displacements for source atoms."""
    x, a, source_order = _sorted_points_and_weights(source_points, source_weights)
    y, b, _ = _sorted_points_and_weights(target_points, target_weights)
    _, pushed_moment = _solve_sorted_monotone_coupling(x, a, y, b)

    barycenter_target = x.copy()
    nonzero = a > 1e-15
    barycenter_target[nonzero] = pushed_moment[nonzero] / a[nonzero]

    displacements_sorted = (barycenter_target - x).reshape(-1, 1)
    displacements = np.zeros_like(displacements_sorted)
    displacements[source_order] = displacements_sorted
    return displacements


def quantile_barycenter_support_1d(
    point_sets: list[np.ndarray],
    weight_sets: list[np.ndarray],
    barycenter_weights: np.ndarray | None,
    n_support: int,
) -> np.ndarray:
    """Compute a 1D W2 barycenter support via quantile averaging."""
    if n_support <= 0:
        raise ValueError("n_support must be positive")
    if len(point_sets) == 0:
        raise ValueError("point_sets must be non-empty")
    if len(point_sets) != len(weight_sets):
        raise ValueError("point_sets and weight_sets must have matching length")

    n_measures = len(point_sets)
    if barycenter_weights is None:
        lambdas = np.full(n_measures, 1.0 / float(n_measures), dtype=float)
    else:
        lambdas = np.asarray(barycenter_weights, dtype=float).reshape(-1)
        if lambdas.shape[0] != n_measures:
            raise ValueError("barycenter_weights must match number of measures")
        if np.any(lambdas < 0):
            raise ValueError("barycenter_weights must be nonnegative")
        total = float(np.sum(lambdas))
        if total <= 0.0:
            raise ValueError("barycenter_weights must sum to a positive value")
        lambdas = lambdas / total

    u = (np.arange(n_support, dtype=float) + 0.5) / float(n_support)
    bary_support = np.zeros(n_support, dtype=float)

    for i, (points, weights) in enumerate(zip(point_sets, weight_sets, strict=True)):
        x, a, _ = _sorted_points_and_weights(points, weights)
        cdf = np.cumsum(a)
        idx = np.searchsorted(cdf, u, side="left")
        idx = np.clip(idx, 0, x.size - 1)
        bary_support += float(lambdas[i]) * x[idx]

    return bary_support
