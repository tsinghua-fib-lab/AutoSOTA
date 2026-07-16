"""Time warping based on cumulative Wasserstein arc-length."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import ot
from scipy.interpolate import PchipInterpolator

from active_wasserstein._warnings import suppress_pot_warnings
from active_wasserstein.geometry.ot_1d import (
    fast_w2_1d_enabled,
    weighted_w2_squared_1d,
)
from active_wasserstein.measures.base import EmpiricalMeasure, ProbabilityMeasure


def _make_strictly_increasing(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if arr.size == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    out = arr.copy()
    min_step = np.finfo(float).eps * max(1.0, float(np.max(np.abs(out))))
    for idx in range(1, out.size):
        if out[idx] <= out[idx - 1]:
            out[idx] = out[idx - 1] + min_step
    return out


def _support_from_measure(
    measure: ProbabilityMeasure,
    n_support: int,
    rng: Optional[np.random.Generator],
) -> np.ndarray:
    """Extract or sample support points from a measure."""
    if isinstance(measure, EmpiricalMeasure) and measure.support.size > 0:
        return measure.support
    rng = rng or np.random.default_rng()
    return measure.sample(n_support, rng=rng)


def _uniform_weights(n: int) -> np.ndarray:
    """Create uniform probability weights."""
    if n <= 0:
        raise ValueError("number of support points must be positive")
    return np.full(n, 1.0 / float(n))


def _normalize_weights(weights: np.ndarray, n_points: int) -> np.ndarray:
    arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.shape[0] != n_points:
        raise ValueError("weights must match number of support points")
    if np.any(arr < 0):
        raise ValueError("weights must be nonnegative")
    total = float(np.sum(arr))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return arr / total


def _downsample_points_and_weights(
    points: np.ndarray,
    weights: np.ndarray,
    n_support: int,
    rng: Optional[np.random.Generator],
) -> tuple[np.ndarray, np.ndarray]:
    n = int(points.shape[0])
    if n_support <= 0:
        raise ValueError("n_support must be positive")
    if n <= n_support:
        return points, weights
    rng = rng or np.random.default_rng()
    idx = rng.choice(n, size=int(n_support), replace=False, p=weights)
    sampled_points = points[idx]
    sampled_weights = weights[idx]
    sampled_weights = sampled_weights / float(np.sum(sampled_weights))
    return sampled_points, sampled_weights


def _support_and_weights_from_measure(
    measure: ProbabilityMeasure,
    n_support: int,
    rng: Optional[np.random.Generator],
    downsample: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    support = getattr(measure, "support", None)
    if support is not None:
        pts = np.asarray(support, dtype=float)
        if pts.ndim == 1:
            pts = pts[:, None]
        if pts.ndim == 2 and pts.shape[0] > 0:
            weights = getattr(measure, "weights", None)
            if weights is not None:
                try:
                    norm_weights = _normalize_weights(weights, pts.shape[0])
                    if downsample:
                        return _downsample_points_and_weights(
                            pts, norm_weights, n_support, rng
                        )
                    return pts, norm_weights
                except ValueError:
                    pass
            uniform = _uniform_weights(pts.shape[0])
            if downsample:
                return _downsample_points_and_weights(
                    pts,
                    uniform,
                    n_support,
                    rng,
                )
            return pts, uniform
    pts = _support_from_measure(measure, n_support, rng)
    return pts, _uniform_weights(pts.shape[0])


def _normalize_w2_backend_name(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized != "pot":
        raise ValueError(
            f"Unsupported Wasserstein distance backend {backend!r}. Expected 'pot'."
        )
    return normalized


def compute_wasserstein_distance(
    measure1: ProbabilityMeasure,
    measure2: ProbabilityMeasure,
    n_support: int = 256,
    reg: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    downsample: bool = False,
    *,
    backend: str = "pot",
) -> float:
    _normalize_w2_backend_name(backend)
    rng = rng or np.random.default_rng()
    pts1, a = _support_and_weights_from_measure(
        measure1,
        n_support,
        rng,
        downsample=downsample,
    )
    pts2, b = _support_and_weights_from_measure(
        measure2,
        n_support,
        rng,
        downsample=downsample,
    )

    if (
        reg <= 0.0
        and fast_w2_1d_enabled()
        and pts1.shape[1] == 1
        and pts2.shape[1] == 1
    ):
        w2_squared = weighted_w2_squared_1d(pts1[:, 0], a, pts2[:, 0], b)
        return float(np.sqrt(max(0.0, w2_squared)))

    cost = ot.dist(pts1, pts2, metric="sqeuclidean")

    with suppress_pot_warnings():
        if reg > 0:
            # Sinkhorn for regularized OT
            plan = ot.sinkhorn(a, b, cost, reg=reg)
            w2_squared = float(np.sum(plan * cost))
        else:
            # EMD for exact OT
            w2_squared = float(ot.emd2(a, b, cost))

    return float(np.sqrt(max(0.0, w2_squared)))


class IdentityWarp:
    """Warp with identity forward/inverse transforms."""

    def __init__(
        self,
        times: np.ndarray | None = None,
        measures: Sequence[ProbabilityMeasure] | None = None,
    ) -> None:
        self.times = None if times is None else np.asarray(times, dtype=float)

    def forward(self, t: float | np.ndarray) -> float | np.ndarray:
        return t

    def inverse(self, s: float | np.ndarray) -> float | np.ndarray:
        return s

    def velocity(self, t: float | np.ndarray) -> float | np.ndarray:
        """Instantaneous warp velocity w'(t)."""
        if np.isscalar(t):
            return 1.0
        t_arr = np.atleast_1d(t)
        return np.ones_like(t_arr, dtype=float)

    @property
    def t_min(self) -> float:
        if self.times is None or len(self.times) == 0:
            return 0.0
        return float(self.times[0])

    @property
    def t_max(self) -> float:
        if self.times is None or len(self.times) == 0:
            return 1.0
        return float(self.times[-1])


@dataclass
class WassersteinArcLengthWarp:
    """Warp based on cumulative Wasserstein arc-length.

    The warp is defined by cumulative 2-Wasserstein arc-length along
    the observed trajectory.

    """

    times: np.ndarray
    measures: Sequence[ProbabilityMeasure]
    n_support: int = 256
    reg: float = 0.0
    downsample: bool = False
    backend: str = "pot"
    rng: Optional[np.random.Generator] = None

    # Computed attributes
    arc_lengths: Optional[np.ndarray] = None
    total_length: Optional[float] = None
    _forward_interp: Optional[PchipInterpolator] = None
    _inverse_interp: Optional[PchipInterpolator] = None

    def _clear_measures(self) -> None:
        # Drop references to heavy measure objects once arc-lengths are computed.
        self.measures = []

    def __post_init__(self) -> None:
        """Validate inputs and compute arc-lengths."""
        self.times = np.asarray(self.times, dtype=float)

        if self.times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if len(self.measures) != len(self.times):
            raise ValueError("must have one measure per time point")

        self.rng = self.rng or np.random.default_rng()

        # Handle empty case: create identity warp
        if len(self.times) == 0:
            self.arc_lengths = np.array([])
            self.total_length = 1.0  # Arbitrary positive value
            self._forward_interp = None
            self._inverse_interp = None
            self._clear_measures()
            return

        # Single time point: identity warp at that point
        if len(self.times) == 1:
            self.arc_lengths = np.array([0.0])
            self.total_length = 1.0  # Arbitrary positive value
            self._forward_interp = None
            self._inverse_interp = None
            self._clear_measures()
            return

        # Sort times and measures if not already sorted
        if not np.all(np.diff(self.times) > 0):
            sorted_indices = np.argsort(self.times)
            self.times = self.times[sorted_indices]
            self.measures = [self.measures[i] for i in sorted_indices]

        self._compute_arc_lengths()
        self._clear_measures()

    def _compute_arc_lengths(self) -> None:
        """Compute cumulative Wasserstein arc-length."""
        n = len(self.times)
        arc_lengths = np.zeros(n)

        # Compute pairwise Wasserstein distances
        for i in range(1, n):
            dist = compute_wasserstein_distance(
                self.measures[i - 1],
                self.measures[i],
                n_support=self.n_support,
                reg=self.reg,
                rng=self.rng,
                downsample=self.downsample,
                backend=self.backend,
            )
            arc_lengths[i] = arc_lengths[i - 1] + dist

        self.arc_lengths = arc_lengths
        self.total_length = float(arc_lengths[-1])

        if self.total_length <= 0:
            raise ValueError("total arc-length must be positive")

        # Build PCHIP interpolators with extrapolation enabled
        # Forward: time -> arc-length
        self._forward_interp = PchipInterpolator(
            self.times, self.arc_lengths, extrapolate=True
        )

        # Inverse: arc-length -> time
        strict_arc = _make_strictly_increasing(self.arc_lengths)
        self._inverse_interp = PchipInterpolator(
            strict_arc, self.times, extrapolate=True
        )

    def forward(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Map time to arc-length coordinate.

        """
        scalar_input = np.isscalar(t)
        t_arr = np.atleast_1d(t)

        # Identity warp if no observations
        if len(self.times) == 0 or self._forward_interp is None:
            return t if scalar_input else t_arr

        result = self._forward_interp(t_arr)
        return float(result[0]) if scalar_input else result

    def inverse(self, s: float | np.ndarray) -> float | np.ndarray:
        """
        Map arc-length coordinate back to time.

        """
        scalar_input = np.isscalar(s)
        s_arr = np.atleast_1d(s)

        # Identity warp if no observations
        if len(self.times) == 0 or self._inverse_interp is None:
            return s if scalar_input else s_arr

        result = self._inverse_interp(s_arr)
        return float(result[0]) if scalar_input else result

    def velocity(self, t: float | np.ndarray) -> float | np.ndarray:
        """Instantaneous warp velocity w'(t)."""
        scalar_input = np.isscalar(t)
        t_arr = np.atleast_1d(t)

        if len(self.times) == 0 or self._forward_interp is None:
            result = np.ones_like(t_arr, dtype=float)
            return float(result[0]) if scalar_input else result

        deriv = self._forward_interp.derivative()
        result = deriv(t_arr)
        return float(result[0]) if scalar_input else result

    @property
    def t_min(self) -> float:
        """Minimum observation time."""
        if len(self.times) == 0:
            return 0.0
        return float(self.times[0])

    @property
    def t_max(self) -> float:
        """Maximum observation time."""
        if len(self.times) == 0:
            return 1.0
        return float(self.times[-1])

    def get_arc_length_at_time(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Get cumulative arc-length at given time(s).


        """
        return self.forward(t)
