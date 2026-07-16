"""Optimal transport helpers built on top of POT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import ot

from active_wasserstein._warnings import suppress_pot_warnings
from active_wasserstein.geometry.ot_1d import (
    barycentric_displacement_1d,
    quantile_barycenter_support_1d,
    fast_w2_1d_enabled,
)
from active_wasserstein.geometry.tangent import TransportResult, TransportSolver
from active_wasserstein.measures.base import EmpiricalMeasure, ProbabilityMeasure


def _uniform_weights(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("number of support points must be positive")
    return np.full(n, 1.0 / float(n))


def _normalize_weights(weights: np.ndarray, n_points: int) -> np.ndarray:
    arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.shape[0] != n_points:
        raise ValueError("weights must match support length")
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


def _support_and_masses_from_measure(
    measure: ProbabilityMeasure,
    n_support: int,
    rng: Optional[np.random.Generator],
    downsample: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    support = getattr(measure, "support", None)
    if support is not None:
        points = np.asarray(support, dtype=float)
        if points.ndim == 1:
            points = points[:, None]
        if points.ndim == 2 and points.shape[0] > 0:
            raw_weights = getattr(measure, "weights", None)
            if raw_weights is not None:
                try:
                    weights = _normalize_weights(raw_weights, points.shape[0])
                    if downsample:
                        return _downsample_points_and_weights(points, weights, n_support, rng)
                    return points, weights
                except ValueError:
                    pass
            uniform = _uniform_weights(points.shape[0])
            if downsample:
                return _downsample_points_and_weights(points, uniform, n_support, rng)
            return points, uniform
    if isinstance(measure, EmpiricalMeasure) and measure.support.size > 0:
        uniform = _uniform_weights(measure.support.shape[0])
        if downsample:
            return _downsample_points_and_weights(
                measure.support,
                uniform,
                n_support,
                rng,
            )
        return measure.support, uniform
    rng = rng or np.random.default_rng()
    points = measure.sample(n_support, rng=rng)
    return points, _uniform_weights(points.shape[0])


def _barycentric_displacement(
    plan: np.ndarray,
    reference: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    mass = plan.sum(axis=1, keepdims=True)
    pushed = plan @ target
    displacements = np.zeros_like(reference)
    nonzero = mass.squeeze() > 0
    displacements[nonzero] = pushed[nonzero] / mass[nonzero] - reference[nonzero]
    return displacements


@dataclass
class POTTransportSolver(TransportSolver):
    """Compute displacements using POT (EMD or Sinkhorn)."""

    n_support: int = 256
    reg: float = 0.0
    max_iter: int = 100000
    downsample: bool = False
    rng: Optional[np.random.Generator] = None

    def __call__(
        self, reference: ProbabilityMeasure, target: ProbabilityMeasure
    ) -> TransportResult:
        ref_points, a = _support_and_masses_from_measure(
            reference,
            self.n_support,
            self.rng,
            downsample=self.downsample,
        )
        tgt_points, b = _support_and_masses_from_measure(
            target,
            self.n_support,
            self.rng,
            downsample=self.downsample,
        )
        if (
            self.reg <= 0.0
            and fast_w2_1d_enabled()
            and ref_points.shape[1] == 1
            and tgt_points.shape[1] == 1
        ):
            displacements = barycentric_displacement_1d(
                source_points=ref_points[:, 0],
                source_weights=a,
                target_points=tgt_points[:, 0],
                target_weights=b,
            )
            return TransportResult(source_points=ref_points, displacements=displacements)
        cost = ot.dist(ref_points, tgt_points, metric="sqeuclidean")
        with suppress_pot_warnings():
            if self.reg > 0:
                plan = ot.sinkhorn(a, b, cost, reg=self.reg, numItermax=self.max_iter)
            else:
                plan = ot.emd(a, b, cost, numItermax=self.max_iter)
        displacements = _barycentric_displacement(plan, ref_points, tgt_points)
        return TransportResult(source_points=ref_points, displacements=displacements)


def wasserstein_barycenter(
    measures: Sequence[ProbabilityMeasure],
    weights: Optional[Sequence[float]] = None,
    barycenter_size: int = 128,
    num_iter: int = 100,
    reg: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> EmpiricalMeasure:
    """Compute a discrete Wasserstein barycenter using POT's free-support solver."""

    if not measures:
        raise ValueError("must supply at least one measure to build a barycenter")
    rng = rng or np.random.default_rng()
    if weights is None:
        bary_weights = np.full(len(measures), 1.0 / float(len(measures)))
    else:
        bary_weights = np.asarray(weights, dtype=float)
        if bary_weights.shape[0] != len(measures):
            raise ValueError("weights must match number of measures")
        bary_weights = bary_weights / bary_weights.sum()
    reg_value = float(reg)
    if reg_value < 0.0:
        raise ValueError("reg must be nonnegative")
    supports = []
    masses = []
    for measure in measures:
        pts, mass = _support_and_masses_from_measure(measure, barycenter_size, rng)
        supports.append(pts)
        masses.append(mass)

    if (
        reg_value <= 0.0
        and fast_w2_1d_enabled()
        and all(pts.shape[1] == 1 for pts in supports)
    ):
        bary_support_1d = quantile_barycenter_support_1d(
            [pts[:, 0] for pts in supports],
            masses,
            bary_weights,
            barycenter_size,
        )
        return EmpiricalMeasure(support=bary_support_1d.reshape(-1, 1))

    stacked = np.vstack(supports)
    if stacked.shape[0] < barycenter_size:
        init_cloud = stacked
    else:
        idx = rng.choice(stacked.shape[0], size=barycenter_size, replace=False)
        init_cloud = stacked[idx]
    with suppress_pot_warnings():
        if reg_value > 0.0:
            bary_support = ot.bregman.free_support_sinkhorn_barycenter(
                supports,
                masses,
                init_cloud,
                reg=reg_value,
                weights=bary_weights,
                numItermax=num_iter,
            )
        else:
            bary_support = ot.lp.free_support_barycenter(
                supports,
                masses,
                init_cloud,
                weights=bary_weights,
                numItermax=num_iter,
            )
    return EmpiricalMeasure(support=bary_support)
