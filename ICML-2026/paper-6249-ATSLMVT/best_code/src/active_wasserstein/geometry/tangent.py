"""Tangent-space representations around a reference measure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from active_wasserstein.measures.base import ProbabilityMeasure

ArrayLike = np.ndarray
VectorField = Callable[[ArrayLike], ArrayLike]


class TransportSolver(Protocol):
    """Protocol for solvers returning displacements from a reference measure."""

    def __call__(
        self, reference: ProbabilityMeasure, target: ProbabilityMeasure
    ) -> "TransportResult": ...


@dataclass
class TransportResult:
    """Container for approximate optimal transport information."""

    source_points: ArrayLike
    displacements: ArrayLike

    def __post_init__(self) -> None:
        if self.source_points.shape != self.displacements.shape:
            raise ValueError("source_points and displacements must share shape")


@dataclass
class TangentBasis:
    """Finite basis of vector fields spanning a tangent subspace."""

    fields: Sequence[VectorField]
    intercept: Optional[VectorField] = None
    atom_scaling: Optional[ArrayLike] = None

    def __post_init__(self) -> None:
        if not self.fields and self.intercept is None:
            raise ValueError("basis must contain at least one field or an intercept")
        if self.atom_scaling is not None:
            scaling = np.asarray(self.atom_scaling, dtype=float).reshape(-1)
            if scaling.size == 0:
                raise ValueError("atom_scaling must be non-empty when provided")
            if not np.all(np.isfinite(scaling)):
                raise ValueError("atom_scaling must be finite")
            if np.any(scaling <= 0):
                raise ValueError("atom_scaling must be strictly positive")
            self.atom_scaling = scaling

    @property
    def rank(self) -> int:
        return len(self.fields)

    def _scaling_for_points(self, points: ArrayLike) -> np.ndarray | None:
        if self.atom_scaling is None:
            return None
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2:
            raise ValueError("points must have shape (n, d)")
        scaling = np.asarray(self.atom_scaling, dtype=float).reshape(-1)
        if scaling.shape[0] != pts.shape[0]:
            raise ValueError(
                "atom_scaling length must match number of points "
                f"(got {scaling.shape[0]} and {pts.shape[0]})"
            )
        return scaling[:, None]

    def _to_scaled(self, points: ArrayLike, field: ArrayLike) -> ArrayLike:
        scaling = self._scaling_for_points(points)
        if scaling is None:
            return np.asarray(field, dtype=float)
        return np.asarray(field, dtype=float) * scaling

    def _from_scaled(self, points: ArrayLike, field: ArrayLike) -> ArrayLike:
        scaling = self._scaling_for_points(points)
        if scaling is None:
            return np.asarray(field, dtype=float)
        return np.asarray(field, dtype=float) / scaling

    def design_tensor(self, points: ArrayLike) -> ArrayLike:
        """Evaluate basis vector fields on a cloud of points."""
        if not self.fields:
            # Log that we have no fields

            pts = np.asarray(points)
            return np.zeros((pts.shape[0], pts.shape[1], 0))
        values = [field(points) for field in self.fields]
        return np.stack(values, axis=-1)

    def project(self, points: ArrayLike, displacements: ArrayLike) -> ArrayLike:
        """Least-squares projection of displacement field onto the basis."""
        target = self._to_scaled(points, displacements)

        if self.intercept is not None:
            target = target - self.intercept(points)

        tensor = self.design_tensor(points)  # (n, d, k)
        n, d, k = tensor.shape

        if k == 0:
            return np.zeros((0,))

        lhs = tensor.reshape(n * d, k)
        rhs = target.reshape(n * d)

        coeffs, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        return coeffs

    def evaluate(self, points: ArrayLike, coefficients: ArrayLike) -> ArrayLike:
        tensor = self.design_tensor(points)
        result = np.tensordot(tensor, coefficients, axes=([2], [0]))
        if self.intercept is not None:
            result = result + self.intercept(points)
        return self._from_scaled(points, result)


@dataclass
class TangentVector:
    """Vector represented in the tangent basis."""

    basis: TangentBasis
    coefficients: ArrayLike

    def evaluate(self, points: ArrayLike) -> ArrayLike:
        return self.basis.evaluate(points, self.coefficients)


def _make_vector_field(vec: ArrayLike) -> VectorField:
    vec = np.asarray(vec, dtype=float)

    def field(points: ArrayLike) -> ArrayLike:
        pts = np.asarray(points, dtype=float)
        n_pts, d_pts = pts.shape
        # Case 1: vec matches ambient dim -> repeat across points, constant field
        if vec.size == d_pts:
            return np.tile(vec, (n_pts, 1))
        # Case 2: vec encodes a per-point field flattened -> reshape
        if vec.size == n_pts * d_pts:
            return vec.reshape(n_pts, d_pts)
        raise ValueError(
            "Vector field dimension mismatch: vec size "
            f"{vec.size} is incompatible with points shape {pts.shape}"
        )

    return field


def pca_vector_fields_with_components(
    displacements: ArrayLike, rank: int = 5, trim_outliers: bool = False
) -> tuple[VectorField, list[VectorField], np.ndarray, np.ndarray, np.ndarray]:
    """Build PCA-aligned fields and return mean field + components + singular values.

    When trim_outliers is True, drops the single displacement with the largest
    L2 norm if it exceeds 3x the median norm, before SVD. This produces a more
    stable basis when a single noisy OT map dominates the displacement cloud.
    """
    # Displacements is an array of shape (n_measurements, n_support*dim)
    disp = np.asarray(displacements, dtype=float)
    if disp.ndim != 2:
        raise ValueError("displacements must have shape (n_samples, d)")
    n, d = disp.shape
    if n == 0 or d == 0:
        raise ValueError("displacements must have nonzero samples and dimension")

    if trim_outliers and n >= 4:
        row_norms = np.linalg.norm(disp, axis=1)
        median_norm = float(np.median(row_norms))
        max_norm = float(np.max(row_norms))
        if max_norm > 3.0 * median_norm and median_norm > 1e-12:
            drop_idx = int(np.argmax(row_norms))
            mask = np.ones(n, dtype=bool)
            mask[drop_idx] = False
            disp = disp[mask]
            n = disp.shape[0]

    mean_disp = disp.mean(axis=0)
    if rank < 0:
        raise ValueError("rank must be non-negative")
    n_pca = rank

    disp_centered = disp - mean_disp
    components = np.zeros((0, d), dtype=float)
    singular_values = np.zeros((0,), dtype=float)
    if n_pca > 0:
        _, s, vh = np.linalg.svd(disp_centered, full_matrices=False)
        n_dirs = min(n_pca, d)
        components = vh[:n_dirs].astype(float)
        singular_values = s[:n_dirs].astype(float)

    # Components is an array of shape (n_dirs, n_support*dim)

    mean_field = _make_vector_field(mean_disp)
    fields = [_make_vector_field(comp) for comp in components]
    return mean_field, fields, mean_disp, components, singular_values


def pca_vector_fields(displacements: ArrayLike, rank: int = 5) -> list[VectorField]:
    """Build vector fields aligned with top PCA directions of displacement cloud."""
    _, fields, _, _, _ = pca_vector_fields_with_components(displacements, rank=rank)
    return fields


def build_pca_tangent_basis(displacements: ArrayLike, rank: int = 3) -> TangentBasis:
    """Return a TangentBasis with mean field intercept + PCA directions."""
    mean_field, fields, _, _, _ = pca_vector_fields_with_components(
        displacements, rank=rank
    )
    return TangentBasis(fields=fields, intercept=mean_field)
