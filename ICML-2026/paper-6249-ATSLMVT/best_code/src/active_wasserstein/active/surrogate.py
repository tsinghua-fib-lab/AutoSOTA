"""Surrogate models used inside the active learning loop."""

from __future__ import annotations

import inspect
import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np

from active_wasserstein.geometry import TangentBasis, TransportSolver
from active_wasserstein.geometry.tangent import pca_vector_fields_with_components
from active_wasserstein.inference import (
    GPyTorchHilbertRegressor,
    PredictiveProcess,
    TangentObservationModel,
)
from active_wasserstein.inference.kernels import KernelSpec
from active_wasserstein.measures import ProbabilityMeasure
from active_wasserstein.utils import WassersteinArcLengthWarp

from .types import AcquiredMeasurement

logger = logging.getLogger(__name__)


def _process_memory_snapshot_mb() -> tuple[float | None, float | None]:
    """Return process RSS and swap in MB when available (Linux /proc)."""
    rss_mb: float | None = None
    swap_mb: float | None = None
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        rss_mb = float(parts[1]) / 1024.0
                elif line.startswith("VmSwap:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        swap_mb = float(parts[1]) / 1024.0
    except OSError:
        pass
    return rss_mb, swap_mb


def _format_memory_snapshot(rss_mb: float | None, swap_mb: float | None) -> str:
    rss_txt = "unknown" if rss_mb is None else f"{rss_mb:.1f}MB"
    swap_txt = "unknown" if swap_mb is None else f"{swap_mb:.1f}MB"
    return f"rss={rss_txt}, swap={swap_txt}"


def _describe_warper(warper: object) -> tuple[str, dict[str, object]]:
    """Return a compact warper name + selected config fields for logs."""
    name = getattr(warper, "__name__", type(warper).__name__)
    cfg: dict[str, object] = {}
    keywords = getattr(warper, "keywords", None)
    if isinstance(keywords, dict):
        for key in ("backend", "n_support", "downsample", "reg"):
            if key in keywords:
                cfg[key] = keywords[key]
    return name, cfg


class SurrogateModel(ABC):
    """Minimal interface exposed to :class:`ActiveLearningLoop`."""

    @abstractmethod
    def fit(self, measurements: Sequence[AcquiredMeasurement]) -> PredictiveProcess: ...

    @property
    @abstractmethod
    def posterior(self) -> PredictiveProcess | None: ...


WarpFactory = Callable[[np.ndarray, Sequence[ProbabilityMeasure]], object]
ScaleInitializer = Callable[[np.ndarray], np.ndarray]


def _default_scale_initializer(coefficients: np.ndarray) -> np.ndarray:
    if coefficients.ndim != 2:
        raise ValueError("coefficients must have shape (rank, n_obs)")
    rank, n_obs = coefficients.shape
    if n_obs == 0:
        return np.ones(rank, dtype=float)
    if n_obs == 1:
        return np.maximum(np.abs(coefficients[:, 0]), 1e-6)
    variances = np.var(coefficients, axis=1, ddof=1)
    return np.maximum(variances, 1e-6)


def _reference_weights(
    reference: ProbabilityMeasure,
    *,
    eps: float,
    n_atoms: int | None = None,
) -> np.ndarray:
    support = getattr(reference, "support", None)
    support_size: int | None = None
    if support is not None:
        points = np.asarray(support, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0:
            raise ValueError("reference support must have shape (n, d) with n > 0")
        support_size = int(points.shape[0])

    if n_atoms is None:
        if support_size is None:
            raise ValueError(
                "n_atoms must be provided when reference does not expose support"
            )
        n_atoms = support_size
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive")

    raw_weights = getattr(reference, "weights", None)
    if raw_weights is None:
        weights = np.full(n_atoms, 1.0 / float(n_atoms), dtype=float)
    else:
        weights = np.asarray(raw_weights, dtype=float).reshape(-1)
        if weights.shape[0] != n_atoms:
            if support_size is not None and weights.shape[0] == support_size:
                weights = np.full(n_atoms, 1.0 / float(n_atoms), dtype=float)
            else:
                raise ValueError(
                    "reference weights must match OT atom count "
                    f"(got {weights.shape[0]} and {n_atoms})"
                )
        if np.any(weights < 0):
            raise ValueError("reference weights must be nonnegative")
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("reference weights must sum to a positive value")
        weights = weights / total

    effective = np.maximum(weights, float(eps))
    effective = effective / float(np.sum(effective))
    return effective


def _scale_displacement(
    displacement: np.ndarray, atom_scaling: np.ndarray
) -> np.ndarray:
    disp = np.asarray(displacement, dtype=float)
    if disp.ndim != 2:
        raise ValueError("displacement must have shape (n, d)")
    scales = np.asarray(atom_scaling, dtype=float).reshape(-1)
    if scales.shape[0] != disp.shape[0]:
        raise ValueError(
            "atom scaling length must match displacement rows "
            f"(got {scales.shape[0]} and {disp.shape[0]})"
        )
    return disp * scales[:, None]


def _unscale_flattened_vectors(
    vectors: np.ndarray, atom_scaling: np.ndarray
) -> np.ndarray:
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim != 2:
        raise ValueError("vectors must have shape (n_vectors, n_atoms * dim)")
    scales = np.asarray(atom_scaling, dtype=float).reshape(-1)
    n_atoms = scales.shape[0]
    if n_atoms <= 0:
        raise ValueError("atom_scaling must be non-empty")
    if arr.shape[1] % n_atoms != 0:
        raise ValueError(
            "flattened vector width must be divisible by number of atoms "
            f"(got width={arr.shape[1]}, n_atoms={n_atoms})"
        )
    dim = arr.shape[1] // n_atoms
    reshaped = arr.reshape(arr.shape[0], n_atoms, dim)
    return (reshaped / scales[None, :, None]).reshape(arr.shape[0], -1)


class LinearizedWassersteinGPSurrogate(SurrogateModel):
    """Linearized OT + Hilbert GP surrogate"""

    def __init__(
        self,
        reference: ProbabilityMeasure,
        transport_solver: TransportSolver,
        basis_rank: int,
        kernel_spec: KernelSpec,
        prior_variance: float,
        base_variance: float = 1.0,
        warper: WarpFactory | None = WassersteinArcLengthWarp,
        scale_initializer: ScaleInitializer | None = None,
        regressor_kwargs: dict | None = None,
        weight_scaling_eps: float = 1.0e-12,
        trim_outliers: bool = False,
    ) -> None:
        if basis_rank <= 0:
            raise ValueError("basis_rank must be positive")
        self.reference = reference
        self.transport_solver = transport_solver
        self.basis_rank = basis_rank
        if kernel_spec is None:
            raise ValueError("kernel_spec must be provided")
        self.kernel_spec = kernel_spec
        if prior_variance <= 0:
            raise ValueError("prior_variance must be positive")
        self.prior_variance = prior_variance
        self.base_variance = base_variance
        self.warper = warper
        self.scale_initializer: ScaleInitializer = (
            scale_initializer or _default_scale_initializer
        )
        self.regressor_kwargs = regressor_kwargs
        self.weight_scaling_eps = float(weight_scaling_eps)
        if self.weight_scaling_eps <= 0.0:
            raise ValueError("weight_scaling_eps must be positive")
        self.trim_outliers = bool(trim_outliers)

        self._posterior: PredictiveProcess | None = None
        self._basis: TangentBasis | None = None
        self._coefficients: np.ndarray | None = None
        self._warp: object | None = None
        self._basis_mean: np.ndarray | None = None
        self._basis_components: np.ndarray | None = None
        self._basis_singular_values: np.ndarray | None = None
        self._last_fit_timing: dict[str, float] = {}

    @property
    def posterior(self) -> PredictiveProcess | None:
        return self._posterior

    @property
    def basis(self) -> TangentBasis | None:
        return self._basis

    @property
    def coefficients(self) -> np.ndarray | None:
        return self._coefficients

    @property
    def basis_mean(self) -> np.ndarray | None:
        return self._basis_mean

    @property
    def basis_components(self) -> np.ndarray | None:
        return self._basis_components

    @property
    def basis_singular_values(self) -> np.ndarray | None:
        return self._basis_singular_values

    @property
    def warp(self) -> object | None:
        return self._warp

    @property
    def last_fit_timing(self) -> dict[str, float]:
        return dict(self._last_fit_timing)

    def _build_basis(
        self,
        displacements: np.ndarray,
        atom_scaling: np.ndarray,
    ) -> TangentBasis:
        mean_field, fields, mean_disp, components, singular_values = (
            pca_vector_fields_with_components(
                displacements, rank=self.basis_rank, trim_outliers=self.trim_outliers
            )
        )
        basis = TangentBasis(
            fields=fields,
            intercept=mean_field,
            atom_scaling=atom_scaling,
        )
        mean_unscaled = _unscale_flattened_vectors(
            mean_disp.reshape(1, -1),
            atom_scaling,
        )[0]
        components_unscaled = _unscale_flattened_vectors(components, atom_scaling)
        self._basis_mean = mean_unscaled
        self._basis_components = components_unscaled
        self._basis_singular_values = singular_values
        if basis.rank == 0:
            raise RuntimeError("PCA basis returned rank zero")
        return basis

    def _build_observations(
        self,
        basis: TangentBasis,
        transport_results: list,
        measurements: Sequence[AcquiredMeasurement],
    ) -> tuple[np.ndarray, list]:
        coefficients = []
        for result in transport_results:
            coeffs = basis.project(result.source_points, result.displacements)
            coefficients.append(coeffs)
        coeff_matrix = (
            np.stack(coefficients, axis=1)
            if coefficients
            else np.zeros((basis.rank, 0))
        )
        obs_model = TangentObservationModel(base_variance=self.base_variance)
        observations = []
        for rec, coeff in zip(measurements, coeff_matrix.T):
            observations.append(
                obs_model.build_observation(
                    time=rec.time,
                    coefficients=coeff,
                    sample_size=rec.sample_size,
                )
            )
        return coeff_matrix, observations

    def _make_regressor(
        self,
        basis: TangentBasis,
        scales: np.ndarray,
    ) -> GPyTorchHilbertRegressor:
        kwargs = dict(self.regressor_kwargs or {})
        return GPyTorchHilbertRegressor(
            basis=basis,
            scales=scales,
            kernel_spec=self.kernel_spec,
            prior_variance=self.prior_variance,
            **kwargs,
        )

    def fit(self, measurements: Sequence[AcquiredMeasurement]) -> PredictiveProcess:
        if not measurements:
            raise ValueError("measurements list cannot be empty")
        self._last_fit_timing = {}
        fit_start = time.perf_counter()
        # Order the set of measurements
        ordered = sorted(measurements, key=lambda m: m.time)
        reference_support = getattr(self.reference, "support", None)
        reference_support_shape: tuple[int, ...] | None = None
        if reference_support is not None:
            ref_arr = np.asarray(reference_support)
            if ref_arr.ndim == 2:
                reference_support_shape = tuple(int(v) for v in ref_arr.shape)
        logger.info(
            "Surrogate fit internals start: n_measurements=%d transport_solver=%s "
            "reference_support=%s mem=%s",
            len(ordered),
            type(self.transport_solver).__name__,
            reference_support_shape
            if reference_support_shape is not None
            else "unknown",
            _format_memory_snapshot(*_process_memory_snapshot_mb()),
        )

        # Compute the OT map with respect to the reference for each measure
        ot_start = time.perf_counter()
        transport_results = []
        n_measurements = len(ordered)
        for idx, rec in enumerate(ordered, start=1):
            logger.info(
                "OT map start: %d/%d time=%.6f sample_size=%s mem=%s",
                idx,
                n_measurements,
                float(rec.time),
                rec.sample_size,
                _format_memory_snapshot(*_process_memory_snapshot_mb()),
            )
            transport_start = time.perf_counter()
            result = self.transport_solver(self.reference, rec.measure)
            transport_seconds = time.perf_counter() - transport_start
            source_shape = tuple(int(v) for v in np.asarray(result.source_points).shape)
            displacement_shape = tuple(
                int(v) for v in np.asarray(result.displacements).shape
            )
            logger.info(
                "OT map done: %d/%d time=%.6f seconds=%.3f source_shape=%s "
                "displacement_shape=%s mem=%s",
                idx,
                n_measurements,
                float(rec.time),
                transport_seconds,
                source_shape,
                displacement_shape,
                _format_memory_snapshot(*_process_memory_snapshot_mb()),
            )
            transport_results.append(result)
        ot_seconds = time.perf_counter() - ot_start
        self._last_fit_timing["ot_displacement_fields"] = ot_seconds
        logger.info(
            "OT displacement phase done: n_measurements=%d total_seconds=%.3f avg_seconds=%.3f mem=%s",
            n_measurements,
            ot_seconds,
            ot_seconds / float(max(1, n_measurements)),
            _format_memory_snapshot(*_process_memory_snapshot_mb()),
        )
        first_points = np.asarray(transport_results[0].source_points, dtype=float)
        if first_points.ndim != 2 or first_points.shape[0] == 0:
            raise ValueError("transport solver returned invalid source points")
        n_atoms = int(first_points.shape[0])
        expected_shape = first_points.shape
        for idx, result in enumerate(transport_results[1:], start=1):
            points = np.asarray(result.source_points, dtype=float)
            if points.shape != expected_shape:
                raise ValueError(
                    "transport solver must return consistent source point shapes "
                    f"(result 0 has {expected_shape}, result {idx} has {points.shape})"
                )

        reference_weights = _reference_weights(
            self.reference,
            eps=self.weight_scaling_eps,
            n_atoms=n_atoms,
        )
        atom_scaling = np.sqrt(reference_weights)

        # Get the point cloud of shape (n_measurements, n_points * dim)
        cloud_start = time.perf_counter()
        displacement_cloud = np.stack(
            [
                _scale_displacement(res.displacements, atom_scaling).reshape(-1)
                for res in transport_results
            ],
            axis=0,
        )
        cloud_seconds = time.perf_counter() - cloud_start
        assert displacement_cloud.shape[0] == len(ordered)
        assert displacement_cloud.ndim == 2
        logger.info(
            "Displacement cloud ready: shape=%s seconds=%.3f mem=%s",
            tuple(int(v) for v in displacement_cloud.shape),
            cloud_seconds,
            _format_memory_snapshot(*_process_memory_snapshot_mb()),
        )

        # Construct the tangent basis
        logger.info(
            "Basis/PCA start: basis_rank=%d n_observations=%d n_atoms=%d",
            int(self.basis_rank),
            len(ordered),
            int(n_atoms),
        )
        pca_start = time.perf_counter()
        basis = self._build_basis(displacement_cloud, atom_scaling=atom_scaling)
        coeff_matrix, observations = self._build_observations(
            basis,
            transport_results,
            ordered,
        )
        pca_seconds = time.perf_counter() - pca_start
        self._last_fit_timing["pca_coefficients"] = pca_seconds
        logger.info(
            "Basis/PCA done: seconds=%.3f basis_rank=%d coeff_shape=%s mem=%s",
            pca_seconds,
            int(basis.rank),
            tuple(int(v) for v in coeff_matrix.shape),
            _format_memory_snapshot(*_process_memory_snapshot_mb()),
        )

        # Build time warp if applicable
        warp_seconds = 0.0
        if self.warper is None:
            warp = None
            logger.info("Warp construction skipped (warper=None)")
        else:
            warper_name, warper_cfg = _describe_warper(self.warper)
            logger.info(
                "Warp construction start: warper=%s config=%s",
                warper_name,
                warper_cfg,
            )
            warp_start = time.perf_counter()
            times = np.array([m.time for m in ordered], dtype=float)
            measures = [m.measure for m in ordered]
            kwargs: dict[str, object] = {}
            try:
                signature = inspect.signature(self.warper)
            except (TypeError, ValueError):
                signature = None
            if signature is not None and "coefficients" in signature.parameters:
                kwargs["coefficients"] = coeff_matrix
            warp = self.warper(times, measures, **kwargs)
            warp_seconds = time.perf_counter() - warp_start
            logger.info(
                "Warp construction done: seconds=%.3f mem=%s",
                warp_seconds,
                _format_memory_snapshot(*_process_memory_snapshot_mb()),
            )
        self._last_fit_timing["warp_construction"] = float(warp_seconds)

        # Initialize scales and fit the regressor
        scales = self.scale_initializer(coeff_matrix)
        if scales.shape[0] != basis.rank:
            raise ValueError(
                "scale initializer must return a vector with length equal to basis rank"
            )

        # Create and condition the regressor
        regressor = self._make_regressor(basis=basis, scales=scales)
        logger.info(
            "GP conditioning start: n_observations=%d basis_rank=%d mem=%s",
            len(observations),
            int(basis.rank),
            _format_memory_snapshot(*_process_memory_snapshot_mb()),
        )
        gp_start = time.perf_counter()
        posterior = regressor.condition(observations, warp=warp)
        gp_seconds = time.perf_counter() - gp_start
        self._last_fit_timing["gp_fit"] = gp_seconds
        logger.info(
            "GP conditioning done: seconds=%.3f mem=%s",
            gp_seconds,
            _format_memory_snapshot(*_process_memory_snapshot_mb()),
        )

        # Store internal state
        self._posterior = posterior
        self._basis = basis
        self._coefficients = coeff_matrix
        self._warp = warp
        logger.info(
            "Surrogate fit internals done: total_seconds=%.3f components={ot=%.3f,pca=%.3f,warp=%.3f,gp=%.3f} mem=%s",
            time.perf_counter() - fit_start,
            self._last_fit_timing.get("ot_displacement_fields", float("nan")),
            self._last_fit_timing.get("pca_coefficients", float("nan")),
            self._last_fit_timing.get("warp_construction", float("nan")),
            self._last_fit_timing.get("gp_fit", float("nan")),
            _format_memory_snapshot(*_process_memory_snapshot_mb()),
        )
        logger.debug(
            "Surrogate fit timings: ot_displacement_fields=%.6f pca_coefficients=%.6f "
            "warp_construction=%.6f gp_fit=%.6f",
            self._last_fit_timing.get("ot_displacement_fields", float("nan")),
            self._last_fit_timing.get("pca_coefficients", float("nan")),
            self._last_fit_timing.get("warp_construction", float("nan")),
            self._last_fit_timing.get("gp_fit", float("nan")),
        )
        return posterior
