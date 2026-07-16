"""Active learning loop orchestration."""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

from active_wasserstein.geometry import wasserstein_barycenter
from active_wasserstein.inference import PredictiveProcess
from active_wasserstein.measures import ProbabilityMeasure

from .surrogate import SurrogateModel
from .types import (
    AcquiredMeasurement,
    AcquisitionFunction,
    AcquisitionRecord,
    MeasurementOracle,
)

logger = logging.getLogger(__name__)

ReferenceBuilder = Callable[[Sequence[ProbabilityMeasure]], ProbabilityMeasure]


class ActiveLearningLoop:
    """High-level orchestration of sequential active learning experiments.

    Parameters
    ----------
    candidate_pool:
        Iterable of times that can be acquired. The loop removes a time once it
        has been sampled unless `remove_from_pool=False` is used when adding a
        measurement manually.
    surrogate:
        Object implementing :class:`SurrogateModel`. The loop re-fits it after
        every acquisition by calling ``surrogate.fit`` on all collected
        measurements.
    acquisition_fn:
        Object that exposes an ``optimize`` method. ``optimize`` must accept
        ``(posterior, candidate_array)`` and return ``(selected_time, scores)``,
        where ``scores`` has shape ``(len(candidate_array),)``.
    oracle:
        Callable that performs the destructive measurement at a requested time
        and returns an :class:`AcquiredMeasurement`.
    recompute_reference_as_barycenter:
        If True, recompute the reference measure as the Wasserstein barycenter
        of all acquired measures before each surrogate fit. If False (default),
        the surrogate's initial reference is kept fixed.
    barycenter_size:
        Number of support points for the barycenter computation.
    barycenter_num_iter:
        Number of iterations for barycenter optimization.
    barycenter_reg:
        Entropic regularization for barycenter computation. ``0.0`` uses exact
        free-support barycenters; values ``> 0`` use Sinkhorn barycenters.
    barycenter_rng:
        Random number generator for barycenter computation.
    reference_builder:
        Optional callable that builds a reference measure from a sequence of
        acquired measures. If provided, this overrides the default barycenter
        builder when the reference is recomputed.
    """

    def __init__(
        self,
        candidate_pool: Iterable[float],
        surrogate: SurrogateModel,
        acquisition_fn: AcquisitionFunction,
        oracle: MeasurementOracle,
        recompute_reference_as_barycenter: bool = False,
        barycenter_size: int = 256,
        barycenter_num_iter: int = 150,
        barycenter_reg: float = 0.0,
        barycenter_rng: Optional[np.random.Generator] = None,
        reference_builder: ReferenceBuilder | None = None,
    ) -> None:
        pool = np.array(list(candidate_pool), dtype=float)
        if pool.size == 0:
            raise ValueError("candidate_pool must contain at least one time")
        self._pool: List[float] = sorted(float(t) for t in pool)
        self.surrogate = surrogate
        self.acquisition_fn = acquisition_fn
        self.oracle = oracle
        self.measurements: list[AcquiredMeasurement] = []
        self.history: list[AcquisitionRecord] = []
        self._posterior: PredictiveProcess | None = None
        self.recompute_reference_as_barycenter = recompute_reference_as_barycenter
        self.barycenter_size = barycenter_size
        self.barycenter_num_iter = barycenter_num_iter
        self.barycenter_reg = float(barycenter_reg)
        if self.barycenter_reg < 0.0:
            raise ValueError("barycenter_reg must be nonnegative")
        self.barycenter_rng = barycenter_rng
        self.reference_builder = reference_builder
        self._last_fit_component_timings: dict[str, float] = {}
        logger.info(
            "ActiveLearningLoop initialized: %d candidates in [%.4f, %.4f]",
            len(self._pool), self._pool[0], self._pool[-1]
        )
        logger.debug(
            "Recompute reference enabled: %s",
            recompute_reference_as_barycenter or reference_builder is not None,
        )

    @property
    def remaining_candidates(self) -> np.ndarray:
        return np.asarray(self._pool, dtype=float)

    @property
    def observed_times(self) -> np.ndarray:
        return np.array([m.time for m in self.measurements], dtype=float)

    @property
    def posterior(self) -> PredictiveProcess | None:
        return self._posterior

    @property
    def last_fit_component_timings(self) -> dict[str, float]:
        return dict(self._last_fit_component_timings)

    def _pop_candidate(self, time: float) -> None:
        """Remove `time` from the pool if it exists."""
        for idx, value in enumerate(self._pool):
            if np.isclose(value, time, atol=1e-12, rtol=0.0):
                self._pool.pop(idx)
                break

    def add_measurement(self, measurement: AcquiredMeasurement, *, remove_from_pool: bool = True) -> None:
        """Register a new measurement, optionally discarding it from the pool."""
        self.measurements.append(measurement)
        support_shape = None
        measure = measurement.measure
        if hasattr(measure, "support"):
            try:
                support = np.asarray(measure.support)
                if support.ndim == 2:
                    support_shape = support.shape
                elif support.ndim == 1:
                    support_shape = (support.shape[0], 1)
            except Exception:
                support_shape = None
        dimension = None
        if hasattr(measure, "dimension"):
            try:
                dimension = int(measure.dimension)
            except Exception:
                dimension = None
        logger.info(
            "Added measurement at t=%.4f (sample_size=%d, support_shape=%s, dimension=%s) - total measurements: %d",
            measurement.time,
            measurement.sample_size,
            support_shape,
            dimension,
            len(self.measurements),
        )
        if remove_from_pool:
            self._pop_candidate(measurement.time)
            logger.debug("Removed t=%.4f from candidate pool - %d candidates remain", measurement.time, len(self._pool))

    def bootstrap(self, times: Sequence[float]) -> None:
        """Collect initial measurements before the adaptive loop starts."""
        logger.info("Bootstrapping with %d initial times: %s", len(times), [float(t) for t in times])
        for t in times:
            measurement = self.oracle(float(t))
            self.add_measurement(measurement)

    @staticmethod
    def _log_timing(
        iteration: int,
        component: str,
        seconds: float,
        **fields: float | int | str,
    ) -> None:
        extras = " ".join(f"{key}={value}" for key, value in fields.items())
        if extras:
            logger.info(
                "TIMING step=%d component=%s seconds=%.6f %s",
                iteration,
                component,
                float(seconds),
                extras,
            )
            return
        logger.info(
            "TIMING step=%d component=%s seconds=%.6f",
            iteration,
            component,
            float(seconds),
        )

    def recompute_reference(self) -> float:
        """Recompute surrogate reference from all acquired measurements."""
        if len(self.measurements) < 1:
            return 0.0
        measures = [m.measure for m in self.measurements]
        builder_label = "custom" if self.reference_builder is not None else "default"
        logger.info(
            "Reference recompute start: n_measurements=%d builder=%s barycenter_size=%d num_iter=%d reg=%.6g",
            len(self.measurements),
            builder_label,
            int(self.barycenter_size),
            int(self.barycenter_num_iter),
            float(self.barycenter_reg),
        )
        start = time.perf_counter()
        if self.reference_builder is None:
            logger.debug(
                "Recomputing reference as Wasserstein barycenter of %d measures",
                len(self.measurements),
            )
            new_reference = wasserstein_barycenter(
                measures=measures,
                barycenter_size=self.barycenter_size,
                num_iter=self.barycenter_num_iter,
                reg=self.barycenter_reg,
                rng=self.barycenter_rng,
            )
        else:
            logger.debug(
                "Recomputing reference from %d measures using custom builder",
                len(self.measurements),
            )
            new_reference = self.reference_builder(measures)
        # Update the surrogate's reference (requires LinearizedWassersteinGPSurrogate)
        if hasattr(self.surrogate, 'reference'):
            self.surrogate.reference = new_reference
            logger.debug("Updated surrogate reference")
        elapsed = time.perf_counter() - start
        logger.info(
            "Reference recompute done in %.2fs (n_measurements=%d)",
            elapsed,
            len(self.measurements),
        )
        return elapsed

    def _update_reference_if_needed(self) -> float | None:
        """Recompute reference if enabled."""
        if not self.recompute_reference_as_barycenter and self.reference_builder is None:
            return None
        return self.recompute_reference()

    def _fit_or_raise(
        self,
        *,
        iteration: int | None = None,
        log_timings: bool = False,
    ) -> PredictiveProcess:
        if not self.measurements:
            raise RuntimeError(
                "at least one measurement is required before fitting the surrogate"
            )
        fit_label = "refit" if iteration is None else f"step={int(iteration)}"
        logger.info(
            "Surrogate fit start (%s): n_measurements=%d",
            fit_label,
            len(self.measurements),
        )
        self._last_fit_component_timings = {}
        reference_seconds = self._update_reference_if_needed()
        if reference_seconds is not None:
            self._last_fit_component_timings["reference_barycenter"] = float(
                reference_seconds
            )
        if (
            log_timings
            and iteration is not None
            and reference_seconds is not None
        ):
            self._log_timing(
                iteration=iteration,
                component="reference_barycenter",
                seconds=reference_seconds,
                n_measurements=len(self.measurements),
                barycenter_reg=f"{self.barycenter_reg:.6g}",
            )
        logger.debug("Fitting surrogate with %d measurements", len(self.measurements))
        surrogate_fit_start = time.perf_counter()
        posterior = self.surrogate.fit(self.measurements)
        surrogate_fit_seconds = time.perf_counter() - surrogate_fit_start
        logger.info(
            "Surrogate fit done (%s) in %.2fs",
            fit_label,
            surrogate_fit_seconds,
        )
        fit_timings = getattr(self.surrogate, "last_fit_timing", None)
        if isinstance(fit_timings, dict):
            for component in (
                "ot_displacement_fields",
                "pca_coefficients",
                "warp_construction",
                "gp_fit",
            ):
                seconds = fit_timings.get(component)
                if seconds is None:
                    continue
                self._last_fit_component_timings[component] = float(seconds)
                if log_timings and iteration is not None:
                    self._log_timing(
                        iteration=iteration,
                        component=component,
                        seconds=float(seconds),
                        n_measurements=len(self.measurements),
                    )
        self._posterior = posterior
        logger.debug("Surrogate fitted successfully")
        return posterior

    def refit(self) -> PredictiveProcess:
        """Fit the surrogate on all collected measurements."""
        return self._fit_or_raise(log_timings=False)

    def step(self) -> AcquisitionRecord:
        """Perform one acquisition step and return the resulting record."""
        iteration = len(self.history) + 1
        logger.info("=== Acquisition step %d ===", iteration)
        
        posterior = self._fit_or_raise(iteration=iteration, log_timings=True)
        candidates = self.remaining_candidates
        if candidates.size == 0:
            raise RuntimeError("no candidates remain to acquire")
        
        logger.debug("Optimizing acquisition over %d candidates", candidates.size)
        acquisition_start = time.perf_counter()
        selected_time, scores = self.acquisition_fn.optimize(posterior, candidates)
        acquisition_seconds = time.perf_counter() - acquisition_start
        self._last_fit_component_timings["acquisition_optimize"] = acquisition_seconds
        self._log_timing(
            iteration=iteration,
            component="acquisition_optimize",
            seconds=acquisition_seconds,
            n_candidates=int(candidates.size),
        )
        scores = np.asarray(scores, dtype=float)
        if scores.shape != candidates.shape:
            raise ValueError("acquisition_fn must return scores matching candidates")

        selected_time = float(selected_time)
        matches = np.where(np.isclose(candidates, selected_time, atol=1e-12, rtol=0.0))[0]
        if matches.size == 0:
            raise ValueError("optimize returned a time not present in the candidate pool")
        idx = int(matches[0])
        max_score = float(scores[idx])

        top_k = min(5, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        logger.debug(
            "Top %d acquisition scores: %s",
            top_k,
            [(float(candidates[i]), float(scores[i])) for i in top_indices]
        )
        logger.info(
            "Selected t=%.4f with score=%.6f (max of %d candidates)",
            selected_time, max_score, candidates.size
        )
        
        measurement = self.oracle(selected_time)
        self.add_measurement(measurement)
        
        record = AcquisitionRecord(
            iteration=len(self.history),
            selected_time=selected_time,
            score=max_score,
            posterior=posterior,
            remaining_candidates=candidates,
        )
        self.history.append(record)
        
        observed = self.observed_times
        logger.debug("Observed times so far: %s", np.sort(observed).tolist())
        if self._last_fit_component_timings:
            summary = " ".join(
                f"{name}={value:.6f}"
                for name, value in sorted(self._last_fit_component_timings.items())
            )
            logger.info("TIMING_SUMMARY step=%d %s", iteration, summary)
        
        return record

    def run(self, num_steps: int, initial_times: Sequence[float] | None = None) -> list[AcquisitionRecord]:
        """Execute ``num_steps`` acquisitions and return the history."""
        logger.info("Starting active learning loop for %d steps", num_steps)
        if initial_times:
            self.bootstrap(initial_times)
        for step_idx in range(num_steps):
            if self.remaining_candidates.size == 0:
                logger.warning("No candidates remain after %d steps, stopping early", step_idx)
                break
            self.step()
        logger.info(
            "Active learning loop completed: %d acquisitions, %d total measurements",
            len(self.history), len(self.measurements)
        )
        logger.info("Final observed times: %s", np.sort(self.observed_times).tolist())
        return list(self.history)
