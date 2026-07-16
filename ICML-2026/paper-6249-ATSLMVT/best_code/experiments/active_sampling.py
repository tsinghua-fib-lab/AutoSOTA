"""Hydra-driven experiment for active vs random vs uniform sampling."""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import hydra
from hydra.utils import get_class, instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from active_wasserstein import (
    ActiveLearningLoop,
)
from experiments.active_sampling_io import (
    _write_acquisition_artifacts,
    _write_checkpoint_errors,
    _write_checkpoint_gp_predictions,
    _write_checkpoint_metrics,
    _write_checkpoint_state,
    _write_error_tables,
    _write_metadata,
    _write_metrics,
    _write_reconstruction_artifacts,
    _write_shared_artifacts,
    _write_timing_table,
)
from experiments.active_sampling_utils import (
    _build_experiment_context,
    _evaluate_strategy_errors,
    _extract_gp_hyperparams,
    _extract_gp_predictions,
    _extract_surrogate_state,
    _resolve_active_loop_reference_settings,
    _resolve_barycenter_backend,
    _resolve_eval_sampling,
    _resolve_reference_source,
    _resolve_sample_size,
    build_candidate_pool,  # re-exported for external helpers/tests
    build_eval_barycenter_from_measures,
    build_eval_barycenter_reference,
    build_eval_times,
    build_uniform_schedule,
    compute_weighted_metric,
    evaluate_observed_reconstruction,
    evaluate_reconstruction_with_true_measures,
)
from experiments.components import build_reference_from_measurements, make_measurement_oracle
from active_wasserstein.acquisition.two_phase import TwoPhaseAcquisition
from active_wasserstein.acquisition.velocity_uncertainty import VelocityWeightedUncertaintySampler
from active_wasserstein.acquisition.uncertainty import UncertaintySampler

logger = logging.getLogger(__name__)

@dataclass
class CheckpointResult:
    """Per-checkpoint evaluation outputs."""

    step: int
    errors: np.ndarray
    uniform_metric: float
    velocity_metric: float
    common_reference_errors: np.ndarray | None = None
    common_reference_uniform_metric: float | None = None
    common_reference_velocity_metric: float | None = None
    gp_predictions: dict | None = None
    checkpoint_state: dict | None = None


@dataclass
class EvaluationSnapshot:
    posterior: object
    reference: object
    errors: np.ndarray
    true_measures: list
    uniform_metric: float
    velocity_metric: float
    common_reference_errors: np.ndarray | None = None
    common_reference_uniform_metric: float | None = None
    common_reference_velocity_metric: float | None = None
    observed_errors: np.ndarray | None = None
    observed_times: np.ndarray | None = None
    observed_sample_sizes: np.ndarray | None = None


@dataclass
class StrategyResult:
    """Container for experiment outputs."""

    name: str
    errors: np.ndarray
    uniform_metric: float
    velocity_metric: float
    common_reference_errors: np.ndarray | None = None
    common_reference_uniform_metric: float | None = None
    common_reference_velocity_metric: float | None = None
    wasserstein_velocities: np.ndarray | None = None
    wasserstein_velocity_grid: np.ndarray | None = None
    velocity_grid_times: np.ndarray | None = None
    # For post-hoc analysis
    observed_times: np.ndarray | None = None
    observed_reconstruction_errors: np.ndarray | None = None
    acquisition_history: list[dict] | None = None
    gp_predictions: dict | None = None  # mean, std at eval times
    gp_hyperparams: dict | None = None
    # Objects for full reconstruction (picklable)
    basis: Any | None = None
    reference: Any | None = None
    warp: Any | None = None
    coefficients: np.ndarray | None = None
    observed_sample_sizes: np.ndarray | None = None
    basis_mean: np.ndarray | None = None
    basis_components: np.ndarray | None = None
    basis_singular_values: np.ndarray | None = None
    checkpoint_results: dict[int, CheckpointResult] | None = None
    timing_records: list[dict[str, Any]] | None = None


def _normalize_checkpoints(checkpoints: Sequence[int] | int | None) -> list[int]:
    if checkpoints is None:
        return []
    if isinstance(checkpoints, (int, np.integer)):
        return [int(checkpoints)]
    return sorted({int(step) for step in checkpoints if step is not None})


def _save_checkpoint_state_enabled(cfg: DictConfig) -> bool:
    """Return whether bulky checkpoint pickle state should be persisted."""
    evaluation = cfg.get("evaluation", {})
    if evaluation is None:
        return False
    return bool(evaluation.get("save_checkpoint_state", False))


def _prune_candidate_pool(
    candidate_pool: Sequence[float],
    exclude_times: Sequence[float] | np.ndarray,
) -> np.ndarray:
    remaining: list[float] = []
    for t in candidate_pool:
        if any(np.isclose(t, s, atol=1e-12, rtol=0.0) for s in exclude_times):
            continue
        remaining.append(float(t))
    return np.asarray(remaining, dtype=float)


def _build_checkpoint_state(
    name: str,
    step: int,
    snapshot: "EvaluationSnapshot",
    surrogate: object,
    candidate_pool: Sequence[float],
    eval_times: Sequence[float],
    acquisition_history: list[dict] | None,
    remaining_candidates: np.ndarray | None,
) -> dict:
    surrogate_state = _extract_surrogate_state(surrogate)
    observed_times = (
        np.asarray(snapshot.observed_times, dtype=float)
        if snapshot.observed_times is not None
        else np.asarray([], dtype=float)
    )
    if observed_times.size > 1:
        observed_times = np.sort(observed_times)
    if remaining_candidates is None:
        remaining_candidates = _prune_candidate_pool(candidate_pool, observed_times)
    return {
        "strategy": str(name),
        "step": int(step),
        "posterior": snapshot.posterior,
        "observed_times": observed_times,
        "coefficients": surrogate_state["coefficients"],
        "basis": getattr(surrogate, "basis", None),
        "basis_mean": surrogate_state["basis_mean"],
        "basis_components": surrogate_state["basis_components"],
        "basis_singular_values": surrogate_state["basis_singular_values"],
        "reference": snapshot.reference,
        "candidate_pool": np.asarray(candidate_pool, dtype=float),
        "eval_times": np.asarray(eval_times, dtype=float),
        "acquisition_history": list(acquisition_history or []),
        "remaining_candidates": np.asarray(remaining_candidates, dtype=float),
    }


def _resolve_strategy_entries(
    cfg: DictConfig,
) -> list[tuple[str, DictConfig | None]]:
    strategies_cfg = cfg.get("strategies")
    if isinstance(strategies_cfg, DictConfig):
        order = cfg.get("strategy_order")
        if order is None:
            order = list(strategies_cfg.keys())
        entries: list[tuple[str, DictConfig | None]] = []
        for key in order:
            if key not in strategies_cfg:
                raise ValueError(f"strategy_order includes unknown strategy '{key}'")
            entry = strategies_cfg[key]
            name = str(key)
            if isinstance(entry, DictConfig):
                entry_name = entry.get("name")
                if entry_name is not None:
                    name = str(entry_name)
            entries.append((name, entry))
        return entries
    if isinstance(strategies_cfg, (ListConfig, list)):
        entries = []
        for item in list(strategies_cfg):
            if isinstance(item, DictConfig):
                entry_name = item.get("name")
                if entry_name is None:
                    raise ValueError("strategy entry is missing a name field")
                entries.append((str(entry_name), item))
            else:
                entries.append((str(item), None))
        if entries:
            return entries
    return [("active", None)] + [(str(name), None) for name in cfg.baselines.keys()]


def _strategy_base(name: str) -> str:
    for base in ("active", "uniform", "random"):
        if name == base or name.startswith(f"{base}:"):
            return base
    return name


def _resolve_strategy_overrides(
    cfg: DictConfig,
    name: str,
    strategy_cfg: DictConfig | None = None,
) -> tuple[DictConfig, DictConfig | None, bool, DictConfig | None]:
    overrides_cfg = cfg.get("strategy_overrides")
    override = None
    if overrides_cfg is not None and name in overrides_cfg:
        override = overrides_cfg.get(name)
    merge_payload: dict[str, Any] = {}
    reference_override_cfg = None
    acquisition_override = None
    for source in (strategy_cfg, override):
        if source is None:
            continue
        for key in ("surrogate", "reference", "active_loop", "acquisition"):
            if key in source and source.get(key) is not None:
                if key == "acquisition":
                    acquisition_override = source.get(key)
                else:
                    merge_payload[key] = source.get(key)
        if "reference_override" in source and source.get("reference_override") is not None:
            reference_override_cfg = source.get("reference_override")
    reference_overridden = bool(merge_payload.get("reference")) or reference_override_cfg is not None
    surrogate_override = merge_payload.pop("surrogate", None)
    if merge_payload or surrogate_override is not None:
        merged_cfg = OmegaConf.merge(cfg)
        if merge_payload:
            merged_cfg = OmegaConf.merge(merged_cfg, merge_payload)
        if surrogate_override is not None:
            merged_cfg = _apply_surrogate_override(merged_cfg, surrogate_override)
    else:
        merged_cfg = cfg
    return merged_cfg, reference_override_cfg, reference_overridden, acquisition_override


def _apply_surrogate_override(cfg: DictConfig, override: Any) -> DictConfig:
    if override is None:
        return cfg
    if not isinstance(override, DictConfig):
        override = OmegaConf.create(override)
    kernel_spec_override = None
    warper_override = None
    surrogate_merge = override
    if isinstance(override, DictConfig):
        if "kernel_spec" in override and override.get("kernel_spec") is not None:
            kernel_spec_override = override.get("kernel_spec")
        if "warper" in override and override.get("warper") is not None:
            warper_override = override.get("warper")
        keys = [key for key in override.keys() if key not in {"kernel_spec", "warper"}]
        surrogate_merge = OmegaConf.masked_copy(override, keys) if keys else None
    merged_cfg = cfg
    if surrogate_merge is not None:
        with open_dict(merged_cfg):
            merged_cfg.surrogate = OmegaConf.merge(merged_cfg.surrogate, surrogate_merge)
    if kernel_spec_override is not None:
        merged_cfg.surrogate.kernel_spec = kernel_spec_override
    if warper_override is not None:
        merged_cfg.surrogate.warper = warper_override
    return merged_cfg


def _resolve_strategy_base(name: str, strategy_cfg: DictConfig | None) -> str:
    if strategy_cfg is not None:
        base = strategy_cfg.get("base")
        if base is not None:
            return str(base)
    return _strategy_base(name)


def _resolve_one_shot_override(
    cfg: DictConfig,
    name: str,
    strategy_cfg: DictConfig | None,
) -> bool | None:
    if strategy_cfg is not None and "one_shot" in strategy_cfg:
        value = strategy_cfg.get("one_shot")
        if value is not None:
            return bool(value)
    overrides_cfg = cfg.get("strategy_overrides")
    if overrides_cfg is not None and name in overrides_cfg:
        override = overrides_cfg.get(name)
        if override is not None and "one_shot" in override:
            value = override.get("one_shot")
            if value is not None:
                return bool(value)
    return None


def _instantiate_reference_override(
    reference_override_cfg: DictConfig,
    reference: object,
    rng: np.random.Generator,
) -> object:
    try:
        return instantiate(reference_override_cfg, reference=reference, rng=rng)
    except TypeError:
        return instantiate(reference_override_cfg, reference=reference)


def _validate_surrogate_config(surrogate_cfg: DictConfig, label: str = "surrogate") -> None:
    if surrogate_cfg is None:
        return
    target = surrogate_cfg.get("_target_")
    if not target:
        return
    try:
        cls = get_class(target)
    except Exception as exc:
        logger.warning("Could not resolve %s target '%s': %s", label, target, exc)
        return
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError) as exc:
        logger.warning("Could not inspect %s target '%s': %s", label, target, exc)
        return
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return
    allowed = {name for name in signature.parameters if name != "self"}
    ignored = {"_target_", "_partial_", "_convert_", "_recursive_", "_args_"}
    extra = [key for key in surrogate_cfg.keys() if key not in allowed and key not in ignored]
    if extra:
        raise ValueError(
            f"{label} config has unexpected keys for {target}: {sorted(extra)}"
        )


def run_strategy(
    name: str,
    cfg: DictConfig,
    trajectory: object,
    candidate_pool: np.ndarray,
    initial_times: Sequence[float],
    initial_measurements: Sequence[object],
    initial_reference: object | None,
    rng: np.random.Generator,
    acquisition_fn,
    num_steps: int,
    checkpoints: Sequence[int] | None = None,
    pre_acquired_times: Sequence[float] | None = None,
    common_reference: object | None = None,
    eval_times: Sequence[float] | None = None,
    eval_sample_method: str | None = None,
    eval_sample_size: int | None = None,
    eval_true_measures: Sequence[object] | None = None,
    eval_velocities: np.ndarray | None = None,
    velocity_grid_times: np.ndarray | None = None,
    velocity_grid: np.ndarray | None = None,
    eval_barycenter_reference: object | None = None,
    reference_override_cfg: DictConfig | None = None,
    reference_overridden: bool = False,
) -> StrategyResult:
    """Run one acquisition strategy and return reconstruction metrics.

    If checkpoints are provided, intermediate evaluations are stored in the result.
    """
    logger.info("="*60)
    logger.info("Running strategy: %s", name)
    logger.info("="*60)

    oracle_sample_size = _resolve_sample_size(cfg.oracle.sample_size)
    logger.debug(
        "Creating measurement oracle (sample_size=%s, method=%s)",
        oracle_sample_size,
        cfg.oracle.sample_method,
    )
    oracle = make_measurement_oracle(
        trajectory=trajectory,
        sample_size=oracle_sample_size,
        sample_method=str(cfg.oracle.sample_method),
        rng=rng,
    )

    if not initial_measurements:
        raise ValueError("initial_measurements must be provided")
    shared_initial_measurements = list(initial_measurements)
    measurement_times = [float(m.time) for m in shared_initial_measurements]
    logger.info(
        "Using %d shared initial measurements at times: %s",
        len(shared_initial_measurements),
        measurement_times,
    )

    if eval_times is None:
        eval_times = build_eval_times(cfg, trajectory)

    reference_source = _resolve_reference_source(cfg)
    reference_backend = _resolve_barycenter_backend(
        cfg=cfg,
        source_cfg=getattr(cfg, "reference", None),
    )
    if reference_source == "eval_barycenter":
        logger.debug(
            "Building reference measure from evaluation barycenter "
            "(size=%d, num_iter=%d, backend=%s, reg=%.4g)",
            int(cfg.reference.barycenter_size),
            int(cfg.reference.num_iter),
            reference_backend,
            float(cfg.reference.get("reg", 0.0)),
        )
        if eval_barycenter_reference is not None and not reference_overridden:
            reference = eval_barycenter_reference
        elif eval_true_measures is not None:
            reference = build_eval_barycenter_from_measures(
                cfg=cfg,
                measures=eval_true_measures,
                rng=rng,
            )
        else:
            reference = build_eval_barycenter_reference(
                cfg=cfg,
                trajectory=trajectory,
                eval_times=eval_times,
                rng=rng,
            )
    elif reference_source == "initial_barycenter":
        if initial_reference is not None and not reference_overridden:
            logger.debug("Using shared initial-barycenter reference")
            reference = initial_reference
        else:
            logger.debug(
                "Building reference measure from initial measurements "
                "(barycenter_size=%d, num_iter=%d, backend=%s, reg=%.4g)",
                int(cfg.reference.barycenter_size),
                int(cfg.reference.num_iter),
                reference_backend,
                float(cfg.reference.get("reg", 0.0)),
            )
            reference = build_reference_from_measurements(
                measurements=shared_initial_measurements,
                barycenter_size=int(cfg.reference.barycenter_size),
                num_iter=int(cfg.reference.num_iter),
                reg=float(cfg.reference.get("reg", 0.0)),
                rng=rng,
                backend=reference_backend,
            )
    else:
        raise ValueError(f"reference.source '{reference_source}' is not supported")
    if reference_override_cfg is not None:
        logger.info("Applying reference override for '%s'", name)
        reference = _instantiate_reference_override(reference_override_cfg, reference, rng)

    logger.debug("Instantiating transport solver")
    transport_solver = instantiate(cfg.transport, rng=rng)

    logger.debug("Instantiating surrogate (basis_rank=%d)", cfg.surrogate.basis_rank)
    _validate_surrogate_config(cfg.surrogate)
    surrogate = instantiate(
        cfg.surrogate,
        reference=reference,
        transport_solver=transport_solver,
        _convert_="object",
    )

    (
        recompute_reference,
        barycenter_size,
        barycenter_num_iter,
        barycenter_reg,
        _active_loop_backend,
    ) = (
        _resolve_active_loop_reference_settings(
            cfg=cfg,
            reference_source=reference_source,
        )
    )
    if reference_source == "eval_barycenter" and recompute_reference:
        logger.warning(
            "reference.source=eval_barycenter but recompute_reference_as_barycenter=True; "
            "reference will be updated during acquisitions"
        )
    barycenter_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
    reference_builder = None

    logger.info("Creating active learning loop with %d candidates", len(candidate_pool))
    loop = ActiveLearningLoop(
        candidate_pool=candidate_pool,
        surrogate=surrogate,
        acquisition_fn=acquisition_fn,
        oracle=oracle,
        recompute_reference_as_barycenter=recompute_reference,
        barycenter_size=barycenter_size,
        barycenter_num_iter=barycenter_num_iter,
        barycenter_reg=barycenter_reg,
        barycenter_rng=barycenter_rng,
        reference_builder=reference_builder,
    )
    for measurement in shared_initial_measurements:
        loop.add_measurement(measurement)
    if pre_acquired_times:
        logger.info("Collecting %d one-shot measurements", len(pre_acquired_times))
        for t in pre_acquired_times:
            loop.add_measurement(oracle(float(t)))

    resolved_eval_sample_method, resolved_eval_sample_size = _resolve_eval_sampling(
        cfg=cfg,
        trajectory=trajectory,
        eval_sample_method=eval_sample_method,
        eval_sample_size=eval_sample_size,
        eval_true_measures=eval_true_measures,
    )
    provided_velocity_grid_times = velocity_grid_times
    provided_velocity_grid = velocity_grid
    timing_records: list[dict[str, Any]] = []

    def _record_timing(
        *,
        stage: str,
        component: str,
        seconds: float,
        step: int | None = None,
        n_measurements: int | None = None,
        n_candidates: int | None = None,
    ) -> None:
        if not np.isfinite(float(seconds)):
            return
        row: dict[str, Any] = {
            "stage": str(stage),
            "component": str(component),
            "seconds": float(seconds),
        }
        if step is not None:
            row["step"] = int(step)
        if n_measurements is not None:
            row["n_measurements"] = int(n_measurements)
        if n_candidates is not None:
            row["n_candidates"] = int(n_candidates)
        timing_records.append(row)

    def _record_loop_fit_timings(stage: str, step: int | None) -> None:
        for component, seconds in loop.last_fit_component_timings.items():
            _record_timing(
                stage=stage,
                component=component,
                seconds=float(seconds),
                step=step,
                n_measurements=len(loop.measurements),
            )

    def _evaluate_loop_state(
        include_observed_errors: bool,
        *,
        timing_stage: str | None = None,
        timing_step: int | None = None,
    ) -> EvaluationSnapshot:
        stage_label = str(timing_stage) if timing_stage is not None else "eval"
        logger.info(
            "Starting evaluation stage='%s' step=%s with %d measurements",
            stage_label,
            "none" if timing_step is None else int(timing_step),
            len(loop.measurements),
        )
        refit_start = time.perf_counter()
        posterior = loop.refit()
        refit_seconds = time.perf_counter() - refit_start
        logger.info(
            "Finished surrogate refit for stage='%s' step=%s in %.2fs",
            stage_label,
            "none" if timing_step is None else int(timing_step),
            refit_seconds,
        )
        if timing_stage is not None:
            _record_loop_fit_timings(timing_stage, timing_step)
        reference_for_eval = getattr(surrogate, "reference", reference)
        if (
            include_observed_errors
            and common_reference is not None
            and hasattr(common_reference, "support")
            and hasattr(reference_for_eval, "support")
        ):
            try:
                common_shape = np.asarray(common_reference.support).shape
                reference_shape = np.asarray(reference_for_eval.support).shape
                if common_shape != reference_shape:
                    logger.warning(
                        "Common reference support shape %s differs from strategy reference %s",
                        common_shape,
                        reference_shape,
                    )
            except Exception:
                logger.debug("Could not compare reference support shapes")
        errors_start = time.perf_counter()
        logger.info(
            "Computing reconstruction errors for stage='%s' step=%s over %d eval times",
            stage_label,
            "none" if timing_step is None else int(timing_step),
            len(eval_times),
        )
        errors, true_measures = _evaluate_strategy_errors(
            posterior=posterior,
            basis=surrogate.basis,
            reference=reference_for_eval,
            eval_times=eval_times,
            trajectory=trajectory,
            eval_sample_method=resolved_eval_sample_method,
            eval_sample_size=resolved_eval_sample_size,
            eval_true_measures=eval_true_measures,
            n_support=int(cfg.evaluation.n_support),
            rng=rng,
        )
        logger.info(
            "Finished reconstruction errors for stage='%s' step=%s in %.2fs",
            stage_label,
            "none" if timing_step is None else int(timing_step),
            time.perf_counter() - errors_start,
        )
        uniform_metric = compute_weighted_metric(errors)
        if eval_velocities is None:
            raise ValueError("eval_velocities must be provided")
        velocity_metric = compute_weighted_metric(errors, eval_velocities)
        common_reference_errors = None
        common_reference_uniform_metric = None
        common_reference_velocity_metric = None
        if common_reference is not None:
            if include_observed_errors:
                logger.info("Refitting surrogate on common reference for evaluation")
            _validate_surrogate_config(cfg.surrogate, label="common_reference surrogate")
            common_surrogate = instantiate(
                cfg.surrogate,
                reference=common_reference,
                transport_solver=transport_solver,
                _convert_="object",
            )
            common_surrogate.fit(loop.measurements)
            common_reference_errors = evaluate_reconstruction_with_true_measures(
                posterior=common_surrogate.posterior,
                basis=common_surrogate.basis,
                reference=common_reference,
                eval_times=eval_times,
                true_measures=true_measures,
                n_support=int(cfg.evaluation.n_support),
                rng=rng,
            )
            common_reference_uniform_metric = compute_weighted_metric(common_reference_errors)
            common_reference_velocity_metric = compute_weighted_metric(
                common_reference_errors,
                eval_velocities,
            )
        observed_errors = None
        if include_observed_errors:
            observed_errors = evaluate_observed_reconstruction(
                posterior=posterior,
                basis=surrogate.basis,
                reference=reference_for_eval,
                measurements=loop.measurements,
                n_support=int(cfg.evaluation.n_support),
                rng=rng,
            )
        observed_times = loop.observed_times.copy()
        observed_sample_sizes = np.array(
            [m.sample_size for m in loop.measurements], dtype=int
        )
        return EvaluationSnapshot(
            posterior=posterior,
            reference=reference_for_eval,
            errors=errors,
            true_measures=true_measures,
            uniform_metric=uniform_metric,
            velocity_metric=velocity_metric,
            common_reference_errors=common_reference_errors,
            common_reference_uniform_metric=common_reference_uniform_metric,
            common_reference_velocity_metric=common_reference_velocity_metric,
            observed_errors=observed_errors,
            observed_times=observed_times,
            observed_sample_sizes=observed_sample_sizes,
        )

    base_name = _strategy_base(name)
    measurement_cache: dict[float, object] = {}

    def _get_measurement(time: float):
        key = float(time)
        if key not in measurement_cache:
            measurement_cache[key] = oracle(key)
        return measurement_cache[key]

    def _filter_candidate_pool(
        pool: Sequence[float],
        initial_times: Sequence[float],
    ) -> list[float]:
        filtered: list[float] = []
        for t in pool:
            if any(np.isclose(t, s, atol=1e-12, rtol=0.0) for s in initial_times):
                continue
            filtered.append(float(t))
        return filtered

    def _select_uniform_times(
        schedule: Sequence[float],
        pool: Sequence[float],
    ) -> list[float]:
        remaining = list(pool)
        selected: list[float] = []
        for idx_step, target in enumerate(schedule, start=1):
            if not remaining:
                break
            select_start = time.perf_counter()
            remaining_arr = np.asarray(remaining, dtype=float)
            idx = int(np.argmin(np.abs(remaining_arr - float(target))))
            selected_time = float(remaining_arr[idx])
            select_seconds = time.perf_counter() - select_start
            _record_timing(
                stage="manual_selection",
                component="acquisition_optimize",
                seconds=select_seconds,
                step=idx_step,
                n_candidates=int(remaining_arr.size),
            )
            selected.append(selected_time)
            remaining.pop(idx)
        return selected

    def _build_uniform_schedule_for_steps(step_count: int) -> list[float]:
        if step_count <= 0:
            return []
        schedule_cfg = OmegaConf.merge(
            cfg,
            {"uniform_schedule": {"num_steps": int(step_count)}},
        )
        return build_uniform_schedule(schedule_cfg, trajectory, initial_times)

    def _evaluate_one_shot(
        selected_times: Sequence[float],
        include_observed_errors: bool,
        keep_measurements: bool = False,
        timing_stage: str | None = None,
        timing_step: int | None = None,
    ) -> EvaluationSnapshot:
        measurements = list(shared_initial_measurements)
        for t in selected_times:
            measurements.append(_get_measurement(float(t)))
        original_measurements = loop.measurements
        loop.measurements = list(measurements)
        snapshot = _evaluate_loop_state(
            include_observed_errors=include_observed_errors,
            timing_stage=timing_stage,
            timing_step=timing_step,
        )
        if not keep_measurements:
            loop.measurements = original_measurements
        return snapshot

    def _select_random_times(
        count: int,
        pool: Sequence[float],
        acquisition,
    ) -> tuple[list[float], list[float]]:
        remaining = list(pool)
        selected: list[float] = []
        scores: list[float] = []
        for idx_step in range(1, min(count, len(remaining)) + 1):
            candidates_arr = np.asarray(remaining, dtype=float)
            optimize_start = time.perf_counter()
            selected_time, score_arr = acquisition.optimize(None, candidates_arr)
            optimize_seconds = time.perf_counter() - optimize_start
            _record_timing(
                stage="manual_selection",
                component="acquisition_optimize",
                seconds=optimize_seconds,
                step=idx_step,
                n_candidates=int(candidates_arr.size),
            )
            deltas = np.abs(candidates_arr - float(selected_time))
            idx = int(np.argmin(deltas))
            selected_time = float(candidates_arr[idx])
            score_val = 0.0
            if isinstance(score_arr, np.ndarray) and score_arr.size > 0:
                if idx < score_arr.size:
                    score_val = float(score_arr[idx])
                else:
                    score_val = float(np.max(score_arr))
            selected.append(selected_time)
            scores.append(score_val)
            remaining.pop(idx)
        return selected, scores

    manual_acquisition_history: list[dict] | None = None
    selected_times_final: list[float] | None = None

    if base_name in {"uniform", "random"}:
        base_pool = _filter_candidate_pool(candidate_pool, initial_times)
        if base_name == "uniform":
            schedule = _build_uniform_schedule_for_steps(int(num_steps))
            selected_times_final = _select_uniform_times(schedule, base_pool)
            scores = [1.0 for _ in selected_times_final]
        else:
            selected_times_final, scores = _select_random_times(
                int(num_steps), base_pool, acquisition_fn
            )
        manual_acquisition_history = [
            {"iteration": idx, "selected_time": float(t), "score": float(s)}
            for idx, (t, s) in enumerate(zip(selected_times_final, scores))
        ]

    def _history_from_manual(step_num: int) -> list[dict]:
        if manual_acquisition_history is None:
            return []
        return [dict(item) for item in manual_acquisition_history[: int(step_num)]]

    def _history_from_loop() -> list[dict]:
        return [
            {
                "iteration": rec.iteration,
                "selected_time": float(rec.selected_time),
                "score": float(rec.score),
            }
            for rec in loop.history
        ]

    checkpoint_results: dict[int, CheckpointResult] | None = None
    normalized_checkpoints = _normalize_checkpoints(checkpoints)
    if normalized_checkpoints:
        checkpoint_results = {}
        checkpoints_set = {step for step in normalized_checkpoints if step >= 0}
        max_steps = int(num_steps)
        ignored = sorted(step for step in checkpoints_set if step > max_steps)
        if ignored:
            logger.warning(
                "Ignoring checkpoints beyond num_steps=%d: %s",
                max_steps,
                ignored,
            )
        checkpoints_set = {step for step in checkpoints_set if step <= max_steps}
        if 0 in checkpoints_set:
            if base_name == "uniform":
                snapshot = _evaluate_one_shot(
                    [],
                    include_observed_errors=False,
                    timing_stage="checkpoint_eval",
                    timing_step=0,
                )
                history = _history_from_manual(0)
                remaining_candidates = np.asarray(base_pool, dtype=float)
            else:
                snapshot = _evaluate_loop_state(
                    include_observed_errors=False,
                    timing_stage="checkpoint_eval",
                    timing_step=0,
                )
                history = _history_from_loop()
                remaining_candidates = loop.remaining_candidates.copy()
            checkpoint_state = _build_checkpoint_state(
                name=name,
                step=0,
                snapshot=snapshot,
                surrogate=surrogate,
                candidate_pool=candidate_pool,
                eval_times=eval_times,
                acquisition_history=history,
                remaining_candidates=remaining_candidates,
            )
            checkpoint_results[0] = CheckpointResult(
                step=0,
                errors=snapshot.errors,
                uniform_metric=snapshot.uniform_metric,
                velocity_metric=snapshot.velocity_metric,
                common_reference_errors=snapshot.common_reference_errors,
                common_reference_uniform_metric=snapshot.common_reference_uniform_metric,
                common_reference_velocity_metric=snapshot.common_reference_velocity_metric,
                gp_predictions=_extract_gp_predictions(snapshot.posterior, eval_times),
                checkpoint_state=checkpoint_state,
            )
        if base_name == "uniform":
            base_pool = _filter_candidate_pool(candidate_pool, initial_times)
            for step_num in sorted(checkpoints_set):
                if step_num == 0:
                    continue
                schedule = _build_uniform_schedule_for_steps(step_num)
                selected_times = _select_uniform_times(schedule, base_pool)
                snapshot = _evaluate_one_shot(
                    selected_times,
                    include_observed_errors=False,
                    timing_stage="checkpoint_eval",
                    timing_step=step_num,
                )
                history = _history_from_manual(step_num)
                remaining_candidates = _prune_candidate_pool(base_pool, selected_times)
                checkpoint_state = _build_checkpoint_state(
                    name=name,
                    step=step_num,
                    snapshot=snapshot,
                    surrogate=surrogate,
                    candidate_pool=candidate_pool,
                    eval_times=eval_times,
                    acquisition_history=history,
                    remaining_candidates=remaining_candidates,
                )
                checkpoint_results[step_num] = CheckpointResult(
                    step=step_num,
                    errors=snapshot.errors,
                    uniform_metric=snapshot.uniform_metric,
                    velocity_metric=snapshot.velocity_metric,
                    common_reference_errors=snapshot.common_reference_errors,
                    common_reference_uniform_metric=snapshot.common_reference_uniform_metric,
                    common_reference_velocity_metric=snapshot.common_reference_velocity_metric,
                    gp_predictions=_extract_gp_predictions(snapshot.posterior, eval_times),
                    checkpoint_state=checkpoint_state,
                )
            if selected_times_final is None:
                selected_times_final = []
            if selected_times_final:
                _evaluate_one_shot(
                    selected_times_final,
                    include_observed_errors=False,
                    keep_measurements=True,
                    timing_stage="one_shot_prepare",
                    timing_step=len(selected_times_final),
                )
            logger.info("Collected %d uniform acquisitions (one-shot)", len(selected_times_final))
        elif base_name == "random":
            if selected_times_final is None:
                selected_times_final = []
            for step_num in sorted(checkpoints_set):
                if step_num == 0:
                    continue
                selected_times = selected_times_final[: int(step_num)]
                snapshot = _evaluate_one_shot(
                    selected_times,
                    include_observed_errors=False,
                    timing_stage="checkpoint_eval",
                    timing_step=step_num,
                )
                history = _history_from_manual(step_num)
                remaining_candidates = _prune_candidate_pool(base_pool, selected_times)
                checkpoint_state = _build_checkpoint_state(
                    name=name,
                    step=step_num,
                    snapshot=snapshot,
                    surrogate=surrogate,
                    candidate_pool=candidate_pool,
                    eval_times=eval_times,
                    acquisition_history=history,
                    remaining_candidates=remaining_candidates,
                )
                checkpoint_results[step_num] = CheckpointResult(
                    step=step_num,
                    errors=snapshot.errors,
                    uniform_metric=snapshot.uniform_metric,
                    velocity_metric=snapshot.velocity_metric,
                    common_reference_errors=snapshot.common_reference_errors,
                    common_reference_uniform_metric=snapshot.common_reference_uniform_metric,
                    common_reference_velocity_metric=snapshot.common_reference_velocity_metric,
                    gp_predictions=_extract_gp_predictions(snapshot.posterior, eval_times),
                    checkpoint_state=checkpoint_state,
                )
            if selected_times_final:
                _evaluate_one_shot(
                    selected_times_final,
                    include_observed_errors=False,
                    keep_measurements=True,
                    timing_stage="one_shot_prepare",
                    timing_step=len(selected_times_final),
                )
            logger.info("Collected %d random acquisitions (one-shot)", len(selected_times_final))
        else:
            logger.info("Running %d acquisition steps", int(num_steps))
            for step_idx in range(int(num_steps)):
                if loop.remaining_candidates.size == 0:
                    logger.warning(
                        "No candidates remain after %d steps, stopping early",
                        step_idx,
                    )
                    break
                loop.step()
                step_num = step_idx + 1
                _record_loop_fit_timings("acquisition_step", step_num)
                if step_num in checkpoints_set:
                    snapshot = _evaluate_loop_state(
                        include_observed_errors=False,
                        timing_stage="checkpoint_eval",
                        timing_step=step_num,
                    )
                    history = _history_from_loop()
                    remaining_candidates = loop.remaining_candidates.copy()
                    checkpoint_state = _build_checkpoint_state(
                        name=name,
                        step=step_num,
                        snapshot=snapshot,
                        surrogate=surrogate,
                        candidate_pool=candidate_pool,
                        eval_times=eval_times,
                        acquisition_history=history,
                        remaining_candidates=remaining_candidates,
                    )
                    checkpoint_results[step_num] = CheckpointResult(
                        step=step_num,
                        errors=snapshot.errors,
                        uniform_metric=snapshot.uniform_metric,
                        velocity_metric=snapshot.velocity_metric,
                        common_reference_errors=snapshot.common_reference_errors,
                        common_reference_uniform_metric=snapshot.common_reference_uniform_metric,
                        common_reference_velocity_metric=snapshot.common_reference_velocity_metric,
                        gp_predictions=_extract_gp_predictions(snapshot.posterior, eval_times),
                        checkpoint_state=checkpoint_state,
                    )
    else:
        if base_name in {"uniform", "random"}:
            if selected_times_final is None:
                selected_times_final = []
            if selected_times_final:
                _evaluate_one_shot(
                    selected_times_final,
                    include_observed_errors=False,
                    keep_measurements=True,
                    timing_stage="one_shot_prepare",
                    timing_step=len(selected_times_final),
                )
            logger.info(
                "Collected %d %s acquisitions (one-shot)",
                len(selected_times_final),
                base_name,
            )
        else:
            logger.info("Running %d acquisition steps", int(num_steps))
            for step_idx in range(int(num_steps)):
                if loop.remaining_candidates.size == 0:
                    logger.warning(
                        "No candidates remain after %d steps, stopping early",
                        step_idx,
                    )
                    break
                loop.step()
                step_num = step_idx + 1
                _record_loop_fit_timings("acquisition_step", step_num)

    logger.info("Final refit with %d measurements", len(loop.measurements))
    logger.info("Evaluating reconstruction at %d time points", len(eval_times))
    final_step = (
        len(manual_acquisition_history)
        if manual_acquisition_history is not None
        else len(loop.history)
    )
    snapshot = _evaluate_loop_state(
        include_observed_errors=True,
        timing_stage="final_eval",
        timing_step=final_step,
    )
    reference_for_eval = snapshot.reference
    posterior = snapshot.posterior
    errors = snapshot.errors
    true_measures = snapshot.true_measures
    uniform_metric = snapshot.uniform_metric
    velocity_metric = snapshot.velocity_metric
    common_reference_errors = snapshot.common_reference_errors
    common_reference_uniform_metric = snapshot.common_reference_uniform_metric
    common_reference_velocity_metric = snapshot.common_reference_velocity_metric
    observed_errors = snapshot.observed_errors
    wasserstein_velocities = eval_velocities
    wasserstein_velocity_grid = provided_velocity_grid
    velocity_grid_times = provided_velocity_grid_times
    logger.info(
        "List of Wasserstein velocities between eval times: %s",
        wasserstein_velocities.tolist(),
    )

    logger.info(
        "Strategy '%s' completed: uniform_metric=%.4f, velocity_metric=%.4f",
        name, uniform_metric, velocity_metric
    )
    if common_reference_uniform_metric is not None and common_reference_velocity_metric is not None:
        logger.info(
            "Strategy '%s' common reference: uniform_metric=%.4f, velocity_metric=%.4f",
            name,
            common_reference_uniform_metric,
            common_reference_velocity_metric,
        )
    logger.debug(
        "Error stats for '%s': min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
        name, float(np.min(errors)), float(np.max(errors)),
        float(np.mean(errors)), float(np.std(errors))
    )
    if observed_errors.size > 0:
        logger.info(
            "Observed-time reconstruction errors for '%s': min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
            name,
            float(np.min(observed_errors)),
            float(np.max(observed_errors)),
            float(np.mean(observed_errors)),
            float(np.std(observed_errors)),
        )
        logger.debug(
            "Observed-time errors for '%s': %s",
            name,
            [
                {"time": float(t), "error": float(err)}
                for t, err in zip(snapshot.observed_times, observed_errors)
            ],
        )
    else:
        logger.info("No observed measurements available for '%s'", name)

    # Collect acquisition history for post-hoc analysis
    observed_times = snapshot.observed_times
    if manual_acquisition_history is not None:
        acquisition_history = manual_acquisition_history
    else:
        acquisition_history = [
            {
                "iteration": rec.iteration,
                "selected_time": float(rec.selected_time),
                "score": float(rec.score),
            }
            for rec in loop.history
        ]
    observed_sample_sizes = snapshot.observed_sample_sizes

    # Get GP predictions at evaluation times for visualization
    gp_predictions = _extract_gp_predictions(posterior, eval_times)
    gp_hyperparams = None
    if posterior is not None:
        try:
            gp_hyperparams = _extract_gp_hyperparams(posterior)
        except Exception as e:
            logger.warning("Could not extract GP hyperparams: %s", e)
    surrogate_state = _extract_surrogate_state(surrogate)
    warp = surrogate_state["warp"]
    coefficients = surrogate_state["coefficients"]
    basis_mean = surrogate_state["basis_mean"]
    basis_components = surrogate_state["basis_components"]
    basis_singular_values = surrogate_state["basis_singular_values"]

    return StrategyResult(
        name=name,
        errors=errors,
        uniform_metric=uniform_metric,
        velocity_metric=velocity_metric,
        common_reference_errors=common_reference_errors,
        common_reference_uniform_metric=common_reference_uniform_metric,
        common_reference_velocity_metric=common_reference_velocity_metric,
        wasserstein_velocities=wasserstein_velocities,
        wasserstein_velocity_grid=wasserstein_velocity_grid,
        velocity_grid_times=velocity_grid_times,
        observed_times=observed_times,
        observed_reconstruction_errors=observed_errors,
        acquisition_history=acquisition_history,
        gp_predictions=gp_predictions,
        gp_hyperparams=gp_hyperparams,
        basis=surrogate.basis,
        reference=reference_for_eval,
        warp=warp,
        coefficients=coefficients,
        observed_sample_sizes=observed_sample_sizes,
        basis_mean=basis_mean,
        basis_components=basis_components,
        basis_singular_values=basis_singular_values,
        checkpoint_results=checkpoint_results,
        timing_records=timing_records,
    )


def _build_acquisition_fn(
    name: str,
    cfg: DictConfig,
    schedule: Sequence[float] | None,
    rng: np.random.Generator,
    base_name: str | None = None,
    acquisition_override: DictConfig | None = None,
    trajectory: object | None = None,
    candidate_pool: np.ndarray | None = None,
    velocity_rng: np.random.Generator | None = None,
):
    if base_name is None:
        base_name = _strategy_base(name)
    if base_name == "active":
        if acquisition_override is None:
            raise ValueError(f"active strategy '{name}' requires an acquisition config")
        acq_cfg = acquisition_override
    else:
        if acquisition_override is not None:
            acq_cfg = acquisition_override
        else:
            baseline_key = name if name in cfg.baselines else base_name
            if baseline_key not in cfg.baselines:
                raise ValueError(f"unknown strategy '{name}'")
            acq_cfg = cfg.baselines[baseline_key].acquisition
    extra = {}
    if base_name == "random":
        extra["rng"] = rng
    if base_name == "uniform":
        if schedule is None:
            raise ValueError(f"{name} baseline requires a schedule")
        extra["schedule"] = schedule
    # Velocity-weighted acquisition (supports single-phase and two-phase modes)
    velocity_weighting = cfg.get("velocity_weighting")
    if velocity_weighting is not None and bool(velocity_weighting.get("enabled", False)):
        if trajectory is None or candidate_pool is None:
            raise ValueError("velocity_weighting requires trajectory and candidate_pool")
        vrng = velocity_rng or np.random.default_rng()
        v_power = float(velocity_weighting.get("power", 1.0))
        two_phase = bool(velocity_weighting.get("two_phase", False))
        switch_step = int(velocity_weighting.get("switch_step", 14))
        try:
            v_weights = trajectory.velocity_proxy(
                np.asarray(candidate_pool, dtype=float),
                n=512,
                rng=vrng,
                noisy=False,
            )
            if len(v_weights) == len(candidate_pool) - 1:
                v_weights = np.concatenate([v_weights, [v_weights[-1]]])
            v_min, v_max = float(np.min(v_weights)), float(np.max(v_weights))
            if v_max > v_min:
                v_weights = 0.05 + 0.95 * (v_weights - v_min) / (v_max - v_min)
            v_times = np.asarray(candidate_pool, dtype=float)

            if two_phase:
                # Phase 1: pure uncertainty; Phase 2: velocity-weighted
                phase1 = UncertaintySampler()
                phase2 = VelocityWeightedUncertaintySampler(
                    velocity_weights=v_weights,
                    velocity_times=v_times,
                    velocity_power=v_power,
                )
                acq = TwoPhaseAcquisition(
                    phase1_fn=phase1,
                    phase2_fn=phase2,
                    switch_step=switch_step,
                )
                logger.info(
                    "Two-phase velocity weighting: switch_step=%d, power=%.2f, weight_range=[%.3f, %.3f]",
                    switch_step, v_power, float(np.min(v_weights)), float(np.max(v_weights)),
                )
                return acq
            else:
                extra["velocity_weights"] = v_weights
                extra["velocity_times"] = v_times
                extra["velocity_power"] = v_power
                logger.info(
                    "Velocity weighting enabled: power=%.2f, weight_range=[%.3f, %.3f]",
                    v_power, float(np.min(v_weights)), float(np.max(v_weights)),
                )
        except Exception as exc:
            logger.warning("Could not compute velocity weights: %s. Falling back to unweighted.", exc)
    return instantiate(acq_cfg, **extra)


def _persist_results(
    output_dir: Path,
    results: Sequence[StrategyResult],
    context: "ExperimentContext",
    cfg: DictConfig,
    strategies: Sequence[str],
    num_steps_by_strategy: dict[str, int],
    trajectory: object,
    checkpoints: Sequence[int] | None,
    include_reconstruction: bool = False,
    log_paths: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(output_dir, results)
    common_errors_path, observed_errors_path = _write_error_tables(
        output_dir,
        results,
        context.eval_times,
        context.eval_velocities,
    )
    checkpoint_metrics_path = _write_checkpoint_metrics(output_dir, results)
    checkpoint_errors_path, checkpoint_common_errors_path = _write_checkpoint_errors(
        output_dir,
        results,
        context.eval_times,
        context.eval_velocities,
    )
    checkpoint_gp_dir = _write_checkpoint_gp_predictions(output_dir, results)
    checkpoint_state_dir = None
    if _save_checkpoint_state_enabled(cfg):
        checkpoint_state_dir = _write_checkpoint_state(output_dir, results)
    else:
        logger.info("Skipping checkpoint state pickle export")
    timing_path = _write_timing_table(output_dir, results)
    acquisition_path = _write_acquisition_artifacts(output_dir, results)
    metadata_path = _write_metadata(
        output_dir,
        cfg,
        context,
        strategies,
        num_steps_by_strategy,
        trajectory,
        checkpoints=checkpoints,
    )
    reconstruction_dir = None
    if include_reconstruction:
        reconstruction_dir = _write_reconstruction_artifacts(
            output_dir,
            results,
            context.common_reference,
        )
    if log_paths:
        logger.info("Saved metrics to %s", output_dir / "metrics.json")
        logger.info("Saved per-time errors to %s", output_dir / "errors.csv")
        if common_errors_path is not None:
            logger.info("Saved common-reference errors to %s", common_errors_path)
        if observed_errors_path is not None:
            logger.info("Saved observed-time errors to %s", observed_errors_path)
        if checkpoint_metrics_path is not None:
            logger.info("Saved checkpoint metrics to %s", checkpoint_metrics_path)
        if checkpoint_errors_path is not None:
            logger.info("Saved checkpoint errors to %s", checkpoint_errors_path)
        if checkpoint_common_errors_path is not None:
            logger.info(
                "Saved checkpoint common-reference errors to %s",
                checkpoint_common_errors_path,
            )
        if checkpoint_gp_dir is not None:
            logger.info("Saved checkpoint GP predictions to %s", checkpoint_gp_dir)
        if checkpoint_state_dir is not None:
            logger.info("Saved checkpoint state to %s", checkpoint_state_dir)
        if timing_path is not None:
            logger.info("Saved timing records to %s", timing_path)
        logger.info("Saved acquisition artifacts to %s", acquisition_path)
        logger.info("Saved metadata to %s", metadata_path)
        if reconstruction_dir is not None:
            logger.info("Saved reconstruction objects to %s", reconstruction_dir)


@hydra.main(version_base=None, config_path="../conf", config_name="exp_sequential_branching")
def main(cfg: DictConfig) -> None:
    """Run the configured active sampling experiment."""
    # Configure logging level based on config if present
    log_level = getattr(logging, str(cfg.get("log_level", "INFO")).upper(), logging.INFO)
    logging.getLogger("active_wasserstein").setLevel(log_level)
    logging.getLogger("experiments").setLevel(log_level)

    logger.info("="*60)
    logger.info("Starting active sampling experiment")
    logger.info("="*60)
    logger.info("Output directory: %s", HydraConfig.get().runtime.output_dir)
    logger.debug("Full configuration:\n%s", OmegaConf.to_yaml(cfg))

    rng = np.random.default_rng(int(cfg.seed))
    logger.info("Random seed: %d", int(cfg.seed))

    logger.info("Instantiating trajectory")
    trajectory = instantiate(cfg.trajectory)
    logger.debug(
        "Trajectory: %s (t_start=%.2f, t_end=%.2f)",
        type(trajectory).__name__,
        trajectory.t_start,
        trajectory.t_end,
    )

    context = _build_experiment_context(cfg, trajectory)
    logger.info(
        "Candidate pool: %d candidates in [%.4f, %.4f]",
        len(context.candidate_pool),
        context.candidate_pool[0],
        context.candidate_pool[-1],
    )
    logger.info("Initial times: %s", context.initial_times)
    logger.info(
        "Evaluation times: %d points in [%.4f, %.4f]",
        len(context.eval_times),
        context.eval_times[0],
        context.eval_times[-1],
    )
    logger.info("Velocity source: %s", context.velocity_source)
    if context.common_reference is not None:
        logger.info("Built common reference distribution for evaluation")

    checkpoints = _normalize_checkpoints(cfg.evaluation.get("checkpoints"))
    if checkpoints:
        logger.info("Checkpoint steps: %s", checkpoints)

    uniform_schedule = build_uniform_schedule(cfg, trajectory, context.initial_times)
    logger.debug("Uniform schedule: %s", uniform_schedule)

    strategy_entries = _resolve_strategy_entries(cfg)
    strategies = [name for name, _ in strategy_entries]
    logger.info("Strategies to run: %s", strategies)
    results: list[StrategyResult] = []
    num_steps_by_strategy: dict[str, int] = {}
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    try:
        shared_path = _write_shared_artifacts(output_dir, context)
        logger.info("Saved shared artifacts to %s", shared_path)
    except Exception as exc:
        logger.warning("Could not save shared artifacts: %s", exc)

    for name, strategy_entry in strategy_entries:
        strategy_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        base_name = _resolve_strategy_base(name, strategy_entry)
        schedule = None
        if base_name == "uniform":
            schedule = uniform_schedule
        strategy_cfg, reference_override_cfg, reference_overridden, acquisition_override = (
            _resolve_strategy_overrides(cfg, name, strategy_entry)
        )
        acquisition_fn = _build_acquisition_fn(
            name,
            strategy_cfg,
            schedule,
            strategy_rng,
            base_name=base_name,
            acquisition_override=acquisition_override,
            trajectory=trajectory,
            candidate_pool=context.candidate_pool,
            velocity_rng=strategy_rng,
        )
        strategy_steps = int(cfg.num_steps)
        pre_acquired_times = None
        baseline_cfg = (
            strategy_cfg.baselines.get(name)
            if name in strategy_cfg.baselines
            else strategy_cfg.baselines.get(base_name)
        )
        one_shot_override = _resolve_one_shot_override(cfg, name, strategy_entry)
        if one_shot_override is True:
            if schedule is None:
                raise ValueError(f"one_shot strategy '{name}' requires a schedule")
            pre_acquired_times = schedule
            strategy_steps = 0
        elif baseline_cfg is not None and bool(baseline_cfg.get("one_shot", False)):
            if schedule is None:
                raise ValueError(f"one_shot baseline '{name}' requires a schedule")
            pre_acquired_times = schedule
            strategy_steps = 0
        num_steps_by_strategy[name] = strategy_steps
        try:
            result = run_strategy(
                name=name,
                cfg=strategy_cfg,
                trajectory=trajectory,
                candidate_pool=context.candidate_pool,
                initial_times=context.initial_times,
                initial_measurements=context.initial_measurements,
                initial_reference=context.initial_reference,
                rng=strategy_rng,
                acquisition_fn=acquisition_fn,
                num_steps=strategy_steps,
                checkpoints=checkpoints,
                pre_acquired_times=pre_acquired_times,
                common_reference=context.common_reference,
                eval_times=context.eval_times,
                eval_sample_method=context.eval_sample_method,
                eval_sample_size=context.eval_sample_size,
                eval_true_measures=context.eval_true_measures,
                eval_velocities=context.eval_velocities,
                velocity_grid_times=context.velocity_grid_times,
                velocity_grid=context.velocity_grid,
                eval_barycenter_reference=context.eval_barycenter_reference,
                reference_override_cfg=reference_override_cfg,
                reference_overridden=reference_overridden,
            )
        except Exception:
            try:
                _persist_results(
                    output_dir,
                    results,
                    context,
                    cfg,
                    strategies,
                    num_steps_by_strategy,
                    trajectory,
                    checkpoints,
                    include_reconstruction=False,
                    log_paths=False,
                )
                logger.info(
                    "Saved partial results before failing on strategy '%s'", name
                )
            except Exception as exc:
                logger.warning(
                    "Could not persist partial results after '%s' failure: %s",
                    name,
                    exc,
                )
            raise
        results.append(result)
        try:
            _persist_results(
                output_dir,
                results,
                context,
                cfg,
                strategies,
                num_steps_by_strategy,
                trajectory,
                checkpoints,
                include_reconstruction=False,
                log_paths=False,
            )
            logger.info("Saved intermediate results after strategy '%s'", name)
        except Exception as exc:
            logger.warning(
                "Could not persist intermediate results after '%s': %s", name, exc
            )

    _persist_results(
        output_dir,
        results,
        context,
        cfg,
        strategies,
        num_steps_by_strategy,
        trajectory,
        checkpoints,
        include_reconstruction=True,
        log_paths=True,
    )
    for r in results:
        msg = (
            f"{r.name}: uniform={r.uniform_metric:.4f}, "
            f"velocity={r.velocity_metric:.4f}"
        )
        if r.common_reference_uniform_metric is not None:
            msg += (
                f", common_uniform={r.common_reference_uniform_metric:.4f}, "
                f"common_velocity={r.common_reference_velocity_metric:.4f}"
            )
        print(msg)


if __name__ == "__main__":
    main()
