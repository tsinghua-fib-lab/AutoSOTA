"""Utility helpers for active sampling experiments."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from active_wasserstein import (
    compute_wasserstein_distance,
    reconstruct_distributions,
    wasserstein_barycenter,
)
from experiments.components import (
    build_reference_from_measurements,
    make_measurement_oracle,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentContext:
    candidate_pool: np.ndarray
    initial_times: list[float]
    initial_measurements: list[object]
    initial_reference: object | None
    eval_times: np.ndarray
    eval_true_measures: list[object]
    eval_velocities: np.ndarray
    velocity_grid_times: np.ndarray | None
    velocity_grid: np.ndarray | None
    eval_barycenter_reference: object | None
    common_reference: object | None
    eval_sample_method: str
    eval_sample_size: int | None
    reference_source: str
    velocity_source: str


def _optional_int(value: Any | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return int(value)


def _first_non_none(*values: Any) -> Any | None:
    for value in values:
        if value is not None:
            return value
    return None


def _resolve_sampler(
    trajectory: object,
    *candidates: Any,
) -> tuple[str, Any]:
    method = _first_non_none(*candidates)
    if method is None:
        raise ValueError("sample_method must be provided")
    method = str(method)
    if not hasattr(trajectory, method):
        raise AttributeError(f"trajectory has no method '{method}'")
    return method, getattr(trajectory, method)


def _resolve_sample_size(*candidates: Any) -> int | None:
    value = _first_non_none(*candidates)
    if value is None:
        return None
    return _optional_int(value)


def _resolve_barycenter_backend(
    cfg: DictConfig,
    source_cfg: DictConfig | None = None,
) -> str:
    reference_cfg = getattr(cfg, "reference", None)
    default_backend = "pot"
    if reference_cfg is not None and reference_cfg.get("backend") is not None:
        default_backend = str(reference_cfg.get("backend"))
    backend_raw = default_backend
    if source_cfg is not None and source_cfg.get("backend") is not None:
        backend_raw = source_cfg.get("backend")
    backend = str(backend_raw).strip().lower()
    if backend != "pot":
        raise ValueError(
            f"Unsupported barycenter backend '{backend_raw}'. Expected 'pot'."
        )
    return backend


def _resolve_active_loop_reference_settings(
    cfg: DictConfig,
    reference_source: str,
) -> tuple[bool, int, int, float, str]:
    active_loop_cfg = cfg.get("active_loop")
    recompute_reference = reference_source != "eval_barycenter"
    reference_cfg = getattr(cfg, "reference", None)
    barycenter_size = int(getattr(reference_cfg, "barycenter_size", 256))
    barycenter_num_iter = int(getattr(reference_cfg, "num_iter", 150))
    barycenter_reg = float(getattr(reference_cfg, "reg", 0.0))
    backend = _resolve_barycenter_backend(cfg, active_loop_cfg)
    if active_loop_cfg is not None:
        if "recompute_reference_as_barycenter" in active_loop_cfg:
            recompute_reference = bool(
                active_loop_cfg.get("recompute_reference_as_barycenter")
            )
        barycenter_size = int(active_loop_cfg.get("barycenter_size", barycenter_size))
        barycenter_num_iter = int(
            active_loop_cfg.get("barycenter_num_iter", barycenter_num_iter)
        )
        barycenter_reg = float(active_loop_cfg.get("barycenter_reg", barycenter_reg))
    return (
        recompute_reference,
        barycenter_size,
        barycenter_num_iter,
        barycenter_reg,
        backend,
    )


def _resolve_barycenter_params(
    cfg: DictConfig, source_cfg: DictConfig | None
) -> tuple[int, int, float, str]:
    backend = _resolve_barycenter_backend(cfg, source_cfg)
    if source_cfg is None:
        return (
            int(cfg.reference.barycenter_size),
            int(cfg.reference.num_iter),
            float(cfg.reference.get("reg", 0.0)),
            backend,
        )
    return (
        int(source_cfg.get("barycenter_size", int(cfg.reference.barycenter_size))),
        int(source_cfg.get("num_iter", int(cfg.reference.num_iter))),
        float(source_cfg.get("reg", float(cfg.reference.get("reg", 0.0)))),
        backend,
    )


def _build_barycenter(
    cfg: DictConfig,
    measures: Sequence[object],
    rng: np.random.Generator,
    source_cfg: DictConfig | None = None,
    error_prefix: str = "barycenter",
) -> object:
    if len(measures) == 0:
        raise ValueError(f"{error_prefix} requires at least one measure")
    if len(measures) == 1:
        return measures[0]
    barycenter_size, num_iter, barycenter_reg, backend = _resolve_barycenter_params(
        cfg,
        source_cfg,
    )
    logger.info(
        "Computing %s using barycenter of %d measures with size %d over %d iterations "
        "(backend=%s, reg=%.4g)",
        error_prefix,
        len(measures),
        barycenter_size,
        num_iter,
        backend,
        barycenter_reg,
    )
    return wasserstein_barycenter(
        measures,
        barycenter_size=barycenter_size,
        num_iter=num_iter,
        reg=barycenter_reg,
        rng=rng,
        backend=backend,
    )


def _to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("value must contain at least one element")
    return float(arr[0])


def _extract_gp_hyperparams(posterior: object) -> dict | None:
    if not hasattr(posterior, "models"):
        return None
    models = getattr(posterior, "models")
    likelihoods = getattr(posterior, "likelihoods", [])
    kernel_spec = getattr(posterior, "kernel_spec", None)
    if models is None:
        return None
    model_params = []
    for idx, model in enumerate(models):
        try:
            lengthscale = _to_float(model.covar_module.base_kernel.lengthscale)
            outputscale = _to_float(model.covar_module.outputscale)
        except Exception:
            lengthscale = None
            outputscale = None
        kernel_snapshot = None
        if kernel_spec is not None and hasattr(model, "covar_module"):
            try:
                kernel_snapshot = kernel_spec.snapshot(model.covar_module)
            except Exception:
                kernel_snapshot = None
        noise_scale = None
        noise_summary = None
        if idx < len(likelihoods):
            lk = likelihoods[idx]
            if hasattr(lk, "noise_scale"):
                try:
                    noise_scale = _to_float(lk.noise_scale)
                except Exception:
                    noise_scale = None
            if hasattr(lk, "noise"):
                try:
                    noise = lk.noise.detach().cpu().numpy()
                    noise_summary = {
                        "mean": float(np.mean(noise)),
                        "min": float(np.min(noise)),
                        "max": float(np.max(noise)),
                    }
                except Exception:
                    noise_summary = None
        model_params.append(
            {
                "lengthscale": lengthscale,
                "outputscale": outputscale,
                "kernel": kernel_snapshot,
                "noise_scale": noise_scale,
                "noise_summary": noise_summary,
            }
        )

    base_params = {}
    for name in ("lengthscale", "variance", "prior_variance", "jitter"):
        if hasattr(posterior, name):
            try:
                base_params[name] = _to_float(getattr(posterior, name))
            except Exception:
                base_params[name] = None
    if hasattr(posterior, "scales"):
        try:
            scales = np.asarray(getattr(posterior, "scales"), dtype=float)
            base_params["scales"] = scales.tolist()
        except Exception:
            base_params["scales"] = None

    input_scaler = getattr(posterior, "input_scaler", None)
    scaler_info = None
    if input_scaler is not None:
        try:
            scaler_info = {
                "t_min": float(input_scaler.t_min),
                "t_max": float(input_scaler.t_max),
            }
        except Exception:
            scaler_info = None
    output_scaler = getattr(posterior, "output_scaler", None)
    output_scaler_info = None
    if output_scaler is not None and hasattr(output_scaler, "scales"):
        try:
            output_scaler_info = {
                "scales": np.asarray(output_scaler.scales, dtype=float).tolist()
            }
        except Exception:
            output_scaler_info = None
    return {
        "base": base_params,
        "per_output": model_params,
        "input_scaler": scaler_info,
        "output_scaler": output_scaler_info,
    }


def _extract_gp_predictions(
    posterior: object | None,
    eval_times: Sequence[float] | np.ndarray,
) -> dict | None:
    if posterior is None:
        return None
    try:
        if hasattr(posterior, "_predict"):
            pred_mean, pred_var = posterior._predict(eval_times)
            pred_std = np.sqrt(pred_var)
        else:
            pred_mean = np.array([posterior.mean(t) for t in eval_times]).T
            pred_var = np.array([posterior.marginal_variance(t) for t in eval_times]).T
            pred_std = np.sqrt(pred_var)
        return {
            "times": np.asarray(eval_times, dtype=float).tolist(),
            "mean": pred_mean.tolist(),
            "std": pred_std.tolist(),
        }
    except Exception as exc:
        logger.warning("Could not extract GP predictions: %s", exc)
        return None


def _extract_surrogate_state(surrogate: object) -> dict:
    posterior = getattr(surrogate, "posterior", None)
    warp = getattr(posterior, "warp", None) if posterior is not None else None

    coefficients = None
    if hasattr(surrogate, "coefficients") and surrogate.coefficients is not None:
        coefficients = surrogate.coefficients.copy()

    basis_mean = None
    if hasattr(surrogate, "basis_mean") and surrogate.basis_mean is not None:
        basis_mean = surrogate.basis_mean.copy()

    basis_components = None
    if (
        hasattr(surrogate, "basis_components")
        and surrogate.basis_components is not None
    ):
        basis_components = surrogate.basis_components.copy()

    basis_singular_values = None
    if (
        hasattr(surrogate, "basis_singular_values")
        and surrogate.basis_singular_values is not None
    ):
        basis_singular_values = surrogate.basis_singular_values.copy()

    return {
        "warp": warp,
        "coefficients": coefficients,
        "basis_mean": basis_mean,
        "basis_components": basis_components,
        "basis_singular_values": basis_singular_values,
    }


def build_candidate_pool(cfg: DictConfig, trajectory: object) -> np.ndarray:
    """Build the candidate pool used by the acquisition loop."""
    if cfg.candidate_pool.times is not None:
        return np.asarray(cfg.candidate_pool.times, dtype=float)
    if hasattr(trajectory, "candidate_times"):
        candidate_times = getattr(trajectory, "candidate_times")
        if callable(candidate_times):
            candidate_times = candidate_times()
        candidate_times = np.asarray(candidate_times, dtype=float)
        if candidate_times.size > 0:
            return candidate_times
    t_start = cfg.candidate_pool.t_start
    t_end = cfg.candidate_pool.t_end
    if t_start is None or t_end is None:
        if not hasattr(trajectory, "t_start") or not hasattr(trajectory, "t_end"):
            raise ValueError(
                "candidate_pool requires t_start/t_end when trajectory lacks them"
            )
        t_start = float(trajectory.t_start)
        t_end = float(trajectory.t_end)
    num = int(cfg.candidate_pool.num_candidates)
    return np.linspace(float(t_start), float(t_end), num)


def build_uniform_schedule(
    cfg: DictConfig,
    trajectory: object,
    initial_times: Sequence[float],
) -> list[float]:
    """Build a uniform acquisition schedule for the baseline."""
    t_start = cfg.uniform_schedule.t_start
    t_end = cfg.uniform_schedule.t_end
    if t_start is None or t_end is None:
        if not hasattr(trajectory, "t_start") or not hasattr(trajectory, "t_end"):
            raise ValueError(
                "uniform_schedule requires t_start/t_end when trajectory lacks them"
            )
        t_start = float(trajectory.t_start)
        t_end = float(trajectory.t_end)
    num_steps = int(cfg.uniform_schedule.num_steps)
    grid = np.linspace(float(t_start), float(t_end), num_steps + 2)
    schedule = list(grid[1:])
    if cfg.uniform_schedule.exclude_initial:
        schedule = [
            t for t in schedule if all(not np.isclose(t, s) for s in initial_times)
        ]
    if len(schedule) < num_steps:
        raise ValueError("uniform schedule shorter than requested num_steps")
    return schedule[:num_steps]


def build_eval_times(cfg: DictConfig, trajectory: object) -> np.ndarray:
    """Build evaluation times from config."""
    if cfg.evaluation.times is not None:
        return np.asarray(cfg.evaluation.times, dtype=float)
    if hasattr(trajectory, "eval_times"):
        eval_times = getattr(trajectory, "eval_times")
        if callable(eval_times):
            eval_times = eval_times()
        eval_times = np.asarray(eval_times, dtype=float)
        if eval_times.size > 0:
            return eval_times

    t_start = cfg.evaluation.t_start
    t_end = cfg.evaluation.t_end
    if t_start is None or t_end is None:
        if not hasattr(trajectory, "t_start") or not hasattr(trajectory, "t_end"):
            raise ValueError(
                "evaluation requires t_start/t_end when trajectory lacks them"
            )
        t_start = float(trajectory.t_start)
        t_end = float(trajectory.t_end)

    return np.linspace(float(t_start), float(t_end), int(cfg.evaluation.num_eval))


def build_common_reference(
    cfg: DictConfig,
    trajectory: object,
    eval_times: Iterable[float],
    rng: np.random.Generator,
) -> object | None:
    """Build a common reference distribution shared across strategies."""
    common_cfg = cfg.get("common_reference")
    if common_cfg is None:
        return None
    if not bool(common_cfg.get("enabled", False)):
        return None
    source = str(common_cfg.get("source", "eval_barycenter"))
    if source == "fixed":
        fixed_cfg = common_cfg.get("fixed")
        if fixed_cfg is None:
            raise ValueError("common_reference.fixed is required when source='fixed'")
        return instantiate(fixed_cfg, _convert_="object")
    if source == "times":
        times_cfg = common_cfg.get("times")
        if times_cfg is None:
            raise ValueError("common_reference.times is required when source='times'")
        if isinstance(times_cfg, (float, int, np.floating, np.integer)):
            times = [float(times_cfg)]
        elif isinstance(times_cfg, str):
            try:
                times = [float(times_cfg)]
            except ValueError as exc:
                raise ValueError(
                    "common_reference.times must be a number or list of numbers"
                ) from exc
        else:
            times = [float(t) for t in list(times_cfg)]
        if not times:
            raise ValueError("common_reference.times must contain at least one time")
        sample_method, sampler = _resolve_sampler(
            trajectory,
            common_cfg.get("sample_method"),
            cfg.evaluation.get("sample_method"),
            cfg.oracle.sample_method,
        )
        sample_size = _resolve_sample_size(
            common_cfg.get("sample_size"),
            cfg.evaluation.get("sample_size"),
        )
        measures = [sampler(float(t), sample_size, rng=rng) for t in times]
        return _build_barycenter(
            cfg=cfg,
            measures=measures,
            rng=rng,
            source_cfg=common_cfg,
            error_prefix="common_reference.times",
        )
    if source != "eval_barycenter":
        raise ValueError(f"common_reference source '{source}' is not supported")
    sample_method, sampler = _resolve_sampler(
        trajectory,
        common_cfg.get("sample_method"),
        cfg.evaluation.get("sample_method"),
        cfg.oracle.sample_method,
    )
    sample_size = _resolve_sample_size(
        common_cfg.get("sample_size"),
        cfg.evaluation.get("sample_size"),
    )
    eval_times = np.asarray(list(eval_times), dtype=float)
    if eval_times.size == 0:
        raise ValueError("common_reference requires at least one evaluation time")
    measures = [sampler(float(t), sample_size, rng=rng) for t in eval_times]
    return _build_barycenter(
        cfg=cfg,
        measures=measures,
        rng=rng,
        source_cfg=common_cfg,
        error_prefix="common_reference",
    )


def build_eval_barycenter_reference(
    cfg: DictConfig,
    trajectory: object,
    eval_times: Iterable[float],
    rng: np.random.Generator,
) -> object:
    """Build a fixed reference as the barycenter of evaluation distributions."""
    reference_cfg = getattr(cfg, "reference", None)
    sample_method, sampler = _resolve_sampler(
        trajectory,
        None if reference_cfg is None else reference_cfg.get("sample_method"),
        cfg.evaluation.get("sample_method"),
        cfg.oracle.sample_method,
    )
    sample_size = _resolve_sample_size(
        None if reference_cfg is None else reference_cfg.get("sample_size"),
        cfg.evaluation.get("sample_size"),
    )
    eval_times = np.asarray(list(eval_times), dtype=float)
    if eval_times.size == 0:
        raise ValueError(
            "eval_barycenter reference requires at least one evaluation time"
        )
    measures = [sampler(float(t), sample_size, rng=rng) for t in eval_times]
    return _build_barycenter(
        cfg=cfg,
        measures=measures,
        rng=rng,
        source_cfg=reference_cfg,
        error_prefix="eval_barycenter reference",
    )


def build_eval_barycenter_from_measures(
    cfg: DictConfig,
    measures: Sequence[object],
    rng: np.random.Generator,
) -> object:
    """Build a fixed reference as the barycenter of provided eval measures."""
    reference_cfg = getattr(cfg, "reference", None)
    return _build_barycenter(
        cfg=cfg,
        measures=measures,
        rng=rng,
        source_cfg=reference_cfg,
        error_prefix="eval_barycenter reference",
    )


def _resolve_reference_source(cfg: DictConfig) -> str:
    reference_cfg = getattr(cfg, "reference", None)
    source = "initial_barycenter"
    if reference_cfg is not None and reference_cfg.get("source") is not None:
        source = str(reference_cfg.get("source"))
    return source


def _resolve_initial_times(cfg: DictConfig, candidate_pool: np.ndarray) -> list[float]:
    initial_cfg = cfg.get("initial_times")
    if initial_cfg is None:
        return [float(candidate_pool[0])]
    initial_list = list(initial_cfg)
    if len(initial_list) == 0:
        return [float(candidate_pool[0])]
    return [float(t) for t in initial_list]


def _build_initial_measurements(
    cfg: DictConfig,
    trajectory: object,
    initial_times: Sequence[float],
    rng: np.random.Generator,
) -> list[object]:
    oracle_sample_size = _resolve_sample_size(cfg.oracle.sample_size)
    oracle = make_measurement_oracle(
        trajectory=trajectory,
        sample_size=oracle_sample_size,
        sample_method=str(cfg.oracle.sample_method),
        rng=rng,
    )
    return [oracle(float(t)) for t in initial_times]


def _build_experiment_context(
    cfg: DictConfig,
    trajectory: object,
) -> ExperimentContext:
    candidate_pool = build_candidate_pool(cfg, trajectory)
    initial_times = _resolve_initial_times(cfg, candidate_pool)
    eval_times = build_eval_times(cfg, trajectory)

    eval_sample_method, eval_sampler = _resolve_sampler(
        trajectory,
        cfg.evaluation.get("sample_method"),
        cfg.oracle.sample_method,
    )
    eval_sample_size = _resolve_sample_size(cfg.evaluation.get("sample_size"))
    oracle_sample_method = str(cfg.oracle.sample_method)
    oracle_sample_size = _resolve_sample_size(cfg.oracle.sample_size)
    persistent_times = np.unique(
        np.concatenate(
            [
                np.asarray(candidate_pool, dtype=float),
                np.asarray(list(eval_times), dtype=float),
                np.asarray(list(initial_times), dtype=float),
            ]
        )
    )
    if hasattr(trajectory, "prepare_persistent_cache"):
        if eval_sample_method in {"sample_persistent", "sample_persistent_clean"}:
            eval_noisy = eval_sample_method != "sample_persistent_clean"
            seed = int(cfg.seed) + 17
            trajectory.prepare_persistent_cache(
                persistent_times,
                n=int(eval_sample_size or cfg.sample_size),
                rng=None,
                noisy=eval_noisy,
                seed=seed,
                overwrite=False,
            )
        if oracle_sample_method in {"sample_persistent", "sample_persistent_clean"}:
            oracle_noisy = oracle_sample_method != "sample_persistent_clean"
            seed = int(cfg.seed) + 23
            trajectory.prepare_persistent_cache(
                persistent_times,
                n=int(oracle_sample_size or cfg.sample_size),
                rng=None,
                noisy=oracle_noisy,
                seed=seed,
                overwrite=False,
            )
    eval_rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), 2]))
    shared_eval_measures = [
        eval_sampler(float(t), eval_sample_size, rng=eval_rng) for t in eval_times
    ]

    velocity_sample_method, velocity_sampler = _resolve_sampler(
        trajectory,
        cfg.evaluation.get("velocity_sample_method"),
        eval_sample_method,
    )
    velocity_sample_size = _resolve_sample_size(
        cfg.evaluation.get("velocity_sample_size"),
        eval_sample_size,
    )
    if (
        velocity_sample_method == eval_sample_method
        and velocity_sample_size == eval_sample_size
    ):
        velocity_eval_measures = shared_eval_measures
        velocity_rng = eval_rng
    else:
        velocity_rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), 5]))
        velocity_eval_measures = [
            velocity_sampler(float(t), velocity_sample_size, rng=velocity_rng)
            for t in eval_times
        ]

    reference_source = _resolve_reference_source(cfg)
    eval_barycenter_reference = None
    if reference_source == "eval_barycenter":
        reference_rng = np.random.default_rng(
            np.random.SeedSequence([int(cfg.seed), 3])
        )
        eval_barycenter_reference = build_eval_barycenter_from_measures(
            cfg=cfg,
            measures=shared_eval_measures,
            rng=reference_rng,
        )

    common_reference_rng = np.random.default_rng(
        np.random.SeedSequence([int(cfg.seed), 1])
    )
    common_reference = build_common_reference(
        cfg=cfg,
        trajectory=trajectory,
        eval_times=eval_times,
        rng=common_reference_rng,
    )

    velocity_source = str(cfg.evaluation.get("velocity_source", "eval")).lower()
    if velocity_source not in {"eval", "grid", "trajectory"}:
        raise ValueError(
            "evaluation.velocity_source must be 'eval', 'grid', or 'trajectory'"
        )
    shared_eval_velocities = None
    shared_velocity_grid_times = None
    shared_velocity_grid = None
    if velocity_source == "trajectory":
        if not hasattr(trajectory, "velocity_proxy"):
            raise AttributeError("trajectory has no method 'velocity_proxy'")
        velocity_noisy = bool(cfg.evaluation.get("velocity_sample_noisy", False))
        shared_eval_velocities = np.asarray(
            trajectory.velocity_proxy(
                np.asarray(list(eval_times), dtype=float),
                n=int(velocity_sample_size or eval_sample_size or cfg.sample_size),
                rng=velocity_rng,
                noisy=velocity_noisy,
            ),
            dtype=float,
        )
        shared_velocity_grid_times = np.asarray(list(eval_times), dtype=float)
        shared_velocity_grid = shared_eval_velocities
    elif velocity_source == "grid":
        use_train_for_grid = True
        time_tolerance = float(cfg.trajectory.get("time_tolerance", 1.0e-6))
        try:
            grid_times, grid_velocities = compute_velocity_grid(
                trajectory=trajectory,
                candidate_pool=candidate_pool,
                eval_times=eval_times,
                eval_measures=velocity_eval_measures,
                eval_sample_method=str(velocity_sample_method),
                eval_sample_size=velocity_sample_size,
                n_support=int(cfg.evaluation.n_support),
                rng=velocity_rng,
                use_train_for_grid=use_train_for_grid,
                time_tolerance=time_tolerance,
            )
            if grid_times is not None and grid_velocities is not None:
                shared_eval_velocities = select_grid_velocities_left_endpoint(
                    grid_times,
                    grid_velocities,
                    np.asarray(list(eval_times), dtype=float),
                    time_tolerance,
                )
                shared_velocity_grid_times = grid_times
                shared_velocity_grid = grid_velocities
        except Exception as exc:
            logger.warning("Falling back to eval-grid velocities: %s", exc)
    if shared_eval_velocities is None:
        shared_eval_velocities = compute_wasserstein_velocities(
            true_measures=velocity_eval_measures,
            n_support=int(cfg.evaluation.n_support),
            eval_times=eval_times,
            rng=velocity_rng,
        )

    initial_rng = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), 0]))
    initial_measurements = _build_initial_measurements(
        cfg=cfg,
        trajectory=trajectory,
        initial_times=initial_times,
        rng=initial_rng,
    )

    initial_reference = None
    if reference_source == "initial_barycenter":
        reference_rng = np.random.default_rng(
            np.random.SeedSequence([int(cfg.seed), 4])
        )
        reference_backend = _resolve_barycenter_backend(
            cfg=cfg,
            source_cfg=getattr(cfg, "reference", None),
        )
        initial_reference = build_reference_from_measurements(
            measurements=initial_measurements,
            barycenter_size=int(cfg.reference.barycenter_size),
            num_iter=int(cfg.reference.num_iter),
            reg=float(cfg.reference.get("reg", 0.0)),
            rng=reference_rng,
            backend=reference_backend,
        )

    return ExperimentContext(
        candidate_pool=candidate_pool,
        initial_times=initial_times,
        initial_measurements=initial_measurements,
        initial_reference=initial_reference,
        eval_times=eval_times,
        eval_true_measures=shared_eval_measures,
        eval_velocities=shared_eval_velocities,
        velocity_grid_times=shared_velocity_grid_times,
        velocity_grid=shared_velocity_grid,
        eval_barycenter_reference=eval_barycenter_reference,
        common_reference=common_reference,
        eval_sample_method=str(eval_sample_method),
        eval_sample_size=eval_sample_size,
        reference_source=reference_source,
        velocity_source=velocity_source,
    )


def _compute_reconstruction_errors(
    posterior: object,
    basis: object,
    reference: object,
    eval_times: Iterable[float],
    true_measures: Sequence[object],
    n_support: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list]:
    eval_times = np.asarray(list(eval_times), dtype=float)
    if len(true_measures) != len(eval_times):
        raise ValueError(
            "true_measures and eval_times must have the same length "
            f"(got {len(true_measures)} and {len(eval_times)})"
        )
    recon_measures = reconstruct_distributions(
        times=eval_times,
        gp_posterior=posterior,
        basis=basis,
        reference=reference,
    )
    n_eval = int(len(eval_times))
    errors = np.empty(n_eval, dtype=float)
    log_progress = n_eval >= 20
    progress_stride = max(1, n_eval // 5)
    start = time.perf_counter()
    for i in range(n_eval):
        errors[i] = compute_wasserstein_distance(
            true_measures[i],
            recon_measures[i],
            n_support=n_support,
            rng=rng,
        )
        if log_progress and ((i + 1) % progress_stride == 0 or (i + 1) == n_eval):
            elapsed = time.perf_counter() - start
            logger.info(
                "Evaluation progress: %d/%d (%.1f%%), elapsed=%.1fs",
                i + 1,
                n_eval,
                100.0 * float(i + 1) / float(n_eval),
                elapsed,
            )
    return errors, recon_measures


def evaluate_reconstruction(
    posterior: object,
    basis: object,
    reference: object,
    eval_times: Iterable[float],
    trajectory: object,
    sample_method: str,
    sample_size: int | None,
    n_support: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list, list]:
    """Reconstruct distributions and evaluate Wasserstein errors."""
    eval_times = np.asarray(list(eval_times), dtype=float)
    _, sampler = _resolve_sampler(trajectory, sample_method)
    true_measures = [sampler(float(t), sample_size, rng=rng) for t in eval_times]
    errors, recon_measures = _compute_reconstruction_errors(
        posterior=posterior,
        basis=basis,
        reference=reference,
        eval_times=eval_times,
        true_measures=true_measures,
        n_support=n_support,
        rng=rng,
    )
    return errors, true_measures, recon_measures


def evaluate_reconstruction_with_true_measures(
    posterior: object,
    basis: object,
    reference: object,
    eval_times: Iterable[float],
    true_measures: Sequence[object],
    n_support: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate reconstruction errors using provided true measures."""
    errors, _ = _compute_reconstruction_errors(
        posterior=posterior,
        basis=basis,
        reference=reference,
        eval_times=eval_times,
        true_measures=true_measures,
        n_support=n_support,
        rng=rng,
    )
    return errors


def _resolve_eval_sampling(
    cfg: DictConfig,
    trajectory: object,
    eval_sample_method: str | None,
    eval_sample_size: int | None,
    eval_true_measures: Sequence[object] | None = None,
) -> tuple[str | None, int | None]:
    if eval_true_measures is not None:
        return eval_sample_method, eval_sample_size
    resolved_method = eval_sample_method
    if resolved_method is None:
        resolved_method, _ = _resolve_sampler(
            trajectory,
            cfg.evaluation.get("sample_method"),
            cfg.oracle.sample_method,
        )
    resolved_size = (
        eval_sample_size
        if eval_sample_size is not None
        else _resolve_sample_size(cfg.evaluation.get("sample_size"))
    )
    return str(resolved_method), resolved_size


def _evaluate_strategy_errors(
    posterior: object,
    basis: object,
    reference: object,
    eval_times: Iterable[float],
    trajectory: object,
    eval_sample_method: str | None,
    eval_sample_size: int | None,
    eval_true_measures: Sequence[object] | None,
    n_support: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list]:
    eval_times_list = list(eval_times)
    if eval_true_measures is not None:
        if len(eval_true_measures) != len(eval_times_list):
            raise ValueError(
                "eval_true_measures length must match eval_times "
                f"(got {len(eval_true_measures)} and {len(eval_times_list)})"
            )
        true_measures = list(eval_true_measures)
        errors = evaluate_reconstruction_with_true_measures(
            posterior=posterior,
            basis=basis,
            reference=reference,
            eval_times=eval_times_list,
            true_measures=true_measures,
            n_support=n_support,
            rng=rng,
        )
        return errors, true_measures
    if eval_sample_method is None:
        raise ValueError(
            "eval_sample_method must be provided when eval_true_measures is None"
        )
    errors, true_measures, _ = evaluate_reconstruction(
        posterior=posterior,
        basis=basis,
        reference=reference,
        eval_times=eval_times_list,
        trajectory=trajectory,
        sample_method=str(eval_sample_method),
        sample_size=eval_sample_size,
        n_support=n_support,
        rng=rng,
    )
    return errors, true_measures


def evaluate_observed_reconstruction(
    posterior: object,
    basis: object,
    reference: object,
    measurements: Sequence[object],
    n_support: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute Wasserstein errors at observed acquisition times."""
    if not measurements:
        return np.asarray([], dtype=float)
    times = np.array([m.time for m in measurements], dtype=float)
    observed_measures = [m.measure for m in measurements]
    recon_measures = reconstruct_distributions(
        times=times,
        gp_posterior=posterior,
        basis=basis,
        reference=reference,
    )
    errors = np.array(
        [
            compute_wasserstein_distance(
                observed_measures[i],
                recon_measures[i],
                n_support=n_support,
                rng=rng,
            )
            for i in range(len(times))
        ],
        dtype=float,
    )
    return errors


def compute_weighted_metric(
    errors: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Compute a weighted mean reconstruction error."""
    if weights is None:
        return float(np.mean(errors))
    weights_arr = np.asarray(weights, dtype=float)
    if weights_arr.size == 0:
        return float(np.mean(errors))
    weight_sum = float(np.sum(weights_arr))
    if weight_sum <= 0:
        return float(np.mean(errors))
    if errors.size == weights_arr.size:
        values = errors
    elif errors.size == weights_arr.size + 1:
        values = errors[:-1]
    else:
        raise ValueError("weights length must match errors or be one shorter")
    normalized = weights_arr / weight_sum
    return float(np.sum(values * normalized))


def compute_uniform_metric(errors: np.ndarray) -> float:
    """Compute the uniform average reconstruction error."""
    return compute_weighted_metric(errors)


def compute_wasserstein_velocities(
    true_measures: Sequence[object],
    n_support: int,
    eval_times: Iterable[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute Wasserstein velocities between successive evaluation times."""
    times = np.asarray(list(eval_times), dtype=float)
    if len(true_measures) != len(times):
        raise ValueError(
            "true_measures and eval_times must have the same length "
            f"(got {len(true_measures)} and {len(times)})"
        )
    if len(true_measures) < 2:
        return np.asarray([], dtype=float)
    dts = np.diff(times)
    velocities = np.array(
        [
            compute_wasserstein_distance(
                true_measures[i],
                true_measures[i + 1],
                n_support=n_support,
                rng=rng,
            )
            / dts[i]
            for i in range(len(true_measures) - 1)
        ],
        dtype=float,
    )
    return velocities


def _time_index(times: np.ndarray, t: float, tol: float) -> int | None:
    idx = np.where(np.isclose(times, t, atol=tol, rtol=0.0))[0]
    if idx.size == 0:
        return None
    return int(idx[0])


def select_grid_velocities_left_endpoint(
    grid_times: np.ndarray,
    grid_velocities: np.ndarray,
    eval_times: np.ndarray,
    tol: float,
) -> np.ndarray:
    if grid_times.size < 2 or grid_velocities.size == 0:
        return np.asarray([], dtype=float)
    if grid_velocities.size != grid_times.size - 1:
        raise ValueError("grid_velocities length must be len(grid_times) - 1")
    eval_times = np.asarray(eval_times, dtype=float)
    eval_velocities: list[float] = []
    for t0 in eval_times[:-1]:
        i0 = _time_index(grid_times, float(t0), tol)
        if i0 is None or i0 >= grid_velocities.size:
            raise ValueError("eval_times must be a subset of grid_times in order")
        eval_velocities.append(float(grid_velocities[i0]))
    return np.asarray(eval_velocities, dtype=float)


def compute_velocity_grid(
    trajectory: object,
    candidate_pool: np.ndarray,
    eval_times: Iterable[float],
    eval_measures: Sequence[object] | None,
    eval_sample_method: str,
    eval_sample_size: int | None,
    n_support: int,
    rng: np.random.Generator,
    use_train_for_grid: bool,
    time_tolerance: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    eval_times_arr = np.asarray(list(eval_times), dtype=float)
    grid_times = np.unique(
        np.concatenate([np.asarray(candidate_pool, dtype=float), eval_times_arr])
    )
    if grid_times.size < eval_times_arr.size:
        return None, None
    if grid_times.size == eval_times_arr.size and not use_train_for_grid:
        return None, None

    if not hasattr(trajectory, eval_sample_method):
        raise AttributeError(f"trajectory has no method '{eval_sample_method}'")
    eval_sampler = getattr(trajectory, eval_sample_method)
    train_sampler = (
        getattr(trajectory, "sample_train", None) if use_train_for_grid else None
    )

    measures = []
    for t in grid_times:
        idx = _time_index(eval_times_arr, float(t), time_tolerance)
        if idx is not None and eval_measures is not None:
            measures.append(eval_measures[idx])
            continue
        if use_train_for_grid and train_sampler is not None and idx is None:
            measures.append(train_sampler(float(t), eval_sample_size, rng=rng))
        else:
            measures.append(eval_sampler(float(t), eval_sample_size, rng=rng))

    grid_velocities = compute_wasserstein_velocities(
        true_measures=measures,
        n_support=n_support,
        eval_times=grid_times,
        rng=rng,
    )
    return grid_times, grid_velocities


def compute_velocity_weighted_metric(
    errors: np.ndarray,
    true_measures: Sequence[object],
    n_support: int,
    eval_times: Iterable[float],
    rng: np.random.Generator,
    velocities: np.ndarray | None = None,
) -> float:
    """Compute the velocity-weighted reconstruction error."""
    if velocities is None:
        velocities = compute_wasserstein_velocities(
            true_measures=true_measures,
            n_support=n_support,
            eval_times=eval_times,
            rng=rng,
        )
    logger.info(
        "List of Wasserstein velocities between eval times: %s", velocities.tolist()
    )
    return compute_weighted_metric(errors, velocities)
