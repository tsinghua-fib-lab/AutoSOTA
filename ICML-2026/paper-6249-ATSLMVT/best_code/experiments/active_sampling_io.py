"""I/O helpers for active sampling experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

from experiments.active_sampling_utils import (
    _resolve_active_loop_reference_settings,
    _resolve_barycenter_backend,
    _first_non_none,
    _resolve_sample_size,
    ExperimentContext,
)

logger = logging.getLogger(__name__)

try:
    import cloudpickle as _checkpoint_pickle
    _CHECKPOINT_PICKLE_NAME = "cloudpickle"
except ImportError:  # pragma: no cover - fallback for minimal environments
    import pickle as _checkpoint_pickle
    _CHECKPOINT_PICKLE_NAME = "pickle"


def _dump_checkpoint_state(path: Path, payload: object) -> None:
    with path.open("wb") as handle:
        _checkpoint_pickle.dump(payload, handle)


def _write_metrics(output_dir: Path, results: Sequence["StrategyResult"]) -> None:
    metrics = {}
    for r in results:
        entry = {
            "uniform_metric": r.uniform_metric,
            "velocity_metric": r.velocity_metric,
        }
        if r.common_reference_uniform_metric is not None:
            entry["common_reference_uniform_metric"] = r.common_reference_uniform_metric
            entry["common_reference_velocity_metric"] = r.common_reference_velocity_metric
        metrics[r.name] = entry
    payload = {"metrics": metrics}
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2))


def _write_error_tables(
    output_dir: Path,
    results: Sequence["StrategyResult"],
    eval_times: np.ndarray,
    eval_velocities: np.ndarray | None = None,
) -> tuple[Path | None, Path | None]:
    def _velocity_weighted_errors(
        errors: np.ndarray,
        velocities: np.ndarray | None,
    ) -> np.ndarray | None:
        if velocities is None:
            return None
        weights = np.asarray(velocities, dtype=float)
        if weights.size == 0:
            return None
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            return None
        values = np.asarray(errors, dtype=float)
        if values.size == weights.size:
            normalized = weights / weight_sum
            return values * normalized
        if values.size == weights.size + 1:
            normalized = weights / weight_sum
            weighted = np.full(values.shape, np.nan, dtype=float)
            weighted[:-1] = values[:-1] * normalized
            return weighted
        logger.warning(
            "Velocity weights length %s does not match errors length %s; "
            "skipping velocity-weighted errors.",
            weights.size,
            values.size,
        )
        return None

    rows = []
    for r in results:
        velocity_weighted = _velocity_weighted_errors(r.errors, eval_velocities)
        for idx, (time, error) in enumerate(zip(eval_times, r.errors)):
            row = {"strategy": r.name, "time": float(time), "error": float(error)}
            if velocity_weighted is not None:
                row["velocity_weighted_error"] = float(velocity_weighted[idx])
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "errors.csv", index=False)

    common_rows = []
    for r in results:
        if r.common_reference_errors is None:
            continue
        velocity_weighted = _velocity_weighted_errors(
            r.common_reference_errors, eval_velocities
        )
        for idx, (time, error) in enumerate(zip(eval_times, r.common_reference_errors)):
            row = {"strategy": r.name, "time": float(time), "error": float(error)}
            if velocity_weighted is not None:
                row["velocity_weighted_error"] = float(velocity_weighted[idx])
            common_rows.append(row)
    common_errors_path = output_dir / "errors_common_reference.csv"
    if common_rows:
        pd.DataFrame(common_rows).to_csv(common_errors_path, index=False)
    else:
        common_errors_path = None

    observed_rows = []
    for r in results:
        if r.observed_times is None or r.observed_reconstruction_errors is None:
            continue
        sample_sizes = r.observed_sample_sizes
        for idx, (time, error) in enumerate(
            zip(r.observed_times, r.observed_reconstruction_errors)
        ):
            row = {
                "strategy": r.name,
                "index": int(idx),
                "time": float(time),
                "error": float(error),
            }
            if sample_sizes is not None and idx < len(sample_sizes):
                row["sample_size"] = int(sample_sizes[idx])
            observed_rows.append(row)
    observed_errors_path = output_dir / "observed_errors.csv"
    if observed_rows:
        pd.DataFrame(observed_rows).to_csv(observed_errors_path, index=False)
    else:
        observed_errors_path = None

    return common_errors_path, observed_errors_path


def _write_checkpoint_metrics(
    output_dir: Path,
    results: Sequence["StrategyResult"],
) -> Path | None:
    rows = []
    for r in results:
        if not getattr(r, "checkpoint_results", None):
            continue
        for step, checkpoint in sorted(r.checkpoint_results.items()):
            row = {
                "strategy": r.name,
                "step": int(step),
                "uniform_metric": checkpoint.uniform_metric,
                "velocity_metric": checkpoint.velocity_metric,
            }
            if checkpoint.common_reference_uniform_metric is not None:
                row["common_reference_uniform_metric"] = (
                    checkpoint.common_reference_uniform_metric
                )
                row["common_reference_velocity_metric"] = (
                    checkpoint.common_reference_velocity_metric
                )
            rows.append(row)
    if not rows:
        return None
    path = output_dir / "metrics_by_step.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_checkpoint_errors(
    output_dir: Path,
    results: Sequence["StrategyResult"],
    eval_times: np.ndarray,
    eval_velocities: np.ndarray | None = None,
) -> tuple[Path | None, Path | None]:
    def _velocity_weighted_errors(
        errors: np.ndarray,
        velocities: np.ndarray | None,
    ) -> np.ndarray | None:
        if velocities is None:
            return None
        weights = np.asarray(velocities, dtype=float)
        if weights.size == 0:
            return None
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            return None
        values = np.asarray(errors, dtype=float)
        if values.size == weights.size:
            normalized = weights / weight_sum
            return values * normalized
        if values.size == weights.size + 1:
            normalized = weights / weight_sum
            weighted = np.full(values.shape, np.nan, dtype=float)
            weighted[:-1] = values[:-1] * normalized
            return weighted
        logger.warning(
            "Velocity weights length %s does not match errors length %s; "
            "skipping velocity-weighted errors.",
            weights.size,
            values.size,
        )
        return None

    rows = []
    for r in results:
        if not getattr(r, "checkpoint_results", None):
            continue
        for step, checkpoint in sorted(r.checkpoint_results.items()):
            velocity_weighted = _velocity_weighted_errors(
                checkpoint.errors, eval_velocities
            )
            for idx, (time, error) in enumerate(zip(eval_times, checkpoint.errors)):
                row = {
                    "strategy": r.name,
                    "step": int(step),
                    "time": float(time),
                    "error": float(error),
                }
                if velocity_weighted is not None:
                    row["velocity_weighted_error"] = float(velocity_weighted[idx])
                rows.append(row)
    errors_path = output_dir / "errors_by_step.csv"
    if rows:
        pd.DataFrame(rows).to_csv(errors_path, index=False)
    else:
        errors_path = None

    common_rows = []
    for r in results:
        if not getattr(r, "checkpoint_results", None):
            continue
        for step, checkpoint in sorted(r.checkpoint_results.items()):
            if checkpoint.common_reference_errors is None:
                continue
            velocity_weighted = _velocity_weighted_errors(
                checkpoint.common_reference_errors, eval_velocities
            )
            for idx, (time, error) in enumerate(
                zip(eval_times, checkpoint.common_reference_errors)
            ):
                row = {
                    "strategy": r.name,
                    "step": int(step),
                    "time": float(time),
                    "error": float(error),
                }
                if velocity_weighted is not None:
                    row["velocity_weighted_error"] = float(velocity_weighted[idx])
                common_rows.append(row)
    common_errors_path = output_dir / "errors_common_reference_by_step.csv"
    if common_rows:
        pd.DataFrame(common_rows).to_csv(common_errors_path, index=False)
    else:
        common_errors_path = None

    return errors_path, common_errors_path


def _write_checkpoint_gp_predictions(
    output_dir: Path,
    results: Sequence["StrategyResult"],
) -> Path | None:
    base_dir = output_dir / "checkpoint_gp_predictions"
    wrote_any = False
    for r in results:
        if not getattr(r, "checkpoint_results", None):
            continue
        safe_name = r.name.replace("/", "_")
        strategy_dir = base_dir / safe_name
        for step, checkpoint in sorted(r.checkpoint_results.items()):
            gp = getattr(checkpoint, "gp_predictions", None)
            if gp is None:
                continue
            try:
                strategy_dir.mkdir(parents=True, exist_ok=True)
                times = np.asarray(gp.get("times", []), dtype=float)
                mean = np.asarray(gp.get("mean", []), dtype=float)
                std = np.asarray(gp.get("std", []), dtype=float)
                np.savez(
                    strategy_dir / f"step_{int(step):03d}.npz",
                    times=times,
                    mean=mean,
                    std=std,
                )
                wrote_any = True
            except Exception as exc:
                logger.warning(
                    "Failed to save checkpoint GP predictions for '%s' step %s: %s",
                    r.name,
                    step,
                    exc,
                )
    if not wrote_any:
        return None
    return base_dir


def _write_checkpoint_state(
    output_dir: Path,
    results: Sequence["StrategyResult"],
) -> Path | None:
    base_dir = output_dir / "checkpoint_state"
    wrote_any = False
    if _CHECKPOINT_PICKLE_NAME == "pickle":
        logger.warning(
            "cloudpickle is unavailable; checkpoint state serialization may fail "
            "for complex objects. Install cloudpickle for full fidelity."
        )
    for r in results:
        if not getattr(r, "checkpoint_results", None):
            continue
        safe_name = r.name.replace("/", "_")
        strategy_dir = base_dir / safe_name
        for step, checkpoint in sorted(r.checkpoint_results.items()):
            state = getattr(checkpoint, "checkpoint_state", None)
            if state is None:
                continue
            try:
                strategy_dir.mkdir(parents=True, exist_ok=True)
                _dump_checkpoint_state(
                    strategy_dir / f"step_{int(step):03d}.pkl",
                    state,
                )
                wrote_any = True
            except Exception as exc:
                logger.warning(
                    "Failed to save checkpoint state for '%s' step %s: %s",
                    r.name,
                    step,
                    exc,
                )
    if not wrote_any:
        return None
    return base_dir


def _write_acquisition_artifacts(
    output_dir: Path,
    results: Sequence["StrategyResult"],
) -> Path:
    acquisition_artifacts = {}
    for r in results:
        acquisition_artifacts[r.name] = {
            "observed_times": r.observed_times.tolist() if r.observed_times is not None else None,
            "observed_sample_sizes": r.observed_sample_sizes.tolist()
            if r.observed_sample_sizes is not None
            else None,
            "observed_reconstruction_errors": r.observed_reconstruction_errors.tolist()
            if r.observed_reconstruction_errors is not None
            else None,
            "acquisition_history": r.acquisition_history,
            "gp_predictions": r.gp_predictions,
            "gp_hyperparams": r.gp_hyperparams,
            "wasserstein_velocities": r.wasserstein_velocities.tolist()
            if r.wasserstein_velocities is not None
            else None,
            "wasserstein_velocity_grid": r.wasserstein_velocity_grid.tolist()
            if r.wasserstein_velocity_grid is not None
            else None,
            "velocity_grid_times": r.velocity_grid_times.tolist()
            if r.velocity_grid_times is not None
            else None,
        }
    path = output_dir / "acquisition_artifacts.json"
    path.write_text(json.dumps(acquisition_artifacts, indent=2))
    return path


def _write_timing_table(
    output_dir: Path,
    results: Sequence["StrategyResult"],
) -> Path | None:
    rows = []
    for r in results:
        records = getattr(r, "timing_records", None)
        if not records:
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            row = {"strategy": r.name}
            row.update(record)
            rows.append(row)
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    preferred = [
        "strategy",
        "stage",
        "step",
        "component",
        "seconds",
        "n_measurements",
        "n_candidates",
    ]
    ordered = [c for c in preferred if c in frame.columns]
    ordered += [c for c in frame.columns if c not in ordered]
    frame = frame[ordered]
    path = output_dir / "timings.csv"
    frame.to_csv(path, index=False)
    return path


def _write_shared_artifacts(
    output_dir: Path,
    context: ExperimentContext,
) -> Path:
    shared_dir = output_dir / "shared_artifacts"
    shared_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "velocity_source": str(context.velocity_source),
        "reference_source": str(context.reference_source),
        "eval_sample_method": str(context.eval_sample_method),
        "eval_sample_size": context.eval_sample_size,
        "initial_times": list(context.initial_times),
    }

    def _save_array(key: str, value: object | None) -> None:
        if value is None:
            return
        arr = np.asarray(value)
        path = shared_dir / f"{key}.npy"
        np.save(path, arr)
        manifest[key] = {"path": path.name, "shape": list(arr.shape)}

    def _save_measure(key: str, measure: object | None) -> None:
        if measure is None:
            return
        info: dict[str, object] = {}
        if hasattr(measure, "support"):
            support = np.asarray(measure.support)
            support_path = shared_dir / f"{key}_support.npy"
            np.save(support_path, support)
            info["support"] = {"path": support_path.name, "shape": list(support.shape)}
        if hasattr(measure, "weights"):
            weights = np.asarray(measure.weights)
            weights_path = shared_dir / f"{key}_weights.npy"
            np.save(weights_path, weights)
            info["weights"] = {"path": weights_path.name, "shape": list(weights.shape)}
        if info:
            manifest[key] = info

    _save_array("eval_times", context.eval_times)
    _save_array("candidate_pool", context.candidate_pool)
    _save_array("eval_velocities", context.eval_velocities)
    _save_array("velocity_grid", context.velocity_grid)
    _save_array("velocity_grid_times", context.velocity_grid_times)
    _save_measure("initial_reference", context.initial_reference)
    _save_measure("eval_barycenter_reference", context.eval_barycenter_reference)
    _save_measure("common_reference", context.common_reference)

    path = shared_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def _build_common_reference_info(
    cfg: DictConfig,
    eval_sample_method: str,
    eval_sample_size: int | None,
    common_reference: object | None,
) -> dict | None:
    common_reference_cfg = cfg.get("common_reference")
    if common_reference_cfg is None:
        return None
    times_cfg = common_reference_cfg.get("times")
    if isinstance(times_cfg, (float, int, np.floating, np.integer)):
        times_list = [float(times_cfg)]
    elif isinstance(times_cfg, str):
        try:
            times_list = [float(times_cfg)]
        except ValueError:
            times_list = None
    elif times_cfg is not None:
        times_list = [float(t) for t in list(times_cfg)]
    else:
        times_list = None
    backend = _resolve_barycenter_backend(cfg=cfg, source_cfg=common_reference_cfg)
    return {
        "enabled": bool(common_reference_cfg.get("enabled", False)),
        "source": str(common_reference_cfg.get("source", "eval_barycenter")),
        "times": times_list,
        "barycenter_size": int(
            common_reference_cfg.get("barycenter_size", int(cfg.reference.barycenter_size))
        ),
        "num_iter": int(common_reference_cfg.get("num_iter", int(cfg.reference.num_iter))),
        "reg": float(common_reference_cfg.get("reg", float(cfg.reference.get("reg", 0.0)))),
        "backend": backend,
        "sample_method": str(
            _first_non_none(common_reference_cfg.get("sample_method"), eval_sample_method)
        ),
        "sample_size": _resolve_sample_size(
            common_reference_cfg.get("sample_size"), eval_sample_size
        ),
        "built": common_reference is not None,
    }


def _build_reference_info(
    cfg: DictConfig,
    reference_source: str,
    eval_sample_method: str,
    eval_sample_size: int | None,
) -> dict:
    backend = _resolve_barycenter_backend(
        cfg=cfg,
        source_cfg=getattr(cfg, "reference", None),
    )
    reference_info = {
        "source": reference_source,
        "barycenter_size": int(cfg.reference.barycenter_size),
        "num_iter": int(cfg.reference.num_iter),
        "reg": float(cfg.reference.get("reg", 0.0)),
        "backend": backend,
    }
    if reference_source == "eval_barycenter":
        reference_cfg = getattr(cfg, "reference", None)
        ref_sample_method = _first_non_none(
            None if reference_cfg is None else reference_cfg.get("sample_method"),
            eval_sample_method,
        )
        ref_sample_size = _resolve_sample_size(
            None if reference_cfg is None else reference_cfg.get("sample_size"),
            eval_sample_size,
        )
        reference_info["sample_method"] = str(ref_sample_method)
        reference_info["sample_size"] = ref_sample_size
    return reference_info


def _write_metadata(
    output_dir: Path,
    cfg: DictConfig,
    context: ExperimentContext,
    strategies: Sequence[str],
    num_steps_by_strategy: dict[str, int],
    trajectory: object,
    checkpoints: Sequence[int] | None = None,
) -> Path:
    oracle_sample_size = _resolve_sample_size(cfg.oracle.sample_size)
    reference_info = _build_reference_info(
        cfg,
        context.reference_source,
        context.eval_sample_method,
        context.eval_sample_size,
    )
    common_reference_info = _build_common_reference_info(
        cfg,
        context.eval_sample_method,
        context.eval_sample_size,
        context.common_reference,
    )
    (
        recompute_reference,
        active_loop_barycenter_size,
        active_loop_barycenter_num_iter,
        active_loop_barycenter_reg,
        active_loop_backend,
    ) = _resolve_active_loop_reference_settings(
        cfg=cfg,
        reference_source=context.reference_source,
    )
    trajectory_metadata = None
    if hasattr(trajectory, "metadata") and callable(getattr(trajectory, "metadata")):
        try:
            trajectory_metadata = trajectory.metadata()
        except Exception as exc:
            logger.warning("Could not collect trajectory metadata: %s", exc)
    metadata = {
        "seed": int(cfg.seed),
        "num_steps": int(cfg.num_steps),
        "num_steps_by_strategy": num_steps_by_strategy,
        "initial_times": context.initial_times,
        "candidate_pool": context.candidate_pool.tolist(),
        "eval_times": context.eval_times.tolist(),
        "strategies": list(strategies),
        "oracle": {
            "sample_method": str(cfg.oracle.sample_method),
            "sample_size": oracle_sample_size,
        },
        "evaluation": {
            "sample_method": str(context.eval_sample_method),
            "sample_size": context.eval_sample_size,
            "n_support": int(cfg.evaluation.n_support),
            "velocity_source": context.velocity_source,
            "save_checkpoint_state": bool(
                cfg.evaluation.get("save_checkpoint_state", False)
            ),
        },
        "reference": reference_info,
        "common_reference": common_reference_info,
        "active_loop": {
            "recompute_reference_as_barycenter": bool(recompute_reference),
            "barycenter_size": int(active_loop_barycenter_size),
            "barycenter_num_iter": int(active_loop_barycenter_num_iter),
            "barycenter_reg": float(active_loop_barycenter_reg),
            "backend": active_loop_backend,
        },
        "trajectory": trajectory_metadata,
    }
    if checkpoints:
        metadata["evaluation"]["checkpoints"] = [int(step) for step in checkpoints]
    strategies_cfg = cfg.get("strategies")
    if isinstance(strategies_cfg, DictConfig):
        try:
            resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(resolved_cfg, dict):
                metadata["strategy_configs"] = resolved_cfg.get("strategies")
            else:
                metadata["strategy_configs"] = OmegaConf.to_container(
                    strategies_cfg,
                    resolve=False,
                )
        except Exception as exc:
            logger.warning("Could not resolve strategy configs: %s", exc)
            metadata["strategy_configs"] = OmegaConf.to_container(
                strategies_cfg,
                resolve=False,
            )
        order = cfg.get("strategy_order")
        if order is not None:
            metadata["strategy_order"] = [str(name) for name in list(order)]
    elif isinstance(strategies_cfg, (ListConfig, list)):
        names = []
        structured = []
        has_structured = False
        for item in list(strategies_cfg):
            if isinstance(item, DictConfig):
                has_structured = True
                item_name = item.get("name")
                if item_name is not None:
                    names.append(str(item_name))
                structured.append(OmegaConf.to_container(item, resolve=True))
            else:
                names.append(str(item))
                structured.append(item)
        if names:
            metadata["strategy_list"] = names
        if has_structured:
            metadata["strategy_configs"] = structured
    strategy_overrides = cfg.get("strategy_overrides")
    if strategy_overrides:
        metadata["strategy_overrides"] = OmegaConf.to_container(
            strategy_overrides,
            resolve=True,
        )
    path = output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2))
    return path


def _write_reconstruction_artifacts(
    output_dir: Path,
    results: Sequence["StrategyResult"],
    common_reference: object | None,
) -> Path:
    reconstruction_dir = output_dir / "reconstruction"
    reconstruction_dir.mkdir(exist_ok=True)

    if common_reference is not None:
        try:
            if hasattr(common_reference, "support"):
                np.save(
                    reconstruction_dir / "common_reference_support.npy",
                    common_reference.support,
                )
            if hasattr(common_reference, "weights"):
                np.save(
                    reconstruction_dir / "common_reference_weights.npy",
                    common_reference.weights,
                )
        except Exception as exc:
            logger.warning("Could not save common reference: %s", exc)

    for r in results:
        safe_name = r.name.replace("/", "_")
        strategy_dir = reconstruction_dir / safe_name
        strategy_dir.mkdir(exist_ok=True)

        if r.reference is not None:
            try:
                if hasattr(r.reference, "support"):
                    np.save(strategy_dir / "reference_support.npy", r.reference.support)
                if hasattr(r.reference, "weights"):
                    np.save(strategy_dir / "reference_weights.npy", r.reference.weights)
            except Exception as exc:
                logger.warning("Could not save reference for %s: %s", r.name, exc)

        if r.warp is not None:
            try:
                warp_data = {}
                if hasattr(r.warp, "times"):
                    warp_data["times"] = np.asarray(r.warp.times).tolist()
                if hasattr(r.warp, "arc_lengths"):
                    warp_data["arc_lengths"] = np.asarray(r.warp.arc_lengths).tolist()
                if warp_data:
                    (strategy_dir / "warp.json").write_text(json.dumps(warp_data, indent=2))
            except Exception as exc:
                logger.warning("Could not save warp for %s: %s", r.name, exc)

        if r.coefficients is not None:
            np.save(strategy_dir / "coefficients.npy", r.coefficients)

        if r.basis_components is not None:
            components = np.asarray(r.basis_components, dtype=float)
            mean_vec = (
                np.asarray(r.basis_mean, dtype=float)
                if r.basis_mean is not None
                else None
            )
            singular_values = (
                np.asarray(r.basis_singular_values, dtype=float)
                if r.basis_singular_values is not None
                else None
            )
            support_shape = None
            if r.reference is not None and hasattr(r.reference, "support"):
                ref_support = np.asarray(r.reference.support)
                if ref_support.ndim == 2:
                    support_shape = ref_support.shape
            components_to_save = components
            mean_to_save = mean_vec
            if support_shape is not None and components.ndim == 2:
                n_support, dim = support_shape
                flat_dim = n_support * dim
                if components.shape[1] == flat_dim:
                    components_to_save = components.reshape((components.shape[0], n_support, dim))
                    if mean_vec is not None and mean_vec.size == flat_dim:
                        mean_to_save = mean_vec.reshape((n_support, dim))
            np.save(strategy_dir / "basis_components.npy", components_to_save)
            if mean_to_save is not None:
                np.save(strategy_dir / "basis_mean.npy", mean_to_save)
            if singular_values is not None:
                np.save(strategy_dir / "basis_singular_values.npy", singular_values)
            basis_meta = {
                "include_mean_field": bool(r.basis is not None and r.basis.intercept is not None),
                "components_shape": list(components_to_save.shape),
                "components_flat_shape": list(components.shape),
                "support_shape": list(support_shape) if support_shape is not None else None,
            }
            if mean_to_save is not None:
                basis_meta["mean_shape"] = list(mean_to_save.shape)
            if singular_values is not None:
                basis_meta["singular_values_shape"] = list(singular_values.shape)
            (strategy_dir / "basis_meta.json").write_text(json.dumps(basis_meta, indent=2))

        if r.observed_times is not None:
            np.save(strategy_dir / "observed_times.npy", r.observed_times)
        if r.observed_reconstruction_errors is not None:
            np.save(strategy_dir / "observed_reconstruction_errors.npy", r.observed_reconstruction_errors)

        if r.wasserstein_velocities is not None:
            np.save(strategy_dir / "wasserstein_velocities.npy", r.wasserstein_velocities)
        if r.wasserstein_velocity_grid is not None:
            np.save(
                strategy_dir / "wasserstein_velocities_grid.npy",
                r.wasserstein_velocity_grid,
            )
        if r.velocity_grid_times is not None:
            np.save(
                strategy_dir / "wasserstein_velocity_grid_times.npy",
                r.velocity_grid_times,
            )
        if r.gp_predictions is not None:
            try:
                gp_times = np.asarray(r.gp_predictions.get("times", []), dtype=float)
                gp_mean = np.asarray(r.gp_predictions.get("mean", []), dtype=float)
                gp_std = np.asarray(r.gp_predictions.get("std", []), dtype=float)
                np.savez(
                    strategy_dir / "gp_predictions.npz",
                    times=gp_times,
                    mean=gp_mean,
                    std=gp_std,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to save GP predictions for '%s': %s",
                    r.name,
                    exc,
                )

    return reconstruction_dir
