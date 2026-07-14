#!/usr/bin/env python3
"""
Inductive CEBRA training on the macaque reaching dataset with k-fold CV.

This mirrors the MARBLE inductive runner:
- Uses the existing macaque preprocessing (filter + PCA) and k-fold splits.
- Trains CEBRA incrementally on TRAIN only via partial_fit.
- Monitors validation accuracy via an SVM trained on TRAIN embeddings after each training increment.
- Early-stops when validation accuracy plateaus; saves best weights/metrics/embeddings.
- For VAL/TEST embeddings, maps held-out trials to TRAIN via Gaussian kernel-weighted
  nearest neighbors (reusing the existing utilities).
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from cebra import CEBRA

# Ensure project root on path for local imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))

from data_processing.macaque_reaching import macaque_prepare_marble_fold_data
from os_utilities import create_experiment_dir, ensure_dir_exists, smart_pickle
from training.macaque_inductive_utils import (
    build_split_features,
    build_split_features_cebra,
    save_embeddings,
    train_and_evaluate_svm,
)
from models.nn_utilities import count_parameters
from training.kfold_results import compute_kfold_results


DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "yaml_files" / "macaque" / "experiment.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CEBRA inductively with k-fold CV on the macaque reaching dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to experiment YAML.")
    parser.add_argument("--day_idx", type=int, help="Day index override (default from config).")
    parser.add_argument("--patience", type=int, default=10, help="Patience (increments) for early stopping.")
    parser.add_argument("--max_iterations", type=int, default=10000, help="Total iteration cap.")
    parser.add_argument("--increment_max_iterations", type=int, default=100, help="Max iterations per partial_fit increment.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size per increment; if absent, use full train set.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="CEBRA learning rate override.")
    parser.add_argument("--temperature", type=float, default=1.0, help="CEBRA temperature override.")
    parser.add_argument("--output_dimension", type=int, default=3, help="CEBRA embedding dimension.")
    parser.add_argument("--distance", type=str, default="euclidean", help="CEBRA distance metric.")
    parser.add_argument("--max_iterations_cebra", type=int, help="(Deprecated) unused; kept for compatibility.")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"), help="Set device (CPU/CUDA).")
    parser.add_argument("--seed", type=int, help="Random seed override.")
    parser.add_argument("--model_name", type=str, default="cebra", help="Experiment model name.")
    parser.add_argument("--experiment_dir", type=Path, help="Existing experiment directory to reuse.")
    parser.add_argument("--quiet", action="store_true", help="Reduce CEBRA verbosity.")
    parser.add_argument("--svm_kernel", type=str, default="rbf", help="SVM kernel.")
    parser.add_argument("--svm_c", type=float, default=1.0, help="SVM C.")
    parser.add_argument("--svm_gamma", type=str, default="scale", help="SVM gamma.")
    parser.add_argument("--k_neighbors", type=int, default=30, help="Neighbors for held-out embedding mapping.")
    parser.add_argument(
        "--use_velocity_inputs",
        action="store_true",
        help="If set, feed CEBRA neural velocities (np.diff) instead of raw spike positions.",
    )
    parser.add_argument(
        "--legacy_log_format",
        action="store_true",
        help="Use the previous per-increment log format (val accuracy only).",
    )
    parser.add_argument("--folds", type=int, nargs="+", help="Space-separated fold indices to run (defaults to all folds).")
    return parser.parse_args()


def load_yaml_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int, set_torch_seed: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if set_torch_seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg == "cuda":
        return "cuda"
    if device_arg == "cpu":
        return "cpu"
    return "cuda_if_available"


def instantiate_cebra(args: argparse.Namespace, device_str: str, default_params: Dict[str, float]) -> CEBRA:
    batch_size = args.batch_size if args.batch_size is not None else int(default_params["batch_size"])
    learning_rate = (
        args.learning_rate if args.learning_rate is not None else float(default_params["learning_rate"])
    )
    temperature = args.temperature if args.temperature is not None else float(default_params["temperature"])
    distance = args.distance if args.distance is not None else str(default_params["distance"])
    return CEBRA(
        model_architecture="offset10-model",
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        output_dimension=int(args.output_dimension),
        max_iterations=int(args.increment_max_iterations),
        distance=distance,
        conditional="time_delta",
        device=device_str,
        verbose=not args.quiet,
        time_offsets=10,
    )


def get_default_cebra_params() -> Dict[str, float]:
    return {
        "batch_size": 512,
        "learning_rate": 0.0001,
        "temperature": 1.0,
        "output_dimension": 3,
        "max_iterations": 100,
        "distance": "euclidean",
    }


def sample_batch(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if batch_size is None or batch_size <= 0 or batch_size >= X.shape[0]:
        return X, y
    idx = np.random.choice(X.shape[0], size=batch_size, replace=True)
    return X[idx], y[idx]


def compute_embeddings_and_svm(
    *,
    cebra_model: CEBRA,
    fold_data,
    expected_nodes: Optional[int],
    use_velocity_inputs: bool,
    svm_kernel: str,
    svm_C: float,
    svm_gamma,
    include_test: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
    transform_start = time.time()
    train_inputs = fold_data.train_velocities if use_velocity_inputs else fold_data.train_positions
    train_embeddings = cebra_model.transform(train_inputs)
    train_feats, train_labels, train_ids = build_split_features(
        fold_data.train_trials,
        train_embeddings,
        expected_nodes,
    )
    val_feats, val_labels, val_ids = build_split_features_cebra(
        fold_data.valid_trials,
        cebra_model,
        expected_nodes,
        use_velocity_inputs=use_velocity_inputs,
    )
    if include_test:
        test_feats, test_labels, test_ids = build_split_features_cebra(
            fold_data.test_trials,
            cebra_model,
            expected_nodes,
            use_velocity_inputs=use_velocity_inputs,
        )
    else:
        test_feats = np.empty((0, train_feats.shape[1] if train_feats.size > 0 else 0), dtype=np.float32)
        test_labels = np.empty((0,), dtype=np.int64)
        test_ids = np.empty((0,), dtype=np.int64)

    transform_time = time.time() - transform_start

    svm_start = time.time()
    svm_stats = train_and_evaluate_svm(
        train_feats=train_feats,
        train_labels=train_labels,
        val_feats=val_feats,
        val_labels=val_labels,
        test_feats=test_feats,
        test_labels=test_labels,
        test_trial_ids=test_ids,
        kernel=svm_kernel,
        C=svm_C,
        gamma=svm_gamma,
    )
    svm_time = time.time() - svm_start

    embeddings = {
        "train": train_feats,
        "val": val_feats,
        "test": test_feats,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }
    timing = {
        "model_infer_time_sec": float(transform_time),
        "svm_time_sec": float(svm_time),
    }
    return embeddings, svm_stats, timing  # type: ignore[return-value]


def run_incremental_training(
    *,
    cebra_model: CEBRA,
    fold_data,
    expected_nodes: Optional[int],
    args: argparse.Namespace,
    log,
) -> Tuple[
    CEBRA,
    List[Dict[str, float]],
    Dict[str, float],
    Dict[str, np.ndarray],
    int,  # best_epoch_increments
    int,  # best_epoch_iterations
]:
    X_train = fold_data.train_velocities if args.use_velocity_inputs else fold_data.train_positions
    y_train = fold_data.train_condition_ids

    best_model = copy.deepcopy(cebra_model)
    best_val = -np.inf
    best_iter = -1
    no_improve = 0
    history: List[Dict[str, float]] = []
    total_iterations = 0
    increment_idx = 0

    while total_iterations < int(args.max_iterations):
        increment_idx += 1
        X_batch, y_batch = sample_batch(X_train, y_train, args.batch_size)
        start = time.time()
        cebra_model.partial_fit(X_batch, y_batch)
        train_time = time.time() - start
        total_iterations += int(args.increment_max_iterations)

        infer_start = time.time()
        embeddings, svm_stats, timing = compute_embeddings_and_svm(
            cebra_model=cebra_model,
            fold_data=fold_data,
            expected_nodes=expected_nodes,
            use_velocity_inputs=args.use_velocity_inputs,
            svm_kernel=args.svm_kernel,
            svm_C=args.svm_c,
            svm_gamma=args.svm_gamma,
            include_test=False,
        )
        full_infer_time = time.time() - infer_start
        model_infer_time = timing.get("model_infer_time_sec", full_infer_time)
        svm_time = timing.get("svm_time_sec", 0.0)
        val_acc = float(svm_stats.get("val_accuracy", float("nan")))
        train_acc = float(svm_stats.get("train_accuracy", float("nan")))

        history.append(
            {
                "increment": increment_idx,
                "total_iterations": total_iterations,
                "train_time_sec": float(train_time),
                "model_infer_time_sec": float(model_infer_time),
                "svm_time_sec": float(svm_time),
                "full_eval_time_sec": float(full_infer_time),
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            }
        )
        if args.legacy_log_format:
            log(
                f"Increment {increment_idx:04d} | total_iters={total_iterations} | "
                f"train_time={train_time:.2f}s | val_acc={val_acc:.4f}"
            )
        else:
            log(
                f"Inc {increment_idx:04d} | total_iters={total_iterations} | "
                f"train_time={train_time:.2f}s | infer_fwd_time={model_infer_time:.2f}s | svm_time={svm_time:.2f}s | "
                f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}"
            )

        val_labels = embeddings["val_labels"]
        has_val = isinstance(val_labels, np.ndarray) and val_labels.size > 0
        if has_val and val_acc > best_val:
            best_val = val_acc
            best_model = copy.deepcopy(cebra_model)
            best_iter = increment_idx
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= int(args.patience):
            log(f"Early stopping triggered after {no_improve} increments without improvement (best_iter={best_iter}).")
            break

    if best_iter < 0:
        best_model = cebra_model

    best_epoch_increments = best_iter if best_iter > 0 else len(history)
    iters_per_increment = max(1, int(args.increment_max_iterations))
    best_epoch_iterations = int(best_epoch_increments * iters_per_increment)

    final_embeddings, final_stats, _ = compute_embeddings_and_svm(
        cebra_model=best_model,
        fold_data=fold_data,
        expected_nodes=expected_nodes,
        use_velocity_inputs=args.use_velocity_inputs,
        svm_kernel=args.svm_kernel,
        svm_C=args.svm_c,
        svm_gamma=args.svm_gamma,
        include_test=True,
    )
    return (
        best_model,
        history,
        final_stats,
        final_embeddings,
        best_epoch_increments,
        best_epoch_iterations,
    )  # type: ignore[return-value]


def safe_count_parameters(
    model: CEBRA,
    log,
) -> Optional[int]:
    """
    Safely count model parameters, logging a warning on failure.
    """
    candidates = [model]
    # The nn.Module might be an attribute of a wrapper class
    for attr in ("model", "model_", "encoder", "net"):
        if hasattr(model, attr):
            candidates.append(getattr(model, attr))

    last_exc: Exception | None = None
    visited: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        try:
            return count_parameters(candidate)
        except Exception as exc:  # keep trying other candidates
            last_exc = exc
            continue

    if last_exc is not None:
        log(f"Warning: failed to count parameters: {last_exc}")
    else:
        log("Warning: failed to count parameters (no valid model candidate found).")
    return None


def main() -> None:
    # Parse CLI and config
    args = parse_args()
    cfg = load_yaml_config(args.config)
    training_cfg = cfg["training"]
    dataset_cfg = cfg["dataset"]

    # Seed and device
    seed = args.seed if args.seed is not None else int(dataset_cfg.get("split_seed", 123456))
    set_global_seed(seed)
    device_str = resolve_device(args.device)
    default_cebra_params = get_default_cebra_params()

    # Resolve paths and folds
    results_root = Path(training_cfg["root_dir"]).expanduser() / training_cfg["save_dir"]
    dataset_root = Path(training_cfg["root_dir"]).expanduser() / dataset_cfg["data_dir"]
    k_folds = int(training_cfg.get("k_folds", dataset_cfg.get("k_folds", 5)))
    if args.folds:
        fold_indices: List[int] = []
        seen = set()
        for idx in args.folds:
            if idx < 0 or idx >= k_folds:
                raise ValueError(f"Fold index {idx} is out of range for k_folds={k_folds}.")
            if idx not in seen:
                fold_indices.append(idx)
                seen.add(idx)
    else:
        fold_indices = list(range(k_folds))

    # Select day
    day_idx = args.day_idx if args.day_idx is not None else dataset_cfg.get("macaque_day_index")
    if day_idx is None:
        raise ValueError("day_idx must be provided via CLI or config.")

    # Experiment directories
    if args.experiment_dir is not None:
        exp_dir_path = args.experiment_dir.expanduser()
        if not exp_dir_path.exists():
            raise FileNotFoundError(f"Experiment directory '{exp_dir_path}' does not exist.")
        ensure_dir_exists(exp_dir_path / "config", raise_exception=True)
        exp_dirs = {
            "exp_dir": str(exp_dir_path),
            "config_save_path": str(exp_dir_path / "config" / "config.yaml"),
        }
    else:
        exp_dirs = create_experiment_dir(
            root_dir=results_root,
            model_name=args.model_name,
            dataset_name=dataset_cfg.get("dataset", "macaque"),
            experiment_id=training_cfg.get("experiment_id"),
            config=cfg,
            verbosity=int(training_cfg.get("verbosity", 0)),
        )

    # Persist resolved config
    ensure_dir_exists(Path(exp_dirs["config_save_path"]).parent, raise_exception=True)
    with open(exp_dirs["config_save_path"], "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # Iterate over k-folds
    for fold_idx in fold_indices:
        fold_root = Path(exp_dirs["exp_dir"]) / f"fold_{fold_idx}"
        metrics_dir = fold_root / "metrics"
        models_dir = fold_root / "models"
        logs_dir = fold_root / "logs"
        config_dir = fold_root / "config"
        embeddings_dir = fold_root / "embeddings"

        # Create fold dirs and log helper
        for dir_path in (metrics_dir, models_dir, logs_dir, config_dir, embeddings_dir):
            ensure_dir_exists(dir_path, raise_exception=True)
        with open(config_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)

        log_path = logs_dir / "training.log"

        def log(msg: str) -> None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] {msg}"
            print(line)
            with open(log_path, "a", encoding="utf-8") as log_f:
                log_f.write(line + "\n")

        # Prepare fold data (splits + preprocessing)
        log(f"\nPreparing fold {fold_idx} data (seed={seed}, day_idx={day_idx})...")
        fold_data = macaque_prepare_marble_fold_data(
            data_root=str(dataset_root),
            day_index=day_idx,
            k_folds=k_folds,
            fold_i=fold_idx,
            seed=seed,
            include_lever_velocity=False,
            k_neighbors=args.k_neighbors,
            apply_savgol_filter_before_pca=True,
        )

        # Train incrementally with val monitoring
        log("Instantiating CEBRA model...")
        cebra_model = instantiate_cebra(args, device_str, default_cebra_params)
        log("Starting incremental training...")

        (
            best_model,
            history,
            final_stats,
            final_embeddings,
            best_epoch_increments,
            best_epoch_iterations,
        ) = run_incremental_training(
            cebra_model=cebra_model,
            fold_data=fold_data,
            expected_nodes=fold_data.nodes_per_trial,
            args=args,
            log=log,
        )
        parameter_count = safe_count_parameters(best_model, log)

        log(
            f"[Fold {fold_idx}] Final metrics | "
            f"train_acc={final_stats.get('train_accuracy', float('nan')):.4f} | "
            f"val_acc={final_stats.get('val_accuracy', float('nan')):.4f} | "
            f"test_acc={final_stats.get('test_accuracy', float('nan')):.4f}"
        )

        # Save artifacts and embeddings
        log("Saving artifacts...")
        best_model_path = models_dir / "best_model.pt"
        best_model.save(str(best_model_path))
        svm_obj = final_stats.get("svm")
        if svm_obj is not None:
            smart_pickle(str(models_dir / "svm.pkl"), svm_obj, overwrite=True)

        history_path = metrics_dir / "train_history.yaml"
        with open(history_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(history, f)

        history_until_best = history[:best_epoch_increments] if best_epoch_increments > 0 else []
        if history_until_best:
            mean_train_time = sum(h.get("train_time_sec", 0.0) for h in history_until_best) / len(history_until_best)
            mean_infer_time = sum(h.get("model_infer_time_sec", 0.0) for h in history_until_best) / len(history_until_best)
        else:
            mean_train_time = float("nan")
            mean_infer_time = float("nan")

        results = {
            "fold": fold_idx,
            "train_accuracy": float(final_stats.get("train_accuracy", float("nan"))),
            "val_accuracy": float(final_stats.get("val_accuracy", float("nan"))),
            "test_accuracy": float(final_stats.get("test_accuracy", float("nan"))),
            "parameter_count": int(parameter_count) if parameter_count is not None else None,
            "best_epoch": int(best_epoch_iterations),
            "mean_train_time": float(mean_train_time),
            "mean_infer_time": float(mean_infer_time),
        }
        with open(metrics_dir / "results.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(results, f)

        save_embeddings(
            embeddings_dir,
            "train",
            final_embeddings["train"],
            final_embeddings["train_labels"],
            final_embeddings["train_ids"],
        )
        save_embeddings(
            embeddings_dir,
            "valid",
            final_embeddings["val"],
            final_embeddings["val_labels"],
            final_embeddings["val_ids"],
        )
        save_embeddings(
            embeddings_dir,
            "test",
            final_embeddings["test"],
            final_embeddings["test_labels"],
            final_embeddings["test_ids"],
        )

        log(f"Completed fold {fold_idx}. Results saved to {fold_root}.")

    if len(fold_indices) > 1:
        compute_kfold_results(
            parent_metrics_dir=Path(exp_dirs["exp_dir"]) / "metrics",
            timing_keys=("mean_train_time", "mean_infer_time"),
            weight_key="best_epoch",
        )


if __name__ == "__main__":
    main()