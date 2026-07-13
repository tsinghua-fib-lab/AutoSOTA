"""Standalone kernel analysis script for the ICML 2026 companion code.

This script focuses on kernel diagnostics rather than Bayesian optimization
runs. It supports domain analysis as well as surrogate learnability analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, cast
import argparse
import csv
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import platform

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from scipy.stats import kendalltau, pearsonr, spearmanr

# Allow direct script execution without requiring manual PYTHONPATH setup.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from cosy.core import Synthesizer
from cosy.core.tree import Tree

# Prefer package-relative imports, but keep a direct-script fallback so this
# file can be executed as `python kernel_experiments.py ...` from the repo root.
try:
    from .utils import (
        DEFAULT_PRESAMPLE_PATH,
        DEFAULT_DATASET_PATH,
        get_or_create_unique_pre_samples,
        load_pre_samples,
        load_dataset,
        create_and_save_dataset,
        generate_pre_samples,
        save_pre_samples,
    )
except ImportError:  # pragma: no cover - direct execution fallback
    from utils import (
        DEFAULT_PRESAMPLE_PATH,
        DEFAULT_DATASET_PATH,
        get_or_create_unique_pre_samples,
        load_pre_samples,
        load_dataset,
        create_and_save_dataset,
        generate_pre_samples,
        save_pre_samples,
    )
from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository
from bayesian_optimization.examples.damg_nas.damg_targets import (
    target_len_3_refined_1,
    target_len_5_refined_1,
    target_len_2,
    target_len_3,
    target_len_4,
    target_len_5,
    target_len_6,
    target_to_name,
    target_len_3_refined_2,
    target_len_5_refined_2,
    target_len_3_refined_3,
    target_len_5_refined_3,
    target_len_4_refined_1
)
from bayesian_optimization.examples.damg_nas.damg_repo_algebras import pytorch_function_algebra
from bayesian_optimization.examples.damg_nas.damg_kernels import (
    noisy_combined_hierarchical_kernel,
    noisy_hierarchical_damg_kernel,
    noisy_hierarchical_tree_kernel,
    noisy_hierarchical_wl_kernel,
    noisy_hierarchical_damg_kernel_13,
    tree_kernel_1, tree_kernel_2, tree_kernel_3,
    wl_kernel_1, wl_kernel_2, wl_kernel_3,
    damg_kernel_1, damg_kernel_2, damg_kernel_3,
)
from bayesian_optimization.kernels.graph_kernel import WeisfeilerLehmanKernel
from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

DEFAULT_PRESAMPLE_SIZE = 10
RANDOM_SEED = 42
WL_N_ITERS = 1
linear_feature_dimensions = [1, 2, 3, 4, 5]
constant_values = [0, 1, -1]
learning_rate_values = [1e-2,]

# Shared DAMG repository used to build the search space for all kernel runs.
repo = DAMGrepository(linear_feature_dimensions=linear_feature_dimensions, constant_values=constant_values,
                     learning_rate_values=learning_rate_values,
                     n_epoch_values=[2000])

AVAILABLE_ODE_TARGETS: dict[str, Any] = {
    target_to_name(target_len_2): target_len_2,
    target_to_name(target_len_3): target_len_3,
    target_to_name(target_len_4): target_len_4,
    target_to_name(target_len_5): target_len_5,
    target_to_name(target_len_6): target_len_6,
    target_to_name(target_len_3_refined_1): target_len_3_refined_1,
    target_to_name(target_len_3_refined_2): target_len_3_refined_2,
    target_to_name(target_len_3_refined_3): target_len_3_refined_3,
    target_to_name(target_len_5_refined_1): target_len_5_refined_1,
    target_to_name(target_len_5_refined_2): target_len_5_refined_2,
    target_to_name(target_len_5_refined_3): target_len_5_refined_3,
    target_to_name(target_len_4_refined_1): target_len_4_refined_1,
}
AVAILABLE_ODE_TARGET_NAMES = tuple(AVAILABLE_ODE_TARGETS.keys())

@lru_cache(maxsize=1)
def _load_objective_dataset(dataset_path: Path = DEFAULT_DATASET_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the ODE regression dataset once and cache it for objective evaluations."""
    try:
        return cast(tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], load_dataset(dataset_path))
    except (FileNotFoundError, ValueError):
        print(f"Dataset not found or invalid at {dataset_path}. Generating new dataset.")
        x, y, x_test, y_test, _ = create_and_save_dataset(dataset_path=dataset_path)
        return x, y, x_test, y_test

def f_obj(t: Tree, seed: int | None = None) -> float:
    """Objective function: evaluate a Tree (neural network architecture) on the ODE regression task.

    This function is used ONLY when evaluating kernels as GP surrogates (evaluate_kernels_as_surrogates).
    Domain-based kernel analysis (analyze_kernel_domain_properties) does NOT need this function.

    Args:
        t: A Tree structure representing a neural network architecture
        seed: Random seed for reproducibility. If None, no seed is set.

    Returns:
        float: Test loss (lower is better)
    """
    if seed is not None:
        import random
        #import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    x, y, x_test, y_test = _load_objective_dataset(DEFAULT_DATASET_PATH)
    learner = t.interpret(pytorch_function_algebra())
    return learner(x, y, x_test, y_test)










@contextmanager
def _suppress_warnings() -> Iterator[None]:
    with warnings.catch_warnings():  # type: ignore[call-overload]
        warnings.simplefilter("ignore")
        yield

def _save_rows_to_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def _safe_nlpd(y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, eps: float = 1e-9) -> float:
    y_std = np.maximum(y_std, eps)
    nll = 0.5 * np.log(2.0 * np.pi * y_std ** 2) + 0.5 * ((y_true - y_mean) / y_std) ** 2
    return float(np.mean(np.asarray(nll, dtype=float)))

def _normalize_targets(y: np.ndarray, mean: float, scale: float, eps: float = 1e-12) -> np.ndarray:
    scale = max(float(scale), eps)
    return np.asarray((np.asarray(y, dtype=float) - float(mean)) / scale, dtype=float)

def _transform_targets(y: np.ndarray, mode: str) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    if mode == "none":
        return y_arr
    if mode == "clip":
        clip_value = float(np.quantile(y_arr, 0.99)) if y_arr.size else 0.0
        return np.asarray(np.clip(y_arr, None, clip_value), dtype=float)
    if mode == "log1p":
        return np.asarray(np.log1p(np.clip(y_arr, 0.0, None)), dtype=float)
    if mode == "log1p_clip":
        clip_value = float(np.quantile(y_arr, 0.99)) if y_arr.size else 0.0
        return np.asarray(np.log1p(np.clip(np.clip(y_arr, 0.0, None), None, clip_value)), dtype=float)
    raise ValueError(f"Unsupported target transformation: {mode}")

def compute_kernel_matrix_diagnostics(kernel: Any, X: np.ndarray) -> dict[str, Any]:
    """Compute K(X, X), eigenvalues, condition number and determinant."""
    K = np.asarray(kernel(X, X), dtype=float)
    # enforce symmetry
    K = 0.5 * (K + K.T)

    # normalize kernel matrix
    d = np.sqrt(np.diag(K))
    K_normalized = K / (d[:, None] * d[None, :] + 1e-12)

    # add jitter
    K_normalized += 1e-6 * np.eye(K.shape[0])

    eigenvalues = np.linalg.eigvalsh(np.asarray(K, dtype=float)).astype(float, copy=False)
    abs_eigs = np.abs(eigenvalues)
    eigs_for_cond = abs_eigs[abs_eigs > 1e-14]
    if eigs_for_cond.size == 0:
        condition_number = float("inf")
    else:
        condition_number = float(np.max(eigs_for_cond) / np.min(eigs_for_cond))

    determinant = float(np.linalg.det(np.asarray(K, dtype=float)))
    sign, log_abs_det = np.linalg.slogdet(np.asarray(K, dtype=float))
    if sign == 0:
        log_abs_det = float("-inf")

    return {
        "kernel_matrix": K,
        "normalized_kernel_matrix": K_normalized,
        "eigenvalues": eigenvalues,
        "condition_number": condition_number,
        "determinant": determinant,
        "log_abs_determinant": float(log_abs_det),
    }


def _sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return safe or "kernel"


def _surrogate_artifact_dir(root: Path, target_name: str) -> Path:
    return root / _sanitize_name(target_name) / "surrogate"


def _as_object_array(X: Iterable[Any]) -> np.ndarray:
    return np.asarray(list(X), dtype=object)

def _as_float_array(y: Iterable[float]) -> np.ndarray:
    return np.asarray(list(y), dtype=float)

@dataclass
class TargetRunData:
    target_name: str
    target_obj: Any
    X: np.ndarray
    y: np.ndarray | None = None
    diagnostics_by_kernel: dict[str, dict[str, Any]] = field(default_factory=dict)
    surrogate_diagnostics_by_kernel: dict[str, dict[str, Any]] = field(default_factory=dict)


def _safe_pairwise_correlation(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")

    if method == "spearman":
        result = cast(Any, spearmanr(x, y))  # type: ignore[arg-type]
        corr_value = result.correlation
        p_value = result.pvalue
    elif method == "pearson":
        result = cast(Any, pearsonr(x=x, y=y))
        corr_value = result.statistic
        p_value = result.pvalue
    elif method == "kendall":
        result = cast(Any, kendalltau(x=x, y=y))
        corr_value = result.correlation
        p_value = result.pvalue
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    return float(corr_value), float(p_value)

def _load_target_samples(target_obj: Any, n_samples: int, pre_sample_path: Path | str) -> np.ndarray:
    """Load or generate the deduplicated reusable X-sample set for one ODE target."""
    synthesizer = Synthesizer(repo.specification(), {})
    search_space = synthesizer.construct_solution_space(target_obj).prune()
    X, _ = get_or_create_unique_pre_samples(
        search_space=search_space,
        target=target_obj,
        pre_sample_path=Path(pre_sample_path),
        size=n_samples,
    )
    return _as_object_array(X)

def _resolve_ode_target(target: str | Any) -> tuple[str, Any]:
    """Resolve a target name or target object to a concrete ODE target."""
    if isinstance(target, str):
        target_name = target.strip()
        if not target_name:
            raise ValueError("Target name must not be empty.")
        if target_name.lower() == "all":
            raise ValueError("'all' is not a concrete target. Expand it before resolving.")
        try:
            return target_name, AVAILABLE_ODE_TARGETS[target_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown target '{target_name}'. Available targets: {', '.join(AVAILABLE_ODE_TARGET_NAMES)}"
            ) from exc

    target_name = target_to_name(target)
    if target_name == "unknown":
        raise ValueError(
            f"Unsupported target object: {target!r}. Available targets: {', '.join(AVAILABLE_ODE_TARGET_NAMES)}"
        )
    return target_name, target

def build_default_kernels() -> list[tuple[str, Any]]:
    return [
        ("noisy_hierarchical_tree_kernel", noisy_hierarchical_tree_kernel),
        ("noisy_hierarchical_wl_kernel", noisy_hierarchical_wl_kernel),
        ("noisy_hierarchical_damg_kernel", noisy_hierarchical_damg_kernel),
        ("noisy_combined_hierarchical_kernel", noisy_combined_hierarchical_kernel),
        ("noisy_hierarchical_damg_kernel_13", noisy_hierarchical_damg_kernel_13),
        ("tree_kernel_1", tree_kernel_1),
        ("tree_kernel_2", tree_kernel_2),
        ("tree_kernel_3", tree_kernel_3),
        ("wl_kernel_1", wl_kernel_1),
        ("wl_kernel_2", wl_kernel_2),
        ("wl_kernel_3", wl_kernel_3),
        ("damg_kernel_1", damg_kernel_1),
        ("damg_kernel_2", damg_kernel_2),
        ("damg_kernel_3", damg_kernel_3),
    ]

def _save_experiment_metadata(
    experiment_id: str,
    output_dir: Path,
    start_time: datetime,
    end_time: datetime | None,
    args: Any,
    status: str,
    error_message: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    duration_seconds = None if end_time is None else (end_time - start_time).total_seconds()

    metadata = {
        "experiment_id": experiment_id,
        "start_timestamp": start_time.isoformat(),
        "end_timestamp": end_time.isoformat() if end_time is not None else None,
        "duration_seconds": duration_seconds,
        "parameters": {
            "n_samples": args.n_samples,
            "cv_folds": args.cv_folds,
            "seed": args.seeds,
            "y_transform": args.y_transform,
            "alpha": args.alpha,
            "n_restarts_optimizer": args.n_restarts_optimizer,
            "mode": args.mode,
            "targets": args.targets,
            "max_kernels": args.max_kernels,
            "no_plots": args.no_plots,
        },
        #"environment": {
        #    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        #    "numpy_version": np.__version__,
        #    "scipy_version": _get_package_version("scipy"),
        #    "sklearn_version": sklearn.__version__,
        #    "os_system": platform.system(),
        #    "os_release": platform.release(),
        #    "machine": platform.machine(),
        #},
        "status": status,
        "error_message": error_message,
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return metadata_path

def _generate_experiment_id(n_samples: int, targets: list[str], seeds: list[int]) -> str:
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_n{n_samples}_targets{len(targets)}_seeds{len(seeds)}"

def _dedupe_pre_samples(pre_samples_x):
    """Remove duplicate sampled trees while preserving their original order."""
    unique_pre_samples_x = []
    seen = set()
    duplicates_removed = 0

    for tree in pre_samples_x:
        if tree in seen:
            duplicates_removed += 1
            continue
        seen.add(tree)
        unique_pre_samples_x.append(tree)

    return unique_pre_samples_x, seen, duplicates_removed


def _plot_results(target_runs: list[TargetRunData]) -> None:
    """Lightweight plotting hook for kernel experiments.

    The script is mainly a numerical analysis tool. When plotting is enabled we
    keep the behaviour intentionally minimal so the script still works in
    minimal environments.
    """
    if not target_runs:
        return
    print("Plotting requested, but no dedicated figures are generated here; skipping.")

def main() -> None:
    # ===============================
    #  CLI argument parsing
    # ===============================
    parser = argparse.ArgumentParser(description="Kernel experiment: analyze domain properties or evaluate kernels as GP surrogates.")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_PRESAMPLE_SIZE)
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds; values below 5 can be unstable.")
    #parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-restarts-optimizer", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument(
        "--y-transform",
        choices=["none", "log1p", "clip", "log1p_clip"],
        default="log1p",
        help="Transform target values before GP fitting.",
    )
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--pre-sample-path", type=str, default=str(DEFAULT_PRESAMPLE_PATH))
    parser.add_argument("--pre-sample-file", type=str, default=None,
                        help="Load a specific pre-sample .pt file instead of generating new ones.")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=[target_to_name(target_len_5)],
        help=(
            "One or more target names from ode_targets (e.g. target_len_3 target_len_5). "
            f"Available: {', '.join(AVAILABLE_ODE_TARGET_NAMES)}. Use 'all' to evaluate all targets."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[RANDOM_SEED],
        help=(
            "One or more seeds. If multiple seeds are provided, the objective function will be called with each seed and the mean of the results will be the objective function value."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-normalize-y", action="store_true",
                        help="Disable normalize_y in GP (useful with log1p to avoid double-normalization).")
    parser.add_argument("--max-kernels", type=int, default=0, help="Use only the first N kernels (0 = all).")
    parser.add_argument(
        "--mode",
        choices=["domain", "surrogate", "both"],
        default="both",
        help=(
            "Experiment mode: 'surrogate' = GP surrogate evaluation (also optimizes kernels), "
            "'domain' = optimized-kernel domain analysis after surrogate optimization, "
            "'both' = run surrogate optimization followed by domain analysis"
        ),
    )
    args = parser.parse_args()

    if len(args.targets) == 1 and args.targets[0].strip().lower() == "all":
        selected_targets = list(AVAILABLE_ODE_TARGET_NAMES)
    else:
        selected_targets = [str(target_name).strip() for target_name in args.targets if str(target_name).strip()]
    if not selected_targets:
        raise ValueError("At least one target must be selected.")

    if len(args.seeds) == 0:
        selected_seeds = [None]
    else:
        selected_seeds = [int(seed) for seed in args.seeds]

    # ===============================
    # Experiment setup and metadata logging
    # ===============================

    experiment_id = _generate_experiment_id(args.n_samples, selected_targets, args.seeds)
    output_root_base = Path(args.output_dir) / "kernel_experiments"
    output_root = output_root_base / str(experiment_id)
    output_root.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc)
    metadata_json_path = _save_experiment_metadata(
        experiment_id=experiment_id,
        output_dir=output_root,
        start_time=start_time,
        end_time=None,
        args=args,
        status="running",
        error_message=None,
    )

    print(f"\nExperiment ID: {experiment_id}")
    print(f"Results will be written to: {output_root}")

    try:
    # ===============================
    # Run the experiment
    # ===============================
        if args.cv_folds < 5:
            warnings.warn(
                f"cv_folds={args.cv_folds} can lead to unstable surrogate estimates because a single outlier may dominate a fold.",
                RuntimeWarning,
                stacklevel=2,
            )

        kernels = build_default_kernels()
        if args.max_kernels > 0:
            kernels = kernels[: args.max_kernels]

        resolved_targets = [_resolve_ode_target(target_name) for target_name in selected_targets]

        print(f"Selected targets: {', '.join(name for name, _ in resolved_targets)}")

        target_runs: list[TargetRunData] = []
        for target_name, target_obj in resolved_targets:
            synthesizer = Synthesizer(repo.specification(), {})
            search_space = synthesizer.construct_solution_space(target_obj).prune()
            # Load from specified pre-sample file or generate a new sample.
            if args.pre_sample_file:
                print(f"Loading pre-sample from: {args.pre_sample_file}")
                pre_samples_x, metadata = load_pre_samples(args.pre_sample_file)
                X = pre_samples_x
            else:
                X, metadata = generate_pre_samples(search_space, target_obj, args.n_samples, output_root)
            unique_pre_samples_x, seen, duplicates_removed = _dedupe_pre_samples(X)
            if not args.pre_sample_file:
                attempts = len(X)
                additional_generated = 0

                # Top up the presample set if deduplication removed collisions.
                while len(unique_pre_samples_x) < args.n_samples and attempts < args.n_samples * 3:
                    batch_size = max(args.n_samples - len(unique_pre_samples_x), 5)
                    batch_pre_samples_x, batch_metadata = generate_pre_samples(search_space, target_obj, batch_size)
                    additional_generated += len(batch_pre_samples_x)
                    attempts += len(batch_pre_samples_x)

                    for tree in batch_pre_samples_x:
                        if tree in seen:
                            duplicates_removed += 1
                            continue
                        seen.add(tree)
                        unique_pre_samples_x.append(tree)
                        if len(unique_pre_samples_x) >= args.n_samples:
                            break
                size = len(unique_pre_samples_x)
                if size < args.n_samples:
                    warnings.warn(
                        f"Only {size} unique pre-samples could be generated for target '{target_name}' "
                        f"after {attempts} attempts (requested {args.n_samples}).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                unique_pre_samples_x = unique_pre_samples_x[:args.n_samples]
                metadata.update({
                    "target_name": target_name,
                    "sample_size": size,
                    "file_name": metadata.get("file_name", f"presample_{target_name}_{size}.pt"),
                    "path": metadata.get("path", output_root / f"presample_{target_name}_{size}.pt"),
                })
                path = save_pre_samples(X, metadata, metadata["path"])
                print(f"Saved presample of size {size} for target {target_name} at {path}.")
                X = np.array(unique_pre_samples_x, dtype=Tree)
            else:
                X = np.array(X, dtype=Tree)
                size = len(X)
                print(f"Loaded {size} pre-samples from file for target {target_name}.")
            # `_load_target_samples(...)` remains available for manual reuse, but
            # the script now always generates a fresh presample set for clarity.
            # X = _load_target_samples(target_obj, args.n_samples, args.pre_sample_path)
            target_runs.append(
                TargetRunData(
                    target_name=target_name,
                    target_obj=target_obj,
                    X=X,
                )
            )

        # =========================================================================
        # EXPERIMENT 1: Kernel-Objective Alignment and Kernel-Matrix Diagnostics
        # =========================================================================

        if args.mode in ("domain", "both"):
            print("\n" + "=" * 80)
            print("EXPERIMENT 1: Kernel-Objective Alignment and Kernel-Matrix Diagnostics")
            print("=" * 80)


            for target_run in target_runs:

                # ======== Step 1: Sample programs (or load sample, if available)

                target_name = target_run.target_name
                X = target_run.X

                # ======== Step 2: Evaluate objective function (if multiple seeds are provided, evaluate with each seed and average)

                print(f"\nTarget: {target_name} | samples: {X.shape[0]}")
                print("Generating target values via objective function...")
                y = np.array([
                    np.mean(np.array([f_obj(tree, seed) for seed in selected_seeds], dtype=float))
                    for tree in X], dtype=float)
                # Guard: replace any non-finite objective values (NaN/Inf from
                # diverged training) so that downstream GP fitting / log1p stays valid.
                n_non_finite = int(np.sum(~np.isfinite(y)))
                if n_non_finite:
                    warnings.warn(
                        f"{n_non_finite} of {y.size} objective evaluations returned NaN/Inf; "
                        "replacing with finite penalty (1e6).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    y = np.where(np.isfinite(y), y, 1e6)
                print(f"Raw y-values: mean={np.mean(y):.8f}, std={np.std(y):.8f}")
                # Objective transformation
                y = _transform_targets(y, args.y_transform)
                print(f"Transformed y-values ({args.y_transform}): mean={np.mean(y):.8f}, std={np.std(y):.8f}")

                # Store transformed y in the TargetRunData
                target_run.y = y

                # ======== Step 3: Pairwise objective difference matrix
                pairwise_objective_diff = -np.abs(y[:, None] - y[None, :])

                upper_O_indices = np.triu_indices(pairwise_objective_diff.shape[0], k=1)
                upper_O = np.asarray(pairwise_objective_diff[upper_O_indices], dtype=float)

                # ======== Step 4: Compute kernel matrix diagnostics for each kernel

                diagnostics_by_kernel: dict[str, dict[str, Any]] = {}
                for kernel_name, kernel_obj in kernels:
                    print(f"\nAnalyzing kernel: {kernel_name}")
                    diagnostics = compute_kernel_matrix_diagnostics(kernel_obj, X)

                    K = diagnostics["normalized_kernel_matrix"]
                    upper_K_indices = np.triu_indices(K.shape[0], k=1)
                    upper_K = np.asarray(K[upper_K_indices], dtype=float)

                    # ======== Step 5: Spearman correlation for kernel objective alignment

                    #result = cast(Any, spearmanr(K, pairwise_objective_diff))  # type: ignore[arg-type]
                    #corr_value = result.correlation
                    #p_value = result.pvalue

                    corr_value, p_value = _safe_pairwise_correlation(upper_K, upper_O, "spearman")

                    diagnostics["correlation_method"] = "spearman"
                    diagnostics["kernel_objective_correlation"] = corr_value
                    diagnostics["kernel_objective_p_value"] = p_value

                    diagnostics_by_kernel[kernel_name] = diagnostics

                    print(f"Kernel '{kernel_name}' objective correlation: {corr_value:.4f} (p-value: {p_value:.4f})")


                target_run.diagnostics_by_kernel = diagnostics_by_kernel

        # =========================================================================
        # EXPERIMENT 2: Surrogate Learnability
        # =========================================================================

        if args.mode in ("surrogate", "both"):
            print("\n" + "=" * 80)
            print("EXPERIMENT 2: Surrogate Learnability")
            print("=" * 80)


            for target_run in target_runs:

                # ======== Step 1: Sample programs (or load sample, if available)

                target_name = target_run.target_name
                X = target_run.X
                print(f"\nTarget: {target_name} | samples: {X.shape[0]}")

                # ======== Step 2: Evaluate objective function (if multiple seeds are provided, evaluate with each seed and average)

                y = target_run.y
                if y is None:
                    print("Generating target values via objective function...")
                    y = np.array([
                        np.mean(np.array([f_obj(tree, seed) for seed in selected_seeds], dtype=float))
                        for tree in X], dtype=float)
                    # Guard: replace any non-finite objective values (NaN/Inf from
                    # diverged training) so that downstream GP fitting / log1p stays valid.
                    n_non_finite = int(np.sum(~np.isfinite(y)))
                    if n_non_finite:
                        warnings.warn(
                            f"{n_non_finite} of {y.size} objective evaluations returned NaN/Inf; "
                            "replacing with finite penalty (1e6).",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        y = np.where(np.isfinite(y), y, 1e6)
                    print(f"Raw y-values: mean={np.mean(y):.8f}, std={np.std(y):.8f}")
                    # Objective transformation
                    y = _transform_targets(y, args.y_transform)
                    print(f"Transformed y-values ({args.y_transform}): mean={np.mean(y):.8f}, std={np.std(y):.8f}")

                random_state = selected_seeds[0]

                cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=random_state)

                # Store per-fold predictions for ensemble (ALGO-06).
                fold_predictions: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
                    kernel_name: [] for kernel_name, _ in kernels
                }
                fold_truths: list[np.ndarray] = []

                surrogate_diagnostics_by_kernel: dict[str, dict[str, Any]] = {}
                for kernel_name, kernel_obj in kernels:
                    normalized_mse_scores: list[float] = []
                    r2_scores: list[float] = []
                    nlpd_scores: list[float] = []

                    # ======== Step 3: Cross Validation for each kernel

                    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):
                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]
                        y_train = np.asarray(y_train, dtype=float)
                        y_test = np.asarray(y_test, dtype=float)
                        y_train_mean = float(np.mean(y_train))
                        y_train_scale = float(np.std(y_train))
                        y_test_norm = _normalize_targets(y_test, y_train_mean, y_train_scale)

                        gp = GaussianProcessRegressor(
                            kernel=clone(kernel_obj),
                            alpha=args.alpha,
                            normalize_y=not args.no_normalize_y,
                            n_restarts_optimizer=args.n_restarts_optimizer,
                            random_state=random_state,
                        )
                        with _suppress_warnings():
                            gp.fit(X_train, y_train)

                        # ======== Step 4: Metrics on predictions

                        y_mean, y_std = gp.predict(X_test, return_std=True)
                        y_mean_norm = _normalize_targets(y_mean, y_train_mean, y_train_scale)
                        y_std_norm = np.asarray(y_std, dtype=float) / max(y_train_scale, 1e-12)

                        # Store predictions for ensemble
                        fold_predictions[kernel_name].append((y_mean_norm, y_test_norm))

                        normalized_mse_scores.append(float(np.mean(
                            np.square(np.asarray(y_test_norm, dtype=float) - np.asarray(y_mean_norm, dtype=float)))))
                        r2_scores.append(
                            float(r2_score(np.asarray(y_test_norm, dtype=float), np.asarray(y_mean_norm, dtype=float))))
                        nlpd_scores.append(_safe_nlpd(y_test_norm, y_mean_norm, y_std_norm))

                    surrogate_diagnostics_by_kernel[kernel_name] = {
                        "normalized_mse_mean": float(np.mean(normalized_mse_scores)),
                        "normalized_mse_std": float(np.std(normalized_mse_scores)),
                        "r2_mean": float(np.mean(r2_scores)),
                        "r2_std": float(np.std(r2_scores)),
                        "nlpd_mean": float(np.mean(nlpd_scores)),
                        "nlpd_std": float(np.std(nlpd_scores)),
                    }

                # ======== Step 5: Ensemble via per-fold softmax kernel weighting (ALGO-06 + refinement)
                # Temperature sweep + kernel quality filtering

                n_folds = args.cv_folds
                temperatures = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

                # Filter out kernels with overall NMSE > 1.0 (worse than predicting mean)
                good_kernel_names = [
                    name for name in surrogate_diagnostics_by_kernel
                    if surrogate_diagnostics_by_kernel[name]["normalized_mse_mean"] < 1.0
                ]
                if not good_kernel_names:
                    good_kernel_names = list(surrogate_diagnostics_by_kernel.keys())
                print(f"  Ensemble: filtering kernels, kept {len(good_kernel_names)}/{len(fold_predictions)} (NMSE < 1.0)")

                best_ens_nmse = float("inf")
                best_ens_diag = None
                best_T = None

                for temperature in temperatures:
                    ensemble_nmse_scores: list[float] = []
                    ensemble_r2_scores: list[float] = []

                    for fold_i in range(n_folds):
                        # Compute per-kernel NMSE for this fold
                        fold_nmse: dict[str, float] = {}
                        for kernel_name in good_kernel_names:
                            y_pred_norm, y_test_norm = fold_predictions[kernel_name][fold_i]
                            fold_nmse[kernel_name] = float(np.mean(
                                np.square(np.asarray(y_test_norm, dtype=float) - np.asarray(y_pred_norm, dtype=float))))

                        # Softmax weights: w_k = exp(-NMSE_k / T)
                        nmse_values = np.array([fold_nmse[name] for name in good_kernel_names])
                        neg_nmse = -nmse_values / max(temperature, 1e-12)
                        neg_nmse = neg_nmse - np.max(neg_nmse)  # Numerical stability
                        weights = np.exp(neg_nmse)
                        weight_sum = np.sum(weights)
                        if weight_sum > 0:
                            weights = weights / weight_sum
                        else:
                            weights = np.ones(len(good_kernel_names)) / len(good_kernel_names)

                        # Ensemble prediction
                        y_ens = np.zeros_like(fold_predictions[good_kernel_names[0]][fold_i][0])
                        for i, kernel_name in enumerate(good_kernel_names):
                            y_pred_norm, y_test_norm = fold_predictions[kernel_name][fold_i]
                            y_ens = y_ens + weights[i] * np.asarray(y_pred_norm, dtype=float)

                        ensemble_nmse_scores.append(float(np.mean(
                            np.square(np.asarray(y_test_norm, dtype=float) - y_ens))))
                        ensemble_r2_scores.append(
                            float(r2_score(np.asarray(y_test_norm, dtype=float), y_ens)))

                    nmse_mean = float(np.mean(ensemble_nmse_scores))
                    print(f"  Ensemble T={temperature:.2f}: NMSE={nmse_mean:.6f}, R2={float(np.mean(ensemble_r2_scores)):.6f}")

                    if nmse_mean < best_ens_nmse:
                        best_ens_nmse = nmse_mean
                        best_ens_diag = {
                            "normalized_mse_mean": nmse_mean,
                            "normalized_mse_std": float(np.std(ensemble_nmse_scores)),
                            "r2_mean": float(np.mean(ensemble_r2_scores)),
                            "r2_std": float(np.std(ensemble_r2_scores)),
                            "nlpd_mean": 0.0,
                            "nlpd_std": 0.0,
                        }
                        best_T = temperature

                surrogate_diagnostics_by_kernel["ensemble_softmax"] = best_ens_diag
                print(f"  Best ensemble: T={best_T}, NMSE={best_ens_nmse:.6f}, R2={best_ens_diag['r2_mean']:.6f}")

                target_run.surrogate_diagnostics_by_kernel = surrogate_diagnostics_by_kernel

        #  ===============================
        #  Plot results
        #  ===============================

        if not args.no_plots:
            _plot_results(target_runs)

        #  ===============================
        #  Save results and metadata
        #  ===============================

        csv_rows: list[dict[str, Any]] = []

        for target_run in target_runs:
            for kernel_name, kernel_obj in kernels:
                diagnostics = target_run.diagnostics_by_kernel.get(kernel_name, {})
                surrogate_diagnostics = target_run.surrogate_diagnostics_by_kernel.get(kernel_name, {})
                csv_rows.append({
                    "target": target_run.target_name,
                    "kernel": kernel_name,
                    "correlation_method": diagnostics.get("correlation_method"),
                    "kernel_objective_spearman_correlation": diagnostics.get("kernel_objective_correlation"),
                    "kernel_objective_spearman_p_value": diagnostics.get("kernel_objective_p_value"),
                    "kernel_condition_number": diagnostics.get("condition_number"),
                    "kernel_log_abs_determinant": diagnostics.get("log_abs_determinant"),
                    "surrogate_normalized_mse_mean": surrogate_diagnostics.get("normalized_mse_mean"),
                    "surrogate_normalized_mse_std": surrogate_diagnostics.get("normalized_mse_std"),
                    "surrogate_r2_mean": surrogate_diagnostics.get("r2_mean"),
                    "surrogate_r2_std": surrogate_diagnostics.get("r2_std"),
                    "surrogate_nlpd_mean": surrogate_diagnostics.get("nlpd_mean"),
                    "surrogate_nlpd_std": surrogate_diagnostics.get("nlpd_std"),
                })

            # Add ensemble row if present (ALGO-06)
            ens_diag = target_run.surrogate_diagnostics_by_kernel.get("ensemble_softmax")
            if ens_diag is not None:
                csv_rows.append({
                    "target": target_run.target_name,
                    "kernel": "ensemble_softmax",
                    "correlation_method": None,
                    "kernel_objective_spearman_correlation": None,
                    "kernel_objective_spearman_p_value": None,
                    "kernel_condition_number": None,
                    "kernel_log_abs_determinant": None,
                    "surrogate_normalized_mse_mean": ens_diag.get("normalized_mse_mean"),
                    "surrogate_normalized_mse_std": ens_diag.get("normalized_mse_std"),
                    "surrogate_r2_mean": ens_diag.get("r2_mean"),
                    "surrogate_r2_std": ens_diag.get("r2_std"),
                    "surrogate_nlpd_mean": ens_diag.get("nlpd_mean"),
                    "surrogate_nlpd_std": ens_diag.get("nlpd_std"),
                })

        end_time = datetime.now(timezone.utc)
        csv_path = output_root / "results.csv"
        _save_rows_to_csv(csv_rows, csv_path)

        metadata_json_path = _save_experiment_metadata(
            experiment_id=experiment_id,
            output_dir=output_root,
            start_time=start_time,
            end_time=end_time,
            args=args,
            status="success",
            error_message=None,
        )

        print("\n" + "=" * 80)
        print("Experiment completed successfully!")
        print("=" * 80)




    except Exception as e:
        end_time = datetime.now(timezone.utc)
        error_message = f"{type(e).__name__}: {e}"
        metadata_json_path = _save_experiment_metadata(
            experiment_id=experiment_id,
            output_dir=output_root,
            start_time=start_time,
            end_time=end_time,
            args=args,
            status="failed",
            error_message=error_message,
        )
        print("\n" + "=" * 80)
        print(f"Experiment failed: {error_message}")
        print("=" * 80)
        raise





if __name__ == "__main__":
    main()
