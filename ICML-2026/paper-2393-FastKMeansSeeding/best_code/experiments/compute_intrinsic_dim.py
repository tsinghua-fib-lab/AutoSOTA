#!/usr/bin/env python3
"""
Compute intrinsic dimension using Levina-Bickel MLE estimator.
For each dataset: sample 10k points, compute ID, repeat 10 times, report average.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


def mle_intrinsic_dim(X: np.ndarray, k: int) -> float:
    """
    Levina-Bickel MLE estimator for intrinsic dimension.

    For each point, computes the MLE estimate based on distances to k nearest neighbors.
    Returns the average over all points.

    Reference: Levina & Bickel (2004) "Maximum Likelihood Estimation of Intrinsic Dimension"
    """
    from scipy.spatial import cKDTree

    n, d = X.shape

    # Use KD-tree for efficient nearest neighbor search
    tree = cKDTree(X)
    # Query k+1 neighbors (includes self)
    dists, _ = tree.query(X, k=k + 1, workers=-1)

    # Exclude self-distance (first column)
    dists = dists[:, 1:k + 1]

    # MLE estimate: d_hat_i = (k-1) / sum_{j=1}^{k-1} log(T_k / T_j)
    T_k = dists[:, -1:]  # Shape (n, 1)
    T_j = dists[:, :-1]   # Shape (n, k-1)

    # Avoid log(0) or division by zero
    T_k = np.maximum(T_k, 1e-10)
    T_j = np.maximum(T_j, 1e-10)

    log_ratios = np.log(T_k / T_j)  # Shape (n, k-1)

    # Sum over j for each point
    sum_log_ratios = np.sum(log_ratios, axis=1)  # Shape (n,)

    # Compute per-point estimate
    d_hat_i = (k - 1) / np.maximum(sum_log_ratios, 1e-10)

    # Return average (excluding outliers)
    d_hat_i = d_hat_i[np.isfinite(d_hat_i)]

    return np.mean(d_hat_i)


def estimate_id_with_subsampling(
    X: np.ndarray,
    k_values: List[int],
    n_samples: int = 10000,
    n_repeats: int = 10,
    seed: int = 42
) -> Tuple[dict, dict]:
    """
    Estimate intrinsic dimension with subsampling.

    Returns:
        means: dict mapping k -> mean ID estimate
        stds: dict mapping k -> std of ID estimates
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    results = {k: [] for k in k_values}

    for repeat in tqdm(range(n_repeats), desc="  Repeats", leave=False):
        # Sample subset
        if n > n_samples:
            idx = rng.choice(n, size=n_samples, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X

        for k in k_values:
            if k >= X_sub.shape[0]:
                continue
            id_est = mle_intrinsic_dim(X_sub, k)
            results[k].append(id_est)

    means = {k: np.mean(v) for k, v in results.items() if len(v) > 0}
    stds = {k: np.std(v) for k, v in results.items() if len(v) > 0}

    return means, stds


def load_dataset(path: Path) -> np.ndarray:
    """Load dataset from .npy or .txt file."""
    if path.suffix == ".npy":
        data = np.load(path)
        # Handle image datasets stored as (n, h, w) or (n, h, w, c)
        # Flatten to (n, h*w) or (n, h*w*c)
        if data.ndim > 2:
            n = data.shape[0]
            data = data.reshape(n, -1)
        return data
    elif path.suffix == ".txt":
        return np.loadtxt(path)
    else:
        raise ValueError(f"Unknown file format: {path.suffix}")


def main():
    datasets_dir = Path(__file__).parent.parent / "datasets"
    results_dir = Path(__file__).parent / "results" / "intrinsic_dim"
    results_dir.mkdir(parents=True, exist_ok=True)

    # k values to try
    k_values = [5, 10, 20, 50, 100]

    # Find all datasets
    dataset_files = list(datasets_dir.glob("*.npy")) + list(datasets_dir.glob("*.txt"))
    dataset_files = [f for f in dataset_files if not f.name.startswith(".")]

    # Skip very large files that are duplicates (prefer .npy)
    seen_names = set()
    filtered_files = []
    for f in sorted(dataset_files):
        name = f.stem
        if name not in seen_names:
            seen_names.add(name)
            filtered_files.append(f)

    print(f"Found {len(filtered_files)} datasets")
    print(f"k values: {k_values}")
    print(f"Subsampling: 10k points, 10 repeats")
    print("=" * 60)

    all_results = []

    for fpath in tqdm(sorted(filtered_files), desc="Datasets"):
        name = fpath.stem
        tqdm.write(f"\nProcessing: {name}")

        try:
            X = load_dataset(fpath)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            n, d = X.shape
            tqdm.write(f"  Shape: n={n}, d={d}")

            means, stds = estimate_id_with_subsampling(
                X, k_values, n_samples=10000, n_repeats=10
            )

            tqdm.write(f"  Intrinsic dimension estimates:")
            for k in k_values:
                if k in means:
                    tqdm.write(f"    k={k:3d}: {means[k]:.2f} +/- {stds[k]:.2f}")

            # Store results
            for k in k_values:
                if k in means:
                    all_results.append({
                        "dataset": name,
                        "n_points": n,
                        "ambient_dim": d,
                        "k": k,
                        "id_mean": means[k],
                        "id_std": stds[k],
                    })

        except Exception as e:
            tqdm.write(f"  Error: {e}")
            continue

    # Save results
    df = pd.DataFrame(all_results)
    output_path = results_dir / "mle_intrinsic_dim.csv"
    df.to_csv(output_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Intrinsic Dimension Estimates (MLE, k=20)")
    print("=" * 60)

    summary = df[df["k"] == 20][["dataset", "n_points", "ambient_dim", "id_mean", "id_std"]]
    summary = summary.sort_values("dataset")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
