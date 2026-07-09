#!/usr/bin/env python3
"""
Compute intrinsic dimension and other dataset metrics.

Usage:
    python scripts/analysis/compute_metrics.py <dataset> [options]
    python scripts/analysis/compute_metrics.py mnist --method mle
    python scripts/analysis/compute_metrics.py --all

Metrics computed:
    - Intrinsic dimension (MLE, correlation dimension)
    - Aspect ratio
    - Dataset statistics (n, d, sparsity)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from scipy import stats

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
RESULTS_DIR = ROOT_DIR / "results"

# Available datasets
DATASETS = [
    "mnist", "fmnist", "cifar10", "cifar100",
    "mnist_clip", "fmnist_clip", "cifar10_clip", "cifar100_clip",
    "har", "susy", "reddit", "stackexchange"
]


def load_dataset(dataset: str) -> np.ndarray:
    """Load dataset from text file."""
    path = DATASETS_DIR / f"{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return np.loadtxt(path)


def mle_intrinsic_dimension(X: np.ndarray, k: int = 10,
                            n_samples: int = 1000) -> float:
    """
    Estimate intrinsic dimension using MLE (Levina & Bickel, 2004).

    For each point, uses k nearest neighbors to estimate local dimension.
    """
    n, d = X.shape
    n_samples = min(n_samples, n)

    # Random sample for efficiency
    indices = np.random.choice(n, n_samples, replace=False)
    X_sample = X[indices]

    # Compute pairwise distances
    from scipy.spatial import KDTree
    tree = KDTree(X)

    dims = []
    for i, x in enumerate(X_sample):
        # Get k+1 nearest neighbors (including self)
        dists, _ = tree.query(x, k=k+1)
        dists = dists[1:]  # Exclude self

        # MLE estimate
        if dists[-1] > 0 and all(dists > 0):
            log_ratios = np.log(dists[-1] / dists[:-1])
            dim_est = (k - 1) / np.sum(log_ratios)
            dims.append(dim_est)

    return np.mean(dims), np.std(dims)


def correlation_dimension(X: np.ndarray, n_samples: int = 1000,
                          eps_range: tuple = (0.01, 0.5)) -> float:
    """
    Estimate correlation dimension using the Grassberger-Procaccia algorithm.

    Fits C(eps) ~ eps^d where C(eps) is the correlation integral.
    """
    n, d = X.shape
    n_samples = min(n_samples, n)

    # Random sample
    indices = np.random.choice(n, n_samples, replace=False)
    X_sample = X[indices]

    # Compute pairwise distances
    dists = pdist(X_sample)
    max_dist = np.max(dists)

    # Range of epsilon values
    eps_values = np.logspace(
        np.log10(eps_range[0] * max_dist),
        np.log10(eps_range[1] * max_dist),
        50
    )

    # Compute correlation integral for each epsilon
    correlations = []
    for eps in eps_values:
        count = np.sum(dists < eps)
        n_pairs = len(dists)
        correlations.append(count / n_pairs)

    # Fit log-log slope
    valid = np.array(correlations) > 0
    if np.sum(valid) < 10:
        return np.nan, np.nan

    log_eps = np.log(eps_values[valid])
    log_corr = np.log(np.array(correlations)[valid])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps, log_corr)

    return slope, r_value**2


def compute_aspect_ratio(X: np.ndarray, n_samples: int = 5000) -> float:
    """
    Compute aspect ratio: max distance / min distance.
    """
    n = X.shape[0]
    n_samples = min(n_samples, n)

    indices = np.random.choice(n, n_samples, replace=False)
    X_sample = X[indices]

    dists = pdist(X_sample)
    dists = dists[dists > 0]  # Remove zeros

    return np.max(dists) / np.min(dists)


def compute_statistics(X: np.ndarray) -> dict:
    """Compute basic dataset statistics."""
    n, d = X.shape

    # Sparsity
    sparsity = np.mean(X == 0)

    # Norms
    norms = np.linalg.norm(X, axis=1)

    return {
        'n': n,
        'd': d,
        'sparsity': sparsity,
        'mean_norm': np.mean(norms),
        'std_norm': np.std(norms),
        'min_val': np.min(X),
        'max_val': np.max(X),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute dataset metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", nargs="?", help="Dataset name")
    parser.add_argument("--all", action="store_true",
                        help="Compute metrics for all datasets")
    parser.add_argument("--method", choices=["mle", "correlation", "all"],
                        default="all", help="Intrinsic dimension method")
    parser.add_argument("--k", type=int, default=10,
                        help="k for MLE method (default: 10)")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of samples for estimation (default: 1000)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV file")

    args = parser.parse_args()

    if args.all:
        datasets = DATASETS
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        sys.exit(1)

    results = []

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print("=" * 60)

        try:
            X = load_dataset(dataset)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        # Basic statistics
        stats_dict = compute_statistics(X)
        print(f"  n={stats_dict['n']}, d={stats_dict['d']}")
        print(f"  Sparsity: {stats_dict['sparsity']:.2%}")
        print(f"  Norm: {stats_dict['mean_norm']:.2f} ± {stats_dict['std_norm']:.2f}")

        result = {'dataset': dataset, **stats_dict}

        # Intrinsic dimension
        if args.method in ["mle", "all"]:
            print(f"  Computing MLE intrinsic dimension (k={args.k})...")
            id_mle, id_mle_std = mle_intrinsic_dimension(X, k=args.k, n_samples=args.samples)
            print(f"  MLE ID: {id_mle:.2f} ± {id_mle_std:.2f}")
            result['id_mle'] = id_mle
            result['id_mle_std'] = id_mle_std

        if args.method in ["correlation", "all"]:
            print(f"  Computing correlation dimension...")
            id_corr, r2 = correlation_dimension(X, n_samples=args.samples)
            print(f"  Correlation ID: {id_corr:.2f} (R²={r2:.3f})")
            result['id_correlation'] = id_corr
            result['id_corr_r2'] = r2

        # Aspect ratio
        print(f"  Computing aspect ratio...")
        aspect = compute_aspect_ratio(X, n_samples=args.samples)
        print(f"  Aspect ratio: {aspect:.2f}")
        result['aspect_ratio'] = aspect

        results.append(result)

    # Save results
    if args.output and results:
        import pandas as pd
        df = pd.DataFrame(results)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    # Print summary table
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Dataset':<20} {'n':>8} {'d':>6} {'ID (MLE)':>12} {'Aspect':>10}")
        print("-" * 80)
        for r in results:
            id_str = f"{r.get('id_mle', 0):.1f}" if 'id_mle' in r else "--"
            print(f"{r['dataset']:<20} {r['n']:>8} {r['d']:>6} {id_str:>12} {r['aspect_ratio']:>10.1f}")


if __name__ == "__main__":
    main()
