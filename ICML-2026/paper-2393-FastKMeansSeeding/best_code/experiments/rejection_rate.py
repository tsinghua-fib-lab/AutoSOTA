#!/usr/bin/env python3
"""
Rejection Rate vs m Experiment

Measures the fraction of center selections where QKMEANS rejection sampling
fails (exhausts all m * ln(k) proposals without acceptance) as a function
of the chain length parameter m.

Reimplements the QKMEANS rejection loop from src/algorithms/qkmeans.hpp
in Python with exact nearest neighbor for instrumentation.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from tqdm import tqdm

# Add parent directory so we can import from scaling.py
sys.path.insert(0, str(Path(__file__).parent))

from scaling import download_dataset, load_dataset

# Publication-quality settings
plt.rcParams.update({
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})


# ============================================================
# QKMEANS Rejection Sampling (Python reimplementation)
# ============================================================

def qkmeans_rejection_sampling(X: np.ndarray, k: int, m: int, seed: int = 42) -> dict:
    """
    Reimplement QKMEANS center selection with rejection tracking.

    Mirrors src/algorithms/qkmeans.hpp lines 113-208.

    Args:
        X: (n, d) float32 data array
        k: number of centers
        m: chain length parameter (max_iter = m * ln(k+1))
        seed: random seed

    Returns:
        dict with:
            n_centers: k-1 (number of center selections after first)
            n_not_accepted: count where rejection loop exhausted budget
            rejection_fraction: n_not_accepted / n_centers
            mean_acceptance_iter: avg iteration of acceptance (for accepted)
            center_indices: list of selected center indices
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape

    # Step 1: Compute squared norms
    sq_norms = np.sum(X ** 2, axis=1)  # (n,)

    # Step 2: Sample first center uniformly
    c1_idx = rng.randint(0, n)
    c1_norm_sq = sq_norms[c1_idx]

    # Step 3: Build kappa weights and proposal weights
    # kappa(x) ∝ ||x||² + ||c1||²
    kappa_weights = sq_norms + c1_norm_sq  # (n,)
    proposal_weights = 2.0 * kappa_weights  # (n,)

    # Normalize kappa for sampling
    kappa_cumsum = np.cumsum(kappa_weights)
    kappa_total = kappa_cumsum[-1]

    # Center storage
    center_indices = [c1_idx]
    centers = [X[c1_idx].copy()]  # list of center vectors

    # Rejection sampling budget
    max_iter = max(10, int(m * np.log(k + 1.0)))

    # Tracking
    n_not_accepted = 0
    acceptance_iters = []

    for t in range(1, k):
        # Fallback: uniform random sample
        s = rng.randint(0, n)
        accepted = False

        for iteration in range(max_iter):
            # Sample x from kappa distribution
            u = rng.uniform(0, kappa_total)
            x_idx = np.searchsorted(kappa_cumsum, u)
            x_idx = min(x_idx, n - 1)

            # Compute exact distance to nearest center
            x_vec = X[x_idx]
            dists = np.array([np.sum((x_vec - c) ** 2) for c in centers])
            dist_to_center = dists.min()

            # Acceptance probability: r = dist / proposal_weight
            r = dist_to_center / proposal_weights[x_idx]
            r = min(r, 1.0)

            # Rejection test
            if rng.uniform() <= r:
                s = x_idx
                accepted = True
                acceptance_iters.append(iteration)
                break

        if not accepted:
            n_not_accepted += 1

        center_indices.append(s)
        centers.append(X[s].copy())

    n_centers = k - 1
    rejection_fraction = n_not_accepted / n_centers if n_centers > 0 else 0.0
    mean_acc_iter = np.mean(acceptance_iters) if acceptance_iters else float('nan')

    return {
        'n_centers': n_centers,
        'n_not_accepted': n_not_accepted,
        'rejection_fraction': rejection_fraction,
        'mean_acceptance_iter': mean_acc_iter,
        'center_indices': center_indices,
    }


# ============================================================
# Experiment Runner
# ============================================================

def run_rejection_experiment(
    X: np.ndarray,
    dataset_name: str,
    m_values: list,
    k_values: list,
    n_runs: int,
) -> pd.DataFrame:
    """
    Run rejection rate experiment across all (m, k) combinations.

    Returns DataFrame with columns:
        dataset, m, k, run, n_centers, n_not_accepted,
        rejection_fraction, mean_acceptance_iter
    """
    results = []
    total = len(m_values) * len(k_values) * n_runs
    pbar = tqdm(total=total, desc=f"{dataset_name}")

    for m in m_values:
        for k in k_values:
            if k > len(X):
                pbar.update(n_runs)
                continue

            for run in range(n_runs):
                seed = run * 10000 + m * 100 + k
                stats = qkmeans_rejection_sampling(X, k, m, seed)

                results.append({
                    'dataset': dataset_name,
                    'm': m,
                    'k': k,
                    'run': run,
                    'n_centers': stats['n_centers'],
                    'n_not_accepted': stats['n_not_accepted'],
                    'rejection_fraction': stats['rejection_fraction'],
                    'mean_acceptance_iter': stats['mean_acceptance_iter'],
                })

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(results)


# ============================================================
# Plotting
# ============================================================

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']

DISPLAY_NAMES = {
    "mnist": "MNIST", "mnist_clip": "MNIST-CLIP",
    "har": "HAR", "stackexchange": "StackExchange",
}


def plot_rejection_grid(all_results: pd.DataFrame, output_dir: Path):
    """
    2x2 grid plot: one subplot per dataset.
    Each subplot: rejection fraction vs m, one line per k.
    """
    datasets = all_results['dataset'].unique()
    n_datasets = len(datasets)

    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 9))
    axes = axes.flatten()

    for idx, ds_name in enumerate(datasets):
        if idx >= nrows * ncols:
            break
        ax = axes[idx]
        df_ds = all_results[all_results['dataset'] == ds_name]

        k_values = sorted(df_ds['k'].unique())

        for ki, k in enumerate(k_values):
            df_k = df_ds[df_ds['k'] == k]

            # Average over runs
            grouped = df_k.groupby('m').agg(
                rej_mean=('rejection_fraction', 'mean'),
                rej_std=('rejection_fraction', 'std'),
            ).reset_index()

            color = COLORS[ki % len(COLORS)]
            ax.errorbar(grouped['m'], grouped['rej_mean'],
                        yerr=grouped['rej_std'],
                        fmt='o-', color=color, markersize=4,
                        linewidth=1.2, capsize=3, capthick=0.8,
                        label=f'$k={k}$')

        display = DISPLAY_NAMES.get(ds_name, ds_name)
        ax.set_xlabel('$m$ (chain length)')
        ax.set_ylabel('Rejection fraction')
        ax.set_title(f'{display}')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused subplots
    for idx in range(n_datasets, nrows * ncols):
        axes[idx].set_visible(False)

    fig.suptitle('Rejection Sampling Failure Rate vs Chain Length $m$', fontsize=13, y=1.01)
    plt.tight_layout()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "rejection_rate_grid.png", dpi=300, facecolor='white')
    fig.savefig(plots_dir / "rejection_rate_grid.pdf")
    plt.close(fig)
    print(f"Saved: {plots_dir / 'rejection_rate_grid.png'}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rejection rate vs m experiment for QKMEANS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--download', type=str, nargs='*',
                        choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100'],
                        default=None, help='Download standard datasets')
    parser.add_argument('-d', '--datasets', type=str, nargs='*',
                        default=None, help='Paths to dataset files')
    parser.add_argument('--names', type=str, nargs='*',
                        default=None, help='Names for each dataset (must match number of datasets)')
    parser.add_argument('--m-values', type=int, nargs='+',
                        default=[10, 20, 50, 100, 150, 200, 300, 500],
                        help='Chain length m values to test')
    parser.add_argument('--k-values', type=int, nargs='+',
                        default=[10, 50, 100, 200, 500],
                        help='Number of centers k to test')
    parser.add_argument('--n-runs', type=int, default=3,
                        help='Number of runs per (m, k) combination')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/results/rejection_rate',
                        help='Output directory')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only regenerate plots from existing CSV')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        csv_path = output_dir / "rejection_rate_results.csv"
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found. Run experiment first.")
            return
        all_results = pd.read_csv(csv_path)
        print(f"Loaded {len(all_results)} rows from {csv_path}")
        plot_rejection_grid(all_results, output_dir)
        print("Done!")
        return

    # Collect datasets
    dataset_list = []  # list of (name, X)

    if args.download:
        for ds in args.download:
            X = download_dataset(ds, cache_dir="datasets")
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)
            X = np.ascontiguousarray(X.astype(np.float32))
            dataset_list.append((ds, X))

    if args.datasets:
        names = args.names if args.names else [Path(p).stem for p in args.datasets]
        for name, path in zip(names, args.datasets):
            X = load_dataset(path)
            if X.ndim > 2:
                X = X.reshape(X.shape[0], -1)
            X = np.ascontiguousarray(X.astype(np.float32))
            dataset_list.append((name, X))

    if not dataset_list:
        parser.error("Must specify --download or -d")

    # Run experiment for each dataset
    all_dfs = []
    for ds_name, X in dataset_list:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} | shape: {X.shape}")
        print(f"m values: {args.m_values}")
        print(f"k values: {args.k_values}")
        print(f"Runs: {args.n_runs}")
        print(f"{'='*60}")

        df = run_rejection_experiment(X, ds_name, args.m_values, args.k_values, args.n_runs)
        all_dfs.append(df)

        # Print summary for this dataset
        summary = df.groupby(['m', 'k']).agg(
            rej_mean=('rejection_fraction', 'mean'),
        ).reset_index()
        print(f"\n{ds_name} summary (mean rejection fraction):")
        pivot = summary.pivot(index='m', columns='k', values='rej_mean')
        print(pivot.to_string())

    # Combine and save
    all_results = pd.concat(all_dfs, ignore_index=True)
    csv_path = output_dir / "rejection_rate_results.csv"
    all_results.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Plot
    print("\nGenerating plots...")
    plot_rejection_grid(all_results, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
