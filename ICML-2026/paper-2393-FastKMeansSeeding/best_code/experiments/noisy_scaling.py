#!/usr/bin/env python3
"""
Noisy Manifold Scaling Law Experiment

Adds Gaussian noise of varying magnitudes to MNIST and measures how the
beta scaling exponent (epsilon) changes. This tests the robustness of the
quantization-theoretic scaling law beta_k ~ k^epsilon under noise.

Noise model: X_noisy = X + nsr * std(X) * N(0, 1)
where nsr is the noise-to-signal ratio.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add parent directory so we can import from scaling.py
sys.path.insert(0, str(Path(__file__).parent))

from scaling import (
    run_kmeans,
    compute_opt1,
    compute_beta,
    compute_eta,
    fit_power_law,
    download_dataset,
    load_dataset,
)

# Publication-quality settings (matching plot_beta_scaling_grid.py)
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
# Noise Injection
# ============================================================

def add_gaussian_noise(X: np.ndarray, nsr: float, seed: int = 42) -> np.ndarray:
    """
    Add Gaussian noise to dataset X.

    Args:
        X: (n, d) data array
        nsr: noise-to-signal ratio. Noise std = nsr * std(X).
        seed: random seed for reproducibility

    Returns:
        X_noisy: (n, d) noisy data (float32, C-contiguous)
    """
    if nsr == 0.0:
        return X.copy()

    rng = np.random.RandomState(seed)
    sigma_signal = X.std()
    sigma_noise = nsr * sigma_signal
    noise = rng.randn(*X.shape).astype(np.float32) * sigma_noise
    X_noisy = X + noise
    return np.ascontiguousarray(X_noisy)


# ============================================================
# Per-NSR Experiment
# ============================================================

def run_scaling_for_noise_level(
    X: np.ndarray,
    nsr: float,
    k_values: list,
    n_runs: int,
    n_iter: int,
    noise_seed: int = 42,
) -> pd.DataFrame:
    """
    Run beta scaling experiment on X with added noise at given NSR.

    Returns DataFrame with columns: k, beta_mean, beta_std, eta_mean, eta_std, n_successful_runs
    """
    X_noisy = add_gaussian_noise(X, nsr, seed=noise_seed)
    opt1 = compute_opt1(X_noisy)

    results = []
    for k in k_values:
        if k > len(X_noisy):
            continue

        betas = []
        etas = []
        for run in range(n_runs):
            seed = run * 1000 + k
            try:
                centers, labels, cost_k = run_kmeans(X_noisy, k, n_iter, seed)
                betas.append(compute_beta(opt1, cost_k))
                etas.append(compute_eta(centers))
            except Exception as e:
                print(f"  Error at nsr={nsr:.3f}, k={k}, run={run}: {e}")
                continue

        if betas:
            results.append({
                'k': k,
                'beta_mean': np.mean(betas),
                'beta_std': np.std(betas),
                'eta_mean': np.mean(etas),
                'eta_std': np.std(etas),
                'n_successful_runs': len(betas),
            })

    return pd.DataFrame(sorted(results, key=lambda x: x['k']))


# ============================================================
# Full Noise Sweep
# ============================================================

def run_noise_experiment(
    X: np.ndarray,
    nsr_values: np.ndarray,
    k_values: list,
    n_runs: int,
    n_iter: int,
    output_dir: Path,
    dataset_name: str = "dataset",
) -> tuple:
    """
    Run scaling law experiment across all noise levels.

    Returns:
        all_results: dict mapping nsr -> DataFrame of per-k results
        summary_df: DataFrame with one row per noise level (epsilon, R^2, etc.)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summaries = []

    for i, nsr in enumerate(tqdm(nsr_values, desc="Noise levels")):
        print(f"\n{'='*60}")
        print(f"NSR = {nsr:.4f} ({i+1}/{len(nsr_values)})")
        print(f"{'='*60}")

        df = run_scaling_for_noise_level(X, nsr, k_values, n_runs, n_iter)

        if len(df) < 2:
            print(f"  Skipping nsr={nsr:.4f}: too few valid k values")
            continue

        # Save per-NSR CSV
        csv_path = output_dir / f"{dataset_name}_nsr{nsr:.4f}_results.csv"
        df.to_csv(csv_path, index=False)

        # Fit power law
        k_arr = df['k'].values
        beta_arr = df['beta_mean'].values
        slope, intercept, r2 = fit_power_law(k_arr, beta_arr)

        sigma_noise = nsr * X.std()

        summaries.append({
            'nsr': nsr,
            'sigma_noise': sigma_noise,
            'epsilon_beta': slope,
            'r2_beta': r2,
            'estimated_intrinsic_dim': 2 / slope if slope > 0 else float('inf'),
        })

        all_results[nsr] = df

        print(f"  epsilon = {slope:.4f}, R^2 = {r2:.4f}, est. d = {2/slope:.1f}")

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "noise_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")

    return all_results, summary_df


# ============================================================
# Plotting
# ============================================================

def plot_beta_overlay(all_results: dict, summary_df: pd.DataFrame, output_dir: Path, dataset_name: str = "MNIST"):
    """
    Single log-log plot with all NSR curves overlaid, colored by NSR.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    nsr_values = sorted(all_results.keys())
    norm = mcolors.Normalize(vmin=min(nsr_values), vmax=max(nsr_values))
    cmap = cm.plasma

    for nsr in nsr_values:
        df = all_results[nsr]
        k_vals = df['k'].values
        beta_vals = df['beta_mean'].values
        color = cmap(norm(nsr))

        ax.plot(k_vals, beta_vals, 'o-', color=color, markersize=3,
                linewidth=1.0, markeredgecolor='none')

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Noise-to-Signal Ratio', fontsize=10)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\log\, k$')
    ax.set_ylabel(r'$\log\, \beta_k$')
    ax.set_title(rf'$\beta$ Scaling Under Gaussian Noise ({dataset_name})')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "beta_scaling_noisy_overlay.png", dpi=300, facecolor='white')
    fig.savefig(plots_dir / "beta_scaling_noisy_overlay.pdf")
    plt.close(fig)
    print(f"Saved: {plots_dir / 'beta_scaling_noisy_overlay.png'}")


def plot_epsilon_vs_noise(summary_df: pd.DataFrame, output_dir: Path, dataset_name: str = "MNIST", ambient_dim: int = None):
    """
    Plot fitted epsilon (beta slope) as a function of noise-to-signal ratio.
    """
    fig, ax1 = plt.subplots(figsize=(6, 4.5))

    nsr = summary_df['nsr'].values
    eps = summary_df['epsilon_beta'].values
    r2 = summary_df['r2_beta'].values

    ax1.scatter(nsr, eps, c='#0072B2',
                s=40, edgecolors='black', linewidths=0.5, zorder=3)
    ax1.plot(nsr, eps, '-', color='#0072B2', linewidth=1.0, alpha=0.6, zorder=2)

    # Reference line at clean-data epsilon
    eps_clean = eps[0] if len(eps) > 0 else None
    if eps_clean is not None:
        ax1.axhline(y=eps_clean, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax1.annotate(f'clean: $\\varepsilon={eps_clean:.3f}$',
                     xy=(nsr[-1] * 0.6, eps_clean),
                     fontsize=8, color='gray', va='bottom')

    # Reference line at 2/D (ambient dimension limit)
    if ambient_dim is not None and ambient_dim > 0:
        eps_ambient = 2.0 / ambient_dim
        ax1.axhline(y=eps_ambient, color='#D55E00', linestyle='--', linewidth=0.8, alpha=0.7)
        ax1.annotate(f'$2/D = 2/{ambient_dim} = {eps_ambient:.4f}$',
                     xy=(nsr[-1] * 0.3, eps_ambient),
                     fontsize=8, color='#D55E00', va='bottom')

    ax1.set_xlabel('Noise-to-Signal Ratio')
    ax1.set_ylabel(r'$\varepsilon$ (scaling exponent)')
    ax1.set_title(rf'Scaling Exponent $\varepsilon$ vs Noise Level ({dataset_name})')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    plt.tight_layout()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "epsilon_vs_noise.png", dpi=300, facecolor='white')
    fig.savefig(plots_dir / "epsilon_vs_noise.pdf")
    plt.close(fig)
    print(f"Saved: {plots_dir / 'epsilon_vs_noise.png'}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Noisy manifold scaling law experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='Path to dataset file (.npy, .npz, or .txt)')
    parser.add_argument('--download', type=str,
                        choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100'],
                        default=None, help='Download a standard dataset')
    parser.add_argument('--name', type=str, default=None,
                        help='Dataset name (for filenames and titles). Inferred from path if not provided.')
    parser.add_argument('--nsr-min', type=float, default=0.0,
                        help='Minimum noise-to-signal ratio')
    parser.add_argument('--nsr-max', type=float, default=2.0,
                        help='Maximum noise-to-signal ratio')
    parser.add_argument('--nsr-steps', type=int, default=20,
                        help='Number of NSR values (linearly spaced)')
    parser.add_argument('--k-values', type=int, nargs='+',
                        default=[5, 10, 50, 100, 250, 500, 750, 1000],
                        help='List of k values to test')
    parser.add_argument('--n-runs', type=int, default=5,
                        help='Number of k-means runs per k value')
    parser.add_argument('--n-iter', type=int, default=100,
                        help='Number of k-means iterations')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/results/noisy_scaling',
                        help='Output directory for results')
    parser.add_argument('--ambient-dim', type=int, default=None,
                        help='Ambient dimension D (for 2/D reference line). Inferred from data if not provided.')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip experiment, only regenerate plots from existing CSVs')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Determine dataset name
    if args.name:
        dataset_name = args.name
    elif args.download:
        dataset_name = args.download
    elif args.dataset:
        dataset_name = Path(args.dataset).stem
    else:
        dataset_name = "dataset"

    # Display name for plot titles
    DISPLAY_NAMES = {
        "mnist": "MNIST", "mnist_clip": "MNIST-CLIP",
        "fmnist": "Fashion-MNIST", "fmnist_clip": "FMNIST-CLIP",
        "cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
        "cifar10_clip": "CIFAR10-CLIP", "cifar100_clip": "CIFAR100-CLIP",
        "reddit": "Reddit", "har": "HAR", "susy": "SUSY",
        "stackexchange": "StackExchange", "fashion_mnist": "Fashion-MNIST",
    }
    display_name = DISPLAY_NAMES.get(dataset_name, dataset_name)

    # NSR values
    nsr_values = np.linspace(args.nsr_min, args.nsr_max, args.nsr_steps)

    if args.plot_only:
        # Load existing results
        summary_path = output_dir / "noise_summary.csv"
        if not summary_path.exists():
            print(f"ERROR: {summary_path} not found. Run experiment first.")
            return

        summary_df = pd.read_csv(summary_path)

        all_results = {}
        for nsr in nsr_values:
            csv_path = output_dir / f"{dataset_name}_nsr{nsr:.4f}_results.csv"
            if csv_path.exists():
                all_results[nsr] = pd.read_csv(csv_path)

        print(f"Loaded {len(all_results)} noise level results")

        ambient_dim = args.ambient_dim

    else:
        # Load dataset
        if args.download:
            X = download_dataset(args.download, cache_dir="datasets")
        elif args.dataset:
            X = load_dataset(args.dataset)
        else:
            parser.error("Must specify --dataset or --download")

        # Flatten if needed (e.g., images stored as (n, h, w) or (n, h, w, c))
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        X = np.ascontiguousarray(X.astype(np.float32))

        print(f"Dataset: {display_name}")
        print(f"Dataset shape: {X.shape}")
        print(f"Signal std: {X.std():.4f}")
        print(f"NSR range: [{args.nsr_min}, {args.nsr_max}] in {args.nsr_steps} steps")
        print(f"Noise std range: [0, {args.nsr_max * X.std():.4f}]")
        print(f"k values: {args.k_values}")
        print(f"Runs per k: {args.n_runs}")

        ambient_dim = args.ambient_dim if args.ambient_dim else X.shape[1]

        all_results, summary_df = run_noise_experiment(
            X, nsr_values, args.k_values, args.n_runs, args.n_iter, output_dir,
            dataset_name=dataset_name,
        )

    # Print summary
    print("\n" + "=" * 70)
    print("NOISE EXPERIMENT SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Generate plots
    print("\nGenerating plots...")
    if all_results:
        plot_beta_overlay(all_results, summary_df, output_dir, dataset_name=display_name)
    plot_epsilon_vs_noise(summary_df, output_dir, dataset_name=display_name, ambient_dim=ambient_dim)

    print("\nDone!")


if __name__ == "__main__":
    main()
