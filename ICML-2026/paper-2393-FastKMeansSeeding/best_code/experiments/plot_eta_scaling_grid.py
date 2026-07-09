#!/usr/bin/env python3
"""
Plot eta scaling for each dataset in a 4x3 grid.
Publication-quality figures for top-tier ML venues.

Creates log-log plots of k vs eta with:
- Best fit line
- R² score
- Estimated slope (epsilon/2)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

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

# Results directory
RESULTS_DIR = Path(__file__).parent / "results" / "scaling_laws"

# Dataset display names
DATASET_NAMES = {
    "mnist": "MNIST",
    "fmnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "mnist_clip": "MNIST-CLIP",
    "fmnist_clip": "FMNIST-CLIP",
    "cifar10_clip": "CIFAR10-CLIP",
    "cifar100_clip": "CIFAR100-CLIP",
    "reddit": "Reddit",
    "har": "HAR",
    "susy": "SUSY",
    "stackexchange": "StackExchange",
}

# Dataset order for plotting (4 rows x 3 columns = 12 subplots)
DATASET_ORDER = [
    "mnist", "fmnist", "cifar10",
    "cifar100", "mnist_clip", "fmnist_clip",
    "cifar10_clip", "cifar100_clip", "reddit",
    "har", "susy", "stackexchange"
]

# Colorblind-friendly colors
COLOR_DATA = '#009E73'  # Bluish green
COLOR_FIT = '#CC79A7'   # Reddish purple


def load_dataset_results(dataset: str) -> pd.DataFrame:
    """Load individual dataset results CSV."""
    results_path = RESULTS_DIR / f"{dataset}_results.csv"
    if not results_path.exists():
        return None
    return pd.read_csv(results_path)


def compute_scaling(k_values: np.ndarray, values: np.ndarray):
    """Compute log-log linear regression for scaling law."""
    log_k = np.log(k_values)
    log_val = np.log(values)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_val)
    r_squared = r_value ** 2

    return slope, intercept, r_squared


def plot_eta_scaling():
    """Create 4x3 subplot grid for eta scaling."""
    fig, axes = plt.subplots(4, 3, figsize=(7, 8))
    axes = axes.flatten()

    for i, dataset in enumerate(DATASET_ORDER):
        ax = axes[i]
        df = load_dataset_results(dataset)

        if df is None:
            ax.text(0.5, 0.5, f"No data",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            ax.set_title(DATASET_NAMES.get(dataset, dataset), fontweight='medium')
            continue

        k_values = df["k"].values
        eta_mean = df["eta_mean"].values
        eta_std = df["eta_std"].values if "eta_std" in df.columns else np.zeros_like(eta_mean)

        # Compute scaling law
        slope, intercept, r_squared = compute_scaling(k_values, eta_mean)

        # Generate fit line
        k_fit = np.logspace(np.log10(k_values.min()), np.log10(k_values.max()), 100)
        eta_fit = np.exp(intercept) * k_fit ** slope

        # Plot data points with error bars
        ax.errorbar(k_values, eta_mean, yerr=eta_std,
                    fmt='o', color=COLOR_DATA, markersize=4,
                    capsize=2, capthick=0.8, elinewidth=0.8,
                    markeredgecolor='white', markeredgewidth=0.3,
                    label='Data', zorder=3)

        # Plot fit line
        ax.plot(k_fit, eta_fit, '--', color=COLOR_FIT, linewidth=1.2,
                label=r'$\eta \sim k^{' + f'{slope:.2f}' + r'}$', zorder=2)

        # Set scales
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Labels and title
        display_name = DATASET_NAMES.get(dataset, dataset)
        ax.set_title(display_name, fontweight='medium')

        # Only add axis labels to edge subplots
        if i >= 9:  # Bottom row
            ax.set_xlabel(r"$k$")
        if i % 3 == 0:  # Left column
            ax.set_ylabel(r"$\eta$")

        # Compute estimated intrinsic dimension from eta slope
        # eta ~ k^(epsilon/2), so d ≈ 1/slope (since epsilon = 2/d implies epsilon/2 = 1/d)
        est_d = 1 / slope if slope > 0 else float('inf')

        # Add annotation box with stats
        textstr = r'$\varepsilon/2=' + f'{slope:.3f}$\n' + r'$R^2=' + f'{r_squared:.3f}$'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='none')
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', bbox=props)

        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=0.5)

    # Save
    output_dir = RESULTS_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "eta_scaling_grid.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "eta_scaling_grid.pdf", bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {output_dir / 'eta_scaling_grid.pdf'}")


def print_summary():
    """Print summary table of eta scaling results."""
    print("\n" + "=" * 70)
    print("ETA SCALING SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<20} {'ε/2 (slope)':<12} {'R²':<10} {'Est. d':<10}")
    print("-" * 70)

    for dataset in DATASET_ORDER:
        df = load_dataset_results(dataset)
        if df is None:
            print(f"{dataset:<20} {'N/A':<12} {'N/A':<10} {'N/A':<10}")
            continue

        k_values = df["k"].values
        eta_mean = df["eta_mean"].values

        slope, intercept, r_squared = compute_scaling(k_values, eta_mean)
        est_d = 1 / slope if slope > 0 else float('inf')

        display_name = DATASET_NAMES.get(dataset, dataset)
        print(f"{display_name:<20} {slope:<12.4f} {r_squared:<10.4f} {est_d:<10.1f}")

    print("=" * 70)


def main():
    print("=" * 60)
    print("Eta Scaling Analysis (4x3 Grid)")
    print("=" * 60)

    # Check if results directory exists
    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        print("Run scaling law experiments first.")
        return

    print(f"\nLooking for results in: {RESULTS_DIR}")

    # Count available datasets
    available = sum(1 for d in DATASET_ORDER if (RESULTS_DIR / f"{d}_results.csv").exists())
    print(f"Found {available}/{len(DATASET_ORDER)} dataset results")

    # Generate plot
    print("\nGenerating eta scaling plot (4x3 grid)...")
    plot_eta_scaling()

    # Print summary
    print_summary()


if __name__ == "__main__":
    main()
