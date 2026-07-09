#!/usr/bin/env python3
"""
Plot d_eps (2/ε from beta scaling) vs d_MLE (intrinsic dimension MLE estimate).
Publication-quality figures for top-tier ML venues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Publication-quality settings
plt.rcParams.update({
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

# Results directories
RESULTS_DIR = Path(__file__).parent / "results" / "scaling_laws"
ID_RESULTS_PATH = Path(__file__).parent / "results" / "intrinsic_dim" / "mle_intrinsic_dim.csv"

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

# Dataset order
DATASET_ORDER = [
    "mnist", "fmnist", "cifar10",
    "cifar100", "mnist_clip", "fmnist_clip",
    "cifar10_clip", "cifar100_clip", "reddit",
    "har", "susy", "stackexchange"
]


def load_scaling_results(dataset: str) -> pd.DataFrame:
    """Load individual dataset scaling results CSV."""
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


def load_mle_intrinsic_dim():
    """Load MLE intrinsic dimension estimates."""
    if not ID_RESULTS_PATH.exists():
        return {}

    df = pd.read_csv(ID_RESULTS_PATH)

    mle_estimates = {}

    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        # Use k=20 as reference
        k20 = subset[subset['k'] == 20]
        if len(k20) > 0:
            mle_estimates[dataset] = k20['id_mean'].values[0]
        else:
            mle_estimates[dataset] = subset['id_mean'].mean()

    return mle_estimates


def plot_d_eps_vs_d_mle():
    """Plot d_eps vs d_MLE with y=x reference line."""

    mle_estimates = load_mle_intrinsic_dim()

    d_eps_list = []
    d_mle_list = []
    labels = []

    for dataset in DATASET_ORDER:
        df = load_scaling_results(dataset)
        if df is None:
            continue

        mle_id = mle_estimates.get(dataset, None)
        if mle_id is None:
            continue

        k_values = df["k"].values
        beta_mean = df["beta_mean"].values

        eps_beta, _, _ = compute_scaling(k_values, beta_mean)
        d_from_eps = 2 / eps_beta if eps_beta > 0 else float('inf')

        if np.isfinite(d_from_eps):
            d_eps_list.append(d_from_eps)
            d_mle_list.append(mle_id)
            labels.append(DATASET_NAMES.get(dataset, dataset))

    d_eps = np.array(d_eps_list)
    d_mle = np.array(d_mle_list)

    # Compute correlation
    correlation, p_value = stats.pearsonr(d_eps, d_mle)

    # Linear regression
    slope, intercept, r_value, _, _ = stats.linregress(d_eps, d_mle)
    r_squared = r_value ** 2

    # Create figure
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # Plot points
    scatter = ax.scatter(d_eps, d_mle, s=60, c='#0072B2', edgecolors='white',
                         linewidths=0.5, zorder=3, alpha=0.9)

    # Add labels for each point
    for i, label in enumerate(labels):
        ax.annotate(label, (d_eps[i], d_mle[i]),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=8, alpha=0.8)

    # Plot y=x reference line
    max_val = max(max(d_eps), max(d_mle)) * 1.1
    min_val = min(min(d_eps), min(d_mle)) * 0.9
    ax.plot([min_val, max_val], [min_val, max_val], '--', color='#999999', linewidth=1.2,
            label=r'$y = x$', alpha=0.7, zorder=1)

    # Plot regression line
    x_fit = np.linspace(min(d_eps) * 0.9, max(d_eps) * 1.1, 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, '-', color='#D55E00', linewidth=1.5,
            label=f'Linear fit ($R^2 = {r_squared:.2f}$)', zorder=2)

    # Labels
    ax.set_xlabel(r'$\hat{d}_{\varepsilon} = 2/\varepsilon$ (from $\beta$ scaling)')
    ax.set_ylabel(r'$\hat{d}_{\mathrm{MLE}}$ (MLE estimate)')

    # Add stats box
    textstr = f'Pearson $r = {correlation:.2f}$\n$R^2 = {r_squared:.2f}$'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='none')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    ax.legend(loc='lower right', framealpha=0.95, edgecolor='none')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    plt.tight_layout(pad=0.5)

    # Save
    output_dir = RESULTS_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "d_eps_vs_d_mle.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "d_eps_vs_d_mle.pdf", bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {output_dir / 'd_eps_vs_d_mle.pdf'}")

    # Print summary
    print(f"\nCorrelation: r = {correlation:.4f} (p = {p_value:.2e})")
    print(f"Linear fit: d_MLE = {slope:.3f} * d_eps + {intercept:.3f}")
    print(f"R² = {r_squared:.4f}")


def main():
    print("=" * 60)
    print("Plotting d_eps vs d_MLE")
    print("=" * 60)

    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        return

    if not ID_RESULTS_PATH.exists():
        print(f"ERROR: MLE intrinsic dim results not found: {ID_RESULTS_PATH}")
        return

    plot_d_eps_vs_d_mle()


if __name__ == "__main__":
    main()
