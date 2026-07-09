#!/usr/bin/env python3
"""
Plot hat{eps} vs hat{d}_eps (= 2/eps) and hat{d}_MLE from beta scaling.
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


def plot_eps_vs_d_eps():
    """Plot hat{eps} vs hat{d}_eps and hat{d}_MLE."""

    mle_estimates = load_mle_intrinsic_dim()

    eps_list = []
    d_eps_list = []
    d_mle_list = []
    labels = []
    datasets_found = []

    for dataset in DATASET_ORDER:
        df = load_scaling_results(dataset)
        if df is None:
            continue

        k_values = df["k"].values
        beta_mean = df["beta_mean"].values

        eps_beta, _, _ = compute_scaling(k_values, beta_mean)
        d_from_eps = 2 / eps_beta if eps_beta > 0 else float('inf')

        mle_id = mle_estimates.get(dataset, None)

        if np.isfinite(d_from_eps) and eps_beta > 0 and mle_id is not None:
            eps_list.append(eps_beta)
            d_eps_list.append(d_from_eps)
            d_mle_list.append(mle_id)
            labels.append(DATASET_NAMES.get(dataset, dataset))
            datasets_found.append(dataset)

    eps_arr = np.array(eps_list)
    d_eps_arr = np.array(d_eps_list)
    d_mle_arr = np.array(d_mle_list)

    # Colorblind-friendly markers for each dataset
    markers = ['o', 's', '^', 'v', 'D', 'p', 'h', '*', 'P', 'X', '<', '>']
    colors = [
        '#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9',
        '#F0E442', '#000000', '#0072B2', '#D55E00', '#009E73', '#E69F00'
    ]

    # Create figure with legend outside
    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Plot theoretical curve d = 2/eps first (background)
    eps_theory = np.linspace(0.04, 0.30, 100)
    d_theory = 2 / eps_theory
    ax.plot(eps_theory, d_theory, '--', color='#999999', linewidth=1.2,
            label=r'$d = 2/\varepsilon$', zorder=1, alpha=0.7)

    # Draw lines connecting d_eps and d_MLE for same dataset
    for i in range(len(eps_arr)):
        ax.plot([eps_arr[i], eps_arr[i]], [d_eps_arr[i], d_mle_arr[i]],
                '-', color='#cccccc', alpha=0.5, linewidth=0.8, zorder=1)

    # Plot each dataset with unique marker
    for i, label in enumerate(labels):
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]

        # d_eps point (filled)
        ax.scatter(eps_arr[i], d_eps_arr[i], s=50, c=color, edgecolors='white',
                   linewidths=0.5, zorder=3, alpha=0.9, marker=marker,
                   label=label)

        # d_MLE point (open marker)
        ax.scatter(eps_arr[i], d_mle_arr[i], s=50, facecolors='none', edgecolors=color,
                   linewidths=1.2, zorder=3, alpha=0.9, marker=marker)

    # Labels
    ax.set_xlabel(r'$\hat{\varepsilon}$ (scaling exponent)')
    ax.set_ylabel(r'Intrinsic dimension estimate $\hat{d}$')

    # Create custom legend entries for marker types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label=r'$\hat{d}_{\varepsilon}$ (filled)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='gray', markersize=8, markeredgewidth=1.2, label=r'$\hat{d}_{\mathrm{MLE}}$ (open)'),
        Line2D([0], [0], linestyle='--', color='#999999', label=r'$d = 2/\varepsilon$'),
    ]

    # Legend outside on right
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, edgecolor='none', fontsize=9)

    # Add dataset legend on right side
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.95,
              edgecolor='none', fontsize=8, ncol=1)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=0.5)

    # Save
    output_dir = RESULTS_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "eps_vs_d_eps.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_dir / "eps_vs_d_eps.pdf", bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {output_dir / 'eps_vs_d_eps.pdf'}")

    # Print summary
    print(f"\nDataset summary:")
    print(f"{'Dataset':<20} {'ε':<10} {'d_ε':<10} {'d_MLE':<10}")
    print("-" * 50)
    for i, label in enumerate(labels):
        print(f"{label:<20} {eps_arr[i]:<10.4f} {d_eps_arr[i]:<10.2f} {d_mle_arr[i]:<10.2f}")


def main():
    print("=" * 60)
    print("Plotting hat{eps} vs hat{d}_eps")
    print("=" * 60)

    if not RESULTS_DIR.exists():
        print(f"ERROR: Results directory not found: {RESULTS_DIR}")
        return

    plot_eps_vs_d_eps()


if __name__ == "__main__":
    main()
