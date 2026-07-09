#!/usr/bin/env python3
"""Plot epsilon_beta vs MLE intrinsic dimension."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# Dataset display names (prettier labels)
DISPLAY_NAMES = {
    "mnist": "MNIST",
    "mnist_test": "MNIST (test)",
    "fmnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "mnist_clip": "MNIST-CLIP",
    "fmnist_clip": "FMNIST-CLIP",
    "cifar10_clip": "CIFAR-10-CLIP",
    "cifar100_clip": "CIFAR-100-CLIP",
    "reddit": "Reddit",
    "har": "HAR",
    "susy": "SUSY",
}


def main():
    results_dir = Path(__file__).parent / "results" / "scaling_laws"
    intrinsic_dim_dir = Path(__file__).parent / "results" / "intrinsic_dim"

    # Load data
    summaries = pd.read_csv(results_dir / 'all_summaries.csv')
    mle_id = pd.read_csv(intrinsic_dim_dir / 'mle_intrinsic_dim.csv')

    # Filter out outliers (e.g., reddit k=5 has id_mean ~400k which is clearly erroneous)
    mle_id_filtered = mle_id[mle_id['id_mean'] < 1000].copy()

    # Average MLE intrinsic dimension across all k values for each dataset
    mle_id_avg = mle_id_filtered.groupby('dataset').agg({
        'id_mean': 'mean',
        'id_std': lambda x: np.sqrt(np.sum(x**2)) / len(x),  # propagate uncertainty
        'n_points': 'first',
        'ambient_dim': 'first',
    }).reset_index()

    # Compute 95% CI
    n_k_per_dataset = mle_id_filtered.groupby('dataset').size()
    mle_id_avg = mle_id_avg.merge(n_k_per_dataset.rename('n_k'), on='dataset')
    mle_id_avg['id_ci95'] = 1.96 * mle_id_avg['id_std'] / np.sqrt(mle_id_avg['n_k'])

    # Merge on dataset name
    df = summaries.merge(mle_id_avg[['dataset', 'id_mean', 'id_std', 'id_ci95']], on='dataset', how='inner')

    # Add display names
    df['display_name'] = df['dataset'].map(lambda x: DISPLAY_NAMES.get(x, x))

    # Sort by epsilon_beta for consistent coloring
    df = df.sort_values('epsilon_beta', ascending=False).reset_index(drop=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    # Viridis colormap
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(df)))

    # Plot each point with 95% CI
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.errorbar(
            row['id_mean'], row['epsilon_beta'],
            xerr=row['id_ci95'],
            fmt='o', markersize=8,
            capsize=3, capthick=1.5,
            color=colors[i], ecolor=colors[i],
            markeredgecolor='white', markeredgewidth=0.8,
            label=row['display_name'],
        )
        

    # Add dataset labels with slight offset
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.annotate(
            row['display_name'],
            (row['id_mean'], row['epsilon_beta']),
            textcoords="offset points", xytext=(6, 4),
            fontsize=9, color=colors[i], fontweight='medium',
        )

    ax.set_xlabel(r'$\hat{d}_{\operatorname{MLE}}$ (averaged over $k$)', fontsize=13)
    ax.set_ylabel(r'$\hat{\varepsilon}$', fontsize=13)
    ax.set_title(r'$\hat{d}_{\operatorname{MLE}}$ vs $\hat{\varepsilon}$', fontsize=14, fontweight='medium')

    # Compute correlations
    pearson_r, pearson_p = stats.pearsonr(df['id_mean'], df['epsilon_beta'])
    spearman_r, spearman_p = stats.spearmanr(df['id_mean'], df['epsilon_beta'])

    # Add correlation text box
    corr_text = (
        f"Pearson $r$ = {pearson_r:.3f} (p = {pearson_p:.3f})\n"
        f"Spearman $\\rho$ = {spearman_r:.3f} (p = {spearman_p:.3f})"
    )
    ax.text(
        0.97, 0.97, corr_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.8', alpha=0.9),
    )

    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save figure
    output_path = results_dir / "eps_beta_vs_mle_id.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to {output_path}")

    # Also save PDF for publication
    pdf_path = results_dir / "eps_beta_vs_mle_id.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved PDF to {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()
