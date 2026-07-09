#!/usr/bin/env python3
"""
Combined 4x3 grid plot for all datasets: Cost vs Runtime (Pareto frontier style)
Publication-quality figures for top-tier ML venues
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Publication-quality settings
plt.rcParams.update({
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
})

# Display names for algorithms
ALGO_DISPLAY = {
    'qkmeans': r'QKMeans',
    'afkmc2': r'AFKMC$^2$',
    'fastcoreset': r'FastKMeans',
    'pronecoreset': r'PRONECoreset',
    'rejectionlsh': r'RejectionSampling',
}

# Publication-quality color palette (colorblind-friendly)
ALGO_STYLES = {
    'qkmeans': {'color': '#0072B2', 'marker': 's', 'linestyle': '-'},
    'afkmc2': {'color': '#D55E00', 'marker': '^', 'linestyle': '-'},
    'fastcoreset': {'color': '#009E73', 'marker': 'D', 'linestyle': '--'},
    'pronecoreset': {'color': '#E69F00', 'marker': 'v', 'linestyle': '--'},
    'rejectionlsh': {'color': '#CC79A7', 'marker': 'o', 'linestyle': ':'},
}

# Dataset order for 4x3 grid (row-major)
DATASET_ORDER = [
    'mnist', 'fmnist', 'cifar10', 'cifar100',
    'mnist_clip', 'fmnist_clip', 'cifar10_clip', 'cifar100_clip',
    'reddit', 'stackexchange', 'har', 'susy'
]

DATASET_INFO = {
    'mnist': {'name': 'MNIST', 'n': 60000, 'd': 784},
    'fmnist': {'name': 'Fashion-MNIST', 'n': 60000, 'd': 784},
    'mnist_clip': {'name': 'MNIST-CLIP', 'n': 60000, 'd': 512},
    'fmnist_clip': {'name': 'FMNIST-CLIP', 'n': 60000, 'd': 512},
    'cifar10': {'name': 'CIFAR-10', 'n': 60000, 'd': 3072},
    'cifar10_clip': {'name': 'CIFAR10-CLIP', 'n': 60000, 'd': 512},
    'cifar100': {'name': 'CIFAR-100', 'n': 60000, 'd': 3072},
    'cifar100_clip': {'name': 'CIFAR100-CLIP', 'n': 60000, 'd': 512},
    'reddit': {'name': 'Reddit', 'n': 100000, 'd': 384},
    'stackexchange': {'name': 'StackExchange', 'n': 100000, 'd': 384},
    'har': {'name': 'HAR', 'n': 10299, 'd': 561},
    'susy': {'name': 'SUSY', 'n': 500000, 'd': 18},
}


def load_dataset_results(dataset: str, results_dir: Path) -> dict:
    """Load all algorithm results for a given dataset."""
    data = {}
    for algo in ALGO_STYLES.keys():
        csv_path = results_dir / f"{algo}_{dataset}.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            try:
                df = pd.read_csv(csv_path)
                if len(df) > 0:
                    data[algo] = {
                        'k': df['k'].tolist(),
                        'cost': df['seeding_cost'].tolist(),
                        'time': df['seeding_time_ms'].tolist(),
                    }
            except Exception as e:
                print(f"  Warning: Could not read {csv_path}: {e}")
    return data


def get_cost_scale(costs: list) -> tuple:
    """Determine appropriate scale factor for costs."""
    max_cost = max(costs)
    if max_cost > 1e12:
        return 1e12, r'$\times 10^{12}$'
    elif max_cost > 1e11:
        return 1e11, r'$\times 10^{11}$'
    elif max_cost > 1e10:
        return 1e10, r'$\times 10^{10}$'
    elif max_cost > 1e9:
        return 1e9, r'$\times 10^{9}$'
    elif max_cost > 1e8:
        return 1e8, r'$\times 10^{8}$'
    elif max_cost > 1e6:
        return 1e6, r'$\times 10^{6}$'
    elif max_cost > 1e3:
        return 1e3, r'$\times 10^{3}$'
    else:
        return 1, ''


def main():
    print("=" * 60)
    print("Generating Combined 4x3 Grid Plot")
    print("=" * 60)

    results_dir = Path("results/benchmark")
    output_dir = results_dir

    # Create 4x3 figure
    fig, axes = plt.subplots(4, 3, figsize=(10, 11))
    axes = axes.flatten()

    # Load data for all datasets
    all_data = {}
    for dataset in DATASET_ORDER:
        all_data[dataset] = load_dataset_results(dataset, results_dir)

    # Plot each dataset
    for idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[idx]
        data = all_data[dataset]
        info = DATASET_INFO.get(dataset, {'name': dataset, 'n': '?', 'd': '?'})

        if not data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{info['name']}")
            continue

        # Get cost scale
        all_costs = []
        for algo_data in data.values():
            all_costs.extend(algo_data['cost'])
        scale, scale_label = get_cost_scale(all_costs)

        # Plot each algorithm (Cost vs Time scatter with lines)
        for algo, vals in data.items():
            style = ALGO_STYLES[algo]
            cost_scaled = [c / scale for c in vals['cost']]

            # Plot line connecting points
            ax.plot(vals['time'], cost_scaled,
                    color=style['color'],
                    linestyle=style['linestyle'],
                    alpha=0.6,
                    linewidth=1.0)

            # Plot scatter points
            ax.scatter(vals['time'], cost_scaled,
                       color=style['color'],
                       marker=style['marker'],
                       s=25,
                       label=ALGO_DISPLAY[algo] if idx == 0 else None,
                       alpha=0.9,
                       edgecolors='white',
                       linewidths=0.3,
                       zorder=10)

        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Title with dataset info
        ax.set_title(f"{info['name']} (n={info['n']:,}, d={info['d']})", fontsize=9, fontweight='bold')

        # Axis labels (only on edges)
        if idx >= 9:  # Bottom row
            ax.set_xlabel('Seeding Time (ms)', fontsize=8)
        if idx % 3 == 0:  # Left column
            ax.set_ylabel(f'Seeding Cost', fontsize=8)

        # Add scale label to y-axis
        if scale_label:
            ax.annotate(scale_label, xy=(0, 1.02), xycoords='axes fraction',
                        fontsize=7, ha='left', va='bottom')

    # Create legend at the bottom
    handles = []
    labels = []
    for algo in ALGO_STYLES.keys():
        style = ALGO_STYLES[algo]
        handle = plt.Line2D([0], [0], color=style['color'], marker=style['marker'],
                            linestyle=style['linestyle'], markersize=6, markeredgecolor='white',
                            markeredgewidth=0.5, linewidth=1.2)
        handles.append(handle)
        labels.append(ALGO_DISPLAY[algo])

    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=True,
               framealpha=0.95, edgecolor='none', fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # Save
    png_path = output_dir / "all_benchmarks_grid.png"
    pdf_path = output_dir / "all_benchmarks_grid.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    # Also create a Time vs k plot grid
    fig2, axes2 = plt.subplots(4, 3, figsize=(10, 11))
    axes2 = axes2.flatten()

    for idx, dataset in enumerate(DATASET_ORDER):
        ax = axes2[idx]
        data = all_data[dataset]
        info = DATASET_INFO.get(dataset, {'name': dataset, 'n': '?', 'd': '?'})

        if not data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{info['name']}")
            continue

        # Plot each algorithm (Time vs k)
        for algo, vals in data.items():
            style = ALGO_STYLES[algo]
            ax.plot(vals['k'], vals['time'],
                    color=style['color'],
                    marker=style['marker'],
                    linestyle=style['linestyle'],
                    markeredgecolor='white',
                    markeredgewidth=0.3,
                    markersize=4,
                    label=ALGO_DISPLAY[algo] if idx == 0 else None)

        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_title(f"{info['name']} (n={info['n']:,}, d={info['d']})", fontsize=9, fontweight='bold')

        if idx >= 9:
            ax.set_xlabel('Number of Centers (k)', fontsize=8)
        if idx % 3 == 0:
            ax.set_ylabel('Seeding Time (ms)', fontsize=8)

    fig2.legend(handles, labels, loc='lower center', ncol=5, frameon=True,
                framealpha=0.95, edgecolor='none', fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    png_path2 = output_dir / "all_benchmarks_time_grid.png"
    pdf_path2 = output_dir / "all_benchmarks_time_grid.pdf"
    plt.savefig(png_path2, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path2, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path2}")
    print(f"Saved: {pdf_path2}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
