#!/usr/bin/env python3
"""
Plot benchmark results for all datasets: Seeding Cost vs Runtime
Reads data from CSV files in results/benchmark/
Publication-quality figures for top-tier ML venues
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import glob

# Publication-quality settings (using mathtext, no LaTeX install needed)
plt.rcParams.update({
    'text.usetex': False,
    'mathtext.fontset': 'cm',  # Computer Modern (LaTeX-like)
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

# Display names for algorithms (mathtext compatible)
ALGO_DISPLAY = {
    'qkmeans': r'QKMeans',
    'afkmc2': r'AFKMC$^2$',
    'fastcoreset': r'FastKMeans',
    'pronecoreset': r'PRONECoreset',
    'rejectionlsh': r'RejectionSampling',
}

# Publication-quality color palette (colorblind-friendly)
ALGO_STYLES = {
    'qkmeans': {'color': '#0072B2', 'marker': 's', 'linestyle': '-'},      # Blue
    'afkmc2': {'color': '#D55E00', 'marker': '^', 'linestyle': '-'},       # Vermillion
    'fastcoreset': {'color': '#009E73', 'marker': 'D', 'linestyle': '--'}, # Bluish green
    'pronecoreset': {'color': '#E69F00', 'marker': 'v', 'linestyle': '--'},# Orange
    'rejectionlsh': {'color': '#CC79A7', 'marker': 'o', 'linestyle': ':'},  # Reddish purple
}

# Dataset display names and info
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
                    # Extract k, cost, and time
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
        return 1e12, r'$(\times 10^{12})$'
    elif max_cost > 1e11:
        return 1e11, r'$(\times 10^{11})$'
    elif max_cost > 1e10:
        return 1e10, r'$(\times 10^{10})$'
    elif max_cost > 1e6:
        return 1e6, r'$(\times 10^{6})$'
    elif max_cost > 1e3:
        return 1e3, r'$(\times 10^{3})$'
    else:
        return 1, ''


def plot_dataset_benchmark(dataset: str, data: dict, output_dir: Path):
    """Create benchmark plots for a single dataset."""
    if not data:
        print(f"  Skipping {dataset}: no data available")
        return

    info = DATASET_INFO.get(dataset, {'name': dataset, 'n': '?', 'd': '?'})

    # Get all costs to determine scale
    all_costs = []
    for algo_data in data.values():
        all_costs.extend(algo_data['cost'])
    scale, scale_label = get_cost_scale(all_costs)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # Plot 1: Runtime vs k
    ax1 = axes[0]
    for algo, vals in data.items():
        style = ALGO_STYLES[algo]
        ax1.plot(vals['k'], vals['time'],
                 color=style['color'],
                 marker=style['marker'],
                 linestyle=style['linestyle'],
                 markeredgecolor='white',
                 markeredgewidth=0.5,
                 label=ALGO_DISPLAY[algo])

    ax1.set_xlabel(r'Number of Centers $(k)$')
    ax1.set_ylabel(r'Seeding Time (ms)')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', framealpha=0.95, edgecolor='none', fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot 2: Cost vs k
    ax2 = axes[1]
    for algo, vals in data.items():
        style = ALGO_STYLES[algo]
        cost_scaled = [c / scale for c in vals['cost']]
        ax2.plot(vals['k'], cost_scaled,
                 color=style['color'],
                 marker=style['marker'],
                 linestyle=style['linestyle'],
                 markeredgecolor='white',
                 markeredgewidth=0.5,
                 label=ALGO_DISPLAY[algo])

    ax2.set_xlabel(r'Number of Centers $(k)$')
    ax2.set_ylabel(f'Seeding Cost {scale_label}')
    ax2.legend(loc='upper right', framealpha=0.95, edgecolor='none', fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle(f"{info['name']} (n={info['n']}, d={info['d']})", fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    png_path = output_dir / f"{dataset}_benchmark_plot.png"
    pdf_path = output_dir / f"{dataset}_benchmark_plot.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {png_path.name}")

    # Also create Cost vs Time scatter plot
    fig2, ax3 = plt.subplots(figsize=(4.5, 3.5))

    for algo, vals in data.items():
        style = ALGO_STYLES[algo]
        cost_scaled = [c / scale for c in vals['cost']]
        ax3.scatter(vals['time'], cost_scaled,
                    color=style['color'],
                    marker=style['marker'],
                    s=50, label=ALGO_DISPLAY[algo], alpha=0.9,
                    edgecolors='white', linewidths=0.5)
        ax3.plot(vals['time'], cost_scaled,
                 color=style['color'],
                 linestyle=style['linestyle'],
                 alpha=0.6)

        # Annotate k values for QKMEANS
        if algo == 'qkmeans':
            for i, k in enumerate(vals['k']):
                if k in [10, 1000] or (len(vals['k']) <= 3):
                    ax3.annotate(r'$k$=' + str(k), (vals['time'][i], cost_scaled[i]),
                                textcoords="offset points", xytext=(5, 5), fontsize=8,
                                color=style['color'])

    ax3.set_xlabel(r'Seeding Time (ms)')
    ax3.set_ylabel(f'Seeding Cost {scale_label}')
    ax3.set_xscale('log')
    ax3.legend(loc='upper right', framealpha=0.95, edgecolor='none', fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_title(f"{info['name']}: Cost vs Time", fontsize=11)

    plt.tight_layout()
    png_path2 = output_dir / f"{dataset}_cost_vs_time.png"
    pdf_path2 = output_dir / f"{dataset}_cost_vs_time.pdf"
    plt.savefig(png_path2, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path2, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {png_path2.name}")


def print_summary(dataset: str, data: dict):
    """Print summary statistics for a dataset."""
    if not data:
        return

    info = DATASET_INFO.get(dataset, {'name': dataset, 'n': '?', 'd': '?'})
    print(f"\n{'='*80}")
    print(f"{info['name']} Benchmark Summary (n={info['n']}, d={info['d']})")
    print('='*80)

    # Get k values from first algorithm
    k_values = list(data.values())[0]['k']

    # Header
    header = f"{'k':>6}"
    for algo in data.keys():
        header += f" | {ALGO_DISPLAY[algo]:>15}"
    print(header)
    print('-'*80)

    # Data rows
    for i, k in enumerate(k_values):
        row = f"{k:>6}"
        for algo, vals in data.items():
            if i < len(vals['time']):
                row += f" | {vals['time'][i]:>13.1f}ms"
            else:
                row += f" | {'N/A':>13}"
        print(row)

    # Speedup at largest k
    if 'qkmeans' in data:
        max_k_idx = -1
        qkmeans_time = data['qkmeans']['time'][max_k_idx]
        print(f"\nSpeedup of QKMeans over other algorithms at k={k_values[max_k_idx]}:")
        for algo, vals in data.items():
            if algo != 'qkmeans':
                speedup = vals['time'][max_k_idx] / qkmeans_time
                print(f"  vs {ALGO_DISPLAY[algo]}: {speedup:.1f}x faster")


def main():
    print("="*60)
    print("Generating Benchmark Plots for All Datasets")
    print("="*60)

    results_dir = Path("results/benchmark")
    output_dir = results_dir

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return

    # Find all datasets by looking at CSV files
    csv_files = list(results_dir.glob("*.csv"))
    datasets = set()
    for f in csv_files:
        name = f.stem
        for algo in ALGO_STYLES.keys():
            if name.startswith(f"{algo}_"):
                dataset = name[len(algo)+1:]
                datasets.add(dataset)

    datasets = sorted(datasets)
    print(f"\nFound {len(datasets)} datasets: {', '.join(datasets)}")

    for dataset in datasets:
        print(f"\n[{dataset}]")
        data = load_dataset_results(dataset, results_dir)
        if data:
            plot_dataset_benchmark(dataset, data, output_dir)
            print_summary(dataset, data)
        else:
            print(f"  No data found for {dataset}")

    print("\n" + "="*60)
    print(f"All plots saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
