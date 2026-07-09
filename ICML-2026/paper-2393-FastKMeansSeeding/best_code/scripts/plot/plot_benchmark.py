#!/usr/bin/env python3
"""
Plot benchmark results from CSV files.

Usage:
    python scripts/plot/plot_benchmark.py <results_csv> [options]
    python scripts/plot/plot_benchmark.py results/comparison_mnist.csv --output figures/mnist.pdf
    python scripts/plot/plot_benchmark.py results/*.csv --combine

Examples:
    python scripts/plot/plot_benchmark.py results/comparison_mnist.csv
    python scripts/plot/plot_benchmark.py results/comparison_mnist.csv --type cost-vs-time
    python scripts/plot/plot_benchmark.py results/comparison_mnist.csv --type runtime --output fig.pdf
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Publication-quality settings
plt.rcParams.update({
    'text.usetex': False,
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
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

# Algorithm display names
DISPLAY_NAMES = {
    'kmeanspp': r'$k$-means++',
    'afkmc2': r'AFKMC$^2$',
    'prone': r'PRONE',
    'pronecoreset': r'PRONECoreset',
    'fastcoreset': r'FastCoreset',
    'rejectionlsh': r'RejectionLSH',
    'qkmeans': r'QKMEANS',
    'qkmeans_anns': r'QKMEANS-ANNS',
}

# Colorblind-friendly palette
COLORS = {
    'kmeanspp': '#000000',      # Black (baseline)
    'afkmc2': '#D55E00',        # Vermillion
    'prone': '#56B4E9',         # Sky blue
    'pronecoreset': '#E69F00',  # Orange
    'fastcoreset': '#009E73',   # Bluish green
    'rejectionlsh': '#CC79A7',  # Reddish purple
    'qkmeans': '#0072B2',       # Blue (our method)
    'qkmeans_anns': '#0072B2',
}

MARKERS = {
    'kmeanspp': 'o',
    'afkmc2': '^',
    'prone': 's',
    'pronecoreset': 'v',
    'fastcoreset': 'D',
    'rejectionlsh': 'p',
    'qkmeans': 's',
    'qkmeans_anns': 's',
}

LINESTYLES = {
    'kmeanspp': '--',
    'afkmc2': '-',
    'prone': ':',
    'pronecoreset': '--',
    'fastcoreset': '--',
    'rejectionlsh': ':',
    'qkmeans': '-',
    'qkmeans_anns': '-',
}


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess results CSV."""
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Handle different column naming conventions
    if 'seeding_time_ms' in df.columns:
        df['time_ms'] = df['seeding_time_ms']
    if 'seeding_cost' in df.columns:
        df['cost'] = df['seeding_cost']

    return df


def plot_runtime_vs_k(df: pd.DataFrame, ax: plt.Axes, title: str = None):
    """Plot runtime vs k for all algorithms."""
    methods = df['method'].unique()

    for method in methods:
        method_df = df[df['method'] == method].sort_values('k')
        ax.plot(
            method_df['k'], method_df['time_ms'],
            color=COLORS.get(method, '#888888'),
            marker=MARKERS.get(method, 'o'),
            linestyle=LINESTYLES.get(method, '-'),
            markeredgecolor='white',
            markeredgewidth=0.5,
            label=DISPLAY_NAMES.get(method, method)
        )

    ax.set_xlabel(r'Number of Centers $(k)$')
    ax.set_ylabel(r'Seeding Time (ms)')
    ax.set_yscale('log')
    ax.legend(loc='best', framealpha=0.95, edgecolor='none')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title:
        ax.set_title(title)


def plot_cost_vs_k(df: pd.DataFrame, ax: plt.Axes, title: str = None):
    """Plot clustering cost vs k for all algorithms."""
    methods = df['method'].unique()

    # Find cost scale
    max_cost = df['cost'].max()
    if max_cost > 1e9:
        scale = 1e9
        scale_label = r'$\times 10^9$'
    elif max_cost > 1e6:
        scale = 1e6
        scale_label = r'$\times 10^6$'
    else:
        scale = 1
        scale_label = ''

    for method in methods:
        method_df = df[df['method'] == method].sort_values('k')
        ax.plot(
            method_df['k'], method_df['cost'] / scale,
            color=COLORS.get(method, '#888888'),
            marker=MARKERS.get(method, 'o'),
            linestyle=LINESTYLES.get(method, '-'),
            markeredgecolor='white',
            markeredgewidth=0.5,
            label=DISPLAY_NAMES.get(method, method)
        )

    ax.set_xlabel(r'Number of Centers $(k)$')
    ylabel = f'Seeding Cost {scale_label}' if scale_label else 'Seeding Cost'
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', framealpha=0.95, edgecolor='none')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title:
        ax.set_title(title)


def plot_cost_vs_time(df: pd.DataFrame, ax: plt.Axes, title: str = None):
    """Plot cost vs time scatter (Pareto-style)."""
    methods = df['method'].unique()

    # Find cost scale
    max_cost = df['cost'].max()
    scale = 1e9 if max_cost > 1e9 else (1e6 if max_cost > 1e6 else 1)

    for method in methods:
        method_df = df[df['method'] == method].sort_values('k')
        ax.scatter(
            method_df['time_ms'], method_df['cost'] / scale,
            color=COLORS.get(method, '#888888'),
            marker=MARKERS.get(method, 'o'),
            s=50, alpha=0.9,
            edgecolors='white', linewidths=0.5,
            label=DISPLAY_NAMES.get(method, method)
        )
        ax.plot(
            method_df['time_ms'], method_df['cost'] / scale,
            color=COLORS.get(method, '#888888'),
            linestyle=LINESTYLES.get(method, '-'),
            alpha=0.5
        )

    ax.set_xlabel(r'Seeding Time (ms)')
    ax.set_ylabel(r'Seeding Cost $(\times 10^9)$' if scale == 1e9 else 'Seeding Cost')
    ax.set_xscale('log')
    ax.legend(loc='best', framealpha=0.95, edgecolor='none')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title:
        ax.set_title(title)


def plot_combined(df: pd.DataFrame, output_path: Path, dataset_name: str = None):
    """Create combined figure with runtime and cost plots."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    title_prefix = f"{dataset_name}: " if dataset_name else ""

    plot_runtime_vs_k(df, axes[0], title=f"{title_prefix}Runtime")
    plot_cost_vs_k(df, axes[1], title=f"{title_prefix}Cost")

    plt.tight_layout(pad=0.5)

    # Save in multiple formats
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if output_path.suffix == '.png':
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')

    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv_files", nargs="+", help="CSV result files to plot")
    parser.add_argument("--type", choices=["runtime", "cost", "cost-vs-time", "combined"],
                        default="combined", help="Type of plot (default: combined)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (default: auto-generated)")
    parser.add_argument("--title", type=str, default=None,
                        help="Plot title")

    args = parser.parse_args()

    # Process each CSV file
    for csv_file in args.csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"ERROR: File not found: {csv_path}")
            continue

        print(f"Processing: {csv_path}")
        df = load_results(csv_path)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_dir = csv_path.parent.parent / "figures"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{csv_path.stem}.png"

        # Get dataset name from CSV
        dataset_name = df['dataset'].iloc[0] if 'dataset' in df.columns else csv_path.stem

        # Create plot
        if args.type == "combined":
            plot_combined(df, output_path, dataset_name)
        else:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))

            if args.type == "runtime":
                plot_runtime_vs_k(df, ax, title=args.title or dataset_name)
            elif args.type == "cost":
                plot_cost_vs_k(df, ax, title=args.title or dataset_name)
            elif args.type == "cost-vs-time":
                plot_cost_vs_time(df, ax, title=args.title or dataset_name)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close()


if __name__ == "__main__":
    main()
