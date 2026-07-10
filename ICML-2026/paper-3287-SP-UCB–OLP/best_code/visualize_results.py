#!/usr/bin/env python3
"""
Visualization Script for Alibaba Experiment Results

Generates:
- Boxplots of competitive ratio by algorithm and rho
- Line plots of performance vs rho
- Summary tables

Usage:
    python visualize_results.py --results ./results/alibaba_results_*.csv
    python visualize_results.py  # Auto-detect latest results
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import glob


# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'SP-UCB-OLP': '#2196F3',  # Blue
    'Greedy': '#FF9800',      # Orange
    'Random': '#9E9E9E',      # Gray
    'Oracle': '#4CAF50',      # Green
}


def load_results(results_path: Optional[str] = None, results_dir: str = './results') -> pd.DataFrame:
    """Load results from CSV file."""
    if results_path:
        path = Path(results_path)
    else:
        # Find latest results file
        results_dir = Path(results_dir)
        csv_files = list(results_dir.glob('alibaba_results_*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No results files found in {results_dir}")
        path = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Loading: {path}")

    df = pd.read_csv(path)
    return df


def plot_boxplots(df: pd.DataFrame, output_dir: Path):
    """Create boxplots of competitive ratio by algorithm and rho."""
    rho_values = sorted(df['rho'].unique())
    algorithms = ['Oracle', 'SP-UCB-OLP', 'Greedy', 'Random']
    algorithms = [a for a in algorithms if a in df['algorithm'].unique()]

    fig, axes = plt.subplots(1, len(rho_values), figsize=(4 * len(rho_values), 5), sharey=True)

    if len(rho_values) == 1:
        axes = [axes]

    for ax, rho in zip(axes, rho_values):
        data = []
        labels = []
        colors = []

        for alg in algorithms:
            subset = df[(df['algorithm'] == alg) & (df['rho'] == rho)]
            if len(subset) > 0:
                data.append(subset['competitive_ratio'].values)
                labels.append(alg)
                colors.append(COLORS.get(alg, '#666666'))

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f'ρ = {rho}', fontsize=12)
        ax.set_ylabel('Competitive Ratio' if ax == axes[0] else '')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Competitive Ratio by Algorithm and Budget (ρ)', fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = output_dir / 'alibaba_boxplots.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_performance_vs_rho(df: pd.DataFrame, output_dir: Path):
    """Create line plot of competitive ratio vs rho."""
    algorithms = ['Oracle', 'SP-UCB-OLP', 'Greedy', 'Random']
    algorithms = [a for a in algorithms if a in df['algorithm'].unique()]

    fig, ax = plt.subplots(figsize=(8, 5))

    for alg in algorithms:
        subset = df[df['algorithm'] == alg]
        grouped = subset.groupby('rho')['competitive_ratio'].agg(['mean', 'std'])

        ax.errorbar(
            grouped.index,
            grouped['mean'],
            yerr=grouped['std'],
            label=alg,
            color=COLORS.get(alg, '#666666'),
            marker='o',
            capsize=3,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel('Budget Scaling (ρ)', fontsize=12)
    ax.set_ylabel('Competitive Ratio', fontsize=12)
    ax.set_title('Performance vs Budget Scaling', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    output_path = output_dir / 'alibaba_performance_vs_rho.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_regret(df: pd.DataFrame, output_dir: Path):
    """Create boxplot of regret by algorithm."""
    algorithms = ['Oracle', 'SP-UCB-OLP', 'Greedy', 'Random']
    algorithms = [a for a in algorithms if a in df['algorithm'].unique()]

    fig, ax = plt.subplots(figsize=(8, 5))

    data = []
    labels = []
    colors = []

    for alg in algorithms:
        subset = df[df['algorithm'] == alg]
        data.append(subset['regret'].values)
        labels.append(alg)
        colors.append(COLORS.get(alg, '#666666'))

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Regret (T·V_mix - R_T)', fontsize=12)
    ax.set_title('Regret by Algorithm', fontsize=14)
    ax.tick_params(axis='x', rotation=45)

    output_path = output_dir / 'alibaba_regret.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_table(df: pd.DataFrame):
    """Print summary statistics table."""
    print("\n" + "=" * 70)
    print("SUMMARY: Mean Competitive Ratio (± std)")
    print("=" * 70)

    pivot = df.pivot_table(
        values='competitive_ratio',
        index='rho',
        columns='algorithm',
        aggfunc=['mean', 'std']
    ).round(4)

    # Reorder columns
    algorithms = ['Oracle', 'SP-UCB-OLP', 'Greedy', 'Random']
    algorithms = [a for a in algorithms if a in df['algorithm'].unique()]

    print("\nBy ρ value:")
    for rho in sorted(df['rho'].unique()):
        print(f"\nρ = {rho}:")
        for alg in algorithms:
            subset = df[(df['algorithm'] == alg) & (df['rho'] == rho)]
            if len(subset) > 0:
                mean = subset['competitive_ratio'].mean()
                std = subset['competitive_ratio'].std()
                print(f"  {alg:15s}: {mean:.4f} ± {std:.4f}")

    print("\n" + "=" * 70)
    print("Overall Mean Competitive Ratio:")
    print("=" * 70)
    for alg in algorithms:
        subset = df[df['algorithm'] == alg]
        if len(subset) > 0:
            mean = subset['competitive_ratio'].mean()
            std = subset['competitive_ratio'].std()
            print(f"  {alg:15s}: {mean:.4f} ± {std:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Alibaba experiment results')
    parser.add_argument('--results', type=str, default=None,
                       help='Path to results CSV file')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Results directory (used if --results not specified)')
    parser.add_argument('--output-dir', type=str, default='./results/figures',
                       help='Output directory for figures')

    args = parser.parse_args()

    # Load results
    try:
        df = load_results(args.results, args.results_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run experiments first with: python run_experiment.py")
        return 1

    print(f"Loaded {len(df)} results")
    print(f"Algorithms: {df['algorithm'].unique()}")
    print(f"Rho values: {df['rho'].unique()}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_boxplots(df, output_dir)
    plot_performance_vs_rho(df, output_dir)
    plot_regret(df, output_dir)

    # Print summary
    print_summary_table(df)

    print(f"\nFigures saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())
