#!/usr/bin/env python3
"""Generate comparison plots from evaluation metrics.

This script reads the aggregated metrics JSON files produced by evaluate.py
and generates comparison plots.

Usage:
    python scripts/plot_evaluation_results.py --task high_dim_gaussian
    python scripts/plot_evaluation_results.py --task wind_tunnel --conditional
    python scripts/plot_evaluation_results.py --task high_dim_gaussian wind_tunnel --save_dir figures
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'fm_post_transform': '#1f77b4',  # Blue
    'rope': '#ff7f0e',               # Orange
    'mf_npe': '#2ca02c',             # Green
    'dpe': '#d62728',                # Red
    'npe': '#9467bd',                # Purple
    'fmpe': '#8c564b',               # Brown
    'fmpe_cal': '#e377c2',           # Pink
    'npe_rs': '#7f7f7f',             # Gray
}

METHOD_NAMES = {
    'fm_post_transform': 'FM Post-Transform',
    'rope': 'RoPE',
    'mf_npe': 'MF-NPE',
    'dpe': 'DPE',
    'npe': 'NPE',
    'fmpe': 'FMPE',
    'fmpe_cal': 'FMPE-Cal',
    'npe_rs': 'NPE-RS',
}

METRIC_NAMES = {
    'joint_c2st': 'Joint C2ST',
    'joint_mmd': 'Joint MMD',
    'joint_wasserstein': 'Joint Wasserstein',
    'c2st': 'C2ST',
    'mmd': 'MMD',
    'wasserstein': 'Wasserstein',
    'lpp': 'LPP',
    'acauc': 'ACAUC',
    'cond_c2st_mean': 'Conditional C2ST',
    'cond_mmd_mean': 'Conditional MMD',
    'cond_wasserstein_mean': 'Conditional Wasserstein',
}

TASK_NAMES = {
    'high_dim_gaussian': 'High-Dim Gaussian',
    'wind_tunnel': 'Wind Tunnel',
    'light_tunnel': 'Light Tunnel',
    'pendulum': 'Pendulum',
    'gaussian': 'Gaussian',
}


def load_metrics(results_dir: Path, experiment: str, task: str, conditional: bool = False) -> Optional[Dict]:
    """Load aggregated metrics from JSON file."""
    if conditional:
        metrics_path = results_dir / experiment / "simulation" / task / "evaluation" / "conditional" / "aggregated_metrics.json"
    else:
        metrics_path = results_dir / experiment / "evaluation" / task / "aggregated" / "metrics.json"

    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return None

    with open(metrics_path, 'r') as f:
        return json.load(f)


def get_all_methods_baselines(metrics: Dict) -> tuple:
    """Extract all methods and baselines from metrics."""
    methods = list(metrics.get('methods', {}).keys())
    baselines = list(metrics.get('baselines', {}).keys())
    return methods, baselines


def get_ncal_values(metrics: Dict) -> List[int]:
    """Extract all ncal values from metrics."""
    ncal_values = set()
    for method_data in metrics.get('methods', {}).values():
        ncal_values.update(int(k) for k in method_data.keys())
    for baseline_data in metrics.get('baselines', {}).values():
        ncal_values.update(int(k) for k in baseline_data.keys())
    return sorted(ncal_values)


def get_metric_values(metrics: Dict, method_or_baseline: str, ncal: int, metric_name: str, is_baseline: bool = False) -> Optional[List[float]]:
    """Get metric values for a method/baseline at a specific ncal."""
    category = 'baselines' if is_baseline else 'methods'
    try:
        data = metrics[category][method_or_baseline][str(ncal)]
        if metric_name in data:
            values = data[metric_name]
            if isinstance(values, list):
                return values
            return [values]
    except KeyError:
        pass
    return None


def plot_metric_vs_ncal(
    metrics: Dict,
    metric_name: str,
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """Plot a metric vs calibration size for all methods and baselines."""
    methods, baselines = get_all_methods_baselines(metrics)
    ncal_values = get_ncal_values(metrics)

    if not ncal_values:
        print(f"No ncal values found for metric {metric_name}")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Plot methods
    for method in methods:
        means, stds = [], []
        valid_ncals = []
        for ncal in ncal_values:
            values = get_metric_values(metrics, method, ncal, metric_name, is_baseline=False)
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values))
                valid_ncals.append(ncal)

        if means:
            color = COLORS.get(method, '#333333')
            label = METHOD_NAMES.get(method, method)
            ax.errorbar(valid_ncals, means, yerr=stds, marker='o', capsize=4,
                       color=color, label=label, linewidth=2, markersize=8)

    # Plot baselines
    for baseline in baselines:
        means, stds = [], []
        valid_ncals = []
        for ncal in ncal_values:
            values = get_metric_values(metrics, baseline, ncal, metric_name, is_baseline=True)
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values))
                valid_ncals.append(ncal)

        if means:
            color = COLORS.get(baseline, '#666666')
            label = METHOD_NAMES.get(baseline, baseline)
            ax.errorbar(valid_ncals, means, yerr=stds, marker='s', capsize=4,
                       color=color, label=label, linewidth=2, markersize=8,
                       linestyle='--')

    ax.set_xlabel('Calibration Size (ncal)', fontsize=12)
    ax.set_ylabel(METRIC_NAMES.get(metric_name, metric_name), fontsize=12)
    ax.set_xscale('log')
    ax.set_xticks(ncal_values)
    ax.set_xticklabels([str(n) for n in ncal_values])

    if title:
        ax.set_title(title, fontsize=14)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def plot_all_metrics_grid(
    metrics: Dict,
    task_name: str,
    save_dir: Path,
    conditional: bool = False,
    figsize: tuple = (15, 10),
):
    """Plot all available metrics in a grid layout."""
    methods, baselines = get_all_methods_baselines(metrics)
    ncal_values = get_ncal_values(metrics)

    if not ncal_values:
        print("No ncal values found")
        return

    # Determine which metrics are available
    available_metrics = set()
    for method_data in metrics.get('methods', {}).values():
        for ncal_data in method_data.values():
            available_metrics.update(ncal_data.keys())
    for baseline_data in metrics.get('baselines', {}).values():
        for ncal_data in baseline_data.values():
            available_metrics.update(ncal_data.keys())

    # Filter out per-obs metrics (they're lists of lists)
    available_metrics = [m for m in available_metrics if '_per_obs' not in m]
    available_metrics = sorted(available_metrics)

    if not available_metrics:
        print("No metrics available to plot")
        return

    # Create grid
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, metric_name in enumerate(available_metrics):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        # Plot methods
        for method in methods:
            means, stds = [], []
            valid_ncals = []
            for ncal in ncal_values:
                values = get_metric_values(metrics, method, ncal, metric_name, is_baseline=False)
                if values:
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                    valid_ncals.append(ncal)

            if means:
                color = COLORS.get(method, '#333333')
                label = METHOD_NAMES.get(method, method)
                ax.errorbar(valid_ncals, means, yerr=stds, marker='o', capsize=3,
                           color=color, label=label, linewidth=1.5, markersize=6)

        # Plot baselines
        for baseline in baselines:
            means, stds = [], []
            valid_ncals = []
            for ncal in ncal_values:
                values = get_metric_values(metrics, baseline, ncal, metric_name, is_baseline=True)
                if values:
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                    valid_ncals.append(ncal)

            if means:
                color = COLORS.get(baseline, '#666666')
                label = METHOD_NAMES.get(baseline, baseline)
                ax.errorbar(valid_ncals, means, yerr=stds, marker='s', capsize=3,
                           color=color, label=label, linewidth=1.5, markersize=6,
                           linestyle='--')

        ax.set_xlabel('ncal')
        ax.set_ylabel(METRIC_NAMES.get(metric_name, metric_name))
        ax.set_xscale('log')
        ax.set_xticks(ncal_values)
        ax.set_xticklabels([str(n) for n in ncal_values])
        ax.set_title(METRIC_NAMES.get(metric_name, metric_name))
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    # Add legend to first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(6, len(handles)),
               bbox_to_anchor=(0.5, 1.02), fontsize=10)

    task_display = TASK_NAMES.get(task_name, task_name)
    suffix = " (Conditional)" if conditional else ""
    fig.suptitle(f'{task_display}{suffix}', fontsize=14, y=1.08)

    plt.tight_layout()

    suffix_str = "_conditional" if conditional else ""
    save_path = save_dir / f"{task_name}_metrics_grid{suffix_str}.pdf"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")

    plt.close()


def plot_comparison_bar(
    metrics: Dict,
    ncal: int,
    metric_name: str,
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """Plot bar chart comparing all methods at a specific ncal."""
    methods, baselines = get_all_methods_baselines(metrics)

    all_names = []
    all_means = []
    all_stds = []
    all_colors = []

    # Collect method data
    for method in methods:
        values = get_metric_values(metrics, method, ncal, metric_name, is_baseline=False)
        if values:
            all_names.append(METHOD_NAMES.get(method, method))
            all_means.append(np.mean(values))
            all_stds.append(np.std(values))
            all_colors.append(COLORS.get(method, '#333333'))

    # Collect baseline data
    for baseline in baselines:
        values = get_metric_values(metrics, baseline, ncal, metric_name, is_baseline=True)
        if values:
            all_names.append(METHOD_NAMES.get(baseline, baseline))
            all_means.append(np.mean(values))
            all_stds.append(np.std(values))
            all_colors.append(COLORS.get(baseline, '#666666'))

    if not all_names:
        print(f"No data available for metric {metric_name} at ncal={ncal}")
        return

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(all_names))
    bars = ax.bar(x, all_means, yerr=all_stds, capsize=5, color=all_colors, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha='right')
    ax.set_ylabel(METRIC_NAMES.get(metric_name, metric_name))

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{METRIC_NAMES.get(metric_name, metric_name)} at ncal={ncal}')

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate comparison plots from evaluation metrics')
    parser.add_argument('--task', nargs='+', required=True, help='Task name(s) to plot')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')
    parser.add_argument('--experiment', type=str, default='exp_50k_v1', help='Experiment name')
    parser.add_argument('--save_dir', type=str, default='./figures', help='Directory to save plots')
    parser.add_argument('--conditional', action='store_true', help='Use conditional metrics')
    parser.add_argument('--metrics', nargs='+', help='Specific metrics to plot')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 60)
    print(f"Results dir: {results_dir}")
    print(f"Experiment: {args.experiment}")
    print(f"Tasks: {args.task}")
    print(f"Conditional: {args.conditional}")
    print(f"Save dir: {save_dir}")
    print("=" * 60)

    for task in args.task:
        print(f"\nProcessing task: {task}")

        metrics = load_metrics(results_dir, args.experiment, task, conditional=args.conditional)
        if metrics is None:
            continue

        task_save_dir = save_dir / task
        task_save_dir.mkdir(parents=True, exist_ok=True)

        # Plot all metrics grid
        print("  Generating metrics grid...")
        plot_all_metrics_grid(metrics, task, task_save_dir, conditional=args.conditional)

        # Plot individual metrics
        ncal_values = get_ncal_values(metrics)

        # Determine which metrics to plot
        available_metrics = set()
        for method_data in metrics.get('methods', {}).values():
            for ncal_data in method_data.values():
                available_metrics.update(ncal_data.keys())
        available_metrics = [m for m in available_metrics if '_per_obs' not in m]

        if args.metrics:
            plot_metrics = [m for m in args.metrics if m in available_metrics]
        else:
            plot_metrics = available_metrics

        for metric_name in plot_metrics:
            print(f"  Plotting {metric_name}...")

            # Line plot vs ncal
            task_display = TASK_NAMES.get(task, task)
            suffix = " (Conditional)" if args.conditional else ""
            plot_metric_vs_ncal(
                metrics, metric_name,
                save_path=task_save_dir / f"{metric_name}_vs_ncal.pdf",
                title=f"{task_display}{suffix} - {METRIC_NAMES.get(metric_name, metric_name)}"
            )

            # Bar plots at specific ncal values
            for ncal in [min(ncal_values), max(ncal_values)]:
                plot_comparison_bar(
                    metrics, ncal, metric_name,
                    save_path=task_save_dir / f"{metric_name}_bar_ncal{ncal}.pdf",
                    title=f"{task_display}{suffix} - {METRIC_NAMES.get(metric_name, metric_name)} (ncal={ncal})"
                )

    print("\n" + "=" * 60)
    print("PLOTS GENERATED SUCCESSFULLY")
    print(f"Saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
