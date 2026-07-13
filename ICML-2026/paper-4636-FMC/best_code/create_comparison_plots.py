#!/usr/bin/env python3
"""Create comprehensive comparison plots from experiment metrics.

This script loads metrics from the comparison_benchmark experiment and creates
an all-in-one plot with:
- Joint metrics by row (c2st, mmd, wasserstein)
- Tasks by column
- All baselines/variants shown
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

# Use Type 1 fonts for publication-quality PDFs
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9


def load_all_metrics(results_dir: Path, seeds: list[int] | None = None) -> Dict:
    """Load all metrics from the comparison benchmark results.

    Args:
        results_dir: Path to comparison_benchmark results directory
        seeds: If provided, only load these seeds. None loads all.

    Returns:
        Dict with structure: {method: {task: {seed: {ncal: metrics}}}}
    """
    all_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Load baselines from shared directory
    shared_baselines = results_dir / "shared" / "baselines"
    if shared_baselines.exists():
        for task_dir in shared_baselines.iterdir():
            if not task_dir.is_dir():
                continue
            task = task_dir.name

            for seed_dir in task_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                seed = int(seed_dir.name.split("_")[1])
                if seeds is not None and seed not in seeds:
                    continue

                for method_dir in seed_dir.iterdir():
                    if not method_dir.is_dir():
                        continue
                    method = method_dir.name

                    for ncal_dir in method_dir.iterdir():
                        if not ncal_dir.is_dir() or not ncal_dir.name.startswith("ncal_"):
                            continue
                        ncal = int(ncal_dir.name.split("_")[1])

                        metrics_file = ncal_dir / "metrics.json"
                        if metrics_file.exists():
                            with open(metrics_file) as f:
                                data = json.load(f)
                                all_metrics[method][task][seed][ncal] = data["metrics"]

    # Load variants
    variants_dir = results_dir / "variants"
    if variants_dir.exists():
        for variant_dir in variants_dir.iterdir():
            if not variant_dir.is_dir():
                continue
            variant_name = variant_dir.name

            eval_dir = variant_dir / "evaluation"
            if not eval_dir.exists():
                continue

            for task_dir in eval_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                task = task_dir.name

                for seed_dir in task_dir.iterdir():
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                        continue
                    seed = int(seed_dir.name.split("_")[1])
                    if seeds is not None and seed not in seeds:
                        continue

                    for method_dir in seed_dir.iterdir():
                        if not method_dir.is_dir():
                            continue

                        for ncal_dir in method_dir.iterdir():
                            if not ncal_dir.is_dir() or not ncal_dir.name.startswith("ncal_"):
                                continue
                            ncal = int(ncal_dir.name.split("_")[1])

                            metrics_file = ncal_dir / "metrics.json"
                            if metrics_file.exists():
                                with open(metrics_file) as f:
                                    data = json.load(f)
                                    all_metrics[variant_name][task][seed][ncal] = data["metrics"]

    return dict(all_metrics)


def aggregate_metrics(metrics: Dict) -> Dict:
    """Aggregate metrics across seeds to get mean and std.

    Returns:
        Dict with structure: {method: {task: {ncal: {metric: (mean, std)}}}}
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for method, tasks in metrics.items():
        for task, seeds in tasks.items():
            # Collect values by ncal and metric
            ncal_metric_values = defaultdict(lambda: defaultdict(list))

            for seed, ncals in seeds.items():
                for ncal, metric_dict in ncals.items():
                    for metric_name, value in metric_dict.items():
                        if isinstance(value, (int, float)):
                            ncal_metric_values[ncal][metric_name].append(value)

            # Compute mean and std
            for ncal, metric_values in ncal_metric_values.items():
                for metric_name, values in metric_values.items():
                    aggregated[method][task][ncal][metric_name] = (
                        np.nanmean(values),
                        np.nanstd(values)
                    )

    return dict(aggregated)


def get_nice_method_name(method: str) -> str:
    """Convert method name to nice display name."""
    name_map = {
        "dpe": "DPE",
        "mf_npe": "MF-NPE",
        "npe_pfn": "NPE-PFN",
        "fm_pt_fmpe": "FMCPE",
        "fm_pt_npe": "FM-PT (NPE)",
        "fm_pt_small": "FM-PT (Small)",
        "fm_post_transform": "FMCPE",
    }
    return name_map.get(method, method.upper())


def get_nice_task_name(task: str) -> str:
    """Convert task name to nice display name."""
    name_map = {
        "high_dim_gaussian": "High-Dim Gaussian",
        "pendulum": "Pendulum",
        "wind_tunnel": "Wind Tunnel",
    }
    return name_map.get(task, task.replace("_", " ").title())


def get_nice_metric_name(metric: str) -> str:
    """Convert metric name to nice display name."""
    name_map = {
        "joint_c2st": "C2ST",
        "joint_mmd": "MMD",
        "joint_wasserstein": "Wasserstein",
        "cond_c2st": "C2ST",
        "cond_mmd": "MMD",
    }
    return name_map.get(metric, metric.upper())


def create_comparison_plot(aggregated_metrics: Dict, save_path: Path):
    """Create all-in-one comparison plot.

    Layout:
    - Rows: metrics (c2st, mmd, wasserstein)
    - Columns: tasks
    - Lines: different methods/variants
    - Uses conditional metrics for high_dim_gaussian, joint for others
    """
    # Get all tasks from the data (dynamically)
    tasks = sorted(set(task for method_data in aggregated_metrics.values()
                      for task in method_data.keys()))

    # Determine which metrics to use for each task
    # For high_dim_gaussian, use conditional metrics; for others, use joint metrics
    task_metrics = {}
    for task in tasks:
        if task == "high_dim_gaussian":
            task_metrics[task] = ["cond_c2st", "cond_mmd"]
        else:
            task_metrics[task] = ["joint_c2st", "joint_mmd", "joint_wasserstein"]

    # Use maximum number of metrics for row count
    max_metrics = max(len(m) for m in task_metrics.values())
    all_metric_names = ["C2ST", "MMD", "Wasserstein"]

    # Get all methods
    methods = sorted(aggregated_metrics.keys())

    # Define colors and markers for each method
    colors = {
        "dpe": "#1f77b4",  # blue
        "mf_npe": "#ff7f0e",  # orange
        "npe_pfn": "#9467bd",  # purple
        "rope": "#d62728",  # red
        "fm_pt_fmpe": "#2ca02c",  # green
        "fm_pt_npe": "#e377c2",  # pink
        "fm_pt_small": "#17becf",  # cyan
    }

    markers = {
        "dpe": "o",
        "mf_npe": "s",
        "npe_pfn": "p",  # pentagon
        "rope": "d",  # diamond
        "fm_pt_fmpe": "^",
        "fm_pt_npe": "D",
        "fm_pt_small": "v",  # inverted triangle
    }

    # Create figure
    fig, axes = plt.subplots(max_metrics, len(tasks), figsize=(15, 12))

    # Plot each metric-task combination
    for j, task in enumerate(tasks):
        metrics_for_task = task_metrics[task]

        for i in range(max_metrics):
            ax = axes[i, j]

            # Skip if this task doesn't have this metric
            if i >= len(metrics_for_task):
                ax.axis('off')
                continue

            metric = metrics_for_task[i]

            # Plot each method
            for method in methods:
                if task not in aggregated_metrics[method]:
                    continue

                task_data = aggregated_metrics[method][task]

                # Get ncal values and corresponding metric values
                ncals = sorted(task_data.keys())
                means = []
                stds = []

                for ncal in ncals:
                    if metric in task_data[ncal]:
                        mean, std = task_data[ncal][metric]
                        # Filter out NaN values
                        if not np.isnan(mean):
                            means.append(mean)
                            stds.append(std)
                        else:
                            means.append(np.nan)
                            stds.append(np.nan)
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)

                means = np.array(means)
                stds = np.array(stds)

                # Plot with error bars
                color = colors.get(method, None)
                marker = markers.get(method, "o")
                label = get_nice_method_name(method)

                ax.errorbar(ncals, means, yerr=stds,
                           label=label, marker=marker, color=color,
                           capsize=3, capthick=1, linewidth=1.5, markersize=6,
                           alpha=0.8)

            # Formatting
            ax.set_xscale('log')
            ax.set_xlabel('Number of Calibration Samples (ncal)')
            ax.set_ylabel(get_nice_metric_name(metric))
            ax.grid(True, alpha=0.3, linestyle='--')

            # Set title only for top row
            if i == 0:
                ax.set_title(get_nice_task_name(task), fontweight='bold')

            # Add legend only to top-right plot
            if i == 0 and j == len(tasks) - 1:
                ax.legend(loc='best', framealpha=0.9)

            # For C2ST, lower is better - add reference line at 0.5
            if "c2st" in metric:
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

            # Set y-axis limits based on metric
            if "c2st" in metric:
                ax.set_ylim([0.45, 1.0])
            elif "mmd" in metric:
                ax.set_ylim(bottom=0)
            elif "wasserstein" in metric:
                ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Save as PDF
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {save_path}")

    plt.close()


def create_individual_task_plots(aggregated_metrics: Dict, save_dir: Path):
    """Create individual plots for each task with all metrics in subplots.
    Uses conditional metrics for high_dim_gaussian, joint for others."""
    tasks = sorted(set(task for method_data in aggregated_metrics.values()
                      for task in method_data.keys()))

    methods = sorted(aggregated_metrics.keys())

    colors = {
        "dpe": "#1f77b4",
        "mf_npe": "#ff7f0e",
        "npe_pfn": "#9467bd",
        "rope": "#d62728",
        "fm_pt_fmpe": "#2ca02c",
        "fm_pt_npe": "#e377c2",
        "fm_pt_small": "#17becf",
    }

    markers = {
        "dpe": "o",
        "mf_npe": "s",
        "npe_pfn": "p",
        "rope": "d",
        "fm_pt_fmpe": "^",
        "fm_pt_npe": "D",
        "fm_pt_small": "v",
    }

    for task in tasks:
        # Choose metrics based on task
        if task == "high_dim_gaussian":
            metrics = ["cond_c2st", "cond_mmd"]
        else:
            metrics = ["joint_c2st", "joint_mmd", "joint_wasserstein"]

        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]

            for method in methods:
                if task not in aggregated_metrics[method]:
                    continue

                task_data = aggregated_metrics[method][task]
                ncals = sorted(task_data.keys())
                means = []
                stds = []

                for ncal in ncals:
                    if metric in task_data[ncal]:
                        mean, std = task_data[ncal][metric]
                        # Filter out NaN values
                        if not np.isnan(mean):
                            means.append(mean)
                            stds.append(std)
                        else:
                            means.append(np.nan)
                            stds.append(np.nan)
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)

                means = np.array(means)
                stds = np.array(stds)

                color = colors.get(method, None)
                marker = markers.get(method, "o")
                label = get_nice_method_name(method)

                ax.errorbar(ncals, means, yerr=stds,
                           label=label, marker=marker, color=color,
                           capsize=3, capthick=1, linewidth=1.5, markersize=6,
                           alpha=0.8)

            ax.set_xscale('log')
            ax.set_xlabel('Number of Calibration Samples (ncal)')
            ax.set_ylabel(get_nice_metric_name(metric))
            ax.set_title(get_nice_metric_name(metric), fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')

            if "c2st" in metric:
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                ax.set_ylim([0.45, 1.0])
            elif "mmd" in metric or "wasserstein" in metric:
                ax.set_ylim(bottom=0)

            if i == len(metrics) - 1:
                ax.legend(loc='best', framealpha=0.9)

        plt.suptitle(get_nice_task_name(task), fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = save_dir / f"{task}_comparison.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")

        plt.close()


def main():
    """Main entry point."""
    results_dir = Path("results/comparison_benchmark")
    figures_dir = Path("figures/comparison_benchmark")

    print("=" * 60)
    print("CREATING COMPARISON PLOTS")
    print("=" * 60)

    # Load all metrics
    print("\nLoading metrics...")
    metrics = load_all_metrics(results_dir, seeds=[33])

    print(f"Found {len(metrics)} methods:")
    for method in sorted(metrics.keys()):
        tasks = list(metrics[method].keys())
        print(f"  {method}: {tasks}")

    # Aggregate across seeds
    print("\nAggregating metrics across seeds...")
    aggregated = aggregate_metrics(metrics)

    # Create all-in-one plot
    print("\nCreating all-in-one comparison plot...")
    create_comparison_plot(aggregated, figures_dir / "all_metrics_comparison.pdf")

    # Create individual task plots
    print("\nCreating individual task plots...")
    create_individual_task_plots(aggregated, figures_dir / "by_task")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
