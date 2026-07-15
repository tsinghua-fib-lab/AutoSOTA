#!/usr/bin/env python3
"""Generate plots from ablation study results."""

import argparse
import sys
from pathlib import Path
import json

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

METHOD_NAMES = {
    "mv": "Majority Vote",
    "ds": "Dawid-Skene",
    "pec": "Posterior Expected Credit"
}

METHOD_COLORS = {
    "mv": "#1f77b4",
    "ds": "#ff7f0e",
    "pec": "#2ca02c"
}

METHOD_MARKERS = {
    "mv": "o",
    "ds": "s",
    "pec": "^"
}


def load_results(results_dir: Path) -> dict:
    """Load results from JSON file."""
    with open(results_dir / "results.json", 'r') as f:
        return json.load(f)


def plot_metric_vs_param(
    results: dict,
    metric: str,
    output_path: Path,
    ylabel: str = None,
    title: str = None
):
    """Plot a metric vs the varying parameter."""
    param = results["param"]
    values = results["values"]
    metrics = results["metrics"]
    
    if param is None:
        print(f"No varying parameter detected, skipping {metric} line plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for method in ["mv", "ds", "pec"]:
        means = [m[method][f"{metric}_mean"] for m in metrics]
        stds = [m[method][f"{metric}_se"] for m in metrics]
        
        ax.errorbar(
            values, means, yerr=stds,
            label=METHOD_NAMES[method],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=8,
            linewidth=2,
            capsize=4
        )
    
    ax.set_xlabel(param.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(ylabel or metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"{metric.replace('_', ' ').title()} vs {param.replace('_', ' ').title()}", fontsize=14)
    ax.legend(frameon=True, fontsize=10)
    
    if metric == "kendall_tau":
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_bar_comparison(
    results: dict,
    metric: str,
    output_path: Path,
    ylabel: str = None,
    title: str = None
):
    """Create bar chart comparing methods across configs."""
    configs = results["configs"]
    metrics = results["metrics"]
    
    config_names = [Path(c).stem for c in configs]
    x = np.arange(len(config_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(max(10, len(configs) * 2), 6))
    
    for i, method in enumerate(["mv", "ds", "pec"]):
        means = [m[method][f"{metric}_mean"] for m in metrics]
        stds = [m[method][f"{metric}_se"] for m in metrics]
        
        ax.bar(
            x + i * width, means, width,
            label=METHOD_NAMES[method],
            color=METHOD_COLORS[method],
            yerr=stds,
            capsize=4
        )
    
    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel(ylabel or metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title or f"{metric.replace('_', ' ').title()} by Configuration", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.legend(frameon=True, fontsize=10)
    
    if metric == "kendall_tau":
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_combined(results: dict, output_path: Path):
    """Create combined figure with MSE and Kendall tau."""
    configs = results["configs"]
    metrics = results["metrics"]
    param = results["param"]
    values = results["values"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    if param and values:
        # Line plots if parameter varies
        for method in ["mv", "ds", "pec"]:
            # MSE
            means = [m[method]["mse_mean"] for m in metrics]
            stds = [m[method]["mse_se"] for m in metrics]
            ax1.errorbar(values, means, yerr=stds, label=METHOD_NAMES[method],
                        color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                        markersize=8, linewidth=2, capsize=4)
            
            # Tau
            means = [m[method]["kendall_tau_mean"] for m in metrics]
            stds = [m[method]["kendall_tau_se"] for m in metrics]
            ax2.errorbar(values, means, yerr=stds, label=METHOD_NAMES[method],
                        color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                        markersize=8, linewidth=2, capsize=4)
        
        ax1.set_xlabel(param.replace("_", " ").title())
        ax2.set_xlabel(param.replace("_", " ").title())
    else:
        # Bar plots if just comparing configs
        config_names = [Path(c).stem for c in configs]
        x = np.arange(len(config_names))
        width = 0.25
        
        for i, method in enumerate(["mv", "ds", "pec"]):
            # MSE
            means = [m[method]["mse_mean"] for m in metrics]
            stds = [m[method]["mse_se"] for m in metrics]
            ax1.bar(x + i * width, means, width, label=METHOD_NAMES[method],
                   color=METHOD_COLORS[method], yerr=stds, capsize=4)
            
            # Tau
            means = [m[method]["kendall_tau_mean"] for m in metrics]
            stds = [m[method]["kendall_tau_se"] for m in metrics]
            ax2.bar(x + i * width, means, width, label=METHOD_NAMES[method],
                   color=METHOD_COLORS[method], yerr=stds, capsize=4)
        
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(config_names, rotation=45, ha='right')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(config_names, rotation=45, ha='right')
        ax1.set_xlabel("Configuration")
        ax2.set_xlabel("Configuration")
    
    ax1.set_ylabel("Mean Squared Error")
    ax1.set_title("Score Estimation Error")
    ax1.legend(frameon=True)
    
    ax2.set_ylabel("Kendall's τ")
    ax2.set_title("Ranking Correlation")
    ax2.legend(frameon=True)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def generate_all_plots(results_dir: Path, output_dir: Path = None):
    """Generate all plots for results."""
    results = load_results(results_dir)
    
    if output_dir is None:
        output_dir = results_dir
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    param = results.get("param")
    
    # Combined plot (always works)
    plot_combined(results, output_dir / "combined.png")
    
    # Bar comparisons (always works)
    plot_bar_comparison(results, "mse", output_dir / "mse_comparison.png",
                       ylabel="Mean Squared Error", title="MSE by Configuration")
    plot_bar_comparison(results, "kendall_tau", output_dir / "tau_comparison.png",
                       ylabel="Kendall's τ", title="Ranking Correlation by Configuration")
    
    # Line plots (only if parameter varies)
    if param:
        plot_metric_vs_param(results, "mse", output_dir / f"mse_vs_{param}.png",
                            ylabel="Mean Squared Error")
        plot_metric_vs_param(results, "kendall_tau", output_dir / f"tau_vs_{param}.png",
                            ylabel="Kendall's τ")
        plot_metric_vs_param(results, "ranking_accuracy", output_dir / f"accuracy_vs_{param}.png",
                            ylabel="Ranking Accuracy")
    
    # Stability plot (if available)
    if results["metrics"] and "stability_mean" in results["metrics"][0].get("mv", {}):
        plot_metric_vs_param(results, "stability", output_dir / f"stability_vs_{param}.png",
                            ylabel="Ranking Stability (τ)")
        plot_bar_comparison(results, "stability", output_dir / "stability_comparison.png",
                           ylabel="Ranking Stability", title="Stability by Configuration")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from experiment results"
    )
    parser.add_argument(
        "results_dir", type=str,
        help="Directory containing results.json"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for plots (default: same as results_dir)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    
    generate_all_plots(results_dir, output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()