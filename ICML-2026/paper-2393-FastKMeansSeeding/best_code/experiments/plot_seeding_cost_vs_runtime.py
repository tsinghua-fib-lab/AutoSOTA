#!/usr/bin/env python3
"""
Plot seeding cost vs runtime from benchmark CSVs.

Reads all *_benchmark.csv files from experiments/results/qkmeans_benchmark/
and creates plots showing cost vs runtime tradeoffs for each method.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# Results directory
RESULTS_DIR = Path(__file__).parent / "results" / "qkmeans_benchmark"

# Method display names and colors
METHOD_STYLES = {
    "AFKMC2": {"color": "#377eb8", "marker": "o", "label": "AFK-MC²"},
    "PRONE": {"color": "#4daf4a", "marker": "s", "label": "PRONE"},
    "PRONECoreset": {"color": "#984ea3", "marker": "^", "label": "PRONE+Coreset"},
    "FastCoresetKMeansPP": {"color": "#ff7f00", "marker": "v", "label": "FastCoreset"},
    "RejectionSamplingLSH": {"color": "#a65628", "marker": "D", "label": "RejectionLSH"},
    "QKMEANS": {"color": "#e41a1c", "marker": "p", "label": "QKMEANS (Ours)"},
}

# Dataset display names
DATASET_NAMES = {
    "mnist": "MNIST",
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
    "stackexchange": "StackExchange",
}


def load_all_benchmarks() -> pd.DataFrame:
    """Load all benchmark CSV files and concatenate them.

    Supports both single-run CSVs and multi-run CSVs (with 'run' column).
    For multi-run CSVs, computes mean and std across runs.
    """
    csv_files = list(RESULTS_DIR.glob("*_benchmark.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No benchmark CSV files found in {RESULTS_DIR}\n"
            "Run the benchmark first."
        )

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Check if this has multiple runs (run column exists)
    if "run" in combined.columns:
        # Aggregate across runs
        agg = combined.groupby(["dataset", "method", "k"]).agg({
            "seeding_cost": ["mean", "std"],
            "time_ms": ["mean", "std"]
        }).reset_index()
        agg.columns = ["dataset", "method", "k", "seeding_cost", "cost_std", "time_ms", "time_std"]
        return agg
    else:
        # Single run - add placeholder std columns
        combined["cost_std"] = 0
        combined["time_std"] = 0
        return combined


def plot_cost_vs_runtime_per_dataset(df: pd.DataFrame, output_dir: Path):
    """Create cost vs runtime plot for each dataset (one subplot per k).

    Shows error bars if std columns are available.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = df["dataset"].unique()
    has_std = "cost_std" in df.columns and "time_std" in df.columns

    for dataset in datasets:
        subset = df[df["dataset"] == dataset]
        k_values = sorted(subset["k"].unique())

        # Determine grid size
        n_k = len(k_values)
        ncols = min(3, n_k)
        nrows = (n_k + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if n_k == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, k in enumerate(k_values):
            ax = axes[i]
            k_data = subset[subset["k"] == k]

            for method, style in METHOD_STYLES.items():
                m_data = k_data[k_data["method"] == method]
                if m_data.empty:
                    continue

                x = m_data["time_ms"].values[0]
                y = m_data["seeding_cost"].values[0]

                if has_std and m_data["cost_std"].values[0] > 0:
                    # Plot with error bars
                    xerr = m_data["time_std"].values[0]
                    yerr = m_data["cost_std"].values[0]
                    ax.errorbar(
                        x, y,
                        xerr=xerr, yerr=yerr,
                        fmt=style["marker"],
                        color=style["color"],
                        markersize=8,
                        capsize=3,
                        capthick=1,
                        elinewidth=1,
                        label=style["label"] if i == 0 else None,
                        zorder=3
                    )
                else:
                    ax.scatter(
                        x, y,
                        marker=style["marker"],
                        color=style["color"],
                        s=100,
                        label=style["label"] if i == 0 else None,
                        zorder=3,
                        edgecolors='white',
                        linewidths=0.5
                    )

            ax.set_xlabel("Runtime (ms)", fontsize=10)
            ax.set_ylabel("Seeding Cost", fontsize=10)
            ax.set_title(f"k = {k}", fontsize=11, fontweight='bold')
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # Add legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 0.02), fontsize=10)

        display_name = DATASET_NAMES.get(dataset, dataset)
        fig.suptitle(f"Seeding Cost vs Runtime: {display_name}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])

        output_path = output_dir / f"cost_vs_runtime_{dataset}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.savefig(output_dir / f"cost_vs_runtime_{dataset}.pdf", bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_path.name}")


def plot_combined_cost_vs_runtime(df: pd.DataFrame, output_dir: Path):
    """Create a combined plot showing cost vs runtime across all datasets and k values."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize cost per (dataset, k) to make them comparable
    df = df.copy()
    df["cost_norm"] = df.groupby(["dataset", "k"])["seeding_cost"].transform(
        lambda x: x / x.min()
    )
    df["time_norm"] = df.groupby(["dataset", "k"])["time_ms"].transform(
        lambda x: x / x.min()
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    for method, style in METHOD_STYLES.items():
        m_data = df[df["method"] == method]
        if m_data.empty:
            continue

        ax.scatter(
            m_data["time_norm"],
            m_data["cost_norm"],
            marker=style["marker"],
            color=style["color"],
            s=60,
            alpha=0.7,
            label=style["label"],
            edgecolors='white',
            linewidths=0.3
        )

    ax.set_xlabel("Normalized Runtime (1 = fastest per dataset/k)", fontsize=12)
    ax.set_ylabel("Normalized Cost (1 = best per dataset/k)", fontsize=12)
    ax.set_title("Seeding Cost vs Runtime (all datasets, all k values)\n"
                 "Lower-left is better", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(loc="upper right", fontsize=10)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "cost_vs_runtime_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "cost_vs_runtime_combined.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_pareto_frontier_per_k(df: pd.DataFrame, output_dir: Path):
    """Create Pareto frontier plot grouped by k value."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize per (dataset, k)
    df = df.copy()
    df["cost_norm"] = df.groupby(["dataset", "k"])["seeding_cost"].transform(
        lambda x: x / x.min()
    )
    df["time_norm"] = df.groupby(["dataset", "k"])["time_ms"].transform(
        lambda x: x / x.min()
    )

    # Aggregate by method and k
    agg = df.groupby(["method", "k"]).agg({
        "cost_norm": ["mean", "std"],
        "time_norm": ["mean", "std"]
    }).reset_index()
    agg.columns = ["method", "k", "cost_mean", "cost_std", "time_mean", "time_std"]

    k_values = sorted(agg["k"].unique())

    fig, ax = plt.subplots(figsize=(12, 8))

    for method, style in METHOD_STYLES.items():
        m_data = agg[agg["method"] == method].sort_values("k")
        if m_data.empty:
            continue

        ax.errorbar(
            m_data["time_mean"],
            m_data["cost_mean"],
            xerr=m_data["time_std"],
            yerr=m_data["cost_std"],
            marker=style["marker"],
            color=style["color"],
            markersize=10,
            linewidth=2,
            capsize=3,
            label=style["label"]
        )

        # Annotate k values
        for _, row in m_data.iterrows():
            ax.annotate(
                f"k={int(row['k'])}",
                (row["time_mean"], row["cost_mean"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                alpha=0.7
            )

    ax.set_xlabel("Normalized Runtime (mean ± std)", fontsize=12)
    ax.set_ylabel("Normalized Cost (mean ± std)", fontsize=12)
    ax.set_title("Pareto Frontier by k Value\n(averaged across datasets)",
                 fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(loc="upper right", fontsize=10)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "pareto_frontier_by_k.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "pareto_frontier_by_k.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_method_comparison_bars(df: pd.DataFrame, output_dir: Path):
    """Create bar chart comparing methods across metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize per (dataset, k)
    df = df.copy()
    df["cost_norm"] = df.groupby(["dataset", "k"])["seeding_cost"].transform(
        lambda x: x / x.min()
    )
    df["time_norm"] = df.groupby(["dataset", "k"])["time_ms"].transform(
        lambda x: x / x.min()
    )

    # Aggregate by method
    agg = df.groupby("method").agg({
        "cost_norm": ["mean", "std"],
        "time_norm": ["mean", "std"],
        "time_ms": "mean"
    }).round(3)
    agg.columns = ["Avg Cost Ratio", "Cost Std", "Avg Time Ratio", "Time Std", "Avg Time (ms)"]

    # Sort by average time
    agg = agg.sort_values("Avg Time (ms)")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    methods = agg.index.tolist()
    colors = [METHOD_STYLES.get(m, {"color": "gray"})["color"] for m in methods]
    labels = [METHOD_STYLES.get(m, {"label": m})["label"] for m in methods]

    x = np.arange(len(methods))
    width = 0.6

    # Cost ratio bars
    bars1 = ax1.bar(x, agg["Avg Cost Ratio"], width, color=colors,
                    yerr=agg["Cost Std"], capsize=5, alpha=0.8)
    ax1.set_ylabel("Normalized Cost (1 = best)", fontsize=11)
    ax1.set_title("Average Cost Ratio by Method", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylim(0.95, None)
    ax1.grid(True, alpha=0.3, axis='y')

    # Time ratio bars
    bars2 = ax2.bar(x, agg["Avg Time Ratio"], width, color=colors,
                    yerr=agg["Time Std"], capsize=5, alpha=0.8)
    ax2.set_ylabel("Normalized Runtime (1 = fastest)", fontsize=11)
    ax2.set_title("Average Time Ratio by Method", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "method_comparison_bars.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "method_comparison_bars.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")

    # Also save the summary table
    agg.to_csv(output_dir / "method_summary.csv")
    print(f"  Saved: method_summary.csv")

    return agg


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Normalize
    df = df.copy()
    df["cost_norm"] = df.groupby(["dataset", "k"])["seeding_cost"].transform(
        lambda x: x / x.min()
    )
    df["time_norm"] = df.groupby(["dataset", "k"])["time_ms"].transform(
        lambda x: x / x.min()
    )

    print(f"\nDatasets: {df['dataset'].nunique()}")
    print(f"k values: {sorted(df['k'].unique())}")
    print(f"Methods: {df['method'].nunique()}")
    print(f"Total data points: {len(df)}")

    # Per method stats
    print("\n" + "-" * 70)
    print(f"{'Method':<25} {'Avg Cost Ratio':<15} {'Avg Time Ratio':<15} {'Avg Time (ms)':<15}")
    print("-" * 70)

    for method in sorted(df["method"].unique()):
        m_data = df[df["method"] == method]
        cost_ratio = m_data["cost_norm"].mean()
        time_ratio = m_data["time_norm"].mean()
        avg_time = m_data["time_ms"].mean()

        label = METHOD_STYLES.get(method, {"label": method})["label"]
        print(f"{label:<25} {cost_ratio:<15.3f} {time_ratio:<15.1f} {avg_time:<15.1f}")

    print("=" * 70)


def main():
    print("=" * 60)
    print("Seeding Cost vs Runtime Plotting")
    print("=" * 60)

    # Load data
    print("\nLoading benchmark data...")
    try:
        df = load_all_benchmarks()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print(f"  Loaded {len(df)} rows from {df['dataset'].nunique()} dataset(s)")

    output_dir = RESULTS_DIR / "plots"

    # Generate plots
    print("\nGenerating plots...")

    print("\n[1/4] Cost vs Runtime (per dataset)...")
    plot_cost_vs_runtime_per_dataset(df, output_dir)

    print("\n[2/4] Combined Cost vs Runtime...")
    plot_combined_cost_vs_runtime(df, output_dir)

    print("\n[3/4] Pareto Frontier by k...")
    plot_pareto_frontier_per_k(df, output_dir)

    print("\n[4/4] Method Comparison Bars...")
    summary = plot_method_comparison_bars(df, output_dir)

    # Print summary
    print_summary(df)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
