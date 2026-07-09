#!/usr/bin/env python3
"""
Plot solution quality vs runtime for QKMEANS benchmark.

Creates Pareto-style plots showing the cost-time tradeoff for each method.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')

# Results directory
RESULTS_DIR = Path(__file__).parent / "results" / "qkmeans_benchmark"

# Method display names and colors
METHOD_STYLES = {
    "QKMEANS": {"color": "#e41a1c", "marker": "o", "label": "QKMEANS (Ours)"},
    "AFKMC2": {"color": "#377eb8", "marker": "s", "label": "AFK-MC²"},
    "PRONE": {"color": "#4daf4a", "marker": "^", "label": "PRONE"},
    "PRONECoreset": {"color": "#984ea3", "marker": "v", "label": "PRONE+Coreset"},
    "FastCoresetKMeansPP": {"color": "#ff7f00", "marker": "D", "label": "FastCoreset"},
    "RejectionSamplingLSH": {"color": "#a65628", "marker": "p", "label": "RejectionLSH"},
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
}


def load_summary():
    """Load the benchmark summary CSV."""
    summary_path = RESULTS_DIR / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}\n"
            "Run run_qkmeans_benchmark.py first."
        )
    return pd.read_csv(summary_path)


def plot_quality_vs_runtime_per_dataset(df: pd.DataFrame, output_dir: Path):
    """Create quality vs runtime plot for each dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = df["dataset"].unique()

    for dataset in datasets:
        subset = df[df["dataset"] == dataset]
        k_values = sorted(subset["k"].unique())

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, k in enumerate(k_values[:6]):  # Show first 6 k values
            ax = axes[i]
            k_data = subset[subset["k"] == k]

            for method, style in METHOD_STYLES.items():
                m_data = k_data[k_data["method"] == method]
                if m_data.empty:
                    continue

                ax.errorbar(
                    m_data["time_mean"].values[0],
                    m_data["cost_mean"].values[0],
                    xerr=m_data["time_std"].values[0],
                    yerr=m_data["cost_std"].values[0],
                    fmt=style["marker"],
                    color=style["color"],
                    markersize=8,
                    capsize=3,
                    label=style["label"] if i == 0 else None
                )

            ax.set_xlabel("Time (ms)", fontsize=10)
            ax.set_ylabel("Clustering Cost", fontsize=10)
            ax.set_title(f"k = {k}", fontsize=11)
            ax.set_xscale("log")

        # Add legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 0.02), fontsize=10)

        display_name = DATASET_NAMES.get(dataset, dataset)
        fig.suptitle(f"Quality vs Runtime: {display_name}", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        output_path = output_dir / f"quality_vs_runtime_{dataset}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {output_path.name}")


def plot_pareto_frontier(df: pd.DataFrame, output_dir: Path):
    """Create Pareto frontier plot aggregated across all datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize costs and times per (dataset, k) to make them comparable
    df = df.copy()
    df["cost_norm"] = df.groupby(["dataset", "k"])["cost_mean"].transform(
        lambda x: x / x.min()
    )
    df["time_norm"] = df.groupby(["dataset", "k"])["time_mean"].transform(
        lambda x: x / x.min()
    )

    # Aggregate across datasets
    agg = df.groupby(["method", "k"]).agg({
        "cost_norm": "mean",
        "time_norm": "mean"
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 7))

    for method, style in METHOD_STYLES.items():
        m_data = agg[agg["method"] == method].sort_values("k")
        if m_data.empty:
            continue

        ax.plot(
            m_data["time_norm"],
            m_data["cost_norm"],
            marker=style["marker"],
            color=style["color"],
            markersize=8,
            linewidth=2,
            label=style["label"]
        )

        # Annotate k values
        for _, row in m_data.iterrows():
            ax.annotate(
                f"k={int(row['k'])}",
                (row["time_norm"], row["cost_norm"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                alpha=0.7
            )

    ax.set_xlabel("Normalized Runtime (1 = fastest)", fontsize=12)
    ax.set_ylabel("Normalized Cost (1 = best)", fontsize=12)
    ax.set_title("Pareto Frontier: Quality vs Runtime\n(averaged across all datasets)",
                 fontsize=14, fontweight="medium")
    ax.set_xscale("log")
    ax.legend(loc="upper right", fontsize=10)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path = output_dir / "pareto_frontier.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "pareto_frontier.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_speedup_vs_k(df: pd.DataFrame, output_dir: Path):
    """Plot speedup of QKMEANS over other methods as a function of k."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute speedup relative to RejectionSamplingLSH
    baseline = df[df["method"] == "RejectionSamplingLSH"][["dataset", "k", "time_mean"]]
    baseline = baseline.rename(columns={"time_mean": "baseline_time"})

    merged = df.merge(baseline, on=["dataset", "k"])
    merged["speedup"] = merged["baseline_time"] / merged["time_mean"]

    # Average speedup per method per k
    speedup_agg = merged.groupby(["method", "k"])["speedup"].agg(["mean", "std"]).reset_index()
    speedup_agg.columns = ["method", "k", "speedup_mean", "speedup_std"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for method, style in METHOD_STYLES.items():
        if method == "RejectionSamplingLSH":
            continue  # Skip baseline

        m_data = speedup_agg[speedup_agg["method"] == method].sort_values("k")
        if m_data.empty:
            continue

        ax.errorbar(
            m_data["k"],
            m_data["speedup_mean"],
            yerr=m_data["speedup_std"],
            marker=style["marker"],
            color=style["color"],
            markersize=8,
            linewidth=2,
            capsize=3,
            label=style["label"]
        )

    ax.set_xlabel("Number of Centers (k)", fontsize=12)
    ax.set_ylabel("Speedup vs RejectionSamplingLSH", fontsize=12)
    ax.set_title("Speedup as a Function of k\n(averaged across all datasets)",
                 fontsize=14, fontweight="medium")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "speedup_vs_k.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "speedup_vs_k.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_cost_ratio_vs_k(df: pd.DataFrame, output_dir: Path):
    """Plot cost ratio (method cost / best cost) as a function of k."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute best cost per (dataset, k)
    best_cost = df.groupby(["dataset", "k"])["cost_mean"].min().reset_index()
    best_cost = best_cost.rename(columns={"cost_mean": "best_cost"})

    merged = df.merge(best_cost, on=["dataset", "k"])
    merged["cost_ratio"] = merged["cost_mean"] / merged["best_cost"]

    # Average cost ratio per method per k
    ratio_agg = merged.groupby(["method", "k"])["cost_ratio"].agg(["mean", "std"]).reset_index()
    ratio_agg.columns = ["method", "k", "ratio_mean", "ratio_std"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for method, style in METHOD_STYLES.items():
        m_data = ratio_agg[ratio_agg["method"] == method].sort_values("k")
        if m_data.empty:
            continue

        ax.errorbar(
            m_data["k"],
            m_data["ratio_mean"],
            yerr=m_data["ratio_std"],
            marker=style["marker"],
            color=style["color"],
            markersize=8,
            linewidth=2,
            capsize=3,
            label=style["label"]
        )

    ax.set_xlabel("Number of Centers (k)", fontsize=12)
    ax.set_ylabel("Cost Ratio (method / best)", fontsize=12)
    ax.set_title("Solution Quality as a Function of k\n(averaged across all datasets)",
                 fontsize=14, fontweight="medium")
    ax.set_xscale("log")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Optimal")
    ax.legend(loc="best", fontsize=10)
    ax.set_ylim(0.95, None)

    plt.tight_layout()
    output_path = output_dir / "cost_ratio_vs_k.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "cost_ratio_vs_k.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_runtime_scaling(df: pd.DataFrame, output_dir: Path):
    """Plot runtime scaling with k for each method."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Average time per method per k
    time_agg = df.groupby(["method", "k"])["time_mean"].agg(["mean", "std"]).reset_index()
    time_agg.columns = ["method", "k", "time_mean", "time_std"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for method, style in METHOD_STYLES.items():
        m_data = time_agg[time_agg["method"] == method].sort_values("k")
        if m_data.empty:
            continue

        ax.errorbar(
            m_data["k"],
            m_data["time_mean"],
            yerr=m_data["time_std"],
            marker=style["marker"],
            color=style["color"],
            markersize=8,
            linewidth=2,
            capsize=3,
            label=style["label"]
        )

    ax.set_xlabel("Number of Centers (k)", fontsize=12)
    ax.set_ylabel("Runtime (ms)", fontsize=12)
    ax.set_title("Runtime Scaling with k\n(averaged across all datasets)",
                 fontsize=14, fontweight="medium")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    output_path = output_dir / "runtime_scaling.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "runtime_scaling.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """Create a summary table comparing methods."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute normalized metrics
    df = df.copy()
    df["cost_norm"] = df.groupby(["dataset", "k"])["cost_mean"].transform(
        lambda x: x / x.min()
    )
    df["time_norm"] = df.groupby(["dataset", "k"])["time_mean"].transform(
        lambda x: x / x.min()
    )

    # Aggregate
    summary = df.groupby("method").agg({
        "cost_norm": ["mean", "std"],
        "time_norm": ["mean", "std"],
        "time_mean": "mean"
    }).round(3)

    summary.columns = ["Avg Cost Ratio", "Cost Std", "Avg Time Ratio", "Time Std", "Avg Time (ms)"]
    summary = summary.sort_values("Avg Time (ms)")

    # Save
    output_path = output_dir / "method_comparison.csv"
    summary.to_csv(output_path)
    print(f"  Saved: {output_path.name}")

    # Print
    print("\n" + "=" * 70)
    print("METHOD COMPARISON SUMMARY")
    print("=" * 70)
    print(summary.to_string())


def main():
    print("=" * 60)
    print("QKMEANS Benchmark Plotting")
    print("=" * 60)

    # Load data
    print("\nLoading summary data...")
    try:
        df = load_summary()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print(f"  Loaded {len(df)} rows")
    print(f"  Datasets: {df['dataset'].nunique()}")
    print(f"  Methods: {df['method'].nunique()}")
    print(f"  k values: {sorted(df['k'].unique())}")

    output_dir = RESULTS_DIR / "plots"

    # Generate plots
    print("\nGenerating plots...")

    print("\n[1/6] Quality vs Runtime (per dataset)...")
    plot_quality_vs_runtime_per_dataset(df, output_dir)

    print("\n[2/6] Pareto Frontier...")
    plot_pareto_frontier(df, output_dir)

    print("\n[3/6] Speedup vs k...")
    plot_speedup_vs_k(df, output_dir)

    print("\n[4/6] Cost Ratio vs k...")
    plot_cost_ratio_vs_k(df, output_dir)

    print("\n[5/6] Runtime Scaling...")
    plot_runtime_scaling(df, output_dir)

    print("\n[6/6] Summary Table...")
    create_summary_table(df, output_dir)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
