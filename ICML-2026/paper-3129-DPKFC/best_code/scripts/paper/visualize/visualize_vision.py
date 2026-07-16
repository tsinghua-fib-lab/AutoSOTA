"""Publication-quality visualization for vision experiment results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dp_kfac.results import load_results_csv, aggregate_results

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

COLORS = {
    "DP-SGD": "#2E86AB",
    "KFAC (Public)": "#A23B72",
    "KFAC (Pink Noise)": "#F18F01",
    "KFAC (White Noise)": "#76B041",
    "Public KFAC": "#A23B72",
    "Pink KFAC": "#F18F01",
    "Private Oracle KFAC": "#E84855",
}

MARKERS = {
    "DP-SGD": "o",
    "KFAC (Public)": "s",
    "KFAC (Pink Noise)": "^",
    "KFAC (White Noise)": "D",
    "Public KFAC": "s",
    "Pink KFAC": "^",
    "Private Oracle KFAC": "*",
}


def plot_accuracy_vs_epsilon(ax, df, title, baseline_acc=None):
    """Plot accuracy vs epsilon for each method with error bands."""
    methods = [m for m in df["Method"].unique() if m != "Plain SGD"]

    for method in methods:
        method_df = df[df["Method"] == method]
        agg = (
            method_df.groupby("Epsilon")["Accuracy"]
            .agg(["mean", "std"])
            .reset_index()
        )
        agg = agg.sort_values("Epsilon")

        color = COLORS.get(method, "#333333")
        marker = MARKERS.get(method, "o")

        ax.plot(
            agg["Epsilon"],
            agg["mean"],
            color=color,
            marker=marker,
            label=method,
            linewidth=1.5,
            markersize=5,
        )
        ax.fill_between(
            agg["Epsilon"],
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            alpha=0.15,
            color=color,
        )

    if baseline_acc is not None:
        ax.axhline(
            y=baseline_acc,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Non-private",
        )

    ax.set_xlabel("Privacy Budget (\u03b5)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize vision experiment results (MNIST CNN + CrossViT CIFAR-100)."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing results CSV files (default: results).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/figures",
        help="Directory for output figures (default: results/figures).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mnist_path = results_dir / "cnn_mnist_results.csv"
    cifar_path = results_dir / "crossvit_cifar100_results.csv"

    mnist_df = None
    cifar_df = None

    if mnist_path.exists():
        mnist_df = load_results_csv(mnist_path)
    else:
        print(f"Warning: {mnist_path} not found -- skipping MNIST subplot.")

    if cifar_path.exists():
        cifar_df = load_results_csv(cifar_path)
    else:
        print(f"Warning: {cifar_path} not found -- skipping CIFAR-100 subplot.")

    if mnist_df is None and cifar_df is None:
        print("No result files found. Nothing to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: CNN on MNIST
    if mnist_df is not None:
        baseline_rows = mnist_df[mnist_df["Method"] == "Plain SGD"]
        baseline_acc = (
            baseline_rows["Accuracy"].mean() if not baseline_rows.empty else None
        )
        plot_accuracy_vs_epsilon(axes[0], mnist_df, "CNN on MNIST", baseline_acc)
    else:
        axes[0].set_visible(False)

    # Right: CrossViT on CIFAR-100
    if cifar_df is not None:
        baseline_rows = cifar_df[cifar_df["Method"] == "Plain SGD"]
        baseline_acc = (
            baseline_rows["Accuracy"].mean() if not baseline_rows.empty else None
        )
        plot_accuracy_vs_epsilon(
            axes[1], cifar_df, "CrossViT on CIFAR-100", baseline_acc
        )
    else:
        axes[1].set_visible(False)

    fig.tight_layout()

    pdf_path = output_dir / "vision_results.pdf"
    png_path = output_dir / "vision_results.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
