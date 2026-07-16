"""Publication-quality visualization for NLP experiment results."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dp_kfac.results import load_results_csv, aggregate_results

COLORS = {
    "DP-SGD": "#2E86AB",
    "KFAC (Public)": "#A23B72",
    "KFAC (Synthetic)": "#F18F01",
    "AdaDPS": "#76B041",
    "Plain SGD": "gray",
}

MARKERS = {
    "DP-SGD": "o",
    "KFAC (Public)": "s",
    "KFAC (Synthetic)": "^",
    "AdaDPS": "D",
}


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,      # TrueType fonts in PDF
        "ps.fonttype": 42,
    })


def _plot_dataset(ax: plt.Axes, csv_path: Path, title: str) -> None:
    df = load_results_csv(csv_path)
    agg = aggregate_results(df, group_by=["Method", "Epsilon"], metric="Accuracy")

    baseline_rows = agg[agg["Method"] == "Plain SGD"]
    if not baseline_rows.empty:
        baseline_acc = baseline_rows["Accuracy_mean"].values[0]
        ax.axhline(
            baseline_acc,
            color=COLORS.get("Plain SGD", "gray"),
            linestyle="--",
            linewidth=1.2,
            label="Plain SGD (non-private)",
            zorder=1,
        )

    dp_methods = [m for m in agg["Method"].unique() if m != "Plain SGD"]

    for method in dp_methods:
        sub = agg[agg["Method"] == method].sort_values("Epsilon")
        epsilons = sub["Epsilon"].values
        means = sub["Accuracy_mean"].values
        stds = sub["Accuracy_std"].values
        # Replace NaN std (single-seed) with zero
        stds = np.nan_to_num(stds, nan=0.0)

        color = COLORS.get(method, "#333333")
        marker = MARKERS.get(method, "x")

        ax.plot(
            epsilons,
            means,
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.5,
            label=method,
            zorder=3,
        )
        ax.fill_between(
            epsilons,
            means - stds,
            means + stds,
            color=color,
            alpha=0.15,
            zorder=2,
        )

    ax.set_title(title)
    ax.set_xlabel(r"Privacy budget $\varepsilon$")
    ax.set_ylabel("Accuracy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)


DATASETS = [
    ("imdb_logreg_results.csv", "IMDB Logistic Regression"),
    ("stackoverflow_results.csv", "StackOverflow"),
    ("sst2_results.csv", "SST-2"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publication-quality NLP results visualization"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing the result CSV files (default: results)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures",
        help="Directory to save the output figures (default: figures)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    available = []
    for csv_name, title in DATASETS:
        csv_path = results_dir / csv_name
        if csv_path.is_file():
            available.append((csv_path, title))
        else:
            print(f"[WARN] {csv_path} not found -- skipping {title}")

    if not available:
        print("[ERROR] No result CSV files found. Nothing to plot.")
        return

    _setup_style()

    n_panels = len(available)
    fig_width = 14 if n_panels == 3 else (9.5 if n_panels == 2 else 5)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 4))

    # Ensure axes is always iterable
    if n_panels == 1:
        axes = [axes]

    for ax, (csv_path, title) in zip(axes, available):
        _plot_dataset(ax, csv_path, title)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)
    for ax in axes[1:]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)

    fig.legend(
        unique_handles,
        unique_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(len(unique_labels), 5),
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgray",
    )

    plt.tight_layout(rect=(0, 0.06, 1, 1))

    pdf_path = output_dir / "nlp_results.pdf"
    png_path = output_dir / "nlp_results.png"
    fig.savefig(str(pdf_path), format="pdf")
    fig.savefig(str(png_path), format="png")
    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
