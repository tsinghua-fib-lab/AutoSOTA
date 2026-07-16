"""Visualize covariance tracking over training epochs."""

import sys
import argparse
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


COLORS = {
    "public": "#A23B72",
    "pink_noise": "#F18F01",
}

SOURCE_LABELS = {
    "public": "Public (FashionMNIST)",
    "pink_noise": "Pink Noise",
}

LINESTYLES = ["-", "--", "-.", ":"]

METRIC_TITLES = {
    "cos_sim_A": "Cosine Similarity (A)",
    "cos_sim_G": "Cosine Similarity (G)",
    "frob_rel_A": "Relative Frobenius Norm (A)",
    "frob_rel_G": "Relative Frobenius Norm (G)",
}

METRIC_YLABELS = {
    "cos_sim_A": "Cosine Similarity",
    "cos_sim_G": "Cosine Similarity",
    "frob_rel_A": "Relative Frobenius Norm",
    "frob_rel_G": "Relative Frobenius Norm",
}

POSITIONS = {
    "cos_sim_A": (0, 0),
    "cos_sim_G": (0, 1),
    "frob_rel_A": (1, 0),
    "frob_rel_G": (1, 1),
}

SOURCES = ["public", "pink_noise"]


def _set_pub_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "text.usetex": False,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
        }
    )


def plot_cov_tracking(tracking: list[dict], output_dir: str) -> None:
    """Create and save the 2x2 covariance-tracking figure."""
    _set_pub_style()

    epochs = [entry["epoch"] for entry in tracking]
    layers = list(tracking[0]["public"].keys())

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for metric, (row, col) in POSITIONS.items():
        ax = axes[row, col]

        for src in SOURCES:
            # Average across layers for each epoch
            values = []
            for entry in tracking:
                layer_vals = [entry[src][layer][metric] for layer in layers]
                values.append(np.mean(layer_vals))

            ax.plot(
                epochs,
                values,
                color=COLORS[src],
                label=SOURCE_LABELS[src],
                linewidth=1.5,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(METRIC_YLABELS[metric])
        ax.set_title(METRIC_TITLES[metric])
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig.savefig(out / "cov_tracking.pdf")
    fig.savefig(out / "cov_tracking.png")
    plt.close(fig)

    print(f"Saved: {out / 'cov_tracking.pdf'}")
    print(f"Saved: {out / 'cov_tracking.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize covariance tracking over training epochs"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing cov_tracking_data.pkl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save the output figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.results_dir) / "cov_tracking_data.pkl"
    if not data_path.exists():
        print(
            f"Error: {data_path} not found. "
            "Run ablation_cov_tracking.py first to generate the tracking data."
        )
        sys.exit(1)

    with open(data_path, "rb") as f:
        tracking = pickle.load(f)

    if not tracking:
        print("Error: Tracking data is empty.")
        sys.exit(1)

    print(f"Loaded {len(tracking)} epoch(s) from {data_path}")
    plot_cov_tracking(tracking, args.output_dir)


if __name__ == "__main__":
    main()
