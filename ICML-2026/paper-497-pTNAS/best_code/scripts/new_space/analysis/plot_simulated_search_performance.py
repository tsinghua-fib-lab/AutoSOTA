#!/usr/bin/env python3
"""Plot simulated pTNAS search performance over explored architectures M."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "run_outputs/data/new_space/simulated_search_performance_5seeds_detail.csv"
DEFAULT_OUTPUT_DIR = ROOT / "run_outputs/data/new_space/search_performance_figs"

DATASET_ORDER = [
    "avito-ad-ctr",
    "event-user-attendance",
    "avito-user-clicks",
    "hm-user-churn",
]

SPACE_STYLE = {
    "ResDNN": {"color": "#3B7EA1", "marker": "o"},
    "BlockMixed": {"color": "#D17842", "marker": "s"},
}


def plot_legend(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.25, 0.35))
    handles = []
    labels = []
    for space, style in SPACE_STYLE.items():
        (line,) = ax.plot(
            [],
            [],
            label=space,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=4.0,
        )
        handles.append(line)
        labels.append(space)
    ax.axis("off")
    legend = fig.legend(handles, labels, loc="center", ncol=2, frameon=True, fontsize=8)
    legend.get_frame().set_edgecolor("#BDBDBD")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "legend.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_dataset(df: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    sub = df[df["dataset"] == dataset].copy()
    metric = sub["metric"].iloc[0]
    x_max = sub[sub["space"] == "BlockMixed"]["M"].max()

    fig, ax = plt.subplots(figsize=(2.35, 1.75), constrained_layout=True)

    for space, style in SPACE_STYLE.items():
        line = sub[sub["space"] == space].sort_values("M")
        line = line[line["M"] <= x_max]
        if line.empty:
            continue
        ax.plot(
            line["M"],
            line["mean_best_so_far"],
            label=space,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.65,
            markersize=3.5,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlim(4.6, x_max * 1.08)
    ax.set_xlabel("Explored architectures M", fontsize=7.6)
    ax.set_ylabel(f"Mean performance ({metric})", fontsize=7.6)
    ax.grid(True, which="major", color="#DCDCDC", linewidth=0.6, alpha=0.9)
    ax.grid(True, which="minor", color="#F0F0F0", linewidth=0.45, alpha=0.65)
    ax.tick_params(axis="both", labelsize=7.1)

    direction = "lower is better" if metric == "MAE" else "higher is better"
    ax.text(0.04, 0.06, direction, transform=ax.transAxes, fontsize=6.1, color="#5A5A5A")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset.replace("-", "_")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def summarize_detail(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["space", "dataset", "metric", "M"], as_index=False)
        .agg(mean_best_so_far=("best_so_far", "mean"))
    )


def plot(input_csv: Path, output_dir: Path) -> None:
    df = summarize_detail(pd.read_csv(input_csv))
    df = df[df["dataset"].isin(DATASET_ORDER)].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    for png in output_dir.glob("*.png"):
        png.unlink()

    for dataset in DATASET_ORDER:
        plot_dataset(df, dataset, output_dir)
    plot_legend(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    input_csv = args.input_csv if args.input_csv.is_absolute() else ROOT / args.input_csv
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    plot(input_csv=input_csv, output_dir=output_dir)
    for path in sorted(output_dir.glob("*.pdf")):
        print(f"Saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
