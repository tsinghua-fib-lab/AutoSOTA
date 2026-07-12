#!/usr/bin/env python3
"""
Plot synthetic-workload end-to-end results as a figure.

This is intended to replace the two LaTeX tables:
  - RTX PRO 6000 (Qwen3-8B)
  - H200 (Qwen3-8B)

Outputs a vector PDF (and optional PNG) summarizing RPS / ITL / TTFT.

Usage:
  python plot_synthetic_e2e_figure.py --out /path/to/fig_synthetic_e2e.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def configure_style() -> None:
    # Consistent, paper-friendly styling.
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["axes.titlesize"] = 17
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["lines.linewidth"] = 2.5
    plt.rcParams["lines.markersize"] = 7
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def main() -> None:
    configure_style()

    parser = argparse.ArgumentParser(
        description="Plot synthetic e2e tables as a figure (vector PDF).")
    parser.add_argument(
        "--out",
        type=str,
        default="fig_synthetic_e2e.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--also-png",
        action="store_true",
        help="Also write a PNG next to the PDF.",
    )
    args = parser.parse_args()
    out_pdf = Path(args.out)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    workloads = ["Decode-heavy", "Balanced", "Prefill-heavy"]
    schedulers = ["v1", "v0", "Ours"]

    # Values copied from the tables in the paper draft.
    # RTX PRO 6000: ITL/TTFT in seconds.
    rtx = {
        "RPS": {
            "v0": [5.58, 8.37, 14.10],
            "v1": [5.36, 7.72, 10.36],
            "Ours": [6.04, 9.21, 14.70],
        },
        "ITL (s)": {
            "v0": [0.096, 0.122, 0.233],
            "v1": [0.046, 0.063, 0.190],
            "Ours": [0.083, 0.083, 0.138],
        },
        "TTFT (s)": {
            "v0": [196.9, 136.0, 90.1],
            "v1": [243.3, 172.1, 132.5],
            "Ours": [179.8, 131.7, 92.2],
        },
    }

    # H200: TTFT is in ms in the draft table (values look like ms); ITL is in seconds.
    # Keep labels as in the table caption, but reflect the units per column header.
    h200 = {
        "RPS": {
            "v0": [15.63, 20.83, 28.32],
            "v1": [13.85, 20.26, 28.55],
            "Ours": [13.58, 20.41, 28.72],
        },
        "ITL (s)": {
            "v0": [0.066, 0.049, 0.174],
            "v1": [0.067, 0.047, 0.169],
            "Ours": [0.063, 0.059, 0.107],
        },
        "TTFT (ms)": {
            "v0": [47.2, 55.4, 41.1],
            "v1": [54.2, 57.2, 40.6],
            "Ours": [60.8, 51.9, 44.6],
        },
    }

    panels = [("RTX PRO 6000", rtx), ("H200", h200)]
    metric_names = ["RPS", "ITL (s)", "TTFT (s)", "TTFT (ms)"]  # used for ordering

    # Define metrics per panel to avoid mismatched units.
    panel_metrics = [
        ["RPS", "ITL (s)", "TTFT (s)"],
        ["RPS", "ITL (s)", "TTFT (ms)"],
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(12.5, 7.4),
        sharex=True,
        constrained_layout=True,
    )

    colors = {"v0": "#7f7f7f", "v1": "#d62728", "Ours": "#1f77b4"}
    display_labels = {"v0": r"v0", "v1": r"v1", "Ours": r"EB($\hat{k}^*$)"}
    x = np.arange(len(workloads))
    width = 0.26
    offsets = {"v1": -width, "v0": 0.0, "Ours": width}

    for row, (title, data) in enumerate(panels):
        for col, metric in enumerate(panel_metrics[row]):
            ax = axes[row, col]
            for s in schedulers:
                vals = np.array(data[metric][s], dtype=float)
                ax.bar(
                    x + offsets[s],
                    vals,
                    width=width,
                    label=display_labels[s] if (row == 0 and col == 0) else None,
                    color=colors[s],
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=0.4,
                )

            # Titles/labels.
            if row == 0:
                ax.set_title(metric, pad=6)
            if col == 0:
                ax.set_ylabel(title)

            # Grid for readability.
            ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)

            # Bold best within each workload for this metric:
            # - RPS: higher is better
            # - ITL / TTFT: lower is better
            higher_is_better = metric == "RPS"
            for i_wl in range(len(workloads)):
                candidates = {s: float(data[metric][s][i_wl]) for s in schedulers}
                best_s = (max if higher_is_better else min)(candidates, key=candidates.get)
                best_v = candidates[best_s]
                # Put a small marker above the best bar.
                ax.text(
                    x[i_wl] + offsets[best_s],
                    best_v,
                    "★",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="black",
                    clip_on=True,
                )

    # Shared x tick labels.
    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(workloads, rotation=0, ha="center")

    # Single legend.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        # Keep legend above column titles to avoid overlap.
        bbox_to_anchor=(0.5, 1.08),
    )

    fig.savefig(out_pdf, bbox_inches="tight")
    if args.also_png:
        fig.savefig(out_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()

