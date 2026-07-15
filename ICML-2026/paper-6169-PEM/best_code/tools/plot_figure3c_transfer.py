#!/usr/bin/env python3
"""
Figure 3(c): Zero-tuning threshold transfer lollipop chart.

Shows improvement in regret (Δregret = always-CMA − transfer) for τ=0.12 and τ=0.22
across multiple target tasks.

Output:
  evidence/paper_figures/figure3c_transfer.(pdf|png)
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from _project import BASE_DIR, repo_relpath
from plot_style import apply_style, save_figure, WIDTHS, add_grid, COLORS


def plot_transfer_lollipop(ax: plt.Axes, evidence_dir: str) -> None:
    transfer_csv = os.path.join(evidence_dir, "probeswitch_transfer_overhead_summary/transfer_summary_compact.csv")
    if not os.path.isfile(transfer_csv):
        ax.text(0.5, 0.5, "Missing transfer_summary_compact.csv", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    rows: list[dict[str, str]] = []
    with open(transfer_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Targets to exclude (boundary cases less relevant to main argument).
    EXCLUDE_TARGETS = {"bbob_B200_d10", "bbob_B200_d20"}

    # Compute deltas for both thresholds: τ=0.12 and τ=0.22.
    # delta = always_cma_regret - transfer_regret (positive = transfer improves).
    lollipop_data: list[dict] = []
    for r in rows:
        target = str(r.get("target", ""))
        if target in EXCLUDE_TARGETS:
            continue
        try:
            always_reg = float(r.get("always_cma_regret_mean", "nan"))
            transfer_reg_012 = float(r.get("bbob_B500_regret_mean", "nan"))
            transfer_reg_022 = float(r.get("safe_regret_mean", "nan"))
        except (ValueError, TypeError):
            continue
        if not np.isfinite(always_reg):
            continue
        delta_012 = always_reg - transfer_reg_012 if np.isfinite(transfer_reg_012) else float("nan")
        delta_022 = always_reg - transfer_reg_022 if np.isfinite(transfer_reg_022) else float("nan")
        label = str(r.get("target_label", r.get("target", "?")))
        # Clean up label text.
        label = label.replace("(BC)", "(breast cancer)")
        label = label.replace("COCO", "BBOB")
        lollipop_data.append({
            "target": target,
            "label": label,
            "delta_012": delta_012,
            "delta_022": delta_022,
        })

    if not lollipop_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Sort by delta_012 descending (best improvement at top).
    all_items = sorted(lollipop_data, key=lambda d: d["delta_012"], reverse=True)

    # Draw dual lollipops: τ=0.12 (blue, upper) and τ=0.22 (gray, lower).
    y_offset = 0.15
    color_012 = COLORS["blue"]
    color_022 = "#888888"

    labels: list[str] = []
    for i, item in enumerate(all_items):
        y_base = i
        labels.append(item["label"])

        # τ=0.12 (upper line).
        d_012 = item["delta_012"]
        if np.isfinite(d_012):
            ax.hlines(y_base - y_offset, 0, d_012, color=color_012, linewidth=1.5)
            ax.plot(d_012, y_base - y_offset, "o", color=color_012, markersize=4)

        # τ=0.22 (lower line).
        d_022 = item["delta_022"]
        if np.isfinite(d_022):
            ax.hlines(y_base + y_offset, 0, d_022, color=color_022, linewidth=1.5)
            ax.plot(d_022, y_base + y_offset, "s", color=color_022, markersize=3.5)

    # Zero reference line.
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)

    # Move x-axis label to top.
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel(r"regret reduction (vs. CMA-ES baseline)", fontsize=7)
    ax.tick_params(axis='x', labelsize=5)

    # Legend for dual thresholds.
    legend_elements = [
        Line2D([0], [0], marker='o', color=color_012, lw=1.5, label=r'$\tau=0.12$', ms=4),
        Line2D([0], [0], marker='s', color=color_022, lw=1.5, label=r'$\tau=0.22$', ms=3.5),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=6, frameon=False)

    ax.invert_yaxis()
    add_grid(ax, axis="x", alpha=0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Figure 3(c) transfer lollipop plot")
    parser.add_argument(
        "--evidence-dir",
        default=os.path.join(BASE_DIR, "evidence"),
        help="Path to evidence directory",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(BASE_DIR, "evidence/paper_figures/figure3c_transfer"),
        help="Output path prefix (without extension)",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)
    apply_style()

    fig, ax = plt.subplots(figsize=(WIDTHS["single"], 2.8))

    plot_transfer_lollipop(ax, str(args.evidence_dir))

    fig.tight_layout()
    save_figure(fig, str(args.output))
    plt.close(fig)

    print("Wrote:", repo_relpath(str(args.output) + ".pdf"))
    print("Wrote:", repo_relpath(str(args.output) + ".png"))


if __name__ == "__main__":
    main()
