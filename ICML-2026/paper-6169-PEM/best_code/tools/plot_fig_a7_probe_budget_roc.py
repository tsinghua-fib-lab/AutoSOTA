#!/usr/bin/env python3
"""
Generate Figure A7: Probe reliability versus probe budget.

This figure shows how probe reliability (AUC and accuracy) scales with probe
budget (population size lambda). Supports claim C5 by demonstrating that the
probe is informative even at low cost, with diminishing returns at higher budgets.

Data source:
  - evidence/bbob_noisy_probe_budget_roc/roc.csv
  - evidence/bbob_noisy_probe_budget_roc/summary.json

Output: evidence/paper_figures/Appendix/fig_a7_probe_budget_roc.pdf

Usage:
    python tools/plot_fig_a7_probe_budget_roc.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _project import BASE_DIR, repo_relpath
from plot_style import apply_style, save_figure, get_figsize, add_grid


def load_summary(path: str) -> dict:
    """Load summary.json with AUC and accuracy by lambda."""
    with open(path) as f:
        return json.load(f)


def load_roc_csv(path: str) -> dict[int, list[dict]]:
    """Load roc.csv and group by lambda."""
    by_lam: dict[int, list[dict]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lam = int(row["lam"])
            if lam not in by_lam:
                by_lam[lam] = []
            by_lam[lam].append({
                "threshold": float(row["threshold"]),
                "tpr": float(row["tpr"]),
                "fpr": float(row["fpr"]),
                "accuracy": float(row["accuracy"]),
            })
    return by_lam


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure A7: Probe budget vs reliability"
    )
    parser.add_argument(
        "--evidence-dir",
        default="evidence/bbob_noisy_probe_budget_roc",
        help="Evidence directory containing roc.csv and summary.json",
    )
    parser.add_argument(
        "--output",
        default="evidence/paper_figures/Appendix/fig_a7_probe_budget_roc",
        help="Output path (without extension)",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    summary_path = os.path.join(args.evidence_dir, "summary.json")
    roc_path = os.path.join(args.evidence_dir, "roc.csv")

    if not os.path.isfile(summary_path):
        raise SystemExit(f"Missing: {repo_relpath(summary_path)}")
    if not os.path.isfile(roc_path):
        raise SystemExit(f"Missing: {repo_relpath(roc_path)}")

    summary = load_summary(summary_path)
    roc_data = load_roc_csv(roc_path)

    by_lam = summary.get("by_lam", {})
    lam_list = sorted(int(k) for k in by_lam.keys())

    if not lam_list:
        raise SystemExit("No lambda values found in summary.json")

    # Extract AUC and accuracy at report threshold for each lambda
    aucs = []
    accs = []
    for lam in lam_list:
        lam_data = by_lam[str(lam)]
        aucs.append(float(lam_data["auc"]))
        accs.append(float(lam_data["accuracy_at_report_threshold"]))

    print(f"Lambda values: {lam_list}")
    print(f"AUC: {aucs}")
    print(f"Accuracy @ tau=0.12: {accs}")

    # Apply style and create single figure
    apply_style()

    fig, ax = plt.subplots(figsize=get_figsize("single", aspect=1.0))

    # Distinct colors for each lambda
    roc_colors = ["#a6cee3", "#1f78b4", "#33a02c", "#e31a1c"]  # light blue, dark blue, green, red
    line_widths = [0.7, 0.8, 0.9, 1.0]

    for idx, lam in enumerate(lam_list):
        if lam not in roc_data:
            continue
        points = roc_data[lam]
        fprs = [p["fpr"] for p in points]
        tprs = [p["tpr"] for p in points]

        # Sort by FPR for proper curve
        order = np.argsort(fprs)
        fprs_sorted = np.array(fprs)[order]
        tprs_sorted = np.array(tprs)[order]

        auc_val = by_lam[str(lam)]["auc"]
        acc_val = by_lam[str(lam)]["accuracy_at_report_threshold"]

        # Add extra space after single-digit λ for alignment
        if lam < 10:
            label = rf"$\lambda$={lam},   AUC={auc_val:.2f}, Acc={acc_val:.2f}"
        else:
            label = rf"$\lambda$={lam}, AUC={auc_val:.2f}, Acc={acc_val:.2f}"

        ax.plot(
            fprs_sorted, tprs_sorted,
            linewidth=line_widths[idx],
            color=roc_colors[idx],
            label=label,
            zorder=2 + idx,
        )

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=0.5, alpha=0.7)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate", fontsize=8)
    ax.set_ylabel("True positive rate", fontsize=8)
    ax.tick_params(axis="both", labelsize=6)
    ax.set_aspect("equal", adjustable="box")

    ax.legend(loc="lower right", fontsize=6.5, framealpha=0.95)
    add_grid(ax, alpha=0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save figure
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    saved = save_figure(fig, out_path)
    plt.close(fig)

    print(f"Saved: {', '.join(repo_relpath(p) for p in saved)}")


if __name__ == "__main__":
    main()
