#!/usr/bin/env python3
"""
Generate Figure A2: Three-axis ablation for residual bootstrapping.

This figure shows Δ log₁₀ regret distributions (boxplots) for each variant
relative to CMA-ES baseline across three ablation axes:
  (a) Boundary reevaluation intensity K_max ∈ {0, 1, 3}
  (b) Bootstrap sample count B_boot ∈ {16, 32, 64}
  (c) Heteroscedastic noise model variants

Data source: evidence/berw_*_ablation_fixed_budget/noisefree/bbob_summary.csv
Output: evidence/paper_figures/Appendix/fig_a2_ablations.pdf

Usage:
    python tools/plot_fig_a2_ablations.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _project import BASE_DIR, repo_relpath
from plot_style import apply_style, save_figure, get_subplot_figsize, COLORS, ALGO_COLORS


# Display name mapping per ablation type
DISPLAY_LABELS_BY_ABLATION = {
    "reeval": {
        "BERW-Hetero(reeval=0)": r"$K_{\max}=0$",
        "BERW-Hetero": r"$K_{\max}=1$",
        "BERW-Hetero(reeval=3)": r"$K_{\max}=3$",
    },
    "bootstrap": {
        "BERW-Hetero(bs=16)": r"$B_{\mathrm{boot}}=16$",
        "BERW-Hetero": r"$B_{\mathrm{boot}}=32$",
        "BERW-Hetero(bs=64)": r"$B_{\mathrm{boot}}=64$",
    },
    "hetero": {
        "BERW-Hetero": "Default",
        "BERW-HeteroRobust": "Robust",
        "BERW-HeteroTMatch": "T-Match",
        "BERW-HeteroVar": "Var-model",
    },
}


def get_label(algo: str, ablation_name: str) -> str:
    """Get display label for algorithm."""
    labels = DISPLAY_LABELS_BY_ABLATION.get(ablation_name, {})
    if algo in labels:
        return labels[algo]
    return algo


def load_bbob_summary(csv_path: str) -> pd.DataFrame | None:
    """Load per-instance bbob_summary.csv."""
    if not os.path.isfile(csv_path):
        return None
    return pd.read_csv(csv_path)


def compute_delta_regret(df: pd.DataFrame, baseline_algo: str = "CMA-ES-sep") -> pd.DataFrame:
    """
    Compute Δ log₁₀ regret for each variant relative to baseline.

    Δ = log₁₀(baseline_regret) - log₁₀(variant_regret)
    Positive values mean the variant is better than baseline.
    """
    # Get baseline performance indexed by (function, instance)
    baseline_df = df[df["algorithm"] == baseline_algo]
    if baseline_df.empty:
        return pd.DataFrame()

    baseline = baseline_df.set_index(["function", "instance"])["best_f"]

    results = []
    for algo in df["algorithm"].unique():
        if algo == baseline_algo:
            continue
        algo_data = df[df["algorithm"] == algo]
        for _, row in algo_data.iterrows():
            key = (row["function"], row["instance"])
            if key not in baseline.index:
                continue
            baseline_regret = baseline[key]
            algo_regret = row["best_f"]
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            # Δ = log(baseline) - log(algo): positive means algo is better
            delta = np.log10(baseline_regret + eps) - np.log10(algo_regret + eps)
            results.append({
                "algorithm": algo,
                "function": row["function"],
                "instance": row["instance"],
                "delta_log10_regret": delta
            })
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure A2: Ablations (boxplot)")
    parser.add_argument(
        "--evidence-dir",
        default="evidence",
        help="Evidence directory (relative to repo/)",
    )
    parser.add_argument(
        "--output",
        default="evidence/paper_figures/Appendix/fig_a2_ablations",
        help="Output path (without extension)",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    # Load all three ablation datasets from noisefree/bbob_summary.csv
    ablations = [
        {
            "name": "reeval",
            "title": r"(a) Reevaluation $K_{\max}$",
            "csv": os.path.join(args.evidence_dir, "berw_reeval_ablation_fixed_budget", "noisefree", "bbob_summary.csv"),
            "order": ["BERW-Hetero(reeval=0)", "BERW-Hetero", "BERW-Hetero(reeval=3)"],
        },
        {
            "name": "bootstrap",
            "title": r"(b) Bootstrap samples $B_{\mathrm{boot}}$",
            "csv": os.path.join(args.evidence_dir, "berw_bootstrap_samples_ablation_fixed_budget", "noisefree", "bbob_summary.csv"),
            "order": ["BERW-Hetero(bs=16)", "BERW-Hetero", "BERW-Hetero(bs=64)"],
        },
        {
            "name": "hetero",
            "title": r"(c) Noise model $\mathcal{M}$",
            "csv": os.path.join(args.evidence_dir, "berw_hetero_model_ablation_fixed_budget", "noisefree", "bbob_summary.csv"),
            "order": ["BERW-Hetero", "BERW-HeteroRobust", "BERW-HeteroVar", "BERW-HeteroTMatch"],
        },
    ]

    # Load data and compute delta regret
    all_values = []
    for abl in ablations:
        raw_df = load_bbob_summary(abl["csv"])
        if raw_df is None:
            print(f"WARNING: Missing {repo_relpath(abl['csv'])}")
            abl["data"] = None
            continue
        abl["data"] = compute_delta_regret(raw_df, baseline_algo="CMA-ES-sep")
        if abl["data"].empty:
            print(f"WARNING: No delta regret computed for {repo_relpath(abl['csv'])}")
            abl["data"] = None
        else:
            all_values.extend(abl["data"]["delta_log10_regret"].values)

    # Check if we have any data
    if all(abl["data"] is None for abl in ablations):
        print("ERROR: No ablation data found")
        sys.exit(1)

    # Compute global y-limits for consistency
    all_values = np.array(all_values)
    global_ymin = np.percentile(all_values, 0.5)  # Slightly trim outliers for display
    global_ymax = np.percentile(all_values, 99.5)
    y_range = global_ymax - global_ymin
    global_ymin = global_ymin - 0.1 * y_range
    global_ymax = global_ymax + 0.18 * y_range  # Extra space for annotations

    # Apply style
    apply_style()

    # Create 1×3 figure with shared y-axis
    fig, axes = plt.subplots(1, 3, figsize=get_subplot_figsize(1, 3, width="double", subplot_aspect=0.9),
                             sharey=True)

    box_color = ALGO_COLORS.get("BERW-Hetero", COLORS["blue"])

    for idx, abl in enumerate(ablations):
        ax = axes[idx]
        df = abl["data"]

        if df is None:
            ax.text(0.5, 0.5, "Data missing", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(abl["title"], fontsize=8, pad=4)
            continue

        # Prepare data for boxplot in specified order
        order = abl["order"]
        box_data = []
        labels = []
        for algo in order:
            algo_df = df[df["algorithm"] == algo]
            if algo_df.empty:
                continue
            box_data.append(algo_df["delta_log10_regret"].values)
            labels.append(get_label(algo, abl["name"]))

        if not box_data:
            ax.text(0.5, 0.5, "No matching data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(abl["title"], fontsize=8, pad=4)
            continue

        # Create boxplot
        bp = ax.boxplot(
            box_data,
            positions=range(len(box_data)),
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='o', markersize=2, alpha=0.3, markerfacecolor='gray', markeredgecolor='none'),
        )

        # Style boxes
        for patch in bp['boxes']:
            patch.set_facecolor(box_color)
            patch.set_alpha(0.75)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.5)
        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(0.5)
        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(0.5)
        for median in bp['medians']:
            median.set_color('white')
            median.set_linewidth(1.2)

        # Draw zero line (parity with CMA-ES)
        ax.axhline(y=0, color='#555555', linestyle='--', linewidth=0.8, zorder=1)

        # Set consistent y-limits
        ax.set_ylim(global_ymin, global_ymax)

        # Add win rate annotations at fixed height (consistent across all panels)
        annotation_y = global_ymax - 0.12 * y_range  # Fixed position near top
        for i, data in enumerate(box_data):
            win_rate = (data > 0).mean() * 100
            # Show win rate as primary metric
            ax.text(i, annotation_y, f"{win_rate:.0f}%",
                    ha="center", va="bottom", fontsize=6.5, fontweight='bold', color='#333333')

        # X-axis labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.tick_params(axis='both', labelsize=6)

        # Only show y-label on leftmost panel
        if idx == 0:
            ax.set_ylabel(r"$\Delta \log_{10}$ regret", fontsize=8)

        # Title
        ax.set_title(abl["title"], fontsize=8, pad=6)

        # Clean up spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    saved = save_figure(fig, out_path)
    plt.close(fig)

    print(f"Saved: {', '.join(repo_relpath(p) for p in saved)}")


if __name__ == "__main__":
    main()
