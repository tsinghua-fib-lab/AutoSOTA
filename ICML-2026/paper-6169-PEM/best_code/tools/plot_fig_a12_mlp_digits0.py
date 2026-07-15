#!/usr/bin/env python3
"""
Generate Figure A12: MLP digits0 external validity experiment.

Demonstrates probe-and-switch effectiveness on nonconvex real data (MLP training).
Shows that Warmstart achieves zero overhead in deterministic regime (batch=256).
"""

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_style, get_figsize, save_figure, ALGO_COLORS

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "Results" / "exp_mlp_digits0_heavytail_sigma1p0_h4_N256_B40_seeds1_50"
EVIDENCE_DIR = BASE_DIR / "evidence" / "application_mlp_minibatch_digits0_heavytail_sigma1p0"
PAPER_FIGURES_DIR = BASE_DIR / "evidence" / "paper_figures" / "Appendix"
LATEX_FIGURES_DIR = BASE_DIR / "docs" / "paper" / "final" / "figures"

# Algorithm display names and colors
ALGO_ORDER = ["CMA-ES", "ProbeSwitch-Noise", "ProbeSwitch-Noise-Warmstart"]
DISPLAY_NAMES = {
    "CMA-ES": "CMA-ES",
    "ProbeSwitch-Noise": "Probe-and-switch",
    "ProbeSwitch-Noise-Warmstart": "P&S (warmstart)",
}
COLORS = {
    "CMA-ES": "#CC3311",           # Red
    "ProbeSwitch-Noise": "#009988",  # Teal
    "ProbeSwitch-Noise-Warmstart": "#0077BB",  # Blue
}

# Batch sizes to plot
BATCH_SIZES = [4, 16, 256]


def load_data(batch_size: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load runs, pairwise test, and probe data for a batch size."""
    runs_path = RESULTS_DIR / f"batch_{batch_size}" / "runs.csv"
    pairwise_path = RESULTS_DIR / f"batch_{batch_size}" / "pairwise_sign_test_post_true.csv"
    probe_path = EVIDENCE_DIR / "probe_values.csv"

    runs = pd.read_csv(runs_path)
    pairwise = pd.read_csv(pairwise_path)
    probe = pd.read_csv(probe_path)

    return runs, pairwise, probe


def get_win_loss_tie(pairwise: pd.DataFrame, algo_a: str, algo_b: str) -> tuple[int, int, int, float]:
    """Get win-loss-tie counts and p-value for algo_a vs algo_b from pairwise test data.

    Returns (wins_a, wins_b, ties, p_value) where wins_a is how many times algo_a beat algo_b.
    """
    mask = (pairwise["algo_a"] == algo_a) & (pairwise["algo_b"] == algo_b)
    if mask.sum() == 0:
        # Try reversed
        mask = (pairwise["algo_a"] == algo_b) & (pairwise["algo_b"] == algo_a)
        if mask.sum() == 0:
            return 0, 0, 0, 1.0
        row = pairwise[mask].iloc[0]
        # Reversed: wins_b for algo_a
        return int(row["wins_b"]), int(row["wins_a"]), int(row["ties"]), float(row["p_two_sided"])

    row = pairwise[mask].iloc[0]
    return int(row["wins_a"]), int(row["wins_b"]), int(row["ties"]), float(row["p_two_sided"])


def get_misranking_range(probe: pd.DataFrame, batch_size: int) -> tuple[float, float]:
    """Get min and max misranking_rd for a batch size."""
    subset = probe[probe["batch_size"] == batch_size]
    if len(subset) == 0:
        return 0.0, 0.0
    return subset["misranking_rd"].min(), subset["misranking_rd"].max()


def format_winrate_with_significance(wins: int, losses: int, ties: int, p_value: float) -> str:
    """Format win-or-tie rate as percentage.

    Returns percentage of times the algorithm was >= CMA-ES (wins + ties).
    """
    total = wins + losses + ties
    if total == 0:
        return ""

    # Probability of being >= CMA-ES (wins + ties)
    win_or_tie_rate = (wins + ties) / total * 100

    return f"{win_or_tie_rate:.0f}%"


def main():
    apply_style()

    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))

    for ax_idx, batch_size in enumerate(BATCH_SIZES):
        ax = axes[ax_idx]
        runs, pairwise, probe = load_data(batch_size)

        # Prepare boxplot data
        box_data = []
        positions = []
        colors = []
        labels = []

        for i, algo in enumerate(ALGO_ORDER):
            algo_data = runs[runs["algorithm"] == algo]["post_true"].values
            box_data.append(algo_data)
            positions.append(i)
            colors.append(COLORS[algo])
            labels.append(DISPLAY_NAMES[algo])

        # Create boxplots
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=1.2),
            whiskerprops=dict(color="black", linewidth=0.6),
            capprops=dict(color="black", linewidth=0.6),
            flierprops=dict(marker="o", markersize=2, markerfacecolor="gray", alpha=0.5),
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.6)

        # Get misranking range
        mr_min, mr_max = get_misranking_range(probe, batch_size)

        # Add win rate annotations (algorithm vs CMA-ES)
        # Note: get_win_loss_tie returns (cma_wins, algo_wins, ties, p_value) when called with "CMA-ES" first
        cma_wins_ps, ps_wins, ties_ps, p_ps = get_win_loss_tie(pairwise, "CMA-ES", "ProbeSwitch-Noise")
        cma_wins_ws, ws_wins, ties_ws, p_ws = get_win_loss_tie(pairwise, "CMA-ES", "ProbeSwitch-Noise-Warmstart")

        # Set y-axis limit first to position annotations correctly
        y_max = max(max(d) for d in box_data if len(d) > 0)
        y_upper = y_max * 1.15  # Leave room for annotations
        ax.set_ylim(bottom=0, top=y_upper)

        # Position annotations inside plot area, near top
        y_annot = y_max * 1.02

        # Annotation for CMA-ES (reference baseline)
        ax.text(0, y_annot, "ref", ha="center", va="bottom", fontsize=7, fontstyle="italic", color="#666666")

        # Annotation for ProbeSwitch vs CMA-ES (above ProbeSwitch box)
        # ps_wins is how many times ProbeSwitch beat CMA-ES
        ps_text = format_winrate_with_significance(ps_wins, cma_wins_ps, ties_ps, p_ps)
        if ps_text:
            ax.text(1, y_annot, ps_text, ha="center", va="bottom", fontsize=7, fontweight="bold")

        # Annotation for Warmstart vs CMA-ES (above Warmstart box)
        ws_text = format_winrate_with_significance(ws_wins, cma_wins_ws, ties_ws, p_ws)
        if ws_text:
            ax.text(2, y_annot, ws_text, ha="center", va="bottom", fontsize=7, fontweight="bold")

        # Set axis properties
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="center")
        ax.set_ylabel("Post-hoc loss" if ax_idx == 0 else "")
        ax.set_title(f"$B_{{\\mathrm{{batch}}}} = {batch_size}$", fontsize=9)
        ax.tick_params(axis='both', labelsize=6)

        # Add misranking annotation below
        if mr_max > 0:
            mr_text = f"$M_{{RD}} \\in [{mr_min:.2f}, {mr_max:.2f}]$"
        else:
            mr_text = "$M_{RD} = 0$ (deterministic)"
        ax.text(0.5, -0.22, mr_text, transform=ax.transAxes,
                ha="center", va="top", fontsize=7)

        # Spine formatting
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, wspace=0.3)

    # Ensure output directories exist
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Save figures
    output_base = PAPER_FIGURES_DIR / "fig_a12_mlp_digits0"
    saved = save_figure(fig, str(output_base))
    print(f"Saved: {saved}")

    # Copy to LaTeX directory
    for ext in [".pdf", ".png"]:
        src = PAPER_FIGURES_DIR / f"fig_a12_mlp_digits0{ext}"
        dst = LATEX_FIGURES_DIR / f"fig_a12_mlp_digits0{ext}"
        if src.exists():
            shutil.copy(src, dst)
            print(f"Copied to: {dst}")

    plt.close(fig)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for batch_size in BATCH_SIZES:
        runs, pairwise, probe = load_data(batch_size)
        mr_min, mr_max = get_misranking_range(probe, batch_size)

        print(f"\nBatch = {batch_size}:")
        print(f"  Misranking range: [{mr_min:.3f}, {mr_max:.3f}]")

        for _, row in pairwise.iterrows():
            print(f"  {row['algo_a']} vs {row['algo_b']}: "
                  f"{int(row['wins_a'])}W-{int(row['wins_b'])}L-{int(row['ties'])}T "
                  f"(p={row['p_two_sided']:.4f})")


if __name__ == "__main__":
    main()
