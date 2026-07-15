#!/usr/bin/env python3
"""
Generate Figure A10: Depth-fidelity trade-off analysis (merged A9+A10).

This figure demonstrates the core depth-fidelity trade-off from two perspectives:
(a) Budget robustness: Residual Bootstrapping beats UH-CMA-ES across all budgets
(b) maxevals sensitivity: UH-CMA-ES loses regardless of maxevals configuration

Key message: Investing evaluations in fitness estimation (resampling) hurts
performance under fixed budget. The NoiseHandler mechanism is a liability.

Data sources:
- Panel (a): evidence/hansen_test_fixed_budget_grid/budget_grid_summary.csv
- Panel (b): evidence/bbob_noisy_uh_cmaes_maxevals_sweep_d40_f1-30_i1-15/bbob_summary.csv

Output: evidence/paper_figures/Appendix/fig_a10_depth_fidelity_tradeoff.pdf
"""

from __future__ import annotations

import os
import sys

# Add parent directory to path for plot_style import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_style import apply_style, save_figure, WIDTHS, COLORS


def load_budget_grid_summary(data_dir: str) -> pd.DataFrame:
    """Load budget_grid_summary.csv from given directory."""
    path = os.path.join(data_dir, "budget_grid_summary.csv")
    return pd.read_csv(path)


def load_bbob_summary(data_dir: str) -> pd.DataFrame:
    """Load bbob_summary.csv from given directory."""
    path = os.path.join(data_dir, "bbob_summary.csv")
    return pd.read_csv(path)


def extract_comparison_data(df: pd.DataFrame, algo_b: str) -> pd.DataFrame:
    """Extract win rates for a specific comparison across budgets."""
    filtered = df[df["algo_b"] == algo_b].copy()
    filtered = filtered.sort_values("budget_mult")
    return filtered


def compute_win_rate(pivot: pd.DataFrame, algo_a: str, algo_b: str) -> tuple[float, int, int]:
    """
    Compute win rate of algo_a over algo_b.
    Returns: (win_rate, wins, total)
    """
    wins = (pivot[algo_a] < pivot[algo_b]).sum()
    total = len(pivot)
    return wins / total, wins, total


def main():
    apply_style()

    # Data directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(repo_dir)

    # Panel (a) data - budget grid
    d40_dir = os.path.join(repo_dir, "evidence", "hansen_test_fixed_budget_grid")
    d20_dir = os.path.join(repo_dir, "evidence", "hansen_test_fixed_budget_grid_d20")

    # Panel (b) data - maxevals sweep
    maxevals_dir = os.path.join(project_root, "evidence", "bbob_noisy_uh_cmaes_maxevals_sweep_d40_f1-30_i1-15")

    # Load data
    df_d40 = load_budget_grid_summary(d40_dir)
    df_d20 = load_budget_grid_summary(d20_dir)
    df_maxevals = load_bbob_summary(maxevals_dir)

    # Create 1x2 figure
    fig_width = WIDTHS["double"]
    fig_height = fig_width * 0.38
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    # =========================================================================
    # Panel (a): Budget grid robustness
    # =========================================================================
    ax_a = axes[0]

    # Define comparisons to plot (algo_b, label, color)
    comparisons_a = [
        ("UH-CMA-ES(maxevals=30)", "vs UH-CMA-ES", COLORS["blue"]),
        ("CMA-ES-Resample(k=10)", "vs Resample", COLORS["green"]),
        ("CMA-ES-sep", "vs CMA-ES", COLORS["red"]),
    ]

    budgets = [50, 100, 200]

    # Add shaded "Res.Boot. wins" region above 50%
    ax_a.fill_between([35, 215], 0.5, 1.0, color='#eef6ee', alpha=0.5, zorder=0)

    # Plot each comparison with D=40 (filled) and D=20 (open)
    for algo_b, label, color in comparisons_a:
        data_d40 = extract_comparison_data(df_d40, algo_b)
        data_d20 = extract_comparison_data(df_d20, algo_b)

        # D=40: filled markers, solid line
        ax_a.plot(data_d40["budget_mult"], data_d40["win_rate_a"],
                  color=color, linewidth=1.2, linestyle="-",
                  marker="o", markersize=5, markerfacecolor=color, zorder=3)

        # D=20: open markers, dashed line (thicker, more visible)
        ax_a.plot(data_d20["budget_mult"], data_d20["win_rate_a"],
                  color=color, linewidth=1.5, linestyle="--", dashes=(6, 3),
                  marker="o", markersize=5, markerfacecolor="white", markeredgecolor=color,
                  markeredgewidth=1.3, zorder=3)

    # Parity line
    ax_a.axhline(y=0.5, color=COLORS["grey"], linewidth=0.8, linestyle=":", zorder=1)

    ax_a.set_xlim(40, 210)
    ax_a.set_ylim(0.46, 1.02)
    ax_a.set_xticks(budgets)
    ax_a.set_xticklabels([f"{b}$D$" for b in budgets], fontsize=6)
    ax_a.set_xlabel("Budget (evaluations)", fontsize=7)
    ax_a.set_ylabel("Win rate of Res. Bootstrapping", fontsize=7)
    ax_a.tick_params(axis='both', labelsize=6)

    # Y-axis as percentage
    ax_a.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_a.set_yticklabels(['50%', '60%', '70%', '80%', '90%', '100%'], fontsize=6)

    ax_a.grid(True, alpha=0.2, linewidth=0.4, color="#888888")
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # Legend for panel (a) - two separate groups for clarity
    # Group 1: comparison baselines (colors)
    leg_colors = [
        Line2D([0], [0], color=COLORS["blue"], linewidth=1.2, label="vs UH-CMA-ES"),
        Line2D([0], [0], color=COLORS["green"], linewidth=1.2, label="vs Resample"),
        Line2D([0], [0], color=COLORS["red"], linewidth=1.2, label="vs CMA-ES"),
    ]
    # Group 2: dimension styles (thicker dashed for clarity)
    leg_styles = [
        Line2D([0], [0], color="#555555", linestyle="-", marker="o", markersize=4,
               markerfacecolor="#555555", linewidth=1.0, label="$D{=}40$"),
        Line2D([0], [0], color="#555555", linestyle="--", dashes=(5, 2.5), marker="o", markersize=4,
               markerfacecolor="white", markeredgecolor="#555555", markeredgewidth=1.2,
               linewidth=1.5, label="$D{=}20$"),
    ]
    leg1 = ax_a.legend(handles=leg_colors, fontsize=5.5, loc="upper left",
                       framealpha=0.95, handlelength=1.8, handletextpad=0.4, borderpad=0.35,
                       labelspacing=0.3, bbox_to_anchor=(0.0, 0.98))
    ax_a.add_artist(leg1)
    ax_a.legend(handles=leg_styles, fontsize=5.5, loc="upper left",
                framealpha=0.95, handlelength=1.8, handletextpad=0.4, borderpad=0.35,
                labelspacing=0.3, bbox_to_anchor=(0.28, 0.98))

    ax_a.set_title("(a) Budget robustness", fontsize=8, fontweight="normal", ha="center")

    # =========================================================================
    # Panel (b): UH-CMA-ES maxevals sensitivity
    # =========================================================================
    ax_b = axes[1]

    # UH-CMA-ES variants
    uh_variants = [
        ("UH-CMA-ES", "m=1"),
        ("UH-CMA-ES(maxevals=10)", "m=10"),
        ("UH-CMA-ES(maxevals=30)", "m=30"),
    ]
    maxevals_values = [1, 10, 30]

    # Baselines to compare against
    baselines_b = [
        ("CMA-ES", "vs CMA-ES", "#CC3311"),
        ("ProbeSwitch-MR(t=0.12)", "vs ProbeSwitch", "#009988"),
    ]

    budget_list = [200, 500]
    # Use filled/open circles instead of squares
    budget_styles = {
        200: {"linestyle": "-", "alpha": 1.0, "markerfacecolor": None},  # filled
        500: {"linestyle": "--", "dashes": (6, 3), "alpha": 1.0, "markerfacecolor": "white"},  # open
    }

    # Compute and plot win rates
    for baseline_algo, baseline_label, baseline_color in baselines_b:
        for budget in budget_list:
            subset = df_maxevals[df_maxevals['budget_multiplier'] == budget]
            pivot = subset.pivot_table(index=['function', 'instance'], columns='algorithm', values='best_f')

            win_rates = []
            for uh_algo, uh_label in uh_variants:
                wr, wins, total = compute_win_rate(pivot, uh_algo, baseline_algo)
                win_rates.append(wr)

            style = budget_styles[budget]

            # Determine marker fill
            if style["markerfacecolor"] is None:
                mfc = baseline_color
                mew = 1.0
            else:
                mfc = "white"
                mew = 1.2

            ax_b.plot(maxevals_values, win_rates,
                      color=baseline_color,
                      linestyle=style["linestyle"],
                      dashes=style.get("dashes", (None, None)) if style["linestyle"] == "--" else (None, None),
                      alpha=style["alpha"],
                      marker="o",
                      markersize=5,
                      markerfacecolor=mfc,
                      markeredgecolor=baseline_color,
                      markeredgewidth=mew,
                      linewidth=1.5 if style["linestyle"] == "--" else 1.2)

    # Parity line
    ax_b.axhline(y=0.5, color='#555555', linestyle=':', linewidth=0.8, zorder=1)

    # Shaded region below 50%
    ax_b.fill_between([0.5, 35], 0, 0.5, color='#ffeeee', alpha=0.4, zorder=0)

    ax_b.set_xlabel("UH-CMA-ES maxevals", fontsize=7)
    ax_b.set_ylabel("UH-CMA-ES win rate", fontsize=7)

    ax_b.set_xlim(0.8, 35)
    ax_b.set_xscale('log')
    ax_b.set_xticks(maxevals_values)
    ax_b.set_xticklabels(["1", "10", "30"], fontsize=6)
    ax_b.minorticks_off()

    ax_b.set_ylim(0, 0.55)
    ax_b.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax_b.set_yticklabels(['0%', '10%', '20%', '30%', '40%', '50%'], fontsize=6)

    ax_b.grid(True, axis='y', alpha=0.25, linewidth=0.4, color="#888888")
    ax_b.set_axisbelow(True)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # Legend for panel (b) - two separate groups
    leg_colors_b = [
        Line2D([0], [0], color="#CC3311", linewidth=1.2, label="vs CMA-ES"),
        Line2D([0], [0], color="#009988", linewidth=1.2, label="vs ProbeSwitch"),
    ]
    leg_styles_b = [
        Line2D([0], [0], color="#555555", linestyle="-", marker="o", markersize=4,
               markerfacecolor="#555555", linewidth=1.0, label="$B{=}200D$"),
        Line2D([0], [0], color="#555555", linestyle="--", dashes=(5, 2.5), marker="o", markersize=4,
               markerfacecolor="white", markeredgecolor="#555555", markeredgewidth=1.2,
               linewidth=1.5, label="$B{=}500D$"),
    ]
    leg1_b = ax_b.legend(handles=leg_colors_b, fontsize=5.5, loc="upper right",
                         framealpha=0.95, handlelength=1.8, handletextpad=0.4, borderpad=0.35,
                         labelspacing=0.3, bbox_to_anchor=(1.0, 0.92))
    ax_b.add_artist(leg1_b)
    ax_b.legend(handles=leg_styles_b, fontsize=5.5, loc="upper right",
                framealpha=0.95, handlelength=1.8, handletextpad=0.4, borderpad=0.35,
                labelspacing=0.3, bbox_to_anchor=(0.97, 0.80))
    ax_b.tick_params(axis='both', labelsize=6)

    ax_b.set_title("(b) UH-CMA-ES maxevals sensitivity", fontsize=8, fontweight="normal", ha="center")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.22)

    # Save figure
    output_dir = os.path.join(repo_dir, "evidence", "paper_figures", "Appendix")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig_a10_depth_fidelity_tradeoff")

    saved = save_figure(fig, output_path)
    print(f"Saved figures: {saved}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Panel (a): Budget Grid Robustness")
    print("=" * 60)
    print("\nD=40:")
    for algo_b, label, *_ in comparisons_a:
        data = extract_comparison_data(df_d40, algo_b)
        wr_min = data["win_rate_a"].min()
        wr_max = data["win_rate_a"].max()
        print(f"  {label}: {wr_min:.1%}--{wr_max:.1%}")

    print("\nD=20:")
    for algo_b, label, *_ in comparisons_a:
        data = extract_comparison_data(df_d20, algo_b)
        wr_min = data["win_rate_a"].min()
        wr_max = data["win_rate_a"].max()
        print(f"  {label}: {wr_min:.1%}--{wr_max:.1%}")

    print("\n" + "=" * 60)
    print("Panel (b): UH-CMA-ES maxevals Sensitivity")
    print("=" * 60)
    for budget in budget_list:
        print(f"\nB = {budget}D:")
        subset = df_maxevals[df_maxevals['budget_multiplier'] == budget]
        pivot = subset.pivot_table(index=['function', 'instance'], columns='algorithm', values='best_f')

        for baseline_algo, baseline_label, _ in baselines_b:
            print(f"  {baseline_label}:")
            for uh_algo, uh_label in uh_variants:
                wr, wins, total = compute_win_rate(pivot, uh_algo, baseline_algo)
                print(f"    UH({uh_label}): {wr:.1%} ({wins}/{total})")

    plt.close(fig)


if __name__ == "__main__":
    main()
