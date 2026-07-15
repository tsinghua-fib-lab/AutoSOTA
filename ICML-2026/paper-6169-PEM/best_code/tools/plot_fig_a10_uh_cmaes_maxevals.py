#!/usr/bin/env python3
"""
Generate Figure A10: UH-CMA-ES maxevals baseline strength analysis.

This figure demonstrates that:
1. UH-CMA-ES loses to simpler methods (CMA-ES, ProbeSwitch) regardless of maxevals configuration
2. Higher maxevals makes UH-CMA-ES worse (win rate decreases as maxevals increases)
3. This is not a tuning issue - it reflects the fundamental depth-fidelity trade-off

Key message: Under fixed budget, UH-CMA-ES's resampling mechanism is harmful.
The NoiseHandler mechanism itself is a liability.

Design: Single-panel line chart showing UH-CMA-ES win rate vs maxevals
- X-axis: maxevals (1, 10, 30)
- Y-axis: UH-CMA-ES win rate (all points below 50%)
- Lines: vs CMA-ES (red) and vs ProbeSwitch (cyan)
- Two budgets: B=200D (solid, dark) and B=500D (dashed, lighter)
- 50% parity reference line

Data source: evidence/bbob_noisy_uh_cmaes_maxevals_sweep_d40_f1-30_i1-15/bbob_summary.csv
Output: evidence/paper_figures/Appendix/fig_a10_uh_cmaes_maxevals.pdf
"""

from __future__ import annotations

import os
import sys

# Add parent directory to path for plot_style import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import apply_style, save_figure, WIDTHS, COLORS


def load_bbob_summary(data_dir: str) -> pd.DataFrame:
    """Load bbob_summary.csv from given directory."""
    path = os.path.join(data_dir, "bbob_summary.csv")
    return pd.read_csv(path)


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

    # Data directory - look in parent's parent (project root) for evidence
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(repo_dir)
    data_dir = os.path.join(project_root, "evidence", "bbob_noisy_uh_cmaes_maxevals_sweep_d40_f1-30_i1-15")

    # Load data
    df = load_bbob_summary(data_dir)

    # UH-CMA-ES variants (in order of maxevals)
    uh_variants = [
        ("UH-CMA-ES", "m=1"),           # maxevals=1
        ("UH-CMA-ES(maxevals=10)", "m=10"),  # maxevals=10
        ("UH-CMA-ES(maxevals=30)", "m=30"),  # maxevals=30
    ]
    maxevals_values = [1, 10, 30]

    # Baselines to compare against
    baselines = [
        ("CMA-ES", "vs CMA-ES", "#CC3311"),           # red
        ("ProbeSwitch-MR(t=0.12)", "vs ProbeSwitch", "#009988"),  # cyan/teal
    ]

    budgets = [200, 500]
    budget_styles = {
        200: {"linestyle": "-", "alpha": 1.0, "marker": "o"},
        500: {"linestyle": "--", "alpha": 0.7, "marker": "s"},
    }

    # Create single-panel figure
    fig_width = WIDTHS["single"]
    fig_height = fig_width * 0.8
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # Compute and plot win rates
    for baseline_algo, baseline_label, baseline_color in baselines:
        for budget in budgets:
            # Get pivot table for this budget
            subset = df[df['budget_multiplier'] == budget]
            pivot = subset.pivot_table(index=['function', 'instance'], columns='algorithm', values='best_f')

            win_rates = []
            for uh_algo, uh_label in uh_variants:
                # Compute UH's win rate against baseline (lower best_f is better)
                wr, wins, total = compute_win_rate(pivot, uh_algo, baseline_algo)
                win_rates.append(wr)

            # Plot line
            style = budget_styles[budget]
            label = f"{baseline_label} (B={budget}D)"
            ax.plot(maxevals_values, win_rates,
                    color=baseline_color,
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                    marker=style["marker"],
                    markersize=5,
                    linewidth=1.2,
                    label=label)

            # Add value annotations - only at first and last points to reduce clutter
            for i, (me, wr) in enumerate(zip(maxevals_values, win_rates)):
                if i == 0 or i == len(maxevals_values) - 1:
                    # Offset based on budget and baseline to avoid overlap
                    if baseline_algo == "CMA-ES":
                        offset_y = 0.018 if budget == 200 else 0.018
                        va = 'bottom'
                    else:  # ProbeSwitch
                        offset_y = -0.018 if budget == 200 else -0.018
                        va = 'top'
                    ax.text(me, wr + offset_y, f"{wr:.1%}",
                            ha='center', va=va,
                            fontsize=5.5, color=baseline_color, alpha=style["alpha"],
                            fontweight='bold')

    # Add parity line at 50%
    ax.axhline(y=0.5, color='#555555', linestyle=':', linewidth=0.8, zorder=1)
    ax.text(30, 0.515, "parity", fontsize=7, color='#666666', ha='right', va='bottom')

    # Configure axes
    ax.set_xlabel("UH-CMA-ES maxevals", fontsize=8)
    ax.set_ylabel("UH-CMA-ES win rate", fontsize=8)
    ax.set_xticks(maxevals_values)
    ax.set_xticklabels(["1", "10", "30"], fontsize=7)

    ax.set_ylim(0, 0.55)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%', '50%'], fontsize=6)

    # Add shaded region below parity to emphasize UH losing
    ax.fill_between([0.5, 32], 0, 0.5, color='#ffeeee', alpha=0.3, zorder=0)
    ax.text(16, 0.02, "UH loses", fontsize=7, color='#aa6666', ha='center', va='bottom', style='italic')

    # Light grid (horizontal only)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.4, color="#888888")
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set x-axis limits
    ax.set_xlim(0.5, 32)
    ax.set_xscale('log')
    ax.set_xticks(maxevals_values)
    ax.set_xticklabels(["1", "10", "30"], fontsize=7)
    ax.minorticks_off()

    # Legend
    ax.legend(loc='upper right', fontsize=6, framealpha=0.9)

    ax.tick_params(axis='both', labelsize=6)

    plt.tight_layout()

    # Save figure - output to repo's evidence folder
    output_dir = os.path.join(repo_dir, "evidence", "paper_figures", "Appendix")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig_a10_uh_cmaes_maxevals")

    saved = save_figure(fig, output_path)
    print(f"Saved figures: {saved}")

    # Print summary statistics
    print("\n=== UH-CMA-ES Win Rate vs Baselines ===")
    for budget in budgets:
        print(f"\nB = {budget}D:")
        subset = df[df['budget_multiplier'] == budget]
        pivot = subset.pivot_table(index=['function', 'instance'], columns='algorithm', values='best_f')

        for baseline_algo, baseline_label, _ in baselines:
            print(f"  {baseline_label}:")
            for uh_algo, uh_label in uh_variants:
                wr, wins, total = compute_win_rate(pivot, uh_algo, baseline_algo)
                print(f"    UH({uh_label}): {wr:.1%} ({wins}/{total})")

    # Also show UH variants against each other
    print("\n=== UH maxevals comparison (internal) ===")
    for budget in budgets:
        print(f"\nB = {budget}D:")
        subset = df[df['budget_multiplier'] == budget]
        pivot = subset.pivot_table(index=['function', 'instance'], columns='algorithm', values='best_f')

        wr_10v1, _, _ = compute_win_rate(pivot, "UH-CMA-ES(maxevals=10)", "UH-CMA-ES")
        wr_30v1, _, _ = compute_win_rate(pivot, "UH-CMA-ES(maxevals=30)", "UH-CMA-ES")
        wr_30v10, _, _ = compute_win_rate(pivot, "UH-CMA-ES(maxevals=30)", "UH-CMA-ES(maxevals=10)")

        print(f"  UH(m=10) vs UH(m=1):  {wr_10v1:.1%}")
        print(f"  UH(m=30) vs UH(m=1):  {wr_30v1:.1%}")
        print(f"  UH(m=30) vs UH(m=10): {wr_30v10:.1%}")

    # Verify key claims from the plan
    print("\n=== Verification of Key Claims ===")
    print("Plan expected win rates (UH vs baselines):")
    print("| Budget | UH(m=1) vs CMA-ES | UH(m=10) vs CMA-ES | UH(m=30) vs CMA-ES |")
    print("|--------|-------------------|--------------------|--------------------|")
    print("| 200D   | 20.4%             | 16.4%              | 15.1%              |")
    print("| 500D   | 26.4%             | 21.3%              | 20.2%              |")
    print()
    print("Actual computed win rates:")
    for budget in budgets:
        subset = df[df['budget_multiplier'] == budget]
        pivot = subset.pivot_table(index=['function', 'instance'], columns='algorithm', values='best_f')

        wr_1, _, _ = compute_win_rate(pivot, "UH-CMA-ES", "CMA-ES")
        wr_10, _, _ = compute_win_rate(pivot, "UH-CMA-ES(maxevals=10)", "CMA-ES")
        wr_30, _, _ = compute_win_rate(pivot, "UH-CMA-ES(maxevals=30)", "CMA-ES")
        print(f"| {budget}D   | {wr_1:.1%}            | {wr_10:.1%}             | {wr_30:.1%}             |")

    plt.close(fig)


if __name__ == "__main__":
    main()
