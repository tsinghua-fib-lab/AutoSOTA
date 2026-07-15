#!/usr/bin/env python3
"""
Plot Figure A9: Budget grid robustness analysis.

This figure demonstrates that residual bootstrapping's advantage is robust across:
1. Different evaluation budgets (B=50D, 100D, 200D)
2. Different dimensions (D=40, D=20)

Key message: BERW consistently outperforms UH-CMA-ES and Resample baselines across
all tested budgets, with win rates improving as budget increases.

Panels:
(a) D=40 - Win rate vs budget for three comparisons
(b) D=20 - Win rate vs budget for three comparisons
"""

from __future__ import annotations

import os
import sys

# Add parent directory to path for plot_style import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import pandas as pd

from plot_style import apply_style, save_figure, COLORS, WIDTHS


def load_budget_grid_summary(data_dir: str) -> pd.DataFrame:
    """Load budget_grid_summary.csv from given directory."""
    path = os.path.join(data_dir, "budget_grid_summary.csv")
    return pd.read_csv(path)


def extract_comparison_data(df: pd.DataFrame, algo_b: str) -> pd.DataFrame:
    """Extract win rates for a specific comparison across budgets."""
    filtered = df[df["algo_b"] == algo_b].copy()
    filtered = filtered.sort_values("budget_mult")
    return filtered


def main():
    apply_style()

    # Data directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d40_dir = os.path.join(base_dir, "evidence", "hansen_test_fixed_budget_grid")
    d20_dir = os.path.join(base_dir, "evidence", "hansen_test_fixed_budget_grid_d20")

    # Load data
    df_d40 = load_budget_grid_summary(d40_dir)
    df_d20 = load_budget_grid_summary(d20_dir)

    # Define comparisons to plot (algo_b, label, color)
    comparisons = [
        ("UH-CMA-ES(maxevals=30)", "vs UH-CMA-ES", COLORS["blue"]),
        ("CMA-ES-Resample(k=10)", "vs Resample", COLORS["green"]),
        ("CMA-ES-sep", "vs CMA-ES", COLORS["red"]),
    ]

    # Budget levels
    budgets = [50, 100, 200]

    # Create single-panel figure
    fig_width = WIDTHS["single"] * 1.2
    fig_height = fig_width * 0.75
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    # Plot each comparison with D=40 (filled) and D=20 (open)
    for algo_b, label, color in comparisons:
        data_d40 = extract_comparison_data(df_d40, algo_b)
        data_d20 = extract_comparison_data(df_d20, algo_b)

        # D=40: filled markers, solid line
        ax.plot(data_d40["budget_mult"], data_d40["win_rate_a"],
                color=color, linewidth=1.0, linestyle="-",
                marker="o", markersize=5, markerfacecolor=color, zorder=3)

        # D=20: open markers, dashed line
        ax.plot(data_d20["budget_mult"], data_d20["win_rate_a"],
                color=color, linewidth=0.9, linestyle="--",
                marker="o", markersize=5, markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.0, zorder=3)

    # Parity line
    ax.axhline(y=0.5, color=COLORS["grey"], linewidth=0.6, linestyle=":", zorder=1)

    # Annotate key points
    # Highest: D=40, B=200D vs UH-CMA-ES
    uh_d40 = extract_comparison_data(df_d40, "UH-CMA-ES(maxevals=30)")
    wr_max = uh_d40[uh_d40["budget_mult"] == 200]["win_rate_a"].values[0]
    ax.annotate(f"{wr_max:.0%}", xy=(200, wr_max), xytext=(193, wr_max + 0.02),
                fontsize=6, color=COLORS["blue"], ha="right", va="bottom")

    # Lowest: D=20, B=100D vs CMA-ES
    cma_d20 = extract_comparison_data(df_d20, "CMA-ES-sep")
    wr_min = cma_d20[cma_d20["budget_mult"] == 100]["win_rate_a"].values[0]
    ax.annotate(f"{wr_min:.0%}", xy=(100, wr_min), xytext=(108, wr_min),
                fontsize=5.5, color=COLORS["red"], ha="left", va="center")

    ax.set_xlim(40, 210)
    ax.set_ylim(0.46, 1.0)
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b}$D$" for b in budgets])
    ax.set_xlabel("Budget (evaluations)", fontsize=7)
    ax.set_ylabel("Win rate of Residual Bootstrapping", fontsize=7)
    ax.tick_params(axis='both', labelsize=6)

    # Light grid
    ax.grid(True, alpha=0.2, linewidth=0.4, color="#888888")

    # Custom legend: color for comparison, line style for dimension
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS["blue"], linewidth=1.0, label="vs UH-CMA-ES"),
        Line2D([0], [0], color=COLORS["green"], linewidth=1.0, label="vs Resample"),
        Line2D([0], [0], color=COLORS["red"], linewidth=1.0, label="vs CMA-ES"),
        Line2D([0], [0], color="grey", linestyle="-", marker="o", markersize=4,
               markerfacecolor="grey", linewidth=0.8, label="$D{=}40$"),
        Line2D([0], [0], color="grey", linestyle="--", marker="o", markersize=4,
               markerfacecolor="white", markeredgecolor="grey", linewidth=0.8, label="$D{=}20$"),
    ]
    ax.legend(handles=legend_elements, fontsize=5.5, loc="lower left",
              framealpha=0.95, handlelength=1.6, handletextpad=0.4, borderpad=0.4,
              ncol=1, labelspacing=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = os.path.join(base_dir, "evidence", "paper_figures", "Appendix")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig_a9_budget_grid_robustness")

    saved = save_figure(fig, output_path)
    print(f"Saved figures: {saved}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print("\nD=40:")
    for algo_b, label, *_ in comparisons:
        data = extract_comparison_data(df_d40, algo_b)
        wr_min = data["win_rate_a"].min()
        wr_max = data["win_rate_a"].max()
        print(f"  {label}: {wr_min:.1%}--{wr_max:.1%}")

    print("\nD=20:")
    for algo_b, label, *_ in comparisons:
        data = extract_comparison_data(df_d20, algo_b)
        wr_min = data["win_rate_a"].min()
        wr_max = data["win_rate_a"].max()
        print(f"  {label}: {wr_min:.1%}--{wr_max:.1%}")

    # Verify all UH-CMA-ES and Resample win rates > 50%
    print("\n=== Verification ===")
    all_above_50 = True
    for df, dim_label in [(df_d40, "D=40"), (df_d20, "D=20")]:
        for algo_b in ["UH-CMA-ES(maxevals=30)", "CMA-ES-Resample(k=10)"]:
            data = extract_comparison_data(df, algo_b)
            for _, row in data.iterrows():
                if row["win_rate_a"] <= 0.5:
                    print(f"WARNING: {dim_label}, {algo_b}, B={row['budget_mult']}D: "
                          f"win_rate={row['win_rate_a']:.1%} <= 50%")
                    all_above_50 = False

    if all_above_50:
        print("All UH-CMA-ES and Resample win rates > 50%")

    # Check win rate trend for UH-CMA-ES
    print("\n=== Budget Trend (UH-CMA-ES) ===")
    for df, dim_label in [(df_d40, "D=40"), (df_d20, "D=20")]:
        data = extract_comparison_data(df, "UH-CMA-ES(maxevals=30)")
        wr_50 = data[data["budget_mult"] == 50]["win_rate_a"].values[0]
        wr_200 = data[data["budget_mult"] == 200]["win_rate_a"].values[0]
        trend = "increasing" if wr_200 > wr_50 else "decreasing"
        print(f"  {dim_label}: {wr_50:.1%} (B=50D) -> {wr_200:.1%} (B=200D) [{trend}]")

    plt.close(fig)


if __name__ == "__main__":
    main()
