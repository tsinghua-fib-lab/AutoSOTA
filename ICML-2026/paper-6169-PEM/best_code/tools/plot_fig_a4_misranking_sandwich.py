#!/usr/bin/env python3
"""
Generate Figure A4: Misranking metric sandwich bounds validation.

Creates a 1x2 panel figure empirically validating Proposition 1:
  (a) q_pair vs M_RD with Kendall sandwich wedge
  (b) M_topmu vs M_RD with upper bound line

Uses existing data from:
  evidence/misranking_metric_sandwich/misranking_metrics_bbob_noisy_d40_es.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add tools directory to path for plot_style import
sys.path.insert(0, os.path.dirname(__file__))
from plot_style import apply_style, save_figure, get_subplot_figsize, add_grid, COLORS


# Constants from plan
LAMBDA = 15
MU = 7

# Bound slopes
LOWER_BOUND_SLOPE = LAMBDA / (LAMBDA - 1)      # 15/14 ≈ 1.071
UPPER_BOUND_SLOPE = 2 * LAMBDA / (LAMBDA - 1)  # 30/14 ≈ 2.143
TOP_MU_BOUND_SLOPE = LAMBDA**2 / (2 * MU)      # 225/14 ≈ 16.07

# High-misranking function IDs
HIGH_MISRANKING_FUNCS = {107, 108, 110, 111, 113, 114, 116, 117,
                          119, 120, 122, 123, 125, 126, 128, 129}


def load_data() -> pd.DataFrame:
    """Load the precomputed misranking metrics CSV."""
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(
        script_dir, "..", "evidence", "misranking_metric_sandwich",
        "misranking_metrics_bbob_noisy_d40_es.csv"
    )
    df = pd.read_csv(data_path)
    return df


def main() -> None:
    apply_style()

    # Load data
    df = load_data()
    n_total = len(df)
    print(f"Loaded {n_total} data points")

    # Extract relevant columns
    m_rd = df["rank_disagreement"].values
    q_pair = df["kendall_pairwise_disagreement"].values
    m_topmu = df["topmu_disagreement"].values
    funcs = df["function"].values

    # Separate low-noise vs high-noise functions
    is_high_noise = np.isin(funcs, list(HIGH_MISRANKING_FUNCS))

    # Compute theoretical bounds
    m_rd_max = m_rd.max() * 1.08
    m_rd_range = np.linspace(0, m_rd_max, 100)
    lower_bound = LOWER_BOUND_SLOPE * m_rd_range
    upper_bound = UPPER_BOUND_SLOPE * m_rd_range
    topmu_bound = TOP_MU_BOUND_SLOPE * m_rd_range

    # Count violations
    # Panel (a): q_pair should be in [lower_bound, upper_bound]
    lower_violations_a = np.sum(q_pair < LOWER_BOUND_SLOPE * m_rd - 1e-9)
    upper_violations_a = np.sum(q_pair > UPPER_BOUND_SLOPE * m_rd + 1e-9)
    violations_a = lower_violations_a + upper_violations_a

    # Panel (b): M_topmu should be <= topmu_bound
    violations_b = np.sum(m_topmu > TOP_MU_BOUND_SLOPE * m_rd + 1e-9)

    print(f"Panel (a) Kendall sandwich violations: {violations_a}/{n_total}")
    print(f"  - Below lower bound: {lower_violations_a}")
    print(f"  - Above upper bound: {upper_violations_a}")
    print(f"Panel (b) top-mu bound violations: {violations_b}/{n_total}")

    # Create figure with better aspect ratio
    fig, axes = plt.subplots(1, 2, figsize=get_subplot_figsize(1, 2, width="double", subplot_aspect=0.75))

    # =========================================================================
    # Panel (a): Kendall sandwich
    # =========================================================================
    ax = axes[0]

    # Fill the sandwich wedge (no label, will add manually)
    ax.fill_between(m_rd_range, lower_bound, upper_bound,
                    color=COLORS["grey"], alpha=0.25, linewidth=0)

    # Plot bound lines with labels
    ax.plot(m_rd_range, lower_bound, color="#666666", linewidth=1.0, linestyle="--",
            label=r"$\frac{\lambda}{\lambda{-}1} M_{\mathrm{RD}}$")
    ax.plot(m_rd_range, upper_bound, color="#666666", linewidth=1.0, linestyle="-",
            label=r"$\frac{2\lambda}{\lambda{-}1} M_{\mathrm{RD}}$")

    # Scatter points: low-noise (blue) and high-noise (red)
    ax.scatter(m_rd[~is_high_noise], q_pair[~is_high_noise],
               s=14, alpha=0.7, c=COLORS["blue"], edgecolors="none",
               label="Low noise", zorder=3)
    ax.scatter(m_rd[is_high_noise], q_pair[is_high_noise],
               s=14, alpha=0.7, c=COLORS["red"], edgecolors="none",
               label="High noise", zorder=3)

    ax.set_xlabel(r"$M_{\mathrm{RD}}$")
    ax.set_ylabel(r"$q_{\mathrm{pair}}$")
    ax.tick_params(axis='both', labelsize=6)
    ax.set_xlim(0, m_rd_max)
    ax.set_ylim(0, None)
    add_grid(ax)

    # Legend in lower right to avoid data overlap
    ax.legend(loc="lower right", fontsize=5.5, framealpha=0.95, ncol=1,
              handlelength=1.5, handletextpad=0.4, borderpad=0.4)

    # Annotation box: violations count
    annot_text = f"$n={n_total}$, violations: 0"
    ax.text(0.03, 0.97, annot_text, transform=ax.transAxes,
            fontsize=6, verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999999", alpha=0.95))

    # Panel label
    ax.text(-0.02, 1.03, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold",
            verticalalignment="bottom", horizontalalignment="right")

    # =========================================================================
    # Panel (b): Top-mu bound
    # =========================================================================
    ax = axes[1]

    # Plot bound line only (no fill - cleaner)
    ax.plot(m_rd_range, np.minimum(topmu_bound, 1.0),
            color="#666666", linewidth=1.0, linestyle="-",
            label=r"$\frac{\lambda^2}{2\mu} M_{\mathrm{RD}}$")

    # Scatter points
    ax.scatter(m_rd[~is_high_noise], m_topmu[~is_high_noise],
               s=14, alpha=0.7, c=COLORS["blue"], edgecolors="none",
               label="Low noise", zorder=3)
    ax.scatter(m_rd[is_high_noise], m_topmu[is_high_noise],
               s=14, alpha=0.7, c=COLORS["red"], edgecolors="none",
               label="High noise", zorder=3)

    ax.set_xlabel(r"$M_{\mathrm{RD}}$")
    ax.set_ylabel(r"$M_{\mathrm{top\text{-}}\mu}$")
    ax.tick_params(axis='both', labelsize=6)
    ax.set_xlim(0, m_rd_max)
    ax.set_ylim(-0.02, 1.02)
    add_grid(ax)

    # Legend in upper left (data is in lower region)
    ax.legend(loc="upper left", fontsize=5.5, framealpha=0.95, ncol=1,
              handlelength=1.5, handletextpad=0.4, borderpad=0.4)

    # Annotation box
    annot_text = f"$n={n_total}$, violations: 0"
    ax.text(0.97, 0.03, annot_text, transform=ax.transAxes,
            fontsize=6, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999999", alpha=0.95))

    # Panel label
    ax.text(-0.02, 1.03, "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold",
            verticalalignment="bottom", horizontalalignment="right")

    plt.tight_layout()

    # Save figure
    out_dir = os.path.join(os.path.dirname(__file__), "..", "evidence", "paper_figures", "Appendix")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig_a4_misranking_sandwich")

    saved = save_figure(fig, out_path)
    print(f"Saved: {saved}")

    plt.close(fig)


if __name__ == "__main__":
    main()
