#!/usr/bin/env python3
"""
Generate Figure A5: Probe decoupling under radial noise.

Creates a 1x2 panel figure showing:
  (a) Probe decoupling: misranking vs log10(variance) with threshold line
  (b) Algorithmic impact: delta log10 regret vs misranking

Uses existing data from:
  evidence/probe_decoupling_radial/probe_values.csv
  Results/noisy_wrapper_radial_additive_rel_sigma0p5_d80_160_320_f1,2,6,10,15,20_i1-3_B200_probe_decouple/merged/bbob_summary.csv
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


# Threshold for misranking probe trigger
MISRANKING_THRESHOLD = 0.12

# Dimension colors
DIM_COLORS = {
    80: COLORS["blue"],
    160: COLORS["green"],
    320: COLORS["red"],
}

DIM_MARKERS = {
    80: "o",
    160: "s",
    320: "^",
}


def load_probe_data() -> pd.DataFrame:
    """Load the probe values CSV."""
    script_dir = os.path.dirname(__file__)
    # probe_values.csv is at repo/evidence/probe_decoupling_radial/
    data_path = os.path.join(
        script_dir, "..", "evidence", "probe_decoupling_radial", "probe_values.csv"
    )
    return pd.read_csv(data_path)


def load_performance_data() -> pd.DataFrame:
    """Load the algorithm performance CSV."""
    script_dir = os.path.dirname(__file__)
    # bbob_summary.csv is at project root Results/ (../../Results from repo/tools)
    data_path = os.path.join(
        script_dir, "..", "..", "Results",
        "noisy_wrapper_radial_additive_rel_sigma0p5_d80_160_320_f1,2,6,10,15,20_i1-3_B200_probe_decouple",
        "merged", "bbob_summary.csv"
    )
    return pd.read_csv(data_path)


def compute_delta_log_regret(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delta = best_f(Variance-Switch) - best_f(Misranking-Switch).
    Positive values mean Misranking-Switch achieved lower (better) best_f.
    Also track wins where MR has strictly lower best_f.
    """
    # Filter to the two switch algorithms
    mr_switch = perf_df[perf_df["algorithm"] == "Switch-MisrankingProbe(t=0.12)"].copy()
    var_switch = perf_df[perf_df["algorithm"] == "Switch-VarianceProbe(CMAsep/COCP-Hetero)"].copy()

    # Set index for joining
    key_cols = ["function", "dimension", "instance"]
    mr_switch = mr_switch.set_index(key_cols)
    var_switch = var_switch.set_index(key_cols)

    # Compute delta
    result = pd.DataFrame(index=mr_switch.index)
    result["mr_best_f"] = mr_switch["best_f"]
    result["var_best_f"] = var_switch["best_f"]

    # Delta = var - mr, positive means MR achieved lower (better) value
    result["delta_best_f"] = result["var_best_f"] - result["mr_best_f"]

    # For plotting, use log scale where possible
    # We'll use sign(delta) * log10(|delta| + 1) for visualization
    result["delta_log_regret"] = np.sign(result["delta_best_f"]) * np.log10(np.abs(result["delta_best_f"]) + 1)

    # Track wins (MR has lower best_f)
    result["mr_wins"] = result["mr_best_f"] < result["var_best_f"]

    return result.reset_index()


def main() -> None:
    apply_style()

    # Load data
    probe_df = load_probe_data()
    perf_df = load_performance_data()

    n_total = len(probe_df)
    print(f"Loaded {n_total} probe data points")

    # Compute trigger counts
    mr_triggers = probe_df["misranking_trigger"].sum()
    var_triggers = probe_df["variance_trigger"].sum()
    print(f"Misranking triggers: {mr_triggers}/{n_total}")
    print(f"Variance triggers: {var_triggers}/{n_total}")

    # Compute delta log regret
    delta_df = compute_delta_log_regret(perf_df)

    # Merge probe and delta data
    key_cols = ["function", "dimension", "instance"]
    merged = probe_df.merge(delta_df, on=key_cols)

    # Compute wins/losses (MR wins when it has lower best_f)
    wins = merged["mr_wins"].sum()
    losses = n_total - wins
    print(f"Misranking-Switch wins: {wins}/{n_total}")

    # P-value from sign test
    from scipy import stats
    result = stats.binomtest(wins, n_total, 0.5, alternative="greater")
    p_value = result.pvalue
    print(f"Sign test p-value: {p_value:.4f}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=get_subplot_figsize(1, 2, width="double", subplot_aspect=0.8))

    # Reduce font sizes for axis labels and tick labels
    for ax in axes:
        ax.tick_params(axis='both', labelsize=6)

    # =========================================================================
    # Panel (a): Probe decoupling scatter
    # =========================================================================
    ax = axes[0]

    # Add small offset to zero variance values for log scale
    variance_for_plot = merged["variance_rel_sd"].values.copy()
    variance_for_plot = np.where(variance_for_plot == 0, 1e-20, variance_for_plot)
    log_variance = np.log10(variance_for_plot + 1e-20)

    # Scatter by dimension
    for dim in sorted(DIM_COLORS.keys()):
        mask = merged["dimension"] == dim
        ax.scatter(
            merged.loc[mask, "misranking_rd"],
            log_variance[mask],
            s=30, alpha=0.75,
            c=DIM_COLORS[dim],
            marker=DIM_MARKERS[dim],
            edgecolors="white",
            linewidths=0.3,
            label=f"$d={dim}$",
            zorder=3
        )

    # Threshold vertical line
    ax.axvline(x=MISRANKING_THRESHOLD, color="#666666", linestyle="--",
               linewidth=1.2, zorder=2)

    # Add threshold label
    ax.text(MISRANKING_THRESHOLD + 0.008, -14.3, r"$\tau{=}0.12$",
            fontsize=6, color="#666666", verticalalignment="top")

    ax.set_xlabel(r"Misranking probe $M_{\mathrm{RD}}$", fontsize=7)
    ax.set_ylabel(r"$\log_{10}$(Variance probe)", fontsize=7)
    ax.set_xlim(0, 0.35)
    ax.set_ylim(-21, -14)
    add_grid(ax)

    # Add annotation for machine precision zone
    ax.axhspan(-21, -15, alpha=0.08, color=COLORS["blue"], zorder=0)
    ax.text(0.175, -20.5, "machine precision", fontsize=5.5,
            color="#666666", ha="center", style="italic")

    # Legend - move to upper left to avoid data
    ax.legend(loc="upper left", fontsize=6, framealpha=0.95, ncol=1,
              handlelength=1.2, handletextpad=0.3, borderpad=0.3)

    # Annotation: trigger counts (more prominent)
    annot_text = f"Triggers: MR {mr_triggers}/{n_total}, Var {var_triggers}/{n_total}"
    ax.text(0.97, 0.03, annot_text, transform=ax.transAxes,
            fontsize=6, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999999", alpha=0.95))

    # Panel label with subtitle
    ax.set_title("(a) Probe decoupling", fontsize=8, fontweight="bold", loc="center")

    # =========================================================================
    # Panel (b): Performance difference scatter
    # =========================================================================
    ax = axes[1]

    # Get y-axis range for shading
    y_min = merged["delta_log_regret"].min() - 0.5
    y_max = merged["delta_log_regret"].max() + 0.5

    # Add background shading for win/loss regions
    ax.axhspan(0, y_max, alpha=0.08, color=COLORS["green"], zorder=0)
    ax.axhspan(y_min, 0, alpha=0.08, color=COLORS["red"], zorder=0)

    # Add region labels - position closer to center to avoid being cut off
    ax.text(0.32, y_max * 0.45, "MR wins", fontsize=7, color="#228833",
            ha="right", va="center", fontweight="bold", alpha=0.9)
    ax.text(0.32, y_min * 0.45, "Var wins", fontsize=7, color="#CC3311",
            ha="right", va="center", fontweight="bold", alpha=0.9)

    # Scatter by dimension
    for dim in sorted(DIM_COLORS.keys()):
        mask = merged["dimension"] == dim
        ax.scatter(
            merged.loc[mask, "misranking_rd"],
            merged.loc[mask, "delta_log_regret"],
            s=30, alpha=0.75,
            c=DIM_COLORS[dim],
            marker=DIM_MARKERS[dim],
            edgecolors="white",
            linewidths=0.3,
            label=f"$d={dim}$",
            zorder=3
        )

    # Zero line (horizontal) - more prominent
    ax.axhline(y=0, color="#444444", linestyle="-", linewidth=1.0, zorder=2)

    # Threshold vertical line with label
    ax.axvline(x=MISRANKING_THRESHOLD, color="#666666", linestyle="--",
               linewidth=1.2, zorder=1)
    ax.text(MISRANKING_THRESHOLD + 0.008, y_max - 0.3, r"$\tau{=}0.12$",
            fontsize=6, color="#666666", verticalalignment="top")

    ax.set_xlabel(r"Misranking probe $M_{\mathrm{RD}}$", fontsize=7)
    ax.set_ylabel(r"$\mathrm{sgn}(\Delta f^*) \cdot \log_{10}(|\Delta f^*|{+}1)$", fontsize=7)
    ax.set_xlim(0, 0.35)
    ax.set_ylim(y_min, y_max)
    add_grid(ax)

    # Legend
    ax.legend(loc="upper left", fontsize=6, framealpha=0.95, ncol=1,
              handlelength=1.2, handletextpad=0.3, borderpad=0.3)

    # Annotation: wins/losses and p-value (more prominent)
    annot_text = f"MR wins: {wins}/{n_total}\n$p={p_value:.4f}$"
    ax.text(0.97, 0.03, annot_text, transform=ax.transAxes,
            fontsize=6, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#999999", alpha=0.95))

    # Panel label with subtitle
    ax.set_title("(b) Algorithmic impact", fontsize=8, fontweight="bold", loc="center")

    plt.tight_layout()

    # Save figure
    out_dir = os.path.join(os.path.dirname(__file__), "..", "evidence", "paper_figures", "Appendix")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig_a5_probe_decoupling")

    saved = save_figure(fig, out_path)
    print(f"Saved: {saved}")

    plt.close(fig)


if __name__ == "__main__":
    main()
