#!/usr/bin/env python3
"""
Plot Figure A8: Threshold sensitivity analysis for probe-and-switch.

This figure demonstrates that:
1. Classification accuracy is robust to threshold choice in [0.08, 0.18]
2. Misranking probe consistently outperforms variance probe
3. Optimal threshold shifts with budget (B=200D vs B=500D)

Panels (side by side):
(a) Accuracy vs threshold τ for different configurations
(b) Regret vs threshold trade-off (train/test)
"""

from __future__ import annotations

import os
import sys

# Add parent directory to path for plot_style import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from plot_style import apply_style, save_figure, COLORS, WIDTHS


def load_threshold_sweep(data_dir: str) -> pd.DataFrame:
    """Load threshold_sweep.csv from given directory."""
    path = os.path.join(data_dir, "threshold_sweep.csv")
    return pd.read_csv(path)


def load_regret_sweep(data_dir: str) -> pd.DataFrame:
    """Load train_test_threshold_sweep_misranking_rd_log10_regret_mean.csv."""
    path = os.path.join(data_dir, "train_test_threshold_sweep_misranking_rd_log10_regret_mean.csv")
    return pd.read_csv(path)


def main():
    apply_style()

    # Data directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    b200_dir = os.path.join(base_dir, "evidence", "bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200")
    b500_dir = os.path.join(base_dir, "evidence", "bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500")

    # Load data
    df_b200 = load_threshold_sweep(b200_dir)
    df_b500 = load_threshold_sweep(b500_dir)
    df_regret = load_regret_sweep(b200_dir)

    # Extract probe-specific data
    b200_mr = df_b200[df_b200["probe_value_key"] == "misranking_rd"].copy()
    b200_var = df_b200[df_b200["probe_value_key"] == "variance_rel_sd"].copy()
    b500_mr = df_b500[df_b500["probe_value_key"] == "misranking_rd"].copy()

    # Create figure with 2 horizontal panels (side by side)
    fig_width = WIDTHS["double"]  # Full width for two panels
    fig_height = fig_width * 0.38  # Shorter for horizontal layout
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    # =========================================================================
    # Panel (a): Accuracy vs threshold
    # =========================================================================
    ax = axes[0]

    # Plot recommended threshold region as shaded band
    ax.axvspan(0.08, 0.18, alpha=0.15, color=COLORS["grey"], zorder=0)

    # Add boundary annotations for recommended range (at top of plot)
    ax.annotate("", xy=(0.08, 0.80), xytext=(0.18, 0.80),
                arrowprops=dict(arrowstyle="<->", color="#888888", lw=0.7))
    ax.text(0.138, 0.81, "plateau", fontsize=6, ha="center", color="#666666")

    # Plot curves - main results
    ax.plot(b200_mr["threshold"], b200_mr["accuracy"],
            color=COLORS["blue"], linewidth=1.0, label="$B{=}200D$, MR", zorder=3)
    ax.plot(b500_mr["threshold"], b500_mr["accuracy"],
            color=COLORS["green"], linewidth=1.0, label="$B{=}500D$, MR", zorder=3)
    ax.plot(b200_var["threshold"], b200_var["accuracy"],
            color=COLORS["red"], linewidth=0.9, linestyle="--",
            label="$B{=}200D$, Var", zorder=2)

    # Chance level
    ax.axhline(y=0.5, color=COLORS["grey"], linewidth=0.5, linestyle=":", zorder=1)
    ax.text(0.29, 0.51, "chance", fontsize=5.5, color="#888888", ha="right")

    # Default threshold vertical line
    ax.axvline(x=0.12, color="#888888", linewidth=0.7, linestyle="--", zorder=1)

    # Find peak accuracies
    b200_mr_peak_idx = b200_mr["accuracy"].idxmax()
    b200_mr_peak_tau = b200_mr.loc[b200_mr_peak_idx, "threshold"]
    b200_mr_peak_acc = b200_mr.loc[b200_mr_peak_idx, "accuracy"]

    b500_mr_peak_idx = b500_mr["accuracy"].idxmax()
    b500_mr_peak_tau = b500_mr.loc[b500_mr_peak_idx, "threshold"]
    b500_mr_peak_acc = b500_mr.loc[b500_mr_peak_idx, "accuracy"]

    # Mark peaks with small dots
    ax.plot(b200_mr_peak_tau, b200_mr_peak_acc, 'o', color=COLORS["blue"],
            markersize=3, zorder=5)
    ax.plot(b500_mr_peak_tau, b500_mr_peak_acc, 'o', color=COLORS["green"],
            markersize=3, zorder=5)

    # Annotate peaks - position closer to the dots
    ax.text(b200_mr_peak_tau, b200_mr_peak_acc + 0.02,
            f"{b200_mr_peak_acc:.1%}", fontsize=6, color=COLORS["blue"],
            ha="center", va="bottom")
    ax.text(b500_mr_peak_tau, b500_mr_peak_acc + 0.012,
            f"{b500_mr_peak_acc:.1%}", fontsize=6, color=COLORS["green"],
            ha="center", va="bottom")

    # Annotate default threshold - position at bottom
    ax.text(0.125, 0.49, r"$\tau{=}0.12$", fontsize=5.5, color="#666666",
            ha="left", va="bottom")

    ax.set_xlim(0.0, 0.30)
    ax.set_ylim(0.48, 0.84)
    ax.set_xlabel(r"Threshold $\tau$", fontsize=7)
    ax.set_ylabel("Classification accuracy", fontsize=7)
    ax.set_title("(a) Threshold sensitivity", fontsize=8, loc="center", fontweight="normal")
    ax.tick_params(axis='both', labelsize=6)

    # Compact legend
    ax.legend(fontsize=6, loc="upper right", framealpha=0.95,
              handlelength=1.5, handletextpad=0.4, borderpad=0.3)

    # Light grid
    ax.grid(True, alpha=0.2, linewidth=0.4, color="#888888")

    # =========================================================================
    # Panel (b): Regret vs threshold
    # =========================================================================
    ax = axes[1]

    # Recommended region shading
    ax.axvspan(0.08, 0.18, alpha=0.15, color=COLORS["grey"], zorder=0)

    # Default threshold
    ax.axvline(x=0.12, color="#888888", linewidth=0.7, linestyle="--", zorder=1)
    # Label at top right of the line
    ax.text(0.125, 0.48, r"$\tau{=}0.12$", fontsize=5.5, color="#666666",
            ha="left", va="top")

    # Plot train regret curve first (so it appears first in legend)
    ax.plot(df_regret["threshold"], df_regret["train_regret_mean"],
            color=COLORS["blue"], linewidth=0.7, linestyle=":",
            label="Train (i1–5)", alpha=0.6, zorder=2)

    # Plot test regret curve (main focus)
    ax.plot(df_regret["threshold"], df_regret["test_regret_mean"],
            color=COLORS["blue"], linewidth=1.0, label="Test (i6–15)", zorder=3)

    # Find minimum regret point
    min_regret_idx = df_regret["test_regret_mean"].idxmin()
    min_regret_tau = df_regret.loc[min_regret_idx, "threshold"]
    min_regret_val = df_regret.loc[min_regret_idx, "test_regret_mean"]

    ax.plot(min_regret_tau, min_regret_val, 'o', color=COLORS["blue"],
            markersize=3, zorder=5)

    # Min label
    ax.text(min_regret_tau + 0.015, min_regret_val - 0.01,
            "min", fontsize=5.5, color=COLORS["blue"], va="top")

    # Region labels - use hyphenated form for academic style
    ax.text(0.008, 0.48, "over-switching", fontsize=5.5, color="#666666",
            ha="left", va="top")
    ax.text(0.215, 0.085, "under-switching", fontsize=5.5, color="#666666",
            ha="left", va="bottom")

    # Tighten Y-axis
    ax.set_xlim(0.0, 0.30)
    ax.set_ylim(-0.02, 0.50)
    ax.set_xlabel(r"Threshold $\tau$", fontsize=7)
    ax.set_ylabel(r"Mean $\log_{10}$ regret", fontsize=7)
    ax.set_title("(b) Regret trade-off", fontsize=8, loc="center", fontweight="normal")
    ax.tick_params(axis='both', labelsize=6)

    ax.legend(fontsize=6, loc="upper right", framealpha=0.95,
              handlelength=1.5, handletextpad=0.4, borderpad=0.3)

    # Light grid
    ax.grid(True, alpha=0.2, linewidth=0.4, color="#888888")

    plt.tight_layout()

    # Save figure
    output_dir = os.path.join(base_dir, "evidence", "paper_figures", "Appendix")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig_a8_threshold_sensitivity")

    saved = save_figure(fig, output_path)
    print(f"Saved figures: {saved}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"B=200D MR: peak accuracy = {b200_mr_peak_acc:.3f} at τ = {b200_mr_peak_tau:.3f}")
    print(f"B=500D MR: peak accuracy = {b500_mr_peak_acc:.3f} at τ = {b500_mr_peak_tau:.3f}")

    acc_mr_at_default = b200_mr[b200_mr["threshold"] == 0.12]["accuracy"].values[0]
    acc_var_at_default = b200_var[b200_var["threshold"] == 0.12]["accuracy"].values[0]
    print(f"At τ=0.12: MR accuracy = {acc_mr_at_default:.3f}, Var accuracy = {acc_var_at_default:.3f}")
    print(f"MR advantage: +{(acc_mr_at_default - acc_var_at_default)*100:.1f}%")
    print(f"\nTest regret minimum: {min_regret_val:.4f} at τ = {min_regret_tau:.3f}")

    plt.close(fig)


if __name__ == "__main__":
    main()
