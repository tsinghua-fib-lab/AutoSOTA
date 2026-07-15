#!/usr/bin/env python3
"""
Generate Figure A3: Residual pool diagnostic snapshots.

This figure supports Claim C4: diagnostic metrics provide visibility into
assumption violations, making theoretical assumptions falsifiable.

Outputs:
  - fig_a3a_diagnostics_boxplot.pdf: 4-metric grouped boxplot comparing
    Good vs Bad runs across drift W₁, shape W₁, scale R², and center rel
  - fig_a3b_diagnostics_traces.pdf: Single-panel overlaid traces showing
    Good (f110, i2) vs Bad (f111, i8) Shape W₁ over generations

Data source:
  - evidence/hansen_test_fixed_budget/diagnostics/perf_vs_diagnostics.csv
  - evidence/hansen_test_fixed_budget/diagnostics/traces/*.csv

Usage:
    python tools/plot_fig_a3_diagnostics.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _project import BASE_DIR, repo_relpath
from plot_style import apply_style, save_figure, get_figsize, COLORS


def load_perf_diagnostics(csv_path: str) -> pd.DataFrame:
    """Load the merged performance + diagnostics table."""
    df = pd.read_csv(csv_path)
    return df


def load_trace(trace_dir: str, func: int, inst: int) -> pd.DataFrame | None:
    """Load per-generation trace for a specific (function, instance)."""
    fname = f"berw_hetero_B100_f{func}_d40_i{inst}.csv"
    path = os.path.join(trace_dir, fname)
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def format_pvalue(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "$p<.001$"
    elif p < 0.01:
        return f"$p={p:.3f}$"
    elif p < 0.05:
        return f"$p={p:.2f}$"
    else:
        return f"$p={p:.2f}$"


def significance_stars(p: float) -> str:
    """Return significance stars for annotation."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def plot_a3a_boxplot(df: pd.DataFrame, out_path: str) -> list[str]:
    """
    Figure A3a: 4-metric grouped boxplot comparing Good vs Bad runs.

    Shows boxplots for drift W₁, shape W₁, scale R², and center rel metrics,
    with Good (blue) and Bad (red) groups side by side. Individual points
    are shown as scatter overlay. Mann-Whitney p-values annotated above each pair.
    """
    apply_style()

    # Metrics to plot (column name, display label)
    metrics = [
        ("mean_noise_drift_w1", r"Drift $W_1$"),
        ("mean_noise_shape_w1", r"Shape $W_1$"),
        ("mean_noise_scale_fit_r2", r"Scale $R^2$"),
        ("mean_noise_center_split_rel", r"Center rel"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(get_figsize("single", aspect=0.8)[0] * 1.4, 3.2))

    color_good = COLORS["blue"]
    color_bad = COLORS["red"]

    good_df = df[~df["berw_worse"]]
    bad_df = df[df["berw_worse"]]

    n_good = len(good_df)
    n_bad = len(bad_df)

    # Positions for grouped boxplots
    n_metrics = len(metrics)
    positions_good = np.arange(n_metrics) * 2 - 0.35
    positions_bad = np.arange(n_metrics) * 2 + 0.35
    box_width = 0.55

    # Collect data for boxplots
    good_data = [good_df[m].dropna().values for m, _ in metrics]
    bad_data = [bad_df[m].dropna().values for m, _ in metrics]

    # Plot Good boxplots
    bp_good = ax.boxplot(
        good_data,
        positions=positions_good,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='white', linewidth=1.2),
        whiskerprops=dict(color=color_good, linewidth=0.8),
        capprops=dict(color=color_good, linewidth=0.8),
    )
    for patch in bp_good['boxes']:
        patch.set_facecolor(color_good)
        patch.set_alpha(0.7)
        patch.set_edgecolor(color_good)

    # Plot Bad boxplots
    bp_bad = ax.boxplot(
        bad_data,
        positions=positions_bad,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='white', linewidth=1.2),
        whiskerprops=dict(color=color_bad, linewidth=0.8),
        capprops=dict(color=color_bad, linewidth=0.8),
    )
    for patch in bp_bad['boxes']:
        patch.set_facecolor(color_bad)
        patch.set_alpha(0.7)
        patch.set_edgecolor(color_bad)

    # Scatter overlay for individual points
    np.random.seed(42)
    for i, (metric, _) in enumerate(metrics):
        # Good points
        y_good = good_df[metric].dropna().values
        x_good = positions_good[i] + np.random.uniform(-0.15, 0.15, len(y_good))
        ax.scatter(x_good, y_good, c=color_good, s=6, alpha=0.25, edgecolors='none', zorder=3)

        # Bad points
        y_bad = bad_df[metric].dropna().values
        x_bad = positions_bad[i] + np.random.uniform(-0.15, 0.15, len(y_bad))
        ax.scatter(x_bad, y_bad, c=color_bad, s=10, alpha=0.6, edgecolors='none', zorder=3)

    # Mann-Whitney U tests
    p_values = []
    for i, (metric, label) in enumerate(metrics):
        good_vals = good_df[metric].dropna().values
        bad_vals = bad_df[metric].dropna().values

        # For R², lower is worse (bad should have lower R²)
        # For others, higher is worse (bad should have higher values)
        if "r2" in metric.lower():
            # Test if bad < good (one-sided)
            _, p = stats.mannwhitneyu(bad_vals, good_vals, alternative='less')
        else:
            # Test if bad > good (one-sided)
            _, p = stats.mannwhitneyu(bad_vals, good_vals, alternative='greater')

        p_values.append(p)

    # X-axis labels
    ax.set_xticks(np.arange(n_metrics) * 2)
    ax.set_xticklabels([label for _, label in metrics], fontsize=9)

    # Y-axis
    ax.set_ylabel("Metric value", fontsize=9)
    ax.set_ylim(-0.15, 2.25)

    # Add significance annotations: stars above pairs + p-value below
    for i, p in enumerate(p_values):
        x_center = i * 2
        stars = significance_stars(p)
        star_color = '#333333' if p < 0.05 else '#999999'
        # Draw bracket
        y_bracket = max(
            np.percentile(good_data[i], 95) if len(good_data[i]) else 0,
            np.percentile(bad_data[i], 95) if len(bad_data[i]) else 0,
        ) + 0.10
        y_bracket = min(y_bracket, 2.10)
        ax.plot([positions_good[i], positions_good[i], positions_bad[i], positions_bad[i]],
                [y_bracket - 0.04, y_bracket, y_bracket, y_bracket - 0.04],
                color=star_color, linewidth=0.6, clip_on=False)
        ax.text(x_center, y_bracket + 0.03, stars, ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=star_color)

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_good, alpha=0.7, edgecolor=color_good),
        plt.Rectangle((0, 0), 1, 1, facecolor=color_bad, alpha=0.7, edgecolor=color_bad),
    ]
    ax.legend(
        legend_handles,
        [f"Good (n={n_good})", f"Bad (n={n_bad})"],
        loc='upper left',
        fontsize=8,
        framealpha=0.9,
    )

    # Grid
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    saved = save_figure(fig, out_path)
    plt.close(fig)
    return saved


def plot_a3b_traces_overlay(
    trace_dir: str,
    out_path: str,
) -> list[str]:
    """
    Figure A3b: Single-panel overlaid traces showing contrasting diagnostic patterns.

    Shows Good case (f110, i2) vs Bad case (f111, i8) Shape W₁ over generations
    in a single panel with both traces overlaid. Vertical line marks early termination.
    """
    apply_style()

    # Fixed cases as per plan: Good = f110, i2; Bad = f111, i8
    good_f, good_i = 110, 2
    bad_f, bad_i = 111, 8

    # Load traces
    good_trace = load_trace(trace_dir, good_f, good_i)
    bad_trace = load_trace(trace_dir, bad_f, bad_i)

    if good_trace is None or bad_trace is None:
        print(f"WARNING: Could not load traces for f{good_f}-i{good_i} or f{bad_f}-i{bad_i}")
        return []

    fig, ax = plt.subplots(1, 1, figsize=(get_figsize("single", aspect=0.55)[0] * 1.6, 3.0))

    color_good = COLORS["blue"]
    color_bad = COLORS["red"]

    # Metric to plot
    metric = "noise_shape_w1"

    # Good trace
    valid_good = good_trace[["generation", metric]].dropna()
    valid_bad = bad_trace[["generation", metric]].dropna()

    # Filter out data after generation counter resets
    good_reset_idx = valid_good["generation"].diff() < 0
    if good_reset_idx.any():
        first_reset = good_reset_idx.idxmax()
        valid_good = valid_good.loc[:first_reset - 1]

    bad_reset_idx = valid_bad["generation"].diff() < 0
    if bad_reset_idx.any():
        first_reset = bad_reset_idx.idxmax()
        valid_bad = valid_bad.loc[:first_reset - 1]

    # Apply rolling mean to smooth traces
    window = 5
    good_smooth = valid_good[metric].rolling(window=window, center=True, min_periods=2).mean()
    bad_smooth = valid_bad[metric].rolling(window=window, center=True, min_periods=2).mean()

    # Plot raw data as faint background, smoothed as solid
    ax.plot(valid_good["generation"], valid_good[metric],
            color=color_good, linewidth=0.3, alpha=0.15, zorder=1)
    ax.plot(valid_good["generation"], good_smooth,
            color=color_good, linewidth=1.0, alpha=0.95,
            label=f"Good: f{good_f}, i{good_i}", zorder=3)

    ax.plot(valid_bad["generation"], valid_bad[metric],
            color=color_bad, linewidth=0.3, alpha=0.15, zorder=1)
    ax.plot(valid_bad["generation"], bad_smooth,
            color=color_bad, linewidth=1.0, alpha=0.95,
            label=f"Bad: f{bad_f}, i{bad_i}", zorder=2)

    # Find termination point of bad case
    bad_last_gen = valid_bad["generation"].max()
    good_last_gen = valid_good["generation"].max()

    # Add vertical dashed line at bad case termination
    ax.axvline(x=bad_last_gen, color=color_bad, linestyle='--', linewidth=1.5,
               alpha=0.9, zorder=4)

    # Shaded region showing "missing" generations
    ax.axvspan(bad_last_gen, good_last_gen, alpha=0.12, color=color_bad, zorder=0)

    # Label the termination line directly
    ax.text(bad_last_gen + 2, 1.08, f"Bad ends\ngen {int(bad_last_gen)}",
            fontsize=7, color=color_bad, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

    # Label the gap
    gen_gap = int(good_last_gen - bad_last_gen)
    ax.annotate(
        f"$\\Delta T = {gen_gap}$ gens lost",
        xy=((bad_last_gen + good_last_gen) / 2, 0.55),
        ha='center', va='center',
        fontsize=8, fontweight='bold', color=color_bad, alpha=0.85,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7,
                  edgecolor=color_bad, linewidth=0.5),
    )

    # Labels
    ax.set_xlabel("Generation", fontsize=9)
    ax.set_ylabel(r"Shape $W_1$ (smoothed)", fontsize=9)

    # Y-axis limits
    ax.set_ylim(-0.05, 1.20)
    ax.set_xlim(-5, good_last_gen + 20)

    # Legend (position in upper left, away from traces)
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.95)

    # Grid
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    saved = save_figure(fig, out_path)
    plt.close(fig)
    return saved


def plot_a3_combined(df: pd.DataFrame, trace_dir: str, out_path: str) -> list[str]:
    """
    Figure A3: Combined diagnostic analysis (two panels).

    (a) Boxplot comparing diagnostic metrics between Good and Bad runs
    (b) Traces showing early termination in Bad case
    """
    apply_style()

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8),
                             gridspec_kw={'width_ratios': [1.1, 1]})

    color_good = COLORS["blue"]
    color_bad = COLORS["red"]

    # ==================== Panel (a): Boxplot ====================
    ax = axes[0]

    metrics = [
        ("mean_noise_drift_w1", r"Drift $W_1$"),
        ("mean_noise_shape_w1", r"Shape $W_1$"),
        ("mean_noise_scale_fit_r2", r"Scale $R^2$"),
        ("mean_noise_center_split_rel", r"Center rel"),
    ]

    good_df = df[~df["berw_worse"]]
    bad_df = df[df["berw_worse"]]
    n_good = len(good_df)
    n_bad = len(bad_df)

    n_metrics = len(metrics)
    positions_good = np.arange(n_metrics) * 2 - 0.3
    positions_bad = np.arange(n_metrics) * 2 + 0.3
    box_width = 0.45

    good_data = [good_df[m].dropna().values for m, _ in metrics]
    bad_data = [bad_df[m].dropna().values for m, _ in metrics]

    # Plot boxplots
    bp_good = ax.boxplot(good_data, positions=positions_good, widths=box_width,
                         patch_artist=True, showfliers=False,
                         medianprops=dict(color='white', linewidth=1.0),
                         whiskerprops=dict(color=color_good, linewidth=0.7),
                         capprops=dict(color=color_good, linewidth=0.7))
    for patch in bp_good['boxes']:
        patch.set_facecolor(color_good)
        patch.set_alpha(0.7)
        patch.set_edgecolor(color_good)

    bp_bad = ax.boxplot(bad_data, positions=positions_bad, widths=box_width,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='white', linewidth=1.0),
                        whiskerprops=dict(color=color_bad, linewidth=0.7),
                        capprops=dict(color=color_bad, linewidth=0.7))
    for patch in bp_bad['boxes']:
        patch.set_facecolor(color_bad)
        patch.set_alpha(0.7)
        patch.set_edgecolor(color_bad)

    # Scatter overlay
    np.random.seed(42)
    for i, (metric, _) in enumerate(metrics):
        y_good = good_df[metric].dropna().values
        x_good = positions_good[i] + np.random.uniform(-0.12, 0.12, len(y_good))
        ax.scatter(x_good, y_good, c=color_good, s=4, alpha=0.25, edgecolors='none', zorder=3)

        y_bad = bad_df[metric].dropna().values
        x_bad = positions_bad[i] + np.random.uniform(-0.12, 0.12, len(y_bad))
        ax.scatter(x_bad, y_bad, c=color_bad, s=7, alpha=0.55, edgecolors='none', zorder=3)

    # Mann-Whitney tests
    p_values = []
    for metric, _ in metrics:
        good_vals = good_df[metric].dropna().values
        bad_vals = bad_df[metric].dropna().values
        if "r2" in metric.lower():
            _, p = stats.mannwhitneyu(bad_vals, good_vals, alternative='less')
        else:
            _, p = stats.mannwhitneyu(bad_vals, good_vals, alternative='greater')
        p_values.append(p)

    # Calculate combined AUC using cross-validated logistic regression
    feature_cols = [m[0] for m in metrics]
    X = df[feature_cols].dropna()
    y = df.loc[X.index, "berw_worse"].astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_prob_cv = cross_val_predict(
        LogisticRegression(random_state=42), X_scaled, y, cv=cv, method='predict_proba'
    )[:, 1]
    combined_auc = roc_auc_score(y, y_prob_cv)

    # Significance annotations removed per request — reported in caption only
    ax.set_ylim(-0.05, 2.05)

    ax.set_xticks(np.arange(n_metrics) * 2)
    ax.set_xticklabels([label for _, label in metrics], fontsize=7)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_ylabel("Metric value", fontsize=8)

    # Legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color_good, alpha=0.7, edgecolor=color_good),
        plt.Rectangle((0, 0), 1, 1, facecolor=color_bad, alpha=0.7, edgecolor=color_bad),
    ]
    ax.legend(legend_handles, [f"Good (n={n_good})", f"Bad (n={n_bad})"],
              loc='upper left', fontsize=6.5, framealpha=0.9)

    ax.grid(True, axis='y', alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("(a) Diagnostic metrics: good vs. bad runs", fontsize=8, fontweight='bold')

    # ==================== Panel (b): Traces ====================
    ax = axes[1]

    good_f, good_i = 110, 2
    bad_f, bad_i = 111, 8

    good_trace = load_trace(trace_dir, good_f, good_i)
    bad_trace = load_trace(trace_dir, bad_f, bad_i)

    if good_trace is None or bad_trace is None:
        print(f"WARNING: Could not load traces")
        return []

    metric = "noise_shape_w1"

    valid_good = good_trace[["generation", metric]].dropna()
    valid_bad = bad_trace[["generation", metric]].dropna()

    # Filter out data after generation counter resets (keep only first continuous run)
    # Detect reset: generation decreases
    good_reset_idx = valid_good["generation"].diff() < 0
    if good_reset_idx.any():
        first_reset = good_reset_idx.idxmax()
        valid_good = valid_good.loc[:first_reset - 1]

    bad_reset_idx = valid_bad["generation"].diff() < 0
    if bad_reset_idx.any():
        first_reset = bad_reset_idx.idxmax()
        valid_bad = valid_bad.loc[:first_reset - 1]

    # Apply rolling mean to smooth the traces (smaller window preserves more variation)
    window = 5
    good_smooth = valid_good[metric].rolling(window=window, center=True, min_periods=2).mean()
    bad_smooth = valid_bad[metric].rolling(window=window, center=True, min_periods=2).mean()

    # Plot raw data as very faint background, smoothed as solid
    ax.plot(valid_good["generation"], valid_good[metric],
            color=color_good, linewidth=0.3, alpha=0.15, zorder=1)
    ax.plot(valid_good["generation"], good_smooth,
            color=color_good, linewidth=1.0, alpha=0.95,
            label=f"Good: f{good_f}, i{good_i}", zorder=3)

    ax.plot(valid_bad["generation"], valid_bad[metric],
            color=color_bad, linewidth=0.3, alpha=0.15, zorder=1)
    ax.plot(valid_bad["generation"], bad_smooth,
            color=color_bad, linewidth=1.0, alpha=0.95,
            label=f"Bad: f{bad_f}, i{bad_i}", zorder=2)

    bad_last_gen = valid_bad["generation"].max()
    good_last_gen = valid_good["generation"].max()

    # Vertical line at termination
    ax.axvline(x=bad_last_gen, color=color_bad, linestyle='--', linewidth=1.5,
               alpha=0.9, zorder=4)

    # Shaded region showing "missing" generations for bad case
    ax.axvspan(bad_last_gen, good_last_gen, alpha=0.12, color=color_bad, zorder=0)

    # Labels
    ax.text(bad_last_gen + 3, 0.97, f"Bad stops gen {int(bad_last_gen)}",
            fontsize=6.5, color=color_bad, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9, edgecolor='none'))

    gen_gap = int(good_last_gen - bad_last_gen)
    ax.annotate(
        f"$\\Delta T = {gen_gap}$ gens lost",
        xy=((bad_last_gen + good_last_gen) / 2, 0.68),
        ha='center', va='center',
        fontsize=7.5, fontweight='bold', color=color_bad, alpha=0.85,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor=color_bad, linewidth=0.5),
    )

    ax.set_xlabel("Generation", fontsize=8)
    ax.set_ylabel(r"Shape $W_1$ (smoothed)", fontsize=8)
    ax.tick_params(axis='both', labelsize=6)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-5, good_last_gen + 10)

    ax.legend(loc='upper left', fontsize=6, framealpha=0.95, bbox_to_anchor=(0.02, 0.98))
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("(b) Early termination in bad case", fontsize=8, fontweight='bold')

    plt.tight_layout()

    saved = save_figure(fig, out_path)
    plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure A3: Diagnostic snapshots")
    parser.add_argument(
        "--evidence-dir",
        default="evidence",
        help="Evidence directory (relative to repo/)",
    )
    parser.add_argument(
        "--output-dir",
        default="evidence/paper_figures/Appendix",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    # Paths
    diag_dir = os.path.join(args.evidence_dir, "hansen_test_fixed_budget", "diagnostics")
    perf_csv = os.path.join(diag_dir, "perf_vs_diagnostics.csv")
    trace_dir = os.path.join(diag_dir, "traces")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    if not os.path.isfile(perf_csv):
        print(f"ERROR: Missing {repo_relpath(perf_csv)}")
        sys.exit(1)

    df = load_perf_diagnostics(perf_csv)

    # Verify good/bad split
    n_good = (~df["berw_worse"]).sum()
    n_bad = df["berw_worse"].sum()
    print(f"Data loaded: {n_good} good runs ({100*n_good/len(df):.0f}%), {n_bad} bad runs ({100*n_bad/len(df):.0f}%)")

    # Generate combined Figure A3
    out_a3 = os.path.join(output_dir, "fig_a3_diagnostics")
    saved_a3 = plot_a3_combined(df, trace_dir, out_a3)
    for p in saved_a3:
        print(f"Saved: {repo_relpath(p)}")

    # Also generate separate figures for backward compatibility
    out_a3a = os.path.join(output_dir, "fig_a3a_diagnostics_boxplot")
    saved_a3a = plot_a3a_boxplot(df, out_a3a)
    for p in saved_a3a:
        print(f"Saved: {repo_relpath(p)}")

    out_a3b = os.path.join(output_dir, "fig_a3b_diagnostics_traces")
    saved_a3b = plot_a3b_traces_overlay(trace_dir, out_a3b)
    for p in saved_a3b:
        print(f"Saved: {repo_relpath(p)}")


if __name__ == "__main__":
    main()
