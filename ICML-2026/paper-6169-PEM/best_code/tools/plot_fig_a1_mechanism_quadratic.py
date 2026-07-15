#!/usr/bin/env python3
"""
Generate Figure A1: Mechanism validation on strongly convex quadratic.

This figure validates the core theory chain:
  misranking → update dispersion → curvature loss (Theorem 1)

On a controlled quadratic f(x) = 0.5||x||^2, we show:
  (a) Update dispersion increases with misranking (M_RD)
  (b) Jensen gap equals 0.5 * Var(Δm) for quadratic objectives

Data source: evidence/theory_update_dispersion_quadratic/
Output: evidence/paper_figures/Appendix/fig_a1_mechanism_quadratic.pdf

Usage:
    python tools/plot_fig_a1_mechanism_quadratic.py
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
from plot_style import apply_style, save_figure, get_subplot_figsize, add_grid, COLORS


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure A1: Mechanism quadratic")
    parser.add_argument(
        "--csv",
        default="evidence/theory_update_dispersion_quadratic/update_dispersion_quadratic.csv",
        help="Input CSV file",
    )
    parser.add_argument(
        "--output",
        default="evidence/paper_figures/Appendix/fig_a1_mechanism_quadratic",
        help="Output path (without extension)",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        print(f"ERROR: Missing {repo_relpath(csv_path)}")
        sys.exit(1)

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {repo_relpath(csv_path)}")

    # Extract columns
    m_rd = df["M_RD"].values
    disp_sq = df["update_dispersion_sq"].values
    jensen_gap = df["Jensen_gap"].values
    var_update = df["Var_update"].values
    ratio = df["gap_over_half_var"].values

    half_var = 0.5 * var_update

    # Print summary statistics
    print(f"\nData summary:")
    print(f"  M_RD range: [{m_rd.min():.4f}, {m_rd.max():.4f}]")
    print(f"  Dispersion range: [{disp_sq.min():.4f}, {disp_sq.max():.4f}]")
    print(f"  Jensen gap range: [{jensen_gap.min():.4f}, {jensen_gap.max():.4f}]")
    print(f"  Ratio (gap / 0.5*var) median: {np.median(ratio):.4f}")

    # Compute correlation
    mask = disp_sq > 0
    corr_log = np.corrcoef(m_rd[mask], np.log1p(disp_sq[mask]))[0, 1]
    corr_raw = np.corrcoef(m_rd, disp_sq)[0, 1]
    print(f"  Correlation (M_RD vs log(1+disp)): {corr_log:.3f}")
    print(f"  Correlation (M_RD vs disp): {corr_raw:.3f}")

    # Apply style
    apply_style()

    # Create figure with 1x2 layout
    fig, axes = plt.subplots(1, 2, figsize=get_subplot_figsize(1, 2, width="double", subplot_aspect=0.75))

    # -------------------------------------------------------------------------
    # Panel (a): Update dispersion vs misranking
    # -------------------------------------------------------------------------
    ax = axes[0]

    ax.scatter(
        m_rd,
        disp_sq,
        s=12,
        alpha=0.6,
        c=COLORS["blue"],
        edgecolors="none",
    )

    ax.set_xlabel(r"$M_{\mathrm{RD}}$", fontsize=8)
    ax.set_ylabel(r"$\|\Delta m^{(a)} - \Delta m^{(b)}\|^2$", fontsize=8)
    ax.tick_params(axis='both', labelsize=6)

    add_grid(ax, alpha=0.2)

    # Annotation
    ax.text(
        0.96, 0.04,
        f"$r = {corr_raw:.2f}$",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=7,
    )

    ax.set_title("(a) Dispersion vs. misranking", fontsize=8, pad=4)

    # -------------------------------------------------------------------------
    # Panel (b): Jensen gap vs 0.5 * Var (quadratic identity)
    # -------------------------------------------------------------------------
    ax = axes[1]

    ax.scatter(
        half_var,
        jensen_gap,
        s=12,
        alpha=0.6,
        c=COLORS["blue"],
        edgecolors="none",
    )

    # Diagonal reference line (y = x)
    hi = max(np.nanmax(half_var), np.nanmax(jensen_gap)) * 1.08
    ax.plot([0, hi], [0, hi], color=COLORS["grey"], lw=0.8, ls="--", zorder=1)

    ax.set_xlabel(r"$\frac{1}{2}\mathrm{Var}(\Delta m)$", fontsize=8)
    ax.set_ylabel(r"$\mathbb{E}[f(m{+}\Delta m)] - f(m{+}\mathbb{E}[\Delta m])$", fontsize=8)
    ax.tick_params(axis='both', labelsize=6)

    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal", adjustable="box")

    add_grid(ax, alpha=0.2)

    # Annotation showing slope = 1 (exact for quadratic)
    ax.text(
        0.96, 0.04,
        r"slope $= 1$",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=7,
    )

    ax.set_title(r"(b) Jensen gap identity", fontsize=8, pad=4)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    plt.tight_layout()

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    saved = save_figure(fig, out_path)
    plt.close(fig)

    print(f"\nSaved: {', '.join(repo_relpath(p) for p in saved)}")


if __name__ == "__main__":
    main()
