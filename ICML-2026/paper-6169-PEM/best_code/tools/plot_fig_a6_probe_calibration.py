#!/usr/bin/env python3
"""
Generate Figure A6: Probe calibration curves for B=200d and B=500d.

This figure demonstrates that the misranking probe statistic is a reliable
predictor of algorithm selection (supports claim C5). The calibration curves
show Pr(Residual bootstrapping wins | P) as a function of the probe value P,
with Wilson confidence intervals.

Data source:
  - evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200/decision_points.csv
  - evidence/bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500/decision_points.csv
  - Threshold JSON files for train/test split and selected thresholds

Output: evidence/paper_figures/Appendix/fig_a6_probe_calibration.pdf

Usage:
    python tools/plot_fig_a6_probe_calibration.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _project import BASE_DIR, repo_relpath
from plot_style import apply_style, save_figure, get_subplot_figsize, ALGO_COLORS


def wilson_ci(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    k = int(max(0, min(int(k), int(n))))
    n_f = float(n)
    p = float(k) / n_f
    denom = 1.0 + (z * z) / n_f
    center = (p + (z * z) / (2.0 * n_f)) / denom
    rad = (z / denom) * float(np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n_f)) / n_f))
    return (float(max(0.0, center - rad)), float(min(1.0, center + rad)))


def quantile_bins(x: np.ndarray, n_bins: int) -> list[tuple[float, float]]:
    """Create quantile-based bins from probe values."""
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(x, qs)
    out: list[tuple[float, float]] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if not out:
            out.append((float(a), float(b)))
        else:
            prev_a, prev_b = out[-1]
            if float(a) <= float(prev_b) + 1e-18:
                a = prev_b
            if float(b) <= float(a) + 1e-18:
                continue
            out.append((float(a), float(b)))
    return out


def load_threshold_json(path: str) -> tuple[float | None, set[int] | None]:
    """Load threshold and test split from JSON file."""
    try:
        with open(path) as f:
            obj = json.load(f)
        thr = obj.get("selected_threshold", None)
        thr_val = float(thr) if thr is not None else None
        split = obj.get("split", {})
        test = split.get("test_instances", None)
        test_set = set(int(t) for t in test) if isinstance(test, list) else None
        return (thr_val, test_set)
    except Exception:
        return (None, None)


def load_decision_points(
    csv_path: str,
    probe_key: str,
    test_instances: set[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load decision points and return (probe_values, labels).

    Labels: 1 = residual bootstrapping wins (cocp/berw), 0 = CMA-ES wins.
    Only includes test split instances if test_instances is provided.
    """
    rows: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    pts = []
    for r in rows:
        lab = str(r.get("label", "")).strip().lower()
        # Accept both "berw" and "cocp" as residual bootstrapping wins
        if lab not in {"cma", "berw", "cocp"}:
            continue
        inst = int(float(r.get("instance", "0") or 0))
        if test_instances is not None and inst not in test_instances:
            continue
        probe_val = r.get(probe_key, "")
        try:
            s = float(probe_val)
        except (ValueError, TypeError):
            continue
        if not np.isfinite(s):
            continue
        # y=1 means residual bootstrapping wins
        y = 1 if lab in {"berw", "cocp"} else 0
        pts.append((s, y))

    if not pts:
        return np.array([]), np.array([])

    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=int)
    return x, y


def compute_calibration_bins(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """
    Compute calibration bins with empirical win rates and Wilson CIs.

    Returns list of dicts with keys: x_mid, x_lo, x_hi, n, rate, ci_lo, ci_hi
    """
    bins = quantile_bins(x, n_bins=n_bins)
    out_bins = []

    for i, (lo, hi) in enumerate(bins):
        if i == len(bins) - 1:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)

        yy = y[mask]
        xx = x[mask]
        if yy.size <= 0:
            continue

        k = int(np.sum(yy))
        n = int(yy.size)
        rate = float(k) / float(n)
        ci_lo, ci_hi = wilson_ci(k, n)

        out_bins.append({
            "x_mid": float(np.median(xx)),
            "x_lo": float(lo),
            "x_hi": float(hi),
            "n": n,
            "rate": rate,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })

    return out_bins


def find_crossover_point(xs: np.ndarray, ys: np.ndarray, y_target: float = 0.5) -> float | None:
    """Find the x value where the curve crosses y_target via linear interpolation."""
    for i in range(len(ys) - 1):
        if (ys[i] <= y_target <= ys[i + 1]) or (ys[i] >= y_target >= ys[i + 1]):
            # Linear interpolation
            if abs(ys[i + 1] - ys[i]) < 1e-10:
                continue
            t = (y_target - ys[i]) / (ys[i + 1] - ys[i])
            x_cross = xs[i] + t * (xs[i + 1] - xs[i])
            return float(x_cross)
    return None


def plot_calibration_panel(
    ax: plt.Axes,
    bins: list[dict],
    threshold: float | None,
    title: str,
    show_ylabel: bool = True,
) -> None:
    """Plot a single calibration panel."""
    if not bins:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        return

    xs = np.array([b["x_mid"] for b in bins])
    ys = np.array([b["rate"] for b in bins])
    yerr_lo = ys - np.array([b["ci_lo"] for b in bins])
    yerr_hi = np.array([b["ci_hi"] for b in bins]) - ys

    # BERW blue color from style
    color = ALGO_COLORS.get("BERW-Hetero", "#0077BB")

    # Find where curve actually crosses Y=0.5
    xlim = (-0.02, 0.42)  # Add padding to avoid error bars touching axes
    crossover = find_crossover_point(xs, ys, 0.5)

    # Use crossover point for vertical split (fall back to threshold if not found)
    split_x = crossover if crossover is not None else threshold

    # Plot background shading based on actual crossover point
    # Only shade from x=0 (probe values are non-negative)
    if split_x is not None and np.isfinite(split_x):
        ax.axvspan(0, split_x, alpha=0.10, color="#EE6677", zorder=0)  # Red: CMA-ES better
        ax.axvspan(split_x, xlim[1], alpha=0.10, color="#4477AA", zorder=0)  # Blue: Res. boot. better

        # Labels for regions - both on single line
        ax.text(
            split_x / 2, 0.75, "CMA-ES better",
            ha="center", va="center", fontsize=6, color="#AA4455", fontweight="medium",
        )
        ax.text(
            split_x + (xlim[1] - split_x) / 2, 0.25, "Residual Bootstrapping better",
            ha="center", va="center", fontsize=5.5, color="#3366AA", fontweight="medium",
        )

    # Reference line: Y=0.5 is the decision boundary
    ax.axhline(0.5, color="#666666", linestyle="-", linewidth=0.8, zorder=1, alpha=0.6)

    # Threshold vertical line - this is our chosen decision rule
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(
            threshold, color="#CC3311", linestyle="--", linewidth=1.0,
            zorder=2, alpha=0.95,
        )
        ax.text(
            threshold + 0.008, 0.97, f"$\\tau\\!=\\!{threshold:.2f}$",
            color="#CC3311", fontsize=7, fontweight="bold", va="top", ha="left",
        )

    # Plot calibration curve with error bars (thinner line)
    ax.errorbar(
        xs, ys,
        yerr=[yerr_lo, yerr_hi],
        fmt="o-",
        color=color,
        linewidth=1.2,
        markersize=5,
        capsize=2.5,
        capthick=0.7,
        alpha=0.95,
        zorder=3,
    )

    # Axis configuration
    ax.set_xlim(xlim)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Misranking probe $P$", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Pr(Res. boot. wins $|$ $P$)", fontsize=8)
    ax.set_title(title, fontsize=9, pad=6)

    # Smaller tick labels
    ax.tick_params(axis='both', labelsize=6)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure A6: Probe calibration curves"
    )
    parser.add_argument(
        "--evidence-dir",
        default="evidence",
        help="Evidence directory (relative to repo/)",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of quantile bins for calibration curve",
    )
    parser.add_argument(
        "--output",
        default="evidence/paper_figures/Appendix/fig_a6_probe_calibration",
        help="Output path (without extension)",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    # Configuration for the two panels
    panels = [
        {
            "name": "B200",
            "title": r"(a) $B = 200D$",
            "csv": os.path.join(
                args.evidence_dir,
                "bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200",
                "decision_points.csv",
            ),
            "threshold_json": os.path.join(
                args.evidence_dir,
                "bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B200",
                "train_test_threshold_misranking_rd_log10_regret_mean.json",
            ),
        },
        {
            "name": "B500",
            "title": r"(b) $B = 500D$",
            "csv": os.path.join(
                args.evidence_dir,
                "bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500",
                "decision_points.csv",
            ),
            "threshold_json": os.path.join(
                args.evidence_dir,
                "bbob_noisy_probe_decision_accuracy_noisefree_i1-15_B500",
                "train_test_threshold_misranking_rd_log10_regret_mean.json",
            ),
        },
    ]

    # Load data for each panel
    for panel in panels:
        threshold, test_instances = load_threshold_json(panel["threshold_json"])
        panel["threshold"] = threshold
        panel["test_instances"] = test_instances

        if not os.path.isfile(panel["csv"]):
            print(f"WARNING: Missing {repo_relpath(panel['csv'])}")
            panel["bins"] = []
            continue

        x, y = load_decision_points(
            panel["csv"],
            probe_key="misranking_rd",
            test_instances=test_instances,
        )

        if x.size == 0:
            print(f"WARNING: No data after filtering for {panel['name']}")
            panel["bins"] = []
            continue

        panel["bins"] = compute_calibration_bins(x, y, n_bins=args.n_bins)
        print(
            f"{panel['name']}: {x.size} points, {len(panel['bins'])} bins, "
            f"threshold={threshold}"
        )

    # Apply style and create figure
    apply_style()

    fig, axes = plt.subplots(
        1, 2,
        figsize=get_subplot_figsize(1, 2, width="double", subplot_aspect=0.75),
    )

    for idx, panel in enumerate(panels):
        plot_calibration_panel(
            axes[idx],
            panel["bins"],
            panel["threshold"],
            panel["title"],
            show_ylabel=True,
        )

    plt.tight_layout()

    # Save figure
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    saved = save_figure(fig, out_path)
    plt.close(fig)

    print(f"Saved: {', '.join(repo_relpath(p) for p in saved)}")


if __name__ == "__main__":
    main()
