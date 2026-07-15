#!/usr/bin/env python3
"""
Plot an empirical conditional advantage curve for ProbeSwitch:

  Δ(p) = E[ log10(best_f_CMA) - log10(best_f_BERW) | probe_value = p ]

If Δ(p) is approximately increasing and crosses 0 once, it supports the
single-crossing assumption used to justify a threshold policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from _project import BASE_DIR, repo_relpath
from plot_style import apply_style, get_figsize, add_grid, save_figure, ALGO_COLORS


@dataclass(frozen=True)
class BinStats:
    p_mid: float
    p_lo: float
    p_hi: float
    n: int
    mean: float
    stderr: float


def _load_threshold(path: str) -> float | None:
    try:
        with open(path) as f:
            obj = json.load(f)
        v = obj.get("selected_threshold", None)
        return float(v) if v is not None else None
    except Exception:
        return None


def _quantile_bins(p: np.ndarray, n_bins: int) -> list[tuple[float, float]]:
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(p, qs)
    # Ensure strict monotonic edges (collapse duplicates).
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


def _compute_bins(p: np.ndarray, delta: np.ndarray, *, n_bins: int) -> list[BinStats]:
    bins = _quantile_bins(p, n_bins=n_bins)
    out: list[BinStats] = []
    for lo, hi in bins:
        # Include left-closed; right-open except for last bin.
        if hi == bins[-1][1]:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        vals = delta[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size <= 0:
            continue
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0
        stderr = sd / float(np.sqrt(max(1, vals.size)))
        out.append(
            BinStats(
                p_mid=float(0.5 * (lo + hi)),
                p_lo=float(lo),
                p_hi=float(hi),
                n=int(vals.size),
                mean=mean,
                stderr=float(stderr),
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-points", required=True, help="decision_points.csv")
    parser.add_argument("--probe-key", default="misranking_rd", help="Probe column name.")
    parser.add_argument("--eps", type=float, default=1e-12, help="Epsilon for log10.")
    parser.add_argument("--n-bins", type=int, default=12, help="Quantile bins for probe value.")
    parser.add_argument(
        "--threshold-json",
        default="",
        help="Optional train_test_threshold_*.json file (draws vertical line).",
    )
    parser.add_argument("--threshold", type=float, default=float("nan"), help="Optional threshold value.")
    parser.add_argument("--functions", default="", help="Comma-separated function indices to include (e.g., '10,11,13,26,30').")
    parser.add_argument("--isotonic", action="store_true", help="Apply isotonic regression for monotonic curve.")
    parser.add_argument("--lowess", action="store_true", help="Apply LOWESS smoothing.")
    parser.add_argument("--out", required=True, help="Output path (without extension; saves pdf+png).")
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    rows: list[dict[str, str]] = []
    with open(args.decision_points, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise SystemExit("Empty decision_points.csv")

    def getf(row: dict[str, str], key: str) -> float:
        v = row.get(key, "")
        return float(v) if v not in ("", None) else float("nan")

    # Parse function filter
    func_filter: set[int] | None = None
    if args.functions.strip():
        func_filter = set(int(x.strip()) for x in args.functions.split(",") if x.strip())

    p = np.array([getf(r, args.probe_key) for r in rows], dtype=float)
    best_cma = np.array([getf(r, "best_f_cma") for r in rows], dtype=float)
    best_berw = np.array([getf(r, "best_f_berw") for r in rows], dtype=float)
    func_idx = np.array([int(getf(r, "function_index")) for r in rows], dtype=int)

    eps = float(max(0.0, args.eps))
    delta = np.log10(np.maximum(best_cma, eps)) - np.log10(np.maximum(best_berw, eps))

    ok = np.isfinite(p) & np.isfinite(delta)
    if func_filter is not None:
        ok = ok & np.isin(func_idx, list(func_filter))
        print(f"Filtering to functions: {sorted(func_filter)}")

    p = p[ok]
    delta = delta[ok]
    if p.size <= 0:
        raise SystemExit("No finite rows after filtering.")

    stats = _compute_bins(p, delta, n_bins=int(args.n_bins))
    if not stats:
        raise SystemExit("No bins produced (check probe distribution).")

    xs = np.array([s.p_mid for s in stats], dtype=float)
    ys = np.array([s.mean for s in stats], dtype=float)
    es = np.array([1.96 * s.stderr for s in stats], dtype=float)  # ~95% CI under normal approx

    # Apply isotonic regression if requested (ensures monotonically increasing curve)
    if args.isotonic:
        try:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(increasing=True)
            ys_iso = ir.fit_transform(xs, ys)
            print(f"Applied isotonic regression: original range [{ys.min():.3f}, {ys.max():.3f}] -> [{ys_iso.min():.3f}, {ys_iso.max():.3f}]")
            ys = ys_iso
        except ImportError:
            print("Warning: sklearn not available, skipping isotonic regression")

    # Apply LOWESS smoothing if requested (local regression, more natural smoothing)
    if args.lowess:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(ys, xs, frac=0.4, return_sorted=False)
            print(f"Applied LOWESS smoothing: original range [{ys.min():.3f}, {ys.max():.3f}] -> [{smoothed.min():.3f}, {smoothed.max():.3f}]")
            ys = smoothed
        except ImportError:
            print("Warning: statsmodels not available, skipping LOWESS smoothing")

    thr = float("nan")
    if str(args.threshold_json).strip():
        thr = _load_threshold(str(args.threshold_json).strip()) or float("nan")
    if np.isfinite(float(args.threshold)):
        thr = float(args.threshold)

    # Apply unified style
    apply_style()

    # Create figure with standard sizing
    fig, ax = plt.subplots(figsize=get_figsize("single", aspect=0.7))

    # Determine x limits
    xlim = (xs.min() - 0.02, xs.max() + 0.02)
    ax.set_xlim(xlim)

    # Background region fills — split at the zero-crossing of Δ(p), not at τ.
    # Find where the curve crosses y=0 by linear interpolation.
    zero_x = xlim[0]  # fallback: entire region is RB-better
    for k in range(len(ys) - 1):
        if ys[k] * ys[k + 1] < 0:  # sign change
            # linear interpolation
            zero_x = xs[k] - ys[k] * (xs[k + 1] - xs[k]) / (ys[k + 1] - ys[k])
            break
    ax.axvspan(xlim[0], zero_x, color=ALGO_COLORS["CMA-ES"], alpha=0.08, zorder=0)
    ax.axvspan(zero_x, xlim[1], color=ALGO_COLORS["BERW"], alpha=0.08, zorder=0)

    # Confidence band (95% CI)
    ax.fill_between(xs, ys - es, ys + es, color=ALGO_COLORS["BERW"], alpha=0.2, linewidth=0)

    # Main curve
    ax.plot(xs, ys, "o-", color=ALGO_COLORS["BERW"], lw=1.0, ms=4, zorder=3)

    # Zero reference line
    ax.axhline(0.0, color="#888888", lw=0.6, alpha=0.8, zorder=1)

    # Threshold line and region labels
    if np.isfinite(thr):
        ax.axvline(thr, color=ALGO_COLORS["CMA-ES"], lw=1.0, ls="--", alpha=0.9, zorder=2)

        # Get y limits after plotting data
        ymin, ymax = ax.get_ylim()

        # Threshold label (positioned to not obscure data)
        ax.text(thr + 0.01, ymax * 0.95, f"τ = {thr:.2f}",
                color=ALGO_COLORS["CMA-ES"], fontsize=7, va="top")

        # Decision region labels (centered in their respective colored regions)
        ax.text((xlim[0] + zero_x) / 2, ymax * 0.55,
                "CMA-ES\nbetter", ha="center", va="center",
                fontsize=7, color="#666666", alpha=0.8)
        ax.text((zero_x + xlim[1]) / 2, ymin * 0.35,
                "Residual Bootstrap better", ha="center", va="center",
                fontsize=7, color="#666666", alpha=0.8)

    # Axis labels (simplified)
    ax.set_xlabel("Probe statistic (misranking rate)")
    ax.set_ylabel("Conditional advantage Δ(p)")

    # Grid
    add_grid(ax)

    # Save figure (pdf + png)
    out_path = os.path.abspath(str(args.out))
    saved = save_figure(fig, out_path)
    plt.close(fig)
    for p in saved:
        print("Wrote:", repo_relpath(p))


if __name__ == "__main__":
    main()
