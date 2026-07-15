#!/usr/bin/env python3
"""
Plot an empirical calibration curve for a probe score.

We treat the probe as a *score* and visualize:

  p_bin -> Pr(label="berw" | probe in bin)

This is a sanity check that the probe is not a brittle heuristic:
if the curve is roughly monotone, a threshold policy is well motivated.

Inputs:
  - decision_points.csv (from tools/probe_decision_accuracy.py)
  - optional train_test_threshold_*.json (to select the test split + draw threshold)
Outputs:
  - a PNG (and optionally PDF) calibration plot with Wilson CIs.
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
from utils.display_names import get_display_name

@dataclass(frozen=True)
class Bin:
    x_mid: float
    x_lo: float
    x_hi: float
    n: int
    rate: float
    ci_lo: float
    ci_hi: float


def _wilson_ci(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    k = int(max(0, min(int(k), int(n))))
    n_f = float(n)
    p = float(k) / n_f
    denom = 1.0 + (z * z) / n_f
    center = (p + (z * z) / (2.0 * n_f)) / denom
    rad = (z / denom) * float(np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n_f)) / n_f))
    return (float(max(0.0, center - rad)), float(min(1.0, center + rad)))


def _quantile_bins(x: np.ndarray, n_bins: int) -> list[tuple[float, float]]:
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


def _load_threshold_json(path: str) -> tuple[float | None, set[int] | None]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-points", required=True, help="decision_points.csv")
    parser.add_argument("--probe-key", default="misranking_rd", help="Probe column name.")
    parser.add_argument("--threshold-json", default="", help="Optional train_test_threshold_*.json (draw threshold).")
    parser.add_argument("--threshold", type=float, default=float("nan"), help="Optional threshold value.")
    parser.add_argument("--use-test-split", action="store_true", help="If --threshold-json is given, use its test split.")
    parser.add_argument("--n-bins", type=int, default=12, help="Quantile bins for probe value.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    parser.add_argument("--out-pdf", default="", help="Optional output PDF path.")
    parser.add_argument("--title", default="", help="Plot title.")
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    rows: list[dict[str, str]] = []
    with open(args.decision_points, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        raise SystemExit("Empty decision_points.csv")

    thr = None
    test_set = None
    if str(args.threshold_json).strip():
        thr, test_set = _load_threshold_json(str(args.threshold_json).strip())
    if np.isfinite(float(args.threshold)):
        thr = float(args.threshold)

    def getf(row: dict[str, str], key: str) -> float | None:
        v = row.get(key, "")
        v = str(v).strip()
        if not v:
            return None
        try:
            x = float(v)
        except ValueError:
            return None
        return float(x) if np.isfinite(x) else None

    pts = []
    for r in rows:
        lab = str(r.get("label", "")).strip()
        if lab not in {"cma", "berw"}:
            continue
        inst = int(float(r.get("instance", "0") or 0))
        if args.use_test_split and test_set is not None and inst not in test_set:
            continue
        s = getf(r, str(args.probe_key))
        if s is None:
            continue
        y = 1 if lab == "berw" else 0
        pts.append((float(s), int(y)))

    if not pts:
        raise SystemExit("No usable rows after filtering.")

    x = np.asarray([p[0] for p in pts], dtype=float)
    y = np.asarray([p[1] for p in pts], dtype=int)
    ok = np.isfinite(x)
    x = x[ok]
    y = y[ok]
    if x.size <= 0:
        raise SystemExit("No finite probe values.")

    bins = _quantile_bins(x, n_bins=int(args.n_bins))
    out_bins: list[Bin] = []
    for lo, hi in bins:
        if hi == bins[-1][1]:
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
        ci_lo, ci_hi = _wilson_ci(k, n)
        out_bins.append(
            Bin(
                x_mid=float(np.median(xx)),
                x_lo=float(lo),
                x_hi=float(hi),
                n=n,
                rate=rate,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
            )
        )

    if not out_bins:
        raise SystemExit("No bins produced (check probe distribution).")

    xs = np.asarray([b.x_mid for b in out_bins], dtype=float)
    ys = np.asarray([b.rate for b in out_bins], dtype=float)
    yerr_lo = ys - np.asarray([b.ci_lo for b in out_bins], dtype=float)
    yerr_hi = np.asarray([b.ci_hi for b in out_bins], dtype=float) - ys

    title = str(args.title).strip()
    if not title:
        suffix = " (test split)" if (args.use_test_split and test_set is not None) else ""
        title = f"Probe calibration: Pr({get_display_name('BERW-Hetero', short=True)} wins | probe){suffix}"

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=180, constrained_layout=True)
    ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="o-", lw=2, ms=5, capsize=3, alpha=0.95)
    ax.set_xlabel(args.probe_key)
    ax.set_ylabel(f"Pr(label = {get_display_name('BERW-Hetero', short=True)})")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.set_title(title)

    if thr is not None and np.isfinite(thr):
        ax.axvline(float(thr), color="#EF4444", lw=1.6, alpha=0.9, linestyle="--")
        ax.text(float(thr), 1.0, f"  τ={float(thr):.3f}", color="#B91C1C", va="top", fontsize=9)

    # Annotate sample counts (small, non-intrusive).
    for b in out_bins:
        ax.text(b.x_mid, max(0.02, b.rate - 0.08), f"n={b.n}", ha="center", va="top", fontsize=8, color="#334155")

    out_png = os.path.abspath(str(args.out))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)
    print("Wrote:", repo_relpath(out_png))
    if str(args.out_pdf).strip():
        out_pdf = os.path.abspath(str(args.out_pdf))
        os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
        fig.savefig(out_pdf)
        print("Wrote:", repo_relpath(out_pdf))
    plt.close(fig)


if __name__ == "__main__":
    main()
