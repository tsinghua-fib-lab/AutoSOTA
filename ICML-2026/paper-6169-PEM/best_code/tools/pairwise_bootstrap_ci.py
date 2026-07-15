#!/usr/bin/env python3
"""
Paired bootstrap confidence intervals from a `bbob_summary.csv`.

Use case:
- Sign-test is robust but discards magnitude information.
- Wilcoxon uses ranks of |Δ| but still doesn't give a direct effect-size CI.

This script computes paired differences per (budget,function,dimension,instance)
and reports bootstrap percentile CIs for:
  - mean and median of Δ = transform(best_f_a) - transform(best_f_b)
  - win-rate of A (fraction with best_f_a < best_f_b), ties counted separately

Notes:
- This avoids SciPy and uses NumPy only.
- Default transform is log10(best_f + eps) to stabilize heavy-tailed scales.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

from _project import repo_relpath


def _read_summary(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "algorithm": str(row["algorithm"]),
                    "budget_multiplier": int(row["budget_multiplier"]),
                    "function": int(row["function"]),
                    "dimension": int(row["dimension"]),
                    "instance": int(row["instance"]),
                    "best_f": float(row["best_f"]),
                }
            )
    return rows


def _is_tie(a: float, b: float, *, atol: float, rtol: float) -> bool:
    return abs(a - b) <= float(atol) + float(rtol) * max(abs(a), abs(b), 1.0)


def _transform(x: float, *, kind: str, eps: float) -> float:
    x = float(x)
    if kind == "none":
        return x
    if kind == "log10":
        return math.log10(max(float(eps), x))
    raise ValueError(f"Unknown transform: {kind}")


def _bootstrap_percentile_ci(values: np.ndarray, *, alpha: float) -> tuple[float, float]:
    lo = float(np.quantile(values, alpha / 2.0))
    hi = float(np.quantile(values, 1.0 - alpha / 2.0))
    return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory containing bbob_summary.csv")
    parser.add_argument("--algo-a", required=True)
    parser.add_argument("--algo-b", required=True)
    parser.add_argument("--transform", default="log10", choices=["log10", "none"])
    parser.add_argument("--eps", type=float, default=1e-300, help="Epsilon used for log10 transform.")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for tie counting (on best_f).")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance for tie counting (on best_f).")
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided CI level: 1-alpha.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch", type=int, default=2000, help="Bootstrap batch size (memory control).")
    parser.add_argument(
        "--output-json",
        default="",
        help="Default: <results-dir>/pairwise_bootstrap_ci_<a>_vs_<b>.json",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(str(args.results_dir))
    summary_path = os.path.join(results_dir, "bbob_summary.csv")
    if not os.path.isfile(summary_path):
        raise SystemExit(f"Missing: {summary_path}")

    rows = _read_summary(summary_path)
    algos = sorted({r["algorithm"] for r in rows})
    if str(args.algo_a) not in algos:
        raise SystemExit(f"Unknown --algo-a: {args.algo_a} (available: {', '.join(algos)})")
    if str(args.algo_b) not in algos:
        raise SystemExit(f"Unknown --algo-b: {args.algo_b} (available: {', '.join(algos)})")

    grouped: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r["budget_multiplier"], r["function"], r["dimension"], r["instance"])
        grouped[key][r["algorithm"]] = float(r["best_f"])

    diffs: list[float] = []
    wins_a = 0
    wins_b = 0
    ties = 0
    compared = 0
    for _, vals in grouped.items():
        if str(args.algo_a) not in vals or str(args.algo_b) not in vals:
            continue
        compared += 1
        a = float(vals[str(args.algo_a)])
        b = float(vals[str(args.algo_b)])
        if _is_tie(a, b, atol=float(args.atol), rtol=float(args.rtol)):
            ties += 1
        elif a < b:
            wins_a += 1
        else:
            wins_b += 1

        da = _transform(a, kind=str(args.transform), eps=float(args.eps))
        db = _transform(b, kind=str(args.transform), eps=float(args.eps))
        diffs.append(float(da - db))  # <0 means A better

    n = int(len(diffs))
    if n <= 0:
        raise SystemExit("No comparable pairs found.")

    diffs_np = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(int(args.seed))
    n_boot = int(args.n_bootstrap)
    batch = max(1, int(args.batch))

    boot_mean: list[np.ndarray] = []
    boot_median: list[np.ndarray] = []
    boot_win_rate: list[np.ndarray] = []

    done = 0
    while done < n_boot:
        bsz = min(batch, n_boot - done)
        idx = rng.integers(0, n, size=(bsz, n), dtype=np.int64)
        sample = diffs_np[idx]
        boot_mean.append(sample.mean(axis=1))
        boot_median.append(np.median(sample, axis=1))
        boot_win_rate.append((sample < 0.0).mean(axis=1))
        done += bsz

    boot_mean_np = np.concatenate(boot_mean, axis=0)
    boot_median_np = np.concatenate(boot_median, axis=0)
    boot_win_np = np.concatenate(boot_win_rate, axis=0)

    alpha = float(args.alpha)
    out: dict[str, object] = {
        "results_dir": repo_relpath(results_dir),
        "algo_a": str(args.algo_a),
        "algo_b": str(args.algo_b),
        "transform": str(args.transform),
        "eps": float(args.eps),
        "n_bootstrap": int(n_boot),
        "alpha": float(alpha),
        "seed": int(args.seed),
        "compared": int(compared),
        "wins_a": int(wins_a),
        "wins_b": int(wins_b),
        "ties": int(ties),
        "n_pairs": int(n),
        "delta": {
            "mean": float(diffs_np.mean()),
            "median": float(np.median(diffs_np)),
            "win_rate_a": float((diffs_np < 0.0).mean()),
        },
        "ci_percentile": {
            "mean": dict(zip(["lo", "hi"], _bootstrap_percentile_ci(boot_mean_np, alpha=alpha))),
            "median": dict(zip(["lo", "hi"], _bootstrap_percentile_ci(boot_median_np, alpha=alpha))),
            "win_rate_a": dict(zip(["lo", "hi"], _bootstrap_percentile_ci(boot_win_np, alpha=alpha))),
        },
    }

    out_path = str(args.output_json).strip()
    if not out_path:
        safe_a = str(args.algo_a).replace("/", "_").replace(" ", "_")
        safe_b = str(args.algo_b).replace("/", "_").replace(" ", "_")
        out_path = os.path.join(results_dir, f"pairwise_bootstrap_ci_{safe_a}_vs_{safe_b}.json")
    out_parent = os.path.dirname(out_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("Wrote:", repo_relpath(out_path))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
