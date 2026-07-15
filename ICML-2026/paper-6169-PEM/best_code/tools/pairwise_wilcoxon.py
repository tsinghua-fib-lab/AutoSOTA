#!/usr/bin/env python3
"""
Paired Wilcoxon signed-rank test (normal approximation) from a `bbob_summary.csv`.

Why:
- Sign-test is robust but discards magnitude information.
- Wilcoxon signed-rank uses ranks of |Δ| and is still nonparametric.

This implementation avoids SciPy and is intended for lightweight sanity checks,
not as a replacement for COCO's official statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict

from _project import repo_relpath


def sanitize_token(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower())
    s = s.strip("_")
    return s or "unnamed"


def read_summary(path: str) -> list[dict]:
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


def is_tie(a: float, b: float, *, atol: float, rtol: float) -> bool:
    return abs(a - b) <= float(atol) + float(rtol) * max(abs(a), abs(b), 1.0)


def normal_cdf(z: float) -> float:
    # Φ(z) = 0.5 * erfc(-z / sqrt(2))
    return 0.5 * math.erfc(-float(z) / math.sqrt(2.0))


def rank_abs_with_ties(diffs: list[float]) -> list[float]:
    """
    Return ranks (1..n) for |diff| with average ranks for ties.
    """

    n = int(len(diffs))
    abs_vals = [abs(float(x)) for x in diffs]
    order = sorted(range(n), key=lambda i: abs_vals[i])

    ranks = [0.0] * n
    i = 0
    cur_rank = 1
    eps = 0.0  # exact tie on float value only
    while i < n:
        j = i + 1
        while j < n and abs(abs_vals[order[j]] - abs_vals[order[i]]) <= eps:
            j += 1

        # ranks cur_rank .. cur_rank+(j-i)-1
        lo = cur_rank
        hi = cur_rank + (j - i) - 1
        avg = 0.5 * (lo + hi)
        for k in range(i, j):
            ranks[order[k]] = float(avg)
        cur_rank += j - i
        i = j

    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory containing bbob_summary.csv")
    parser.add_argument("--algo-a", required=True)
    parser.add_argument("--algo-b", required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument(
        "--output-json",
        default="",
        help="Default: <results-dir>/pairwise_wilcoxon_<a>_vs_<b>.json",
    )
    args = parser.parse_args()

    results_dir_abs = os.path.abspath(str(args.results_dir))
    summary_path = os.path.join(results_dir_abs, "bbob_summary.csv")
    if not os.path.isfile(summary_path):
        raise SystemExit(f"Missing: {summary_path}")

    rows = read_summary(summary_path)
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
        if is_tie(a, b, atol=float(args.atol), rtol=float(args.rtol)):
            ties += 1
            continue
        if a < b:
            wins_a += 1
        else:
            wins_b += 1
        diffs.append(a - b)  # <0 means A better

    n = int(len(diffs))
    out: dict[str, object] = {
        "results_dir": repo_relpath(results_dir_abs),
        "algo_a": str(args.algo_a),
        "algo_b": str(args.algo_b),
        "compared": int(compared),
        "ties_dropped": int(ties),
        "n_non_ties": int(n),
        "wins_a": int(wins_a),
        "wins_b": int(wins_b),
        "mean_diff_a_minus_b": float(sum(diffs) / n) if n else float("nan"),
        "median_diff_a_minus_b": float(sorted(diffs)[n // 2]) if n else float("nan"),
    }

    if n <= 0:
        out["wilcoxon"] = {"w_plus": float("nan"), "z": float("nan"), "p_two_sided": float("nan"), "r": float("nan")}
    else:
        ranks = rank_abs_with_ties(diffs)
        w_plus = float(sum(r for r, d in zip(ranks, diffs) if d > 0.0))
        mean = float(n * (n + 1) / 4.0)
        var = 0.25 * float(sum(float(r) * float(r) for r in ranks))
        sd = math.sqrt(var) if var > 0 else float("nan")

        # Continuity correction.
        cc = 0.0
        if w_plus > mean:
            cc = 0.5
        elif w_plus < mean:
            cc = -0.5

        z = (w_plus - mean - cc) / sd if sd and math.isfinite(sd) and sd > 0 else float("nan")
        p = 2.0 * min(normal_cdf(z), 1.0 - normal_cdf(z)) if math.isfinite(z) else float("nan")
        r_eff = abs(z) / math.sqrt(float(n)) if n and math.isfinite(z) else float("nan")
        out["wilcoxon"] = {
            "w_plus": float(w_plus),
            "mean_null": float(mean),
            "var_null": float(var),
            "z": float(z),
            "p_two_sided": float(min(1.0, p)) if math.isfinite(p) else float("nan"),
            "r": float(r_eff),
        }

    out_path = str(args.output_json).strip()
    if not out_path:
        out_path = os.path.join(
            results_dir_abs,
            f"pairwise_wilcoxon_{sanitize_token(args.algo_a)}_vs_{sanitize_token(args.algo_b)}.json",
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print("Wrote:", repo_relpath(out_path))


if __name__ == "__main__":
    main()
