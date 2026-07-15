#!/usr/bin/env python3
"""
Pairwise win/tie/loss + exact sign test for generic per-run CSVs.

This complements `tools/pairwise_sign_test.py` (which is bbob_summary-specific).

Expected input: a CSV with columns:
- algorithm
- seed (or another grouping key)
- a numeric metric column (e.g., post_cvar, post_mean, final_cost, ...)

Within each group key, compare two algorithms on the chosen metric.
Lower is better by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from itertools import combinations

from _project import repo_relpath

def sign_test_p_two_sided(wins: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    wins = int(wins)
    n = int(n)
    denom = 1 << n  # 2**n
    lo = sum(math.comb(n, k) for k in range(0, wins + 1))
    hi = sum(math.comb(n, k) for k in range(wins, n + 1))
    p = 2.0 * min(lo / denom, hi / denom)
    return float(min(1.0, p))


def is_tie(a: float, b: float, *, atol: float, rtol: float) -> bool:
    return abs(a - b) <= float(atol) + float(rtol) * max(abs(a), abs(b), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", required=True, help="Input CSV path (must contain 'algorithm' and metric).")
    parser.add_argument("--metric", required=True, help="Metric column name to compare.")
    parser.add_argument("--group-by", default="seed", help="Column used to group paired comparisons (default: seed).")
    parser.add_argument("--lower-is-better", action="store_true", help="Treat lower metric values as better (default).")
    parser.add_argument("--higher-is-better", action="store_true", help="Treat higher metric values as better.")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--output", default="", help="Output CSV path (default: alongside runs.csv).")
    args = parser.parse_args()

    if bool(args.higher_is_better) and bool(args.lower_is_better):
        raise SystemExit("Choose at most one of --lower-is-better/--higher-is-better.")
    higher_better = bool(args.higher_is_better)

    in_path = os.path.abspath(str(args.runs_csv))
    if not os.path.isfile(in_path):
        raise SystemExit(f"Missing: {in_path}")

    rows = []
    with open(in_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    metric = str(args.metric)
    group_by = str(args.group_by)

    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        algo = str(row.get("algorithm", "")).strip()
        if not algo:
            continue
        group = str(row.get(group_by, "")).strip()
        if not group:
            continue
        if metric not in row:
            raise SystemExit(f"Missing metric column '{metric}' in: {in_path}")
        val = float(row[metric])
        grouped[group][algo] = float(val)

    algos = sorted({a for g in grouped.values() for a in g.keys()})
    if len(algos) < 2:
        raise SystemExit("Need at least 2 algorithms in runs CSV.")

    out_rows = []
    for a, b in combinations(algos, 2):
        wins_a = 0
        wins_b = 0
        ties = 0
        compared = 0
        for _, vals in grouped.items():
            if a not in vals or b not in vals:
                continue
            compared += 1
            va = float(vals[a])
            vb = float(vals[b])
            if is_tie(va, vb, atol=float(args.atol), rtol=float(args.rtol)):
                ties += 1
                continue
            if higher_better:
                better_a = va > vb
            else:
                better_a = va < vb
            if better_a:
                wins_a += 1
            else:
                wins_b += 1
        n = wins_a + wins_b
        p = sign_test_p_two_sided(wins_a, n)
        out_rows.append(
            {
                "algo_a": a,
                "algo_b": b,
                "metric": metric,
                "group_by": group_by,
                "wins_a": wins_a,
                "wins_b": wins_b,
                "ties": ties,
                "compared": compared,
                "n_non_ties": n,
                "win_rate_a": (wins_a / n) if n else float("nan"),
                "p_two_sided": p,
            }
        )

    out_path = str(args.output).strip()
    if not out_path:
        out_path = os.path.join(os.path.dirname(in_path), f"pairwise_sign_test_{metric}.csv")
    out_path = os.path.abspath(out_path)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "algo_a",
                "algo_b",
                "metric",
                "group_by",
                "wins_a",
                "wins_b",
                "ties",
                "compared",
                "n_non_ties",
                "win_rate_a",
                "p_two_sided",
            ],
        )
        w.writeheader()
        for row in sorted(out_rows, key=lambda r: (r["algo_a"], r["algo_b"])):
            w.writerow(row)

    print("Wrote:", repo_relpath(out_path))


if __name__ == "__main__":
    main()
