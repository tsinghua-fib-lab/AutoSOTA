#!/usr/bin/env python3
"""
Pairwise win/tie/loss counts + exact sign test from a `bbob_summary.csv`.

This is intended as a lightweight sanity check:
- win/loss defined on final `best_f` per (budget,function,dimension,instance),
- ties can be controlled via atol/rtol to avoid over-interpreting near-equalities,
- p-values are exact two-sided sign tests (ties dropped).

It is *not* a replacement for COCO's official ERT/ECDF statistics.
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from itertools import combinations

from _project import BASE_DIR, repo_relpath

def read_summary(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "algorithm": row["algorithm"],
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory containing bbob_summary.csv")
    parser.add_argument(
        "--algorithms",
        default="",
        help="Comma-separated subset of algorithms (default: all found in bbob_summary.csv).",
    )
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for ties.")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance for ties.")
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path (default: <results-dir>/pairwise_sign_test.csv).",
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    results_dir = os.path.abspath(args.results_dir)
    summary_path = os.path.join(results_dir, "bbob_summary.csv")
    if not os.path.isfile(summary_path):
        raise SystemExit(f"Missing: {summary_path}")

    rows = read_summary(summary_path)
    algos_all = sorted({r["algorithm"] for r in rows})
    if not algos_all:
        raise SystemExit("No rows found.")

    algos = algos_all
    if str(args.algorithms).strip():
        want = [a.strip() for a in str(args.algorithms).split(",") if a.strip()]
        missing = [a for a in want if a not in algos_all]
        if missing:
            raise SystemExit("Unknown algorithms: " + ", ".join(missing))
        algos = want

    grouped: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["algorithm"] not in algos:
            continue
        key = (r["budget_multiplier"], r["function"], r["dimension"], r["instance"])
        grouped[key][r["algorithm"]] = float(r["best_f"])

    out_rows: list[dict] = []
    for a, b in combinations(algos, 2):
        wins_a = 0
        wins_b = 0
        ties = 0
        compared = 0
        for _, vals in grouped.items():
            if a not in vals or b not in vals:
                continue
            compared += 1
            fa = float(vals[a])
            fb = float(vals[b])
            if is_tie(fa, fb, atol=args.atol, rtol=args.rtol):
                ties += 1
            elif fa < fb:
                wins_a += 1
            else:
                wins_b += 1
        n = wins_a + wins_b
        p = sign_test_p_two_sided(wins_a, n)
        out_rows.append(
            {
                "algo_a": a,
                "algo_b": b,
                "wins_a": wins_a,
                "wins_b": wins_b,
                "ties": ties,
                "compared": compared,
                "n_non_ties": n,
                "win_rate_a": (wins_a / n) if n else float("nan"),
                "p_two_sided": p,
            }
        )

    out_path = str(args.output).strip() or os.path.join(results_dir, "pairwise_sign_test.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algo_a",
                "algo_b",
                "wins_a",
                "wins_b",
                "ties",
                "compared",
                "n_non_ties",
                "win_rate_a",
                "p_two_sided",
            ],
        )
        writer.writeheader()
        for row in sorted(out_rows, key=lambda r: (r["algo_a"], r["algo_b"])):
            writer.writerow(row)

    print("Wrote:", repo_relpath(out_path))


if __name__ == "__main__":
    main()
