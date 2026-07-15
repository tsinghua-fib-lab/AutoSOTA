#!/usr/bin/env python3
"""
Plot bbob-noisy per-function win-rate vs measured misranking severity.

Inputs:
- A merged results dir with `bbob_summary.csv` (e.g. from `run_coco_bbob_noisy_parallel.py`).
- A misranking-per-function CSV (e.g. from `tools/summarize_misranking_by_function.py`).

This helps validate the "misranking regime" story: algorithms designed to handle
misranking should win more often on functions with higher misranking severity.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from _project import repo_relpath

def read_misranking_by_function(path: str) -> dict[int, float]:
    out: dict[int, float] = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            suite = str(row.get("suite", "")).strip()
            if suite and suite != "bbob-noisy":
                continue
            func = int(float(row["function"]))
            dim = int(float(row["dimension"]))
            if dim != 40:
                continue
            out[func] = float(row["rank_disagreement_mean"])
    return out


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Merged results dir containing bbob_summary.csv")
    parser.add_argument("--misranking-by-func", required=True, help="CSV with rank_disagreement_mean per function")
    parser.add_argument("--baseline", default="CMA-ES-sep")
    parser.add_argument("--compare", required=True)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--dimension", type=int, default=40)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary_path = os.path.join(os.path.abspath(args.results_dir), "bbob_summary.csv")
    rows = read_summary(summary_path)
    rows = [r for r in rows if r["budget_multiplier"] == int(args.budget) and r["dimension"] == int(args.dimension)]

    by_key: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r["function"], r["instance"])
        by_key[key][r["algorithm"]] = float(r["best_f"])

    wins = defaultdict(int)
    losses = defaultdict(int)
    ties = defaultdict(int)
    compared = defaultdict(int)

    for (func, _inst), vals in by_key.items():
        if args.baseline not in vals or args.compare not in vals:
            continue
        a = float(vals[args.baseline])
        b = float(vals[args.compare])
        compared[func] += 1
        if b < a:
            wins[func] += 1
        elif b > a:
            losses[func] += 1
        else:
            ties[func] += 1

    mis = read_misranking_by_function(os.path.abspath(args.misranking_by_func))

    xs = []
    ys = []
    labels = []
    sizes = []
    for func, n in sorted(compared.items()):
        if func not in mis:
            continue
        w = wins[func]
        l = losses[func]
        t = ties[func]
        nn = w + l
        win_rate = (w / nn) if nn > 0 else 0.5
        xs.append(float(mis[func]))
        ys.append(float(win_rate))
        labels.append(str(func))
        sizes.append(20.0 + 2.0 * float(n))

    if not xs:
        raise SystemExit("No points to plot (check inputs / filters).")

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)

    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(xs_arr, ys_arr, s=sizes, alpha=0.85, color="#2b6cb0")
    for x, y, lab in zip(xs_arr, ys_arr, labels):
        plt.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)
    plt.axhline(0.5, color="#64748B", linewidth=1.0, alpha=0.7)
    plt.xlabel("misranking severity (mean rank_disagreement)")
    plt.ylabel(f"win-rate of {args.compare} over {args.baseline} (ties ignored)")
    plt.title(f"bbob-noisy D={args.dimension} B={args.budget}x | per-function win-rate vs misranking")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()
    print("Wrote:", repo_relpath(out_path))


if __name__ == "__main__":
    main()
