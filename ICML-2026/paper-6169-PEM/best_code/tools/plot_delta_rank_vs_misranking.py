#!/usr/bin/env python3
"""
Plot performance delta vs measured misranking severity.

Inputs:
- `sweep_summary.csv` from `tools/summarize_noisy_wrapper_sweep.py`
- misranking CSVs from `tools/measure_misranking_severity.py` (one per sigma)

Outputs:
- delta_rank_vs_misranking.png
"""

import argparse
import csv
import glob
import os
import re

import matplotlib.pyplot as plt

from _project import BASE_DIR, repo_relpath

def read_sweep(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "sigma": float(row["sigma"]),
                    "algorithm": row["algorithm"],
                    "avg_rank": float(row["avg_rank"]),
                }
            )
    return rows


def parse_sigma_from_name(path: str) -> float:
    m = re.search(r"sigma([0-9]+(?:p[0-9]+)?)", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse sigma from filename: {path}")
    return float(m.group(1).replace("p", "."))


def read_misranking_mean(path: str) -> tuple[float, float]:
    rds = []
    ovs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rds.append(float(row["rank_disagreement"]))
            ovs.append(float(row["topmu_overlap"]))
    if not rds:
        return (float("nan"), float("nan"))
    return (sum(rds) / len(rds), sum(ovs) / len(ovs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-summary", required=True, help="Path to sweep_summary.csv")
    parser.add_argument("--misranking-glob", required=True, help="Glob for misranking CSVs (one per sigma).")
    parser.add_argument("--baseline", default="CMA-ES-sep", help="Baseline algorithm name.")
    parser.add_argument("--compare", required=True, help="Algorithm name to compare against baseline.")
    parser.add_argument("--baseline-label", default=None, help="Optional display label for the baseline on the y-axis.")
    parser.add_argument("--compare-label", default=None, help="Optional display label for the compared algorithm on the y-axis.")
    parser.add_argument("--title", default="Delta Avg Rank vs Misranking Severity (lower is better)", help="Plot title (empty disables).")
    parser.add_argument("--output", required=True, help="Output path (png/pdf/etc).")
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    sweep = read_sweep(args.sweep_summary)
    by_sigma_algo = {(r["sigma"], r["algorithm"]): r for r in sweep}
    sigmas = sorted({r["sigma"] for r in sweep})

    points = []
    for mpath in sorted(glob.glob(args.misranking_glob)):
        s = parse_sigma_from_name(mpath)
        if s not in sigmas:
            continue
        rd_mean, ov_mean = read_misranking_mean(mpath)
        base = by_sigma_algo.get((s, args.baseline))
        comp = by_sigma_algo.get((s, args.compare))
        if not base or not comp:
            continue
        points.append(
            {
                "sigma": s,
                "rank_disagreement_mean": rd_mean,
                "topmu_overlap_mean": ov_mean,
                "delta_avg_rank": float(comp["avg_rank"]) - float(base["avg_rank"]),
            }
        )

    if not points:
        raise SystemExit("No matching points to plot (check inputs).")

    xs = [p["rank_disagreement_mean"] for p in points]
    ys = [p["delta_avg_rank"] for p in points]
    labels = [str(p["sigma"]) for p in points]

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(xs, ys, marker="o", linewidth=2)
    for x, y, lab in zip(xs, ys, labels):
        plt.annotate(lab, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    plt.axhline(0.0, color="#64748B", linewidth=1.0, alpha=0.7)
    baseline_label = args.baseline_label or args.baseline
    compare_label = args.compare_label or args.compare
    plt.xlabel("misranking severity (mean rank disagreement)")
    plt.ylabel(f"$\\Delta$ avg rank ({compare_label} - {baseline_label})")
    if args.title:
        plt.title(args.title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    out_path = os.path.abspath(str(args.output))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()

    print("Wrote:", repo_relpath(out_path))


if __name__ == "__main__":
    main()
