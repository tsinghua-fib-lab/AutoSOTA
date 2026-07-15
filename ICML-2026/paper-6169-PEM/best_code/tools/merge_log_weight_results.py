#!/usr/bin/env python3
"""
Merge all log-weight ablation CSVs (B={20,50,100,200}d) into one file,
then print the budget-trend analysis.

Usage:
    python3 tools/merge_log_weight_results.py

Automatically finds the latest main run and extra run under Results/log_weight_ablation/.
"""

import csv
import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_BASE = os.path.join(BASE_DIR, "Results", "log_weight_ablation")


def find_latest(pattern):
    """Find the latest directory matching a glob pattern."""
    dirs = sorted(glob.glob(os.path.join(RESULTS_BASE, pattern)))
    dirs = [d for d in dirs if os.path.isdir(d)]
    return dirs[-1] if dirs else None


def main():
    # Find latest main run (B=100d,200d) and extra run (B=50d).
    main_dir = find_latest("20*")  # e.g. 20260329_182934
    extra_dir = find_latest("extra_*")

    csvs = []
    if main_dir:
        p = os.path.join(main_dir, "bbob_summary_merged.csv")
        if os.path.exists(p):
            csvs.append(p)
            print(f"Main run:  {os.path.relpath(p, BASE_DIR)}")
    if extra_dir:
        p = os.path.join(extra_dir, "bbob_summary_merged.csv")
        if os.path.exists(p):
            csvs.append(p)
            print(f"Extra run: {os.path.relpath(p, BASE_DIR)}")

    if not csvs:
        print("No result CSVs found under Results/log_weight_ablation/")
        sys.exit(1)

    # Merge.
    out_path = os.path.join(RESULTS_BASE, "bbob_summary_all_budgets.csv")
    header = None
    rows = []
    for csv_path in csvs:
        with open(csv_path) as f:
            reader = csv.reader(f)
            h = next(reader)
            if header is None:
                header = h
            for row in reader:
                rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nMerged: {len(rows)} rows -> {os.path.relpath(out_path, BASE_DIR)}")

    # Quick completeness check.
    from collections import Counter
    algo_idx = header.index("algorithm")
    bm_idx = header.index("budget_multiplier")
    counts = Counter()
    for row in rows:
        counts[(row[algo_idx], int(row[bm_idx]))] += 1

    print(f"\nCompleteness (expected 1350 per cell = 30f × 15i × 3d):")
    algos = sorted(set(k[0] for k in counts))
    budgets = sorted(set(k[1] for k in counts))
    print(f"  {'':30s}", "  ".join(f"B={b}d" for b in budgets))
    all_ok = True
    for algo in algos:
        vals = []
        for b in budgets:
            n = counts.get((algo, b), 0)
            mark = "  ✓" if n == 1350 else f" ✗({n})"
            vals.append(f"{n:5d}{mark}")
            if n != 1350:
                all_ok = False
        print(f"  {algo:30s}", "  ".join(vals))

    if all_ok:
        print("\n  All cells complete.")

    print(f"\nAnalyze with:")
    print(f"  python3 tools/analyze_log_weight_ablation.py {os.path.relpath(out_path, BASE_DIR)}")


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    main()
