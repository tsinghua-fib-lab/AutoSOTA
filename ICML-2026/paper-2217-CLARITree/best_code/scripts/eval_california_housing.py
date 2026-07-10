"""Reproduce CLARITree Test R2 on California Housing.

Settings: depth=4, thresholds=20, n_runs=5 (outer splits), 80/20 split,
time limit ~600s, eval protocol: best_over_sparsity_sweep (select lambda/kappa
by best mean val_r2 across outers, report corresponding test_r2).

Uses parallel grid evaluation (--workers) to stay within the 120-min budget.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

DATASET = "california_housing"
DEPTH = 4
N_THRESHOLDS = 20
CV_FOLDS = 3
METHOD = "claritree"
N_OUTERS = 5
DEFAULT_WORKERS = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Number of parallel workers per outer fold")
    parser.add_argument("--refine_grid", action="store_true",
                        help="ALGO-05: two-stage grid refinement")
    parser.add_argument("--refine_kappa_factor", type=float, default=1.0,
                        help="ALGO-02: post-training leaf coefficient refinement")
    parser.add_argument("--outer_start", type=int, default=0,
                        help="First outer fold index (default: 0)")
    parser.add_argument("--outer_end", type=int, default=N_OUTERS - 1,
                        help="Last outer fold index inclusive (default: 4)")
    parser.add_argument("--fixed_lambda", type=float, default=None,
                        help="If set, skip grid search and use this lambda")
    parser.add_argument("--fixed_kappa", type=float, default=None,
                        help="If set, skip grid search and use this kappa")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    run_script = repo_root / "scripts" / "run_ours_outer.py"
    merge_script = repo_root / "scripts" / "merge_single_method_outer.py"

    # Step 1: run each outer split
    for outer in range(args.outer_start, args.outer_end + 1):
        print("\n=== Outer %d ===" % outer)
        cmd = [
            sys.executable, str(run_script),
            "--name", DATASET,
            "--outer", str(outer),
            "--method", METHOD,
            "--depth", str(DEPTH),
            "--n_thresholds", str(N_THRESHOLDS),
            "--cv_folds", str(CV_FOLDS),
            "--thresholds_strategy", "quantile",
            "--workers", str(args.workers),
            "--refine_kappa_factor", str(args.refine_kappa_factor),
        ]
        if args.refine_grid:
            cmd.append("--refine_grid")
        result = subprocess.run(cmd, cwd=str(repo_root), check=True)

    # Step 2: merge results
    print("\n=== Merging ===")
    result = subprocess.run(
        [
            sys.executable, str(merge_script),
            "--method", METHOD,
            "--name", DATASET,
            "--depth", str(DEPTH),
            "--n_thresholds", str(N_THRESHOLDS),
        ],
        cwd=str(repo_root),
        check=True,
    )

    # Step 3: extract & report best result
    merged_csv = (
        repo_root / "results" / "ours" / METHOD /
        "linear_regression_tree_depth%d_threshold_%d" % (DEPTH, N_THRESHOLDS) /
        DATASET / "%s_outer0-4_d%d.csv" % (DATASET, DEPTH)
    )
    df = pd.read_csv(merged_csv)
    best = df[df["outer"] == "best_by_mean_val_r2"].iloc[0]
    outer_data = df[~df["outer"].isin(["mean", "std", "best_by_mean_val_r2"])]
    best_outer = outer_data[
        (outer_data["lambda"] == best["lambda"]) &
        (outer_data["kappa"] == best["kappa"])
    ]

    bl = best["lambda"]; bk = best["kappa"]
    bt = best["test_r2"]; bts = best_outer["test_r2"].std()
    btr = best["train_r2"]; btrs = best_outer["train_r2"].std()
    bv = best["val_r2"]; blv = best["n_leaves"]; btt = best["train_time_s"]
    print("\n=== RESULT ===")
    print("lambda=%s, kappa=%s" % (bl, bk))
    print("Test R2:  %.6f +/- %.6f" % (bt, bts))
    print("Train R2: %.6f +/- %.6f" % (btr, btrs))
    print("Val R2:   %.6f" % bv)
    print("Leaves:   %.1f" % blv)
    print("Train time: %.1fs (mean per outer)" % btt)

    return best["test_r2"]

if __name__ == "__main__":
    main()
