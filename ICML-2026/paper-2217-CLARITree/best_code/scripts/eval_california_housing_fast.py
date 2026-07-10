"""Fast evaluation for CLARITree optimization iterations.

Uses fixed best hyperparameters (lambda=0.001, kappa=1e-05) from baseline
to evaluate algorithmic changes in ~5-10 min instead of full grid sweep.

Usage:
    python3 scripts/eval_california_housing_fast.py [--refine_kappa_factor F]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATASET = "california_housing"
DEPTH = 4
N_THRESHOLDS = 20
CV_FOLDS = 3
METHOD = "claritree"
N_OUTERS = 5

# Baseline best hyperparameters (from reproduction)
BEST_LAMBDA = 0.001
BEST_KAPPA = 0.00001


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (ignored for single-combo fast eval)")
    parser.add_argument("--fixed_lambda", type=float, default=BEST_LAMBDA,
                        help="Lambda value (default: best from baseline)")
    parser.add_argument("--fixed_kappa", type=float, default=BEST_KAPPA,
                        help="Kappa value (default: best from baseline)")
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--n_thresholds", type=int, default=N_THRESHOLDS)
    parser.add_argument("--refine_kappa_factor", type=float, default=1.0,
                        help="Factor for post-training leaf coefficient refinement")
    parser.add_argument("--single_outer", type=int, default=None,
                        help="Run only this outer fold index (for quick checks)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    run_script = repo_root / "scripts" / "run_ours_outer_single.py"

    outers = [args.single_outer] if args.single_outer is not None else range(N_OUTERS)

    for outer in outers:
        print("\n=== Outer %d ===" % outer)
        cmd = [
            sys.executable, str(run_script),
            "--name", DATASET,
            "--outer", str(outer),
            "--method", METHOD,
            "--depth", str(args.depth),
            "--n_thresholds", str(args.n_thresholds),
            "--cv_folds", str(CV_FOLDS),
            "--thresholds_strategy", "quantile",
            "--lambda", str(args.fixed_lambda),
            "--kappa", str(args.fixed_kappa),
            "--refine_kappa_factor", str(args.refine_kappa_factor),
        ]
        result = subprocess.run(cmd, cwd=str(repo_root), check=True)

    if args.single_outer is not None:
        outer_dir = (
            repo_root / "results" / "ours" / METHOD /
            "linear_regression_tree_depth%d_threshold_%d" % (args.depth, args.n_thresholds) /
            DATASET / "outer%d" % args.single_outer
        )
        csv_path = outer_dir / "%s_outer%d_d%d.csv" % (DATASET, args.single_outer, args.depth)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            row = df.iloc[0]
            print("\n=== SINGLE OUTER RESULT ===")
            print("lambda=%s, kappa=%s" % (args.fixed_lambda, args.fixed_kappa))
            print("Test R2:  %.6f" % row["test_r2"])
            print("Train R2: %.6f" % row["train_r2"])
            print("Val R2:   %.6f" % row["val_r2"])
            print("Leaves:   %d" % row["n_leaves"])
        return

    # Multi-outer: compute mean and std
    rows = []
    for outer in range(N_OUTERS):
        outer_dir = (
            repo_root / "results" / "ours" / METHOD /
            "linear_regression_tree_depth%d_threshold_%d" % (args.depth, args.n_thresholds) /
            DATASET / "outer%d" % outer
        )
        csv_path = outer_dir / "%s_outer%d_d%d.csv" % (DATASET, outer, args.depth)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            rows.append(df.iloc[0])

    if rows:
        df_all = pd.DataFrame(rows)
        test_mean = df_all["test_r2"].mean()
        test_std = df_all["test_r2"].std()
        train_mean = df_all["train_r2"].mean()
        train_std = df_all["train_r2"].std()
        val_mean = df_all["val_r2"].mean()
        leaves_mean = df_all["n_leaves"].mean()

        print("\n=== FAST EVAL RESULT (fixed lambda=%s, kappa=%s) ===" % (args.fixed_lambda, args.fixed_kappa))
        print("Test R2:  %.6f +/- %.6f" % (test_mean, test_std))
        print("Train R2: %.6f +/- %.6f" % (train_mean, train_std))
        print("Val R2:   %.6f" % val_mean)
        print("Leaves:   %.1f" % leaves_mean)
        print("N outers: %d" % len(rows))


if __name__ == "__main__":
    main()
