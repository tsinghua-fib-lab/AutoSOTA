"""
BLOB experiment: Varying reference size N with fixed query size M=50.
Reproduces Table 1 (left, BLOB) in the paper.

All LoTT parameters exposed via CLI for systematic optimization.
"""

import numpy as np
import argparse
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from lott.wrapper import run_lott

parser = argparse.ArgumentParser(description="BLOB: Varying N (reference size)")
# --- Experiment parameters ---
parser.add_argument("--check", default=1, type=int, help="1=test power, 0=type I error")
parser.add_argument("--N_ref", default=[100, 150, 200, 300, 400, 500, 1000], nargs="+", type=int,
                    help="Reference sample sizes N to evaluate")
parser.add_argument("--N_query", default=50, type=int, help="Query sample size M")
parser.add_argument("--n_exp", default=10, type=int, help="Number of independent experiments")
parser.add_argument("--n_test", default=100, type=int, help="Number of test repetitions per experiment")
parser.add_argument("--alpha", default=0.05, type=float, help="Significance level")
parser.add_argument("--seed", default=819, type=int, help="Base random seed")

# --- CODE-01 + ALGO-06 + ALGO-01: Exposed LoTT parameters ---
parser.add_argument("--selection_method", default="precision_weight", type=str,
                    choices=["precision_weight", "sensitivity_weight", "top_n", "threshold"],
                    help="RDR selection method")
parser.add_argument("--n_select", default=2, type=int, help="Number of RDRs to select (top_n only)")
parser.add_argument("--variance_threshold", default=None, type=float,
                    help="Variance threshold (threshold method only)")
parser.add_argument("--perturbation_scale", default=0.01, type=float,
                    help="Perturbation scale for sensitivity_weight")
parser.add_argument("--M", default=10, type=int, help="Number of landmark RDRs")
parser.add_argument("--subset_size", default=10, type=int, help="Subset size per landmark RDR")
parser.add_argument("--n_permutations", default=500, type=int, help="Number of permutations")
parser.add_argument("--train_frac", default=0.4, type=float, help="Fraction of X for RDR training")
parser.add_argument("--calib_frac", default=0.1, type=float, help="Fraction of X for calibration")
parser.add_argument("--k_knn", default=5, type=int, help="k for KNN_RDR")
parser.add_argument("--k_lof", default=20, type=int, help="k for LOF_RDR")
parser.add_argument("--statistic_formulation", default="mean_of_squares", type=str,
                    choices=["mean_of_squares", "square_of_mean", "hybrid"],
                    help="Test statistic formulation (ALGO-06)")
parser.add_argument("--use_multiscale_me", action="store_true",
                    help="Use MultiScaleME_RDR instead of ME_RDR (ALGO-01)")
parser.add_argument("--verbose", action="store_true", help="Print RDR selection details")
args = parser.parse_args()

exp_path = os.path.dirname(os.path.abspath(__file__))

SEP = "=" * 60

for N in args.N_ref:
    print()
    print(SEP)
    print(f"BLOB | N={N}, M={args.N_query}, check={args.check}")
    print(SEP)

    results = np.zeros(args.n_exp)
    for kk in range(args.n_exp):
        rs = kk * 1000 + args.seed + N
        H = run_lott("blob", N, args.N_query, rs, args.check,
                     n_test=args.n_test, alpha=args.alpha, is_selection=True,
                     selection_method=args.selection_method,
                     n_select=args.n_select,
                     variance_threshold=args.variance_threshold,
                     perturbation_scale=args.perturbation_scale,
                     M=args.M,
                     subset_size=args.subset_size,
                     n_permutations=args.n_permutations,
                     train_frac=args.train_frac,
                     calib_frac=args.calib_frac,
                     k_knn=args.k_knn,
                     k_lof=args.k_lof,
                     statistic_formulation=args.statistic_formulation,
                     use_multiscale_me=args.use_multiscale_me,
                     verbose=args.verbose)
        results[kk] = np.mean(H)

    mean_power = np.mean(results)
    std_power = np.std(results) / np.sqrt(args.n_exp)
    print(f"LoTT: {mean_power:.3f} +/- {std_power:.3f}")

    tag = "test_power" if args.check else "typeI_error"
    out_dir = os.path.join(exp_path, "Results", tag, str(args.alpha))
    os.makedirs(out_dir, exist_ok=True)
    fname = f"blob_N{N}_M{args.N_query}_seed{args.seed}"
    np.savetxt(os.path.join(out_dir, fname), [mean_power, std_power], fmt="%.4f")
