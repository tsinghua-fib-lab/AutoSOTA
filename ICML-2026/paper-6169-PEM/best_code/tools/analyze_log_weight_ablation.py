#!/usr/bin/env python3
"""
Analyze log-weight ablation experiment results.

Reads the merged bbob_summary.csv produced by run_log_weight_ablation.sh and outputs:
  1. Per-algorithm win rates vs CMA-ES-sep (pairwise sign test)
  2. Wilcoxon signed-rank p-values
  3. Effect sizes (median Δlog₁₀ regret)
  4. Breakdown by high-misranking (15 funcs) vs all 30 functions

Usage:
    python3 tools/analyze_log_weight_ablation.py Results/log_weight_ablation/<timestamp>/bbob_summary_merged.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# The 15 high-misranking functions used in the paper (bbob-noisy id_function).
HIGH_MISRANKING_FIDS = {108, 110, 111, 113, 114, 116, 117, 119, 120, 122, 123, 125, 126, 128, 129}
LOW_MISRANKING_FIDS = set(range(101, 131)) - HIGH_MISRANKING_FIDS


def load_summary(path):
    """Load bbob_summary.csv into a dict keyed by (algorithm, budget_mult, function, dimension, instance)."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["algorithm"],
                int(row["budget_multiplier"]),
                int(row["function"]),
                int(row["dimension"]),
                int(row["instance"]),
            )
            data[key] = float(row["best_f"])
    return data


def pairwise_comparison(data, algo_a, algo_b, budget, dim, func_set=None):
    """
    Compare algo_a vs algo_b on matched problems.

    Returns: (wins_a, wins_b, ties, deltas)
      where deltas[i] = log10_regret(algo_a) - log10_regret(algo_b)
      so negative delta means algo_a is better.
    """
    wins_a, wins_b, ties = 0, 0, 0
    deltas = []

    # Collect all (func, inst) pairs where both algorithms ran.
    problems_a = {(k[2], k[4]): v for k, v in data.items()
                  if k[0] == algo_a and k[1] == budget and k[3] == dim}
    problems_b = {(k[2], k[4]): v for k, v in data.items()
                  if k[0] == algo_b and k[1] == budget and k[3] == dim}

    common = set(problems_a.keys()) & set(problems_b.keys())
    if func_set is not None:
        common = {(fid, inst) for fid, inst in common if fid in func_set}

    for fid, inst in sorted(common):
        fa = problems_a[(fid, inst)]
        fb = problems_b[(fid, inst)]
        if fa < fb:
            wins_a += 1
        elif fb < fa:
            wins_b += 1
        else:
            ties += 1
        deltas.append(fa - fb)

    return wins_a, wins_b, ties, np.array(deltas)


def wilcoxon_test(deltas):
    """Wilcoxon signed-rank test (two-sided). Returns (statistic, p-value)."""
    deltas = deltas[deltas != 0.0]
    if len(deltas) < 5:
        return float("nan"), float("nan")
    try:
        from scipy.stats import wilcoxon
        stat, pval = wilcoxon(deltas)
        return float(stat), float(pval)
    except ImportError:
        # Manual approximation for large n.
        n = len(deltas)
        ranks = np.argsort(np.abs(deltas)).argsort() + 1
        W_plus = float(np.sum(ranks[deltas > 0]))
        W_minus = float(np.sum(ranks[deltas < 0]))
        W = min(W_plus, W_minus)
        mu = n * (n + 1) / 4.0
        sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        if sigma < 1e-12:
            return W, 1.0
        z = (W - mu) / sigma
        # Approximate p-value via normal CDF.
        p = 2.0 * (1.0 - 0.5 * (1.0 + np.sign(z) * (1.0 - np.exp(-2.0 * z * z / np.pi))))
        return W, max(0.0, min(1.0, p))


def print_comparison_table(data, methods, baseline, budget, dim, func_set, label):
    """Print a comparison table for a set of methods vs baseline."""
    print(f"\n{'─' * 70}")
    print(f"  {label}  |  baseline: {baseline}  |  B={budget}d, d={dim}")
    print(f"{'─' * 70}")
    print(f"  {'Method':<32s} {'W/L/T':>10s} {'Win%':>7s} {'med Δ':>8s} {'p-val':>8s}")
    print(f"  {'─' * 68}")

    for method in methods:
        wa, wb, t, deltas = pairwise_comparison(data, method, baseline, budget, dim, func_set)
        total = wa + wb + t
        if total == 0:
            print(f"  {method:<32s} {'(no data)':>10s}")
            continue

        win_rate = wa / max(1, wa + wb) * 100.0
        med_delta = float(np.median(deltas)) if len(deltas) > 0 else float("nan")
        _, pval = wilcoxon_test(deltas)

        pval_str = f"{pval:.4f}" if np.isfinite(pval) else "  n/a"
        delta_str = f"{med_delta:+.4f}" if np.isfinite(med_delta) else "  n/a"
        print(f"  {method:<32s} {wa:>3d}/{wb:>3d}/{t:>2d} {win_rate:>6.1f}% {delta_str:>8s} {pval_str:>8s}")

    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary_csv", help="Path to bbob_summary_merged.csv")
    args = parser.parse_args()

    data = load_summary(args.summary_csv)

    # Discover available algorithms, budgets, dims.
    algos = sorted(set(k[0] for k in data))
    budgets = sorted(set(k[1] for k in data))
    dims = sorted(set(k[3] for k in data))
    funcs = sorted(set(k[2] for k in data))

    print("=" * 70)
    print("  Log-Weight Ablation Experiment Analysis")
    print("=" * 70)
    print(f"  Algorithms: {', '.join(algos)}")
    print(f"  Budgets: {budgets}")
    print(f"  Dimensions: {dims}")
    print(f"  Functions: {len(funcs)} ({min(funcs)}-{max(funcs)})")
    n_high = len([f for f in funcs if f in HIGH_MISRANKING_FIDS])
    n_low = len([f for f in funcs if f in LOW_MISRANKING_FIDS])
    print(f"  High-misranking: {n_high}, Low-misranking: {n_low}")

    baseline = "CMA-ES-sep"
    methods = [a for a in algos if a != baseline]

    for dim in dims:
        for budget in budgets:
            # (a) High-misranking subset (paper's original 15 functions)
            print_comparison_table(
                data, methods, baseline, budget, dim,
                HIGH_MISRANKING_FIDS,
                f"High-misranking functions ({n_high}f)")

            # (b) Low-misranking subset
            if n_low > 0:
                print_comparison_table(
                    data, methods, baseline, budget, dim,
                    LOW_MISRANKING_FIDS,
                    f"Low-misranking functions ({n_low}f)")

            # (c) All functions
            if len(funcs) > 15:
                print_comparison_table(
                    data, methods, baseline, budget, dim,
                    None,
                    f"All functions ({len(funcs)}f)")

    # Cross-comparison: LogW vs original Hetero (same baseline)
    if "BERW-Hetero" in algos and "BERW-Hetero-LogW" in algos:
        print(f"\n{'═' * 70}")
        print(f"  Direct comparison: BERW-Hetero-LogW vs BERW-Hetero")
        print(f"{'═' * 70}")
        for dim in dims:
            for budget in budgets:
                for func_set, label in [
                    (HIGH_MISRANKING_FIDS, "High-misranking"),
                    (None, "All functions"),
                ]:
                    wa, wb, t, deltas = pairwise_comparison(
                        data, "BERW-Hetero-LogW", "BERW-Hetero", budget, dim, func_set)
                    total = wa + wb + t
                    if total == 0:
                        continue
                    med = float(np.median(deltas)) if len(deltas) > 0 else float("nan")
                    print(f"  {label}: LogW {wa}W / Hetero {wb}W / {t}T "
                          f"(med Δ={med:+.4f}, n={total})")
        print()


if __name__ == "__main__":
    main()
