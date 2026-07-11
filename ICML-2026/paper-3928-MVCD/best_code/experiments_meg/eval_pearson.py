#!/usr/bin/env python3
"""Evaluation script: computes the Pearson Correlation stability metric for LiMVAM on Cam-CAN MEG.

This script loads pre-computed causal discovery results (B matrices from 50 runs of
PairwiseLiMVAM on 30 random subjects) and computes the average Pearson correlation
between element-wise median B matrices across all pairs of runs.

This is the metric reported in Section 6.2 / Figure 3b of the paper:
"average correlation: 0.67, demonstrating the stability of our method"

Usage:
    python3 experiments_meg/eval_pearson.py
    python3 experiments_meg/eval_pearson.py --results_dir experiments_meg/4_results/aparc_sub_30_random_subjects_50_runs_pairwise_limvam
"""
import numpy as np
from scipy.stats import pearsonr
import argparse
from pathlib import Path
import sys


def compute_pearson_stability(results_dir, algo_name="PairwiseLiMVAM"):
    """Compute average Pearson correlation across runs from saved B matrices."""
    results_dir = Path(results_dir)
    
    B_total = np.load(results_dir / "B_total.npy")
    n_runs = B_total.shape[0]
    
    # Compute Pearson coefficients between median B matrices for all pairs of runs
    pearson_matrix = np.zeros((n_runs, n_runs))
    for i in range(n_runs):
        for j in range(n_runs):
            B1_median = np.median(B_total[i], axis=0)
            B2_median = np.median(B_total[j], axis=0)
            rho, _ = pearsonr(B1_median.flatten(), B2_median.flatten())
            pearson_matrix[i, j] = rho
    np.fill_diagonal(pearson_matrix, 0)
    
    # Average upper-triangular values (excluding diagonal)
    upper_tri = pearson_matrix[np.triu_indices(n_runs, k=1)]
    avg_corr = np.mean(upper_tri)
    std_corr = np.std(upper_tri)
    
    return avg_corr, std_corr, pearson_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LiMVAM Pearson correlation stability on Cam-CAN MEG"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiments_meg/4_results/aparc_sub_30_random_subjects_50_runs_pairwise_limvam",
        help="Path to directory containing B_total.npy from multi-run experiment",
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="experiments_meg/4_results/aparc_sub_30_random_subjects_50_runs_ica_limvam",
        help="Path to directory containing baseline ICA-LiMVAM B_total.npy",
    )
    args = parser.parse_args()
    
    # Compute PairwiseLiMVAM metric
    avg_corr, std_corr, _ = compute_pearson_stability(args.results_dir, "PairwiseLiMVAM")
    
    print(f"=== LiMVAM Pearson Correlation Stability ===")
    print(f"Method: PairwiseLiMVAM")
    print(f"Average Pearson Correlation: {avg_corr:.4f}")
    print(f"Standard Deviation: {std_corr:.4f}")
    print(f"Paper reported value: 0.67")
    
    # Compute baseline if available
    baseline_path = Path(args.baseline_dir)
    if baseline_path.exists():
        avg_corr_base, std_corr_base, _ = compute_pearson_stability(
            args.baseline_dir, "ICA-LiMVAM"
        )
        print(f"\n=== Baseline (ICA-LiMVAM-ML) ===")
        print(f"Average Pearson Correlation: {avg_corr_base:.4f}")
        print(f"Standard Deviation: {std_corr_base:.4f}")
        print(f"Paper reported baseline: 0.27")
    
    # Print summary line for parsing
    print(f"\n{{pearson_correlation: {avg_corr:.4f}, method: PairwiseLiMVAM}}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
