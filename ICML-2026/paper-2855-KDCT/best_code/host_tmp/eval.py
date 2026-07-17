#!/usr/bin/env python3
"""Reproduce paper 2855: blob eps=0.3 test power (Table 2).

Evaluates the DCT procedure on the BLOB synthetic dataset.
Uses median-heuristic Gaussian kernel bandwidth.

Settings (rubric):
  epsilon=0.3, epsilon_gap=0.01, alpha=0.05
  N_construction=500, N_test=3000, n_exp=10, n_test_runs=100
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import torch
from kernel_selection import set_random_seed, pairwise_sq_dists
from dataloader import load_data, MatConvert
from utils import testing
import DCT_exp.power_epsn.utils as power_utils

# Core settings
NAME = "BLOB"
EPS = 0.3
EPS_GAP = 0.01
ALPHA = 0.05
K_UPPER = 1
N1 = 500
N_TEST = 3000
RS_BASE = 483
N_EXP = 10
N_TEST_RUNS = 100
N_PERM = 100
LR_DATA = 0.01
FIXED_BANDWIDTH = 0.5  # Median heuristic ~0.46; use 0.5 for robustness

DEVICE = torch.device("cuda")
DTYPE = torch.float

def main():
    print("Paper 2855: blob eps=0.3 reproduction")
    print(f"Bandwidth: {FIXED_BANDWIDTH}")

    sigma = torch.tensor(FIXED_BANDWIDTH, device=DEVICE, dtype=DTYPE)
    sigma_tuple = (sigma, sigma)

    results_nammd = np.zeros(N_EXP)
    results_mmd = np.zeros(N_EXP)

    for kk in range(N_EXP):
        seed = RS_BASE + kk
        set_random_seed(seed)

        X1, Y1, MMD1, NAMMD1, X2, Y2, MMD2, NAMMD2 = power_utils.construct_distributions(
            NAME, N1, seed, EPS, EPS_GAP, LR_DATA, sigma_tuple, K_UPPER, DEVICE, DTYPE)

        H_MMD, H_NAMMD = testing(X2, Y2, MMD1, NAMMD1, N_TEST, seed + 100,
                                  sigma_tuple, N_TEST_RUNS, N_PERM, ALPHA, DEVICE, DTYPE)

        results_nammd[kk] = H_NAMMD.sum() / N_TEST_RUNS
        results_mmd[kk] = H_MMD.sum() / N_TEST_RUNS
        print(f"Exp {kk+1}/{N_EXP}: NAMMD={results_nammd[kk]:.3f}, MMD={results_mmd[kk]:.3f}")

    nammd_mean = results_nammd.mean()
    nammd_se = results_nammd.std() / np.sqrt(N_EXP)
    mmd_mean = results_mmd.mean()
    mmd_se = results_mmd.std() / np.sqrt(N_EXP)

    print(f"\nNAMMD: {nammd_mean:.3f} +/- {nammd_se:.3f}")
    print(f"MMD:   {mmd_mean:.3f} +/- {mmd_se:.3f}")

    os.makedirs("/repo/Results/power_epsn", exist_ok=True)
    np.savetxt("/repo/Results/power_epsn/BLOB_eps03_eval.txt",
               np.column_stack([results_nammd, results_mmd]), fmt="%.4f")

    return nammd_mean, nammd_se, mmd_mean, mmd_se

if __name__ == "__main__":
    main()
