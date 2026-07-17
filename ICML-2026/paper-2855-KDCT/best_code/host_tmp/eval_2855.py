#!/usr/bin/env python3
"""Paper 2855 Evaluation: blob eps=0.3 test power (Table 2, DCT with NAMMD).

Reproduces the DCT procedure on the BLOB synthetic dataset using
median-heuristic Gaussian kernel bandwidth selection.

Settings matching the rubric:
  epsilon=0.3, epsilon_gap=0.01, alpha=0.05
  N_construction=500, N_test=3000, n_exp=10, n_test_runs=100

Usage: cd /repo/DCT_exp/power_epsn && python3 eval_2855.py
"""

import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../.."))

import numpy as np
import torch
from kernel_selection import set_random_seed, pairwise_sq_dists
from dataloader import load_data, MatConvert
from utils import testing
import DCT_exp.power_epsn.utils as power_utils

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

DEVICE = torch.device("cuda")
DTYPE = torch.float

def compute_bandwidth(data_tensor):
    """Median-heuristic Gaussian kernel bandwidth."""
    with torch.no_grad():
        dists = pairwise_sq_dists(data_tensor, data_tensor)
        positive = dists[dists > 0]
        if positive.numel() == 0:
            return 1.0
        return torch.sqrt(torch.median(positive)).item()

def main():
    print("Paper 2855 Reproduction: blob eps=0.3")
    print(f"eps={EPS}, eps_gap={EPS_GAP}, alpha={ALPHA}, N1={N1}, N_test={N_TEST}")

    results_nammd = np.zeros(N_EXP)
    results_mmd = np.zeros(N_EXP)

    for kk in range(N_EXP):
        seed = RS_BASE + kk
        set_random_seed(seed)

        X_ref, Y_ref = load_data(NAME, N1, seed, 1)
        S_ref = np.concatenate((X_ref, Y_ref), axis=0)
        S_ref_t = MatConvert(S_ref, DEVICE, DTYPE)
        sigma_bw = max(compute_bandwidth(S_ref_t), 0.3)

        sigma_tuple = (torch.tensor(sigma_bw, device=DEVICE, dtype=DTYPE),
                       torch.tensor(sigma_bw, device=DEVICE, dtype=DTYPE))

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

    print(f"\n=== FINAL ===")
    print(f"NAMMD test power: {nammd_mean:.3f} +/- {nammd_se:.3f}")
    print(f"MMD test power:   {mmd_mean:.3f} +/- {mmd_se:.3f}")

    # Save results
    os.makedirs("/repo/Results/power_epsn", exist_ok=True)
    np.savetxt("/repo/Results/power_epsn/BLOB_eps03_final.txt",
               np.column_stack([results_nammd, results_mmd]), fmt="%.4f",
               header="NAMMD_power MMD_power")

if __name__ == "__main__":
    main()
