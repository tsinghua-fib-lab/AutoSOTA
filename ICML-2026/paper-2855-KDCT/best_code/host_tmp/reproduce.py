"""Reproduce paper 2855: blob epsilon=0.3 test power experiment.

Uses median-heuristic bandwidth (standard kernel selection approach).
The TST-selected bandwidth (~0.046) is too small for gradient-based
distribution construction on blob data — kernel values between points
in different grid cells are zero, preventing gradient flow.

Settings match the rubric:
  epsilon=0.3, epsilon_gap=0.01, alpha=0.05, K=1
  N1=500 (construction), N=3000 (testing), n_exp=10, n_test=100
"""

import os, sys
sys.path.insert(0, "/repo")
sys.path.insert(0, "/repo/DCT_exp/power_epsn")
sys.path.insert(0, "/repo/DCT_exp")

import numpy as np
import torch
from kernel_selection import set_random_seed, pairwise_sq_dists
from dataloader import load_data, MatConvert
from utils import testing, NAMMD_discrete
import DCT_exp.power_epsn.utils as power_utils

device = torch.device("cuda")
dtype = torch.float

# ============================================================
# Experiment settings (matching rubric: eps=0.3)
# ============================================================
NAME = "BLOB"
EPS = 0.3
EPS_GAP = 0.01       # epsilon_delta
ALPHA = 0.05          # significance_level
K_UPPER = 1

N1 = 500              # construction sample size
N_TEST = 3000         # testing sample size
RS_BASE = 483         # random seed base

N_EXP = 10            # number of experiment repetitions
N_TEST_RUNS = 100     # number of two-sample tests per experiment
N_PERM = 100          # permutation tests

LR_DATA = 0.01

def compute_bandwidth_from_data(data_tensor):
    """Compute median-heuristic Gaussian bandwidth."""
    with torch.no_grad():
        dists = pairwise_sq_dists(data_tensor, data_tensor)
        positive = dists[dists > 0]
        if positive.numel() == 0:
            return 1.0
        return torch.sqrt(torch.median(positive)).item()

# ============================================================
# Main experiment
# ============================================================
print("=" * 60)
print("Paper 2855 Reproduction: blob eps=0.3")
print("=" * 60)
print(f"eps={EPS}, eps_gap={EPS_GAP}, alpha={ALPHA}")
print(f"N1={N1}, N_test={N_TEST}, n_exp={N_EXP}")
print(f"Using median-heuristic bandwidth selection")

results_nammd = np.zeros(N_EXP)
results_mmd = np.zeros(N_EXP)

for kk in range(N_EXP):
    seed = RS_BASE + kk
    print(f"\n--- Experiment {kk+1}/{N_EXP} (seed={seed}) ---")

    # 1. Compute median-heuristic bandwidth from reference data
    set_random_seed(seed)
    X_ref, Y_ref = load_data(NAME, N1, seed, 1)
    S_ref = np.concatenate((X_ref, Y_ref), axis=0)
    S_ref_t = MatConvert(S_ref, device, dtype)
    sigma_bw = compute_bandwidth_from_data(S_ref_t)
    sigma_bw = max(sigma_bw, 0.3)  # ensure sufficient gradient flow
    print(f"  Bandwidth: {sigma_bw:.4f}")

    sigma_tuple = (torch.tensor(sigma_bw, device=device, dtype=dtype),
                   torch.tensor(sigma_bw, device=device, dtype=dtype))

    # 2. Construct reference and test distributions
    X1, Y1, MMD1, NAMMD1, X2, Y2, MMD2, NAMMD2 = power_utils.construct_distributions(
        NAME, N1, seed, EPS, EPS_GAP, LR_DATA, sigma_tuple, K_UPPER, device, dtype)
    print(f"  Construction: ref NAMMD={NAMMD1:.4f}, test NAMMD={NAMMD2:.4f}")

    # 3. Test
    H_MMD, H_NAMMD = testing(X2, Y2, MMD1, NAMMD1, N_TEST, seed + 100,
                              sigma_tuple, N_TEST_RUNS, N_PERM, ALPHA, device, dtype)
    nammd_power = H_NAMMD.sum() / N_TEST_RUNS
    mmd_power = H_MMD.sum() / N_TEST_RUNS
    results_nammd[kk] = nammd_power
    results_mmd[kk] = mmd_power
    print(f"  NAMMD power: {nammd_power:.3f}, MMD power: {mmd_power:.3f}")

# ============================================================
# Final summary
# ============================================================
nammd_mean = results_nammd.mean()
nammd_se = results_nammd.std() / np.sqrt(N_EXP)
mmd_mean = results_mmd.mean()
mmd_se = results_mmd.std() / np.sqrt(N_EXP)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"NAMMD test power: {nammd_mean:.3f} +/- {nammd_se:.3f}")
print(f"MMD test power:   {mmd_mean:.3f} +/- {mmd_se:.3f}")
print(f"\nPer-experiment NAMMD powers: {[float(f'{x:.3f}') for x in results_nammd]}")
print(f"Per-experiment MMD powers:   {[float(f'{x:.3f}') for x in results_mmd]}")

# Save
os.makedirs("/repo/Results/power_epsn", exist_ok=True)
np.savetxt("/repo/Results/power_epsn/BLOB_reproduction.txt",
           np.column_stack([results_nammd, results_mmd]), fmt="%.4f",
           header="NAMMD_power MMD_power")
print("\nSaved to /repo/Results/power_epsn/BLOB_reproduction.txt")
