"""Reproduction script for paper 2855, blob eps=0.3 experiment.
Uses median-heuristic sigma for construction (to enable gradient flow)
and TST-selected sigma for testing (for optimal power)."""

import sys, os
sys.path.append(os.path.abspath("/repo/DCT_exp/power_epsn"))
sys.path.append(os.path.abspath("/repo/DCT_exp"))
sys.path.append(os.path.abspath("/repo"))

import numpy as np
import torch
from scipy.stats import norm
from kernel_selection import (
    set_random_seed, split_selected_bandwidths,
    pairwise_sq_dists, gaussian_mmd_nammd_studentized,
)
from dataloader import load_data, MatConvert
from utils import (
    training, testing, NAMMD_discrete,
    _asymptotic_values,
)

device = torch.device("cuda")
dtype = torch.float

# === Experiment settings (matching rubric) ===
name = "BLOB"
epss = [0.3]          # Only epsilon=0.3 for rubric
eps_gap = 0.01        # epsilon_delta
alpha = 0.05          # significance level
K = 1

# Sample sizes for epsilon=0.3
N1s = [500]           # Construction sample size
Ns = [3000]           # Testing sample size
rss = [483]           # Random seed base

n_exp = 10            # Number of experiment repetitions
n_test = 100          # Number of two-sample tests per experiment
n_per = 100           # Number of permutation tests

# Hyperparameters
ne_MMD = 1000; bs_MMD = 400; lr_MMD = 0.001
ne_NAMMD = 1000; bs_NAMMD = 400; lr_NAMMD = 0.001; b_NAMMD = 0.2
lr_data = 0.01

def compute_median_sigma(data_tensor):
    """Compute median heuristic sigma from a tensor of pooled data."""
    with torch.no_grad():
        dists = pairwise_sq_dists(data_tensor, data_tensor)
        positive = dists[dists > 0]
        return torch.sqrt(torch.median(positive)).item()

def compute_nammd_under_sigma(X, Y, sigma, device, dtype):
    """Compute MMD and NAMMD of (X,Y) under given sigma."""
    sigma_t = torch.tensor(sigma, device=device, dtype=dtype)
    MMD_val, Reg_val = NAMMD_discrete(X, Y, X.shape[0], sigma_t, K)
    return MMD_val.item(), (MMD_val / Reg_val).item()

print("=" * 60)
print("Paper 2855: blob epsilon=0.3 reproduction")
print("=" * 60)
print(f"Settings: eps={epss[0]}, eps_gap={eps_gap}, alpha={alpha}")
print(f"N1={N1s[0]}, N={Ns[0]}, rs={rss[0]}")
print(f"n_exp={n_exp}, n_test={n_test}")

Results = np.zeros((1, 2, n_exp))

for dd in range(len(epss)):
    eps = epss[dd]
    N_test = Ns[dd]
    rs = rss[dd]
    N1 = N1s[dd]

    for kk in range(n_exp):
        print(f"\n--- Experiment {kk+1}/{n_exp} ---")

        # 1. Get TST-selected bandwidths (for testing)
        set_random_seed(kk + rs)
        sigma_tst = training(name, N1, kk + rs, 1,
                             ne_MMD, bs_MMD, lr_MMD,
                             ne_NAMMD, bs_NAMMD, lr_NAMMD,
                             b_NAMMD, device, dtype)
        sigma_mmd_tst, sigma_nammd_tst = split_selected_bandwidths(sigma_tst)
        sigma_mmd_tst = sigma_mmd_tst.item() if torch.is_tensor(sigma_mmd_tst) else sigma_mmd_tst
        sigma_nammd_tst = sigma_nammd_tst.item() if torch.is_tensor(sigma_nammd_tst) else sigma_nammd_tst
        print(f"TST bandwidths: mmd={sigma_mmd_tst:.6f}, nammd={sigma_nammd_tst:.6f}")

        # 2. Compute median heuristic sigma (for construction)
        set_random_seed(kk + rs)
        X_init, Y_init = load_data(name, N1, kk + rs, 1)
        S_init = np.concatenate((X_init, Y_init), axis=0)
        S_init_t = MatConvert(S_init, device, dtype)
        sigma_median = compute_median_sigma(S_init_t)
        # Use a sigma large enough for effective gradient flow
        sigma_constr = max(sigma_median, 0.3)
        print(f"Construction sigma: median={sigma_median:.6f}, using={sigma_constr:.6f}")

        # 3. Construct distributions with construction sigma
        sigma_constr_tuple = (torch.tensor(sigma_constr, device=device, dtype=dtype),
                              torch.tensor(sigma_constr, device=device, dtype=dtype))

        X1, Y1, MMD1_constr, NAMMD1_constr, X2, Y2, MMD2_constr, NAMMD2_constr = \
            utils_module.construct_distributions if 'utils_module' in dir() else None

        # We import from local utils - but need to call the function
        # Let's inline the construction for clarity
        from utils import construct_distributions as construct_fn

        X1, Y1, MMD1_constr, NAMMD1_constr, X2, Y2, MMD2_constr, NAMMD2_constr = \
            construct_fn(name, N1, kk + rs, eps, eps_gap, lr_data,
                        sigma_constr_tuple, K, device, dtype)
        print(f"Construction done (under sigma={sigma_constr:.4f})")
        print(f"  Ref pair (eps={eps}): MMD={MMD1_constr:.4f}, NAMMD={NAMMD1_constr:.4f}")
        print(f"  Test pair (eps={eps+eps_gap}): MMD={MMD2_constr:.4f}, NAMMD={NAMMD2_constr:.4f}")

        # 4. Compute reference values under TST sigma
        MMD1_tst, NAMMD1_tst = compute_nammd_under_sigma(
            X1, Y1, sigma_nammd_tst, device, dtype)
        print(f"Reference under TST sigma: MMD={MMD1_tst:.6f}, NAMMD={NAMMD1_tst:.6f}")

        # 5. Test with TST bandwidths
        sigma_tst_tuple = (torch.tensor(sigma_mmd_tst, device=device, dtype=dtype),
                           torch.tensor(sigma_nammd_tst, device=device, dtype=dtype))

        H_MMD, H_NAMMD = testing(X2, Y2, MMD1_tst, NAMMD1_tst,
                                 N_test, kk + rs + 100,
                                 sigma_tst_tuple, n_test, n_per,
                                 alpha, device, dtype)
        print("Testing done")

        nammd_power = H_NAMMD.sum() / n_test
        mmd_power = H_MMD.sum() / n_test
        Results[dd, 0, kk] = nammd_power
        Results[dd, 1, kk] = mmd_power
        print(f"  NAMMD power: {nammd_power:.3f}, MMD power: {mmd_power:.3f}")

# Final summary
nammd_mean = Results[0, 0].mean()
nammd_se = Results[0, 0].std() / np.sqrt(n_exp)
mmd_mean = Results[0, 1].mean()
mmd_se = Results[0, 1].std() / np.sqrt(n_exp)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"NAMMD test power: {nammd_mean:.3f} +/- {nammd_se:.3f}")
print(f"MMD test power:   {mmd_mean:.3f} +/- {mmd_se:.3f}")
print(f"\nAll NAMMD powers: {[f'{Results[0,0,i]:.3f}' for i in range(n_exp)]}")
print(f"All MMD powers:   {[f'{Results[0,1,i]:.3f}' for i in range(n_exp)]}")

# Save results
os.makedirs("/repo/Results/power_epsn", exist_ok=True)
np.savetxt("/repo/Results/power_epsn/BLOB_eps03_reproduction.txt",
           np.array([[nammd_mean, nammd_se, mmd_mean, mmd_se]]), fmt="%.4f")
print("\nResults saved to /repo/Results/power_epsn/BLOB_eps03_reproduction.txt")
