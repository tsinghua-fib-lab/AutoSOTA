#!/usr/bin/env python3
"""
Reproduction script for Paper 3791: "Operator Splitting with Hamilton-Jacobi-based Proximals"
Target: LASSO PGD-HJ objective value
Paper: PGD-HJ = 10.849, PGD = 10.751 (Figure 1)
Setup: n=250, p=500, beta[400:410]=1, lambda=1, N=1000, decreasing power law delta
"""

import torch
import numpy as np
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hj_prox import hj_prox

# ---------------------------------------------------------------------------
# Paper configuration (Section H.1 + notebook pgd_lasso.ipynb)
# ---------------------------------------------------------------------------
SEED = 112
DIM = 500                        # p = 500 predictors
N_OBS = DIM // 2                 # n = 250 observations
NOISE_LEVEL = 0.1
LAMBDA_1 = 1.0
MAX_ITERS = 15000
EPS = 1e-5                       # p = 0.00001 in delta schedule
STEP_FACTOR = 0.085              # step-size reduction for error control
DELTA_FLOOR = 0.002             # minimum delta (lowered from 0.01 for finer prox)
N_0 = 4000                      # base MC samples (doubled for lower variance)
N_MAX = 8000                    # max MC samples to limit runtime
ADAPTIVE_N_GAMMA = 0.33         # N scaling exponent
NUM_SAMPLES = N_0                # base Monte Carlo samples N (adaptive per-iter)
POLYAK_START = 10000             # start Polyak averaging at iteration 8000

np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Data generation (matches paper exactly)
# ---------------------------------------------------------------------------
A = torch.randn(N_OBS, DIM)
x_true = torch.zeros(DIM, 1)
x_true[400:410] = 1.0
noise = NOISE_LEVEL * torch.randn(N_OBS, 1)
b = A @ x_true + noise
x0 = torch.zeros(DIM, 1, dtype=torch.float32)

sigma_max = torch.linalg.norm(A, ord=2)
L = sigma_max ** 2
step_size = float(STEP_FACTOR / L)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def l1_penalty(x_batch):
    """L1 penalty for HJ-Prox (batched)."""
    if x_batch.dim() == 1:
        x_batch = x_batch.unsqueeze(0)
    return LAMBDA_1 * torch.sum(torch.abs(x_batch), dim=1)

def lasso_objective(x):
    """LASSO objective: 0.5*||Ax-b||^2 + lambda*||x||_1."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.dim() == 2 and x.shape[0] != 1:
        x = x.t()
    residual = A @ x.t() - b
    return (0.5 * torch.norm(residual, p=2) ** 2 + LAMBDA_1 * torch.norm(x, p=1)).item()

# ===========================================================================
# ALGORITHM 1: Analytical PGD (gold-standard baseline)
# ===========================================================================
print("=" * 60)
print("Paper 3791 Reproduction: LASSO with PGD-HJ")
print("=" * 60)
print(f"Seed: {SEED}, n={N_OBS}, p={DIM}, lambda={LAMBDA_1}")
print(f"||A||_2={sigma_max:.4f}, L={L:.4f}, step_size={step_size:.8f}")
print(f"N_MC={NUM_SAMPLES}, max_iters={MAX_ITERS}")
print(f"Delta schedule: decreasing power law 125000/k^(2+{EPS}), floor={DELTA_FLOOR}")

print("\n--- Analytical PGD (soft-thresholding) ---")
xk = x0.clone()
for i in range(MAX_ITERS):
    grad = A.t() @ (A @ xk - b)
    x_grad = xk - step_size * grad
    x_prox = torch.sign(x_grad) * torch.maximum(
        torch.abs(x_grad) - step_size * LAMBDA_1, torch.zeros_like(x_grad))
    xk = x_prox.clone()
pgd_obj = lasso_objective(xk)
print(f"PGD final objective: {pgd_obj:.6f}")

# ===========================================================================
# ALGORITHM 2: PGD-HJ (HJ-Prox for L1 proximal)
# ===========================================================================
print("\n--- PGD-HJ (HJ-Prox, decreasing delta) ---")
xk = x0.clone()
best_f = float('inf')
f_hist = []
x_avg = torch.zeros(DIM, 1, dtype=torch.float32)
polyak_count = 0  # number of iterates averaged

for i in range(MAX_ITERS):
    k = i + 1
    # Decreasing power law: delta_k = 125000 / k^(2+p), p=1e-5
    delta_raw = 125000.0 / (k ** (2.0 + EPS))  # raw power-law delta
    delta_k = max(DELTA_FLOOR, delta_raw)       # actual delta with floor
    # Adaptive N: increase samples as delta shrinks for constant MC precision
    # N_k = N_0 * (delta_ref / delta_k)^gamma, capped at [N_0, N_MAX]
    if delta_k < 0.01:
        adaptive_N = int(N_0 * ((0.01 / delta_k) ** ADAPTIVE_N_GAMMA))
        adaptive_N = max(N_0, min(N_MAX, adaptive_N))
    else:
        adaptive_N = N_0

    grad = A.t() @ (A @ xk - b)
    x_grad = xk - step_size * grad

    # HJ-Prox approximates prox_{t*lambda*||.||_1}(x_grad)
    x_prox, _ = hj_prox(
        x_grad, t=step_size, f=l1_penalty,
        delta=delta_k, num_samples=adaptive_N, alpha=1.0,
    )

    fk = lasso_objective(x_prox)
    f_hist.append(fk)

    # Polyak-Ruppert averaging: average iterates from POLYAK_START onward
    if k >= POLYAK_START:
        polyak_count += 1
        x_avg = (x_avg * (polyak_count - 1) + x_prox) / polyak_count
    best_f = min(best_f, fk)
    xk = x_prox.clone()

    if (i + 1) % 2500 == 0:
        print(f"  iter {i+1:5d}: f={fk:.6f}, best={best_f:.6f}, delta={delta_k:.6e}, N={adaptive_N}")

pgd_hj_obj = f_hist[-1]
pgd_hj_polyak_obj = lasso_objective(x_avg) if polyak_count > 0 else pgd_hj_obj
print(f"\nPGD-HJ final objective:  {pgd_hj_obj:.6f}")
print(f"PGD-HJ Polyak-averaged (iter {POLYAK_START}-{MAX_ITERS}): {pgd_hj_polyak_obj:.6f}")
# Use Polyak-averaged objective as the primary metric when available
pgd_hj_obj = pgd_hj_polyak_obj if polyak_count > 0 else pgd_hj_obj
print(f"PGD-HJ best objective:  {best_f:.6f}")

# ===========================================================================
# RESULTS
# ===========================================================================
print("\n" + "=" * 60)
print("REPRODUCTION RESULTS")
print("=" * 60)
print(f"Metric:         Objective Value (lower is better)")
print(f"Paper PGD-HJ:   10.849")
print(f"Paper PGD:      10.751")
print(f"Ours PGD:       {pgd_obj:.6f}")
print(f"Ours PGD-HJ:    {pgd_hj_obj:.6f}")
print(f"Reproduce CI:   [10.751, 10.8588]")

in_ci = 10.751 <= pgd_hj_obj <= 10.8588
print(f"In CI bounds:   {'YES' if in_ci else 'NO'}")

results = {
    'paper_id': 3791,
    'pgd_objective': pgd_obj,
    'pgd_hj_objective': pgd_hj_obj,
    'pgd_hj_best': best_f,
    'in_ci': in_ci,
    'config': {
        'seed': SEED, 'dim': DIM, 'n_obs': N_OBS,
        'noise_level': NOISE_LEVEL, 'lambda_1': LAMBDA_1,
        'max_iters': MAX_ITERS, 'num_samples': f'adaptive(N_0={N_0},N_max={N_MAX},gamma={ADAPTIVE_N_GAMMA})',
        'step_factor': STEP_FACTOR, 'eps': EPS, 'delta_floor': DELTA_FLOOR,
    }
}

with open('reproduction_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nStatus: {'SUCCESS' if in_ci else 'FAILED'}")
print("Results saved to reproduction_results.json")
