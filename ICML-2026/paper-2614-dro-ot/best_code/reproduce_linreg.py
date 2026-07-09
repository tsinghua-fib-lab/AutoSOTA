#!/usr/bin/env python3
"""
Reproduction script for paper 2614, Section 6.2, Figure 5 (left).
Linear regression with: k=1, w=1, sigma=10, J=20, nb=20, wasserstein_type=1.
Metrics:
  - e_wc: worst-case expected absolute error
  - e_oos: out-of-sample expected absolute error (10^7 samples)
"""
import numpy as np
import time
import os
import json
import sys

from trainable_ot_dro.utils.wasserstein_distance import wasserstein_gradient_L, wasserstein_distance_L
from trainable_ot_dro.reformulations import reform_linreg_dro
from trainable_ot_dro.bilevel_optimization import optimize_transportation_matrix

# ── Paper Settings (Section 6.2, Figure 5 left) ──────────────────────────
SEED = 0

# Linear model parameters: y = w_true * x + noise, noise ~ N(0, var_eps)
w_true_val = 1.0
x_min = -10.0
x_max = 10.0
sigma = 10.0
var_eps = sigma**2  # 100

# Feature dimension (k = 1)
k = 1
dim_randomness = k + 1  # = 2, since xi = (x, y)

# Number of samples and bootstrap distributions
J = 20
n_bootstrap = 20

# Wasserstein type 1 = absolute error
wasserstein_type = 1

# Initial transport cost matrix
# L_init will be computed from sample covariance after data generation

# ── Penalisation & optimisation parameters (Table 1) ─────────────────────
penalization = {"lambda": 1.0, "eta": 100.0}

stopping_tols = {
    "diff_decision": 0.0,
    "diff_objective": 0.0,
    "diff_objective_pen": 0.0,
    "diff_weightmat": 0.0,
    "weightmat_gradient": 0.0,
    "rel_impr_obj": 1e-6,
    "rel_diff_obj_pen": 0.0,
}

opt_params = {
    "n_iter_max": 1_000_000,
    "learning_rate": 1e-4,
    "store_every": 100,
    "stopping_tols": stopping_tols,
    "penalization": penalization,
}

distribution_distance = {
    "gradient": wasserstein_gradient_L,
    "type": wasserstein_type,
}

constraint_params = {"gamma": 0.1}

# ── Generate data ─────────────────────────────────────────────────────────
rng = np.random.default_rng(seed=SEED)
w_true = np.full(k, w_true_val)  # shape (k,)

X = rng.uniform(x_min, x_max, (J, k))
y = np.dot(X, w_true) + rng.normal(0, sigma, J)
samples = np.hstack((X, y.reshape(-1, 1)))

# Data-driven L initialization from sample covariance
Sigma = np.cov(samples.T)
# Add small ridge for numerical stability
Sigma_reg = Sigma + 1e-6 * np.eye(dim_randomness)
L_init = np.linalg.cholesky(np.linalg.inv(Sigma_reg))
L_init = np.tril(L_init)  # ensure lower triangular
print(f"L_init from sample covariance:\n{L_init}")
print(f"  (vs identity:\n{np.eye(dim_randomness)})")

# Reference (empirical) distribution
supp_ref = np.unique(samples, axis=0)
prob_ref = np.zeros(supp_ref.shape[0])
for i in range(supp_ref.shape[0]):
    prob_ref[i] = np.sum(np.all(samples == supp_ref[i], axis=1)) / J
ref_dist = (supp_ref, prob_ref)

# Bootstrap distributions
bootstrap_dists = []
bootstrap_initial_distances = []
for i in range(n_bootstrap):
    indices = rng.choice(J, J, replace=True)
    boot_samples = samples[indices, :]
    supp_boots = np.unique(boot_samples, axis=0)
    prob_boots = np.zeros(supp_boots.shape[0])
    for j in range(supp_boots.shape[0]):
        prob_boots[j] = np.sum(np.all(boot_samples == supp_boots[j], axis=1)) / J
    bootstrap_dists.append((supp_boots, prob_boots))
    dist_val, _, _ = wasserstein_distance_L(ref_dist, (supp_boots, prob_boots), L_init, wasserstein_type=wasserstein_type)
    bootstrap_initial_distances.append(dist_val)

distributions = {"reference": ref_dist, "bootstrap": bootstrap_dists}

# epsilon = (1-gamma)-quantile of bootstrap distances
eps_0 = np.quantile(bootstrap_initial_distances, 1 - constraint_params["gamma"])
constraint_params["eps"] = eps_0

# ── Conic reformulation ───────────────────────────────────────────────────
conic_program = reform_linreg_dro(X, y, eps_0, L_init, wasserstein_type=wasserstein_type)

# ── Run bilevel optimisation ─────────────────────────────────────────────
print("=" * 60)
print("Starting bilevel optimisation for linear regression DRO")
print(f"  k={k}, w_true={w_true_val}, sigma={sigma}, J={J}, nb={n_bootstrap}")
print(f"  wasserstein_type={wasserstein_type}, eps_0={eps_0:.6f}")
print("=" * 60)

start_opt = time.time()
result = optimize_transportation_matrix(
    L_init, conic_program, distributions,
    distribution_distance, constraint_params, opt_params
)
end_opt = time.time()
print(f"\nOptimisation wall time: {end_opt - start_opt:.1f} s")

# ── Extract metrics ───────────────────────────────────────────────────────
opt_decisions = result["decisions"]
wc_objectives = np.array(result["objective_values"])

# For wasserstein_type=1, objective = e_wc directly
print(f"\n--- Iterations stored: {len(wc_objectives)} ---")
print(f"Initial e_wc: {wc_objectives[0]:.4f}")
print(f"Final   e_wc: {wc_objectives[-1]:.4f}")

# ── Compute e_oos with 10^7 samples ──────────────────────────────────────
print("\nComputing e_oos with 10^7 OOS samples...")
n_samples_OOS = 10_000_000
batch_size = 1_000_000  # process in batches to manage memory

oos_errors = []
for idx, decision in enumerate(opt_decisions):
    total_loss = 0.0
    for batch_start in range(0, n_samples_OOS, batch_size):
        batch_n = min(batch_size, n_samples_OOS - batch_start)
        x_batch = np.random.uniform(x_min, x_max, (batch_n, k))
        e_batch = np.random.normal(0, sigma, batch_n)
        decision_diff = decision - w_true
        losses = np.abs(np.dot(x_batch, decision_diff) + e_batch)
        total_loss += np.sum(losses)
    oos_error = total_loss / n_samples_OOS
    oos_errors.append(oos_error)
    if idx % 10 == 0 or idx == len(opt_decisions) - 1:
        print(f"  decision {idx}: e_oos = {oos_error:.6f}")

print(f"\nInitial e_oos: {oos_errors[0]:.4f}")
print(f"Final   e_oos: {oos_errors[-1]:.4f}")

# ── Final metrics ─────────────────────────────────────────────────────────
final_e_wc = float(wc_objectives[-1])
final_e_oos = float(oos_errors[-1])

# Best-checkpoint selection by e_oos
best_idx = int(np.argmin(oos_errors))
best_e_wc = float(wc_objectives[best_idx])
best_e_oos = float(oos_errors[best_idx])

print("\n" + "=" * 60)
print("REPRODUCTION RESULT")
print(f"  Final checkpoint  (iter {len(wc_objectives)-1}): e_wc = {final_e_wc:.4f}, e_oos = {final_e_oos:.4f}")
print(f"  Best checkpoint   (iter {best_idx}): e_wc = {best_e_wc:.4f}, e_oos = {best_e_oos:.4f}")
print(f"  (paper ~17.0, CI [16.8, 19.0] for e_wc; ~8.0, CI [7.95, 8.5] for e_oos)")
print("=" * 60)

# Report best-checkpoint metrics as primary (early stopping)
final_e_wc = best_e_wc
final_e_oos = best_e_oos

# ── Save results ─────────────────────────────────────────────────────────
output = {
    "e_wc": final_e_wc,
    "e_oos": final_e_oos,
    "all_e_wc": [float(v) for v in wc_objectives],
    "all_e_oos": [float(v) for v in oos_errors],
    "params": {
        "k": k, "w_true": w_true_val, "sigma": sigma, "J": J,
        "n_bootstrap": n_bootstrap, "wasserstein_type": wasserstein_type,
        "seed": SEED, "n_iter_max": opt_params["n_iter_max"],
        "learning_rate": opt_params["learning_rate"],
    },
    "wall_time_s": end_opt - start_opt,
}
os.makedirs("/repo/results", exist_ok=True)
ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
result_path = f"/repo/results/reproduce_linreg_{ts}.json"
with open(result_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {result_path}")
