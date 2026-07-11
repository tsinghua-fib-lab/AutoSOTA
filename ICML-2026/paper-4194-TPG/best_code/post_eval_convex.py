#!/usr/bin/env python3
"""
Post-processing evaluation: convex combination TPG(k=1) + DM(k=0)
Uses existing pickle files from results/ directory.
"""
import numpy as np
import pickle
import os

def compute_convex_combination(results, alpha_values):
    gate = np.array(results["Ground Truth ATE"])
    dm_est = np.array(results["k=0"])
    tpg_est = np.array(results["k=1"])
    results_by_alpha = {}
    for alpha in alpha_values:
        combined = alpha * tpg_est + (1 - alpha) * dm_est
        bias_pct = np.abs(100 * (combined - gate) / gate)
        results_by_alpha[alpha] = {"mae": np.mean(bias_pct), "std": np.std(combined)}
    return results_by_alpha

results_dir = "results"
treatment_bias = 0.1
smoothness = 0.5
mixing_coeffs = np.round(np.linspace(0.01, 0.99, num=20), 2)
alpha_values = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]

# Also compute baseline k=1 for reference
k1_mae_all = []
k1_std_all = []

mae_comb = {a: [] for a in alpha_values}
std_comb = {a: [] for a in alpha_values}

for mc in mixing_coeffs:
    fn = f"{results_dir}/results_mix{mc:.2f}_bias{treatment_bias}_smooth{smoothness}.pkl"
    if not os.path.exists(fn):
        print(f"Missing: {fn}")
        continue
    with open(fn, "rb") as f:
        results = pickle.load(f)
    
    gate = np.array(results["Ground Truth ATE"])
    k1_est = np.array(results["k=1"])
    k1_bias = np.abs(100 * (k1_est - gate) / gate)
    k1_mae_all.append(np.mean(k1_bias))
    k1_std_all.append(np.std(k1_est))
    
    combo = compute_convex_combination(results, alpha_values)
    for a in alpha_values:
        mae_comb[a].append(combo[a]["mae"])
        std_comb[a].append(combo[a]["std"])

print("=" * 60)
print("Convex combination: alpha*TPG(k=1) + (1-alpha)*DM(k=0)")
print("=" * 60)
print(f"{'alpha':<12} {'MAE (%)':<12} {'STD':<12}")
print("-" * 36)

best_a, best_m = None, float("inf")
for a in alpha_values:
    m = np.mean(mae_comb[a]) if mae_comb[a] else 0
    s = np.mean(std_comb[a]) if std_comb[a] else 0
    print(f"{a:<12.2f} {m:<12.2f} {s:<12.3f}")
    if m < best_m:
        best_m, best_a = m, a

print()
print(f"Baseline k=1:       MAE={np.mean(k1_mae_all):.2f}%, STD={np.mean(k1_std_all):.3f}")
if best_a is not None:
    bs = np.mean(std_comb[best_a])
    print(f"Best alpha={best_a:.2f}:    MAE={best_m:.2f}%, STD={bs:.3f}")
    delta_mae = best_m - np.mean(k1_mae_all)
    delta_std = bs - np.mean(k1_std_all)
    print(f"Delta vs k=1:        MAE delta={delta_mae:+.2f}%, STD delta={delta_std:+.3f}")
