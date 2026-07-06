#!/usr/bin/env python3
"""Run MORetro* benchmark on full USPTO-190 and compute paper metrics."""
import ast
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import gin
import numpy as np
from pymoo.indicators.hv import HV

warnings.filterwarnings("ignore")

def compute_r2(costs, ref_point, weights):
    """Compute R2 indicator. Lower is better.
    
    R2(A, U) = (1/|U|) * sum_{u in U} min_{a in A} max_{j} u_j * (ref_j - a_j)
    
    Uses augmented Tchebycheff: max_j w_j * (ref_j - a_j) + 0.01 * sum_j (ref_j - a_j)
    """
    if len(costs) == 0:
        return float("inf")
    
    r2_sum = 0.0
    rho = 0.01  # Small augmentation constant
    
    for w in weights:
        min_tch = float("inf")
        for a in costs:
            diff = ref_point - a
            tch = np.max(w * diff) + rho * np.sum(diff)
            if tch < min_tch:
                min_tch = tch
        r2_sum += min_tch
    
    return r2_sum / len(weights)


def get_weights_uniform(n_obj, n_points_per_dim=10):
    """Generate uniform weight vectors using Das-Dennis method."""
    from itertools import product
    ref_dirs = []
    for combo in product(range(n_points_per_dim + 1), repeat=n_obj):
        if sum(combo) == n_points_per_dim:
            w = np.array(combo) / n_points_per_dim
            # Filter out zero vectors
            if np.all(w > 0):
                ref_dirs.append(w)
    return np.array(ref_dirs)


# Read targets
targets = []
with open("/paper_data/uspto_190_targets.txt") as f:
    for line in f:
        line = line.strip()
        if line:
            tup = ast.literal_eval(line)
            targets.append(tup[0])

print(f"Running on {len(targets)} targets", flush=True)

# Gin config
gin.parse_config_file("/repo/configs/search_config.gin")

all_costs = {}  # mol -> cost array
success_count = 0
total_time = 0.0

for idx, smiles in enumerate(targets):
    t0 = time.time()
    try:
        from moretro.moretro_star import MORetro
        moretro = MORetro(
            smiles,
            output_dir="output/full_benchmark",
            visualize_plots=False,
            save_json=False,
        )
        moretro.search()
        
        safe_name = moretro._safe_smiles_dirname(smiles)
        cost_path = Path(f"output/full_benchmark/{safe_name}/solution_costs.pkl")
        if cost_path.exists():
            with open(cost_path, "rb") as f:
                cost_dict = pickle.load(f)
            all_costs[smiles] = np.array(list(cost_dict.keys()))
        else:
            all_costs[smiles] = np.array([])
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{idx+1:3d}/{len(targets)}] ERROR: {type(e).__name__}: {str(e)[:80]} ({elapsed:.0f}s)", flush=True)
        all_costs[smiles] = np.array([])

    elapsed = time.time() - t0
    total_time += elapsed
    n_sol = len(all_costs[smiles])
    success = n_sol > 0
    if success:
        success_count += 1
    print(f"[{idx+1:3d}/{len(targets)}] success={success} sols={n_sol} time={elapsed:.0f}s", flush=True)

print(f"\nTotal time: {total_time:.1f}s ({total_time/3600:.2f}h)")
print(f"Success rate: {success_count}/{len(targets)} = {100*success_count/len(targets):.1f}%")

# --- Metric Computation ---
successful_costs = []
for smiles, costs in all_costs.items():
    if len(costs) > 0 and costs.shape[1] >= 3:
        successful_costs.append(costs[:, :3])

if not successful_costs:
    print("ERROR: No successful molecules")
    # Save partial results
    results = {
        "error": "No successful molecules",
        "targets_processed": len(targets),
        "success_count": success_count,
    }
    with open("output/benchmark_full_results.json", "w") as f:
        json.dump(results, f, indent=2)
    sys.exit(1)

# Flatten all costs for percentile computation
all_costs_flat = np.vstack(successful_costs)
p5 = np.percentile(all_costs_flat, 5, axis=0)
p95 = np.percentile(all_costs_flat, 95, axis=0)
print(f"Percentile 5: {p5}")
print(f"Percentile 95: {p95}")

def normalize(costs, p5, p95):
    denom = p95 - p5
    denom[denom == 0] = 1.0
    return (costs - p5) / denom

ref_point = np.array([1.1, 1.1, 1.1])
weights = get_weights_uniform(3, n_points_per_dim=10)
print(f"Number of weight vectors for R2: {len(weights)}")

# Compute per molecule
hv_list = []
r2_list = []

for costs in successful_costs:
    costs_norm = normalize(costs, p5, p95)
    
    # HV: higher is better
    try:
        hv_ind = HV(ref_point=ref_point)
        hv = hv_ind(costs_norm)
        hv_list.append(hv)
    except Exception:
        hv_list.append(0.0)
    
    # R2: lower is better
    try:
        r2 = compute_r2(costs_norm, ref_point, weights)
        r2_list.append(r2)
    except Exception as e:
        print(f"R2 computation error: {e}")
        r2_list.append(float("inf"))

hv_list = np.array(hv_list)
r2_list = np.array(r2_list)

# Filter out inf R2 values
valid_r2 = r2_list[np.isfinite(r2_list)]

print(f"\n=== RESULTS ===")
print(f"HV: mean={hv_list.mean():.4f} std={hv_list.std():.4f}")
print(f"R2: mean={valid_r2.mean():.4f} std={valid_r2.std():.4f}")
print(f"Success Rate: {success_count}/{len(targets)} = {100*success_count/len(targets):.1f}%")
print(f"Num mols with solutions: {len(successful_costs)}")

# Save
results = {
    "hv_mean": float(hv_list.mean()),
    "hv_std": float(hv_list.std()),
    "r2_mean": float(valid_r2.mean()),
    "r2_std": float(valid_r2.std()),
    "success_rate_pct": 100 * success_count / len(targets),
    "success_count": success_count,
    "total_targets": len(targets),
    "total_time_seconds": total_time,
    "num_with_solutions": len(successful_costs),
    "percentile_5": p5.tolist(),
    "percentile_95": p95.tolist(),
    "r2_weight_count": len(weights),
}

with open("output/benchmark_full_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to output/benchmark_full_results.json")
