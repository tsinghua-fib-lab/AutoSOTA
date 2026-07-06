"""Compute paper metrics from benchmark results."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from pymoo.indicators.hv import HV
warnings = __import__("warnings")
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("output/full_benchmark")

def compute_r2(costs, ref_point, weights):
    """R2 indicator: lower is better."""
    if len(costs) == 0:
        return float("inf")
    r2_sum = 0.0
    rho = 0.01
    for w in weights:
        min_tch = float("inf")
        for a in costs:
            diff = ref_point - a
            tch = np.max(w * diff) + rho * np.sum(diff)
            if tch < min_tch:
                min_tch = tch
        r2_sum += min_tch
    return r2_sum / len(weights)

def get_weights(n_obj, n_per_dim=10):
    """Das-Dennis uniform weights."""
    from itertools import product
    ref_dirs = []
    for combo in product(range(n_per_dim + 1), repeat=n_obj):
        if sum(combo) == n_per_dim:
            w = np.array(combo) / n_per_dim
            if np.all(w > 0):
                ref_dirs.append(w)
    return np.array(ref_dirs)

# Load all costs
all_costs = []
success_count = 0
total_targets = 0

for mol_dir in sorted(OUTPUT_DIR.iterdir()):
    if mol_dir.is_dir():
        total_targets += 1
        cost_file = mol_dir / "solution_costs.pkl"
        if cost_file.exists():
            with open(cost_file, "rb") as f:
                cost_dict = pickle.load(f)
            if cost_dict:
                costs_arr = np.array(list(cost_dict.keys()))
                if costs_arr.shape[1] >= 3:
                    all_costs.append(costs_arr[:, :3])
                    success_count += 1

print(f"Total targets: {total_targets}")
print(f"Successful: {success_count}")
print(f"Success Rate: {100*success_count/total_targets:.1f}%")

if not all_costs:
    print("No successful molecules")
    sys.exit(1)

# Compute percentiles
all_flat = np.vstack(all_costs)
p5 = np.percentile(all_flat, 5, axis=0)
p95 = np.percentile(all_flat, 95, axis=0)
print(f"P5: {p5}")
print(f"P95: {p95}")

def normalize(costs):
    denom = p95 - p5
    denom[denom == 0] = 1.0
    return (costs - p5) / denom

ref_point = np.array([1.1, 1.1, 1.1])
weights = get_weights(3)
print(f"R2 weights: {len(weights)}")

hv_list, r2_list = [], []
for costs in all_costs:
    c_norm = normalize(costs)
    try:
        hv = HV(ref_point=ref_point)(c_norm)
        hv_list.append(hv)
    except Exception:
        hv_list.append(0.0)
    try:
        r2 = compute_r2(c_norm, ref_point, weights)
        r2_list.append(r2)
    except Exception:
        r2_list.append(float("inf"))

hv_arr = np.array(hv_list)
r2_arr = np.array([r for r in r2_list if np.isfinite(r)])

print(f"\n=== RESULTS ===")
print(f"HV:  {hv_arr.mean():.4f} +/- {hv_arr.std():.4f}")
print(f"R2:  {r2_arr.mean():.4f} +/- {r2_arr.std():.4f}")
print(f"SR:  {100*success_count/total_targets:.1f}%")

# Save
results = {
    "hv_mean": float(hv_arr.mean()),
    "hv_std": float(hv_arr.std()),
    "r2_mean": float(r2_arr.mean()),
    "r2_std": float(r2_arr.std()),
    "success_rate_pct": round(100*success_count/total_targets, 1),
    "num_targets": total_targets,
    "num_successful": success_count,
    "p5": p5.tolist(),
    "p95": p95.tolist(),
}
with open("output/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to output/metrics.json")
