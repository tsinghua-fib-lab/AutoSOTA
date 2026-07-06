"""Compute paper metrics exactly as described in the paper."""
import json, pickle, sys
from pathlib import Path
import numpy as np
from pymoo.indicators.hv import HV

OUTPUT_DIR = Path("output/full_benchmark")
TOTAL_TARGETS = 190

def compute_r2(costs, ref_point, weights):
    if len(costs) == 0:
        return float("inf")
    r2_sum = 0.0
    rho = 0.01
    for w in weights:
        min_tch = min(np.max(w * (ref_point - a)) + rho * np.sum(ref_point - a) for a in costs)
        r2_sum += min_tch
    return r2_sum / len(weights)

def get_weights(n_obj, n_per_dim=10):
    from itertools import product
    ref_dirs = []
    for combo in product(range(n_per_dim + 1), repeat=n_obj):
        if sum(combo) == n_per_dim:
            w = np.array(combo) / n_per_dim
            if np.all(w > 0):
                ref_dirs.append(w)
    return np.array(ref_dirs)

# Load all costs from all molecules
all_per_mol = {}  # smiles -> cost array or None
for mol_dir in sorted(OUTPUT_DIR.iterdir()):
    if mol_dir.is_dir():
        cost_file = mol_dir / "solution_costs.pkl"
        if cost_file.exists():
            with open(cost_file, "rb") as f:
                cost_dict = pickle.load(f)
            if cost_dict:
                costs_arr = np.array(list(cost_dict.keys()))
                if costs_arr.shape[1] >= 3:
                    all_per_mol[mol_dir.name] = costs_arr[:, :3]
                else:
                    all_per_mol[mol_dir.name] = None
            else:
                all_per_mol[mol_dir.name] = None
        else:
            all_per_mol[mol_dir.name] = None

n_processed = len(all_per_mol)
n_success = sum(1 for v in all_per_mol.values() if v is not None)

# Collect all costs for percentile computation
all_costs_flat = []
for costs in all_per_mol.values():
    if costs is not None:
        all_costs_flat.append(costs)

if not all_costs_flat:
    print("No successful molecules yet")
    sys.exit(0)

all_flat = np.vstack(all_costs_flat)
p5 = np.percentile(all_flat, 5, axis=0)
p95 = np.percentile(all_flat, 95, axis=0)

def normalize(costs):
    denom = p95 - p5
    denom[denom == 0] = 1.0
    return (costs - p5) / denom

ref_point = np.array([1.1, 1.1, 1.1])
weights = get_weights(3)

# Per molecule: HV and R2. Failed molecules get HV=0, R2 excluded
hv_per_mol = []
r2_per_mol = []
for costs in all_per_mol.values():
    if costs is not None:
        c_norm = normalize(costs)
        try:
            hv = float(HV(ref_point=ref_point)(c_norm))
        except Exception:
            hv = 0.0
        hv_per_mol.append(hv)
        try:
            r2 = compute_r2(c_norm, ref_point, weights)
            r2_per_mol.append(r2)
        except Exception:
            r2_per_mol.append(float("inf"))
    else:
        hv_per_mol.append(0.0)

hv_arr = np.array(hv_per_mol)
r2_valid = [r for r in r2_per_mol if np.isfinite(r)]

print(f"Processed: {n_processed}/{TOTAL_TARGETS}")
print(f"Successful: {n_success} ({100*n_success/n_processed:.1f}%)")
print(f"P5: {p5}")
print(f"P95: {p95}")
print(f"HV:  {hv_arr.mean():.4f} +/- {hv_arr.std():.4f}")
if r2_valid:
    print(f"R2:  {np.mean(r2_valid):.4f} +/- {np.std(r2_valid):.4f}")
print(f"SR:  {100*n_success/n_processed:.1f}%")

results = {
    "hv_mean": float(hv_arr.mean()), "hv_std": float(hv_arr.std()),
    "r2_mean": float(np.mean(r2_valid)) if r2_valid else None,
    "r2_std": float(np.std(r2_valid)) if r2_valid else None,
    "success_rate_pct": round(100*n_success/n_processed, 1),
    "n_processed": n_processed, "n_success": n_success,
    "p5": p5.tolist(), "p95": p95.tolist(),
}
with open("output/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to output/metrics.json")
