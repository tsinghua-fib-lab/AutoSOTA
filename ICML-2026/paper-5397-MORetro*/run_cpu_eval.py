#!/usr/bin/env python3
"""CPU evaluation wrapper for SOTA optimization testing.
Processes a subset of molecules and computes paper metrics."""
import json, pickle, sys, time, warnings, ast
from pathlib import Path
import numpy as np
import gin
from pymoo.indicators.hv import HV

warnings.filterwarnings("ignore")

# ===== Configuration =====
CONFIG_FILE = sys.argv[1] if len(sys.argv) > 1 else "configs/search_config_cpu.gin"
DATASET_FILE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/subset_10mol.csv"
OUTPUT_DIR = sys.argv[3] if len(sys.argv) > 3 else "output/cpu_eval"

# ===== Parse targets =====
targets = []
with open(DATASET_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            targets.append(line)

print(f"Running CPU eval on {len(targets)} targets")
print(f"Config: {CONFIG_FILE}")
print(f"Output: {OUTPUT_DIR}")

# ===== Load gin config =====
gin.parse_config_file(CONFIG_FILE)

# ===== Process each molecule =====
results = {"targets": [], "solutions": [], "success": [], "times": []}
start_time = time.time()

for idx, smiles in enumerate(targets):
    t0 = time.time()
    try:
        from moretro.moretro_star import MORetro
        moretro = MORetro(
            smiles,
            output_dir=OUTPUT_DIR,
            visualize_plots=False,
            save_json=False,
        )
        moretro.search()
        
        safe_name = moretro._safe_smiles_dirname(smiles)
        cost_path = Path(f"{OUTPUT_DIR}/{safe_name}/solution_costs.pkl")
        if cost_path.exists():
            with open(cost_path, "rb") as f:
                cost_dict = pickle.load(f)
            costs = np.array(list(cost_dict.keys()))
            n_solutions = len(costs)
        else:
            costs = np.array([])
            n_solutions = 0
            
        success = n_solutions > 0
        results["targets"].append(smiles)
        results["solutions"].append(n_solutions)
        results["success"].append(success)
        
        elapsed = time.time() - t0
        results["times"].append(elapsed)
        rate = 100 * sum(results["success"]) / len(results["success"])
        print(f"[{idx+1}/{len(targets)}] success={success} sols={n_solutions} time={elapsed:.0f}s rate={rate:.1f}%")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{idx+1}/{len(targets)}] ERROR: {e} time={elapsed:.0f}s")
        results["targets"].append(smiles)
        results["solutions"].append(0)
        results["success"].append(False)
        results["times"].append(elapsed)

total_time = time.time() - start_time
n_success = sum(results["success"])
n_total = len(results["targets"])

# ===== Compute metrics =====
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

# Collect all costs
all_per_mol = {}
for smiles in results["targets"]:
    safe_name = smiles.translate(str.maketrans({"/": "_", "\\": "_", ":": "_", "*": "_", "?": "_", "\"": "_", "<": "_", ">": "_", "|": "_"}))
    cost_path = Path(f"{OUTPUT_DIR}/{safe_name}/solution_costs.pkl")
    if cost_path.exists():
        with open(cost_path, "rb") as f:
            cost_dict = pickle.load(f)
        if cost_dict:
            costs_arr = np.array(list(cost_dict.keys()))
            if costs_arr.shape[1] >= 3:
                all_per_mol[smiles] = costs_arr[:, :3]
            else:
                all_per_mol[smiles] = None
        else:
            all_per_mol[smiles] = None
    else:
        all_per_mol[smiles] = None

# Global normalization
all_flat = []
for costs in all_per_mol.values():
    if costs is not None:
        all_flat.append(costs)

if not all_flat:
    print("No successful molecules!")
    metrics = {"HV": 0.0, "R2": float("inf"), "Success Rate": 0.0}
else:
    all_flat = np.vstack(all_flat)
    p5 = np.percentile(all_flat, 5, axis=0)
    p95 = np.percentile(all_flat, 95, axis=0)
    
    def normalize(costs):
        denom = p95 - p5
        denom[denom == 0] = 1.0
        return (costs - p5) / denom
    
    ref_point = np.array([1.1, 1.1, 1.1])
    weights = get_weights(3)
    
    hv_list = []
    r2_list = []
    for costs in all_per_mol.values():
        if costs is not None:
            c_norm = normalize(costs)
            try:
                hv_list.append(float(HV(ref_point=ref_point)(c_norm)))
            except Exception:
                hv_list.append(0.0)
            try:
                r2_list.append(compute_r2(c_norm, ref_point, weights))
            except Exception:
                pass
        else:
            hv_list.append(0.0)
    
    hv_arr = np.array(hv_list)
    r2_valid = [r for r in r2_list if np.isfinite(r)]
    
    metrics = {
        "HV": round(float(hv_arr.mean()), 4),
        "R2": round(float(np.mean(r2_valid)), 4) if r2_valid else float("inf"),
        "Success Rate": round(100 * n_success / n_total, 1),
        "n_processed": n_total,
        "n_success": n_success,
        "total_time_s": round(total_time, 1),
    }

print(f"\n===== RESULTS =====")
print(json.dumps(metrics, indent=2))

# Save metrics
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR}/metrics.json")
