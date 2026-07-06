#!/usr/bin/env python3
"""Run MORetro* benchmark on USPTO-190 and compute metrics."""
import ast
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import gin
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

warnings.filterwarnings("ignore")

# Parse targets
targets = []
with open("/paper_data/uspto_190_targets.txt") as f:
    for line in f:
        line = line.strip()
        if line:
            tup = ast.literal_eval(line)
            targets.append(tup[0])

# Subset for testing (comment out for full run)
# targets = targets[:2]

print(f"Running benchmark on {len(targets)} targets")

# Gin config
gin.parse_config_file("/repo/configs/search_config.gin")

results = {
    "targets": [],
    "solutions": [],
    "success": [],
    "hv_list": [],
    "r2_list": [],
}

start_time = time.time()

for idx, smiles in enumerate(targets):
    t0 = time.time()
    try:
        from moretro.moretro_star import MORetro
        moretro = MORetro(
            smiles,
            output_dir=f"output/benchmark",
            visualize_plots=False,
            save_json=False,
        )
        moretro.search()

        # Load solution costs
        safe_name = moretro._safe_smiles_dirname(smiles)
        cost_path = Path(f"output/benchmark/{safe_name}/solution_costs.pkl")
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
        print(f"[{idx+1}/{len(targets)}] {smiles[:50]}... success={success} sols={n_solutions} time={elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{idx+1}/{len(targets)}] {smiles[:50]}... ERROR: {e} time={elapsed:.1f}s")
        results["targets"].append(smiles)
        results["solutions"].append(0)
        results["success"].append(False)

total_time = time.time() - start_time
print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")
print(f"Success rate: {sum(results[success])}/{len(targets)} = {100*sum(results[success])/len(targets):.1f}%")

# Save raw results
with open("output/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to output/benchmark_results.json")
