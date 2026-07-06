#!/usr/bin/env python3
"""Robust MORetro* benchmark with intermediate saves and resume capability."""
import ast
import json
import os
import pickle
import signal
import sys
import time
import warnings
from pathlib import Path

import gin
import numpy as np

warnings.filterwarnings("ignore")

RESULTS_FILE = "output/benchmark_intermediate.json"
STATE_FILE = "output/benchmark_state.json"

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    def exit_gracefully(self, *args):
        self.kill_now = True

killer = GracefulKiller()

# Read targets
targets = []
with open("/paper_data/uspto_190_targets.txt") as f:
    for line in f:
        line = line.strip()
        if line:
            tup = ast.literal_eval(line)
            targets.append(tup[0])

gin.parse_config_file("/repo/configs/search_config.gin")

# Load or initialize state
if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        state = json.load(f)
    start_idx = state.get("last_idx", 0)
    all_costs = state.get("all_costs", {})
    success_count = state.get("success_count", 0)
    total_time = state.get("total_time", 0.0)
    print(f"Resuming from molecule {start_idx}/{len(targets)}")
else:
    start_idx = 0
    all_costs = {}
    success_count = 0
    total_time = 0.0

for idx in range(start_idx, len(targets)):
    if killer.kill_now:
        print(f"\nGraceful shutdown at molecule {idx}")
        break

    smiles = targets[idx]
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
            all_costs[smiles] = cost_dict
        else:
            all_costs[smiles] = {}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{idx+1:3d}/{len(targets)}] ERROR: {type(e).__name__}: {str(e)[:100]} ({elapsed:.0f}s)", flush=True)
        all_costs[smiles] = {}

    elapsed = time.time() - t0
    total_time += elapsed
    n_sol = len(all_costs[smiles])
    success = n_sol > 0
    if success:
        success_count += 1
    print(f"[{idx+1:3d}/{len(targets)}] success={success} sols={n_sol} time={elapsed:.0f}s", flush=True)

    # Save intermediate results every 5 molecules
    if (idx + 1) % 5 == 0 or idx == len(targets) - 1:
        # Count successes and collect costs
        successful_list = []
        for s, costs in all_costs.items():
            if len(costs) > 0:
                costs_arr = np.array(list(costs.keys()))
                if costs_arr.shape[1] >= 3:
                    successful_list.append(s)
        
        intermediate = {
            "processed": idx + 1,
            "total": len(targets),
            "success_count": success_count,
            "success_rate_pct": round(100 * success_count / max(1, idx + 1), 1),
            "total_time_seconds": total_time,
            "num_with_solutions": len(successful_list),
        }
        with open(RESULTS_FILE, "w") as f:
            json.dump(intermediate, f, indent=2)
        
        # Save state for resume
        with open(STATE_FILE, "w") as f:
            state_save = {
                "last_idx": idx + 1,
                "all_costs": all_costs,
                "success_count": success_count,
                "total_time": total_time,
            }
            json.dump(state_save, f, indent=2, default=str)
        
        print(f"  -> Intermediate save: {success_count}/{idx+1} successful ({intermediate[success_rate_pct]}%)", flush=True)

# Final save
final_results = {
    "all_costs": {k: list(v.keys()) if v else [] for k, v in all_costs.items()},
    "success_count": success_count,
    "total_targets": len(targets),
    "total_time_seconds": total_time,
}
with open("output/benchmark_final_costs.json", "w") as f:
    json.dump(final_results, f, indent=2, default=str)

print(f"\nDone. {success_count}/{len(targets)} successful. Results saved.", flush=True)
