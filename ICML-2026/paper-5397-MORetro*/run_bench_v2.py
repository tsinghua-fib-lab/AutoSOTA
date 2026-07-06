import ast, json, os, pickle, sys, time, warnings
from pathlib import Path
import gin, numpy as np
warnings.filterwarnings("ignore")

LOG_FILE = "/repo/output/benchmark_run.log"
STATE_FILE = "/repo/output/benchmark_state.json"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

targets = []
with open("/paper_data/uspto_190_targets.txt") as f:
    for line in f:
        line = line.strip()
        if line:
            tup = ast.literal_eval(line)
            targets.append(tup[0])

log(f"Running on {len(targets)} targets")
gin.parse_config_file("/repo/configs/search_config.gin")

if os.path.exists(STATE_FILE):
    with open(STATE_FILE) as f:
        state = json.load(f)
    start_idx = state["last_idx"]
    success_count = state["success_count"]
    total_time = state["total_time"]
    log(f"Resuming from molecule {start_idx}/{len(targets)}")
else:
    start_idx = 0
    success_count = 0
    total_time = 0.0

for idx in range(start_idx, len(targets)):
    smiles = targets[idx]
    t0 = time.time()
    try:
        from moretro.moretro_star import MORetro
        moretro = MORetro(smiles, output_dir="output/full_benchmark", visualize_plots=False, save_json=False)
        moretro.search()
        safe_name = moretro._safe_smiles_dirname(smiles)
        cost_path = Path(f"output/full_benchmark/{safe_name}/solution_costs.pkl")
        if cost_path.exists():
            with open(cost_path, "rb") as f:
                costs = pickle.load(f)
            n_sol = len(costs)
        else:
            n_sol = 0
    except Exception as e:
        elapsed = time.time() - t0
        log(f"[{idx+1:3d}/{len(targets)}] ERROR: {type(e).__name__}: {str(e)[:100]} ({elapsed:.0f}s)")
        n_sol = 0

    elapsed = time.time() - t0
    total_time += elapsed
    if n_sol > 0:
        success_count += 1
    log(f"[{idx+1:3d}/{len(targets)}] success={n_sol>0} sols={n_sol} time={elapsed:.0f}s rate={100*success_count/(idx+1):.1f}%")
    
    if (idx + 1) % 5 == 0:
        with open(STATE_FILE, "w") as f:
            json.dump({"last_idx": idx+1, "success_count": success_count, "total_time": total_time}, f)
        log(f"  -> Checkpoint at mol {idx+1}: {success_count} success, {total_time:.0f}s")

log(f"DONE. {success_count}/{len(targets)} successful. Time: {total_time:.0f}s")
