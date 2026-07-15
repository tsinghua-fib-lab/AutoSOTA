"""Run all Credit Pareto models in a single Python process to avoid JIT recompilation."""

import json, os, math, time, gc
import numpy as np
import veritas

os.environ["PRADA_DATA_DIR"] = "/datasets/prada"
os.makedirs("/datasets/prada", exist_ok=True)

# Import repo modules
import sys
sys.path.insert(0, "/repo/src")
import util
from verification import run_adversarial_robustness_tasks
from depth_first_ocenum import enumerate_ocs_to_disk

COMPRESSED_FILE = "/repo/data/raw/OC-space_paper_compression.txt"
SAVE_DIR = "/autosota_cache/oc_space"
SEED = 5823

# Step 1: Find Credit Pareto front models
print("=== Step 1: Finding Credit Pareto models ===")
results = {}
with open(COMPRESSED_FILE, "r") as f:
    for line in f:
        if not line.startswith("{"):
            continue
        line_dict = json.loads(line.strip())
        if line_dict["dname"] != "Credit":
            continue
        p = line_dict["params"]
        key = (int(p["n_estimators"]), int(p["max_depth"]), float(p["learning_rate"]), int(line_dict["fold"]))
        results[key] = line_dict

import pandas as pd

refinements_idx = None
for v in list(results.values())[:1]:
    refinements = v["refinements"]
    refinements_idx = [i for i, r in enumerate(refinements) if r["penalty"] in ["lop", "ours"]][0]
    break

df_data = []
for depth in [4, 6, 8]:
    for n_est in [10, 25, 50, 100]:
        for lr in [0.1, 0.25, 0.5, 1.0]:
            avg_mtest = 0
            avg_nleafs = 0
            for fold in range(5):
                key = (n_est, depth, lr, fold)
                if key in results:
                    r = results[key]["refinements"][refinements_idx]
                    avg_mtest += r["mtest"] / 5
                    avg_nleafs += r["nleafs"] / 5
            df_data.append([depth, n_est, lr, avg_mtest, avg_nleafs])

df = pd.DataFrame(df_data, columns=["max_depth", "n_estimators", "learning_rate", "mtest", "nleafs"])
on_front, _ = util.pareto_front_xy(df["nleafs"].to_numpy(), df["mtest"].to_numpy())
df["on_front"] = on_front

pareto_configs = []
for row in df[df["on_front"] == True].itertuples():
    for fold in range(5):
        pareto_configs.append({
            "fold": fold,
            "learning_rate": float(row.learning_rate),
            "max_depth": int(row.max_depth),
            "n_estimators": int(row.n_estimators),
        })

print(f"Pareto front models to process: {len(pareto_configs)}")

# Step 2: Process each model - enumerate + verify
all_results = []

for i, cfg in enumerate(pareto_configs):
    fold = cfg["fold"]
    lr = cfg["learning_rate"]
    md = cfg["max_depth"]
    ne = cfg["n_estimators"]
    
    key = (ne, md, lr, fold)
    if key not in results:
        print(f"[{i+1}/{len(pareto_configs)}] SKIP: no data for {cfg}")
        continue
    
    line_dict = results[key]
    refinements = line_dict["refinements"]
    idx = refinements_idx
    model_json = line_dict["refinements"][idx]["model_json"]
    
    print(f"[{i+1}/{len(pareto_configs)}] Credit fold={fold} lr={lr} depth={md} trees={ne}", end=" ")
    
    try:
        model = veritas.AddTree.from_json(model_json)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        continue
    
    # Enumerate OC-space
    run_id = time.strftime("%Y%m%d_%H%M%S")
    oc_file = f"{SAVE_DIR}/OC_enum_Credit_{ne}_{md}_{lr}_fold{fold}_{run_id}.zarr"
    
    t0 = time.time()
    enum_result = enumerate_ocs_to_disk(model, buffer_size=1000*8194, filename=oc_file, timeout=3600)
    enum_time = time.time() - t0
    oc_space = enum_result.get("oc_space", 0)
    failed = enum_result.get("failed", False)
    
    if failed:
        print(f"ENUM_FAILED (time={enum_time:.1f}s)")
        continue
    
    print(f"|O|={oc_space} enum={enum_time:.1f}s", end=" ")
    
    # Get dataset for verification
    d, dtrain, dvalid, dtest = util.get_dataset("Credit", SEED, fold, True)
    
    # Run verification
    storage_options = None
    t0 = time.time()
    try:
        verify_result = run_adversarial_robustness_tasks(
            model, dtest.X, dtest.y, 
            oc_file=oc_file, storage_options=storage_options, 
            timeout=3600, n=500
        )
    except Exception as e:
        print(f"VERIFY_ERROR: {e}")
        continue
    
    verify_time = time.time() - t0
    
    # Extract OC-Tree metrics
    octree = verify_result.get("octree_index", {})
    if octree:
        n_inst = octree.get("emp_rob_n", 0)
        total_t = octree.get("emp_rob_time", 0)
        if n_inst > 0:
            avg_ms = total_t / n_inst * 1000
        else:
            avg_ms = 0
        idx_build = octree.get("index_building_time", 0)
        
        # Individual times for analysis
        indiv_times = octree.get("individual_verification_times", [])
        
        # Average excluding first call (JIT warmup)
        if len(indiv_times) > 1:
            rest_times = indiv_times[1:]
            rest_avg_ms = sum(rest_times) / len(rest_times) * 1000
        else:
            rest_avg_ms = avg_ms
        
        print(f"avg={avg_ms:.4f}ms rest_avg={rest_avg_ms:.4f}ms idx={idx_build:.1f}s")
        
        all_results.append({
            "fold": fold, "lr": lr, "depth": md, "trees": ne,
            "oc_space": oc_space,
            "avg_time_ms": avg_ms,
            "rest_avg_ms": rest_avg_ms,  # excluding first warmup call
            "total_time_s": total_t,
            "n_instances": n_inst,
            "index_build_s": idx_build,
            "indiv_times_n": len(indiv_times),
        })
    else:
        print("NO_OCTREE_RESULT")
    
    gc.collect()

# Step 3: Compute geometric mean
print("\n=== RESULTS ===")
if all_results:
    # Standard geometric mean
    times_full = [r["avg_time_ms"] for r in all_results]
    log_sum = sum(math.log(t) for t in times_full if t > 0)
    geo_full = math.exp(log_sum / len(times_full))
    
    # Geometric mean excluding first warmup call
    times_rest = [r["rest_avg_ms"] for r in all_results]
    log_sum_r = sum(math.log(t) for t in times_rest if t > 0)
    geo_rest = math.exp(log_sum_r / len(times_rest))
    
    print(f"Models verified: {len(all_results)}")
    print(f"Geometric mean (full avg): {geo_full:.6f} ms")
    print(f"Geometric mean (excl 1st call): {geo_rest:.6f} ms")
    print(f"Paper value: 0.027 ms")
    print(f"Min avg: {min(times_full):.6f} ms, Max avg: {max(times_full):.6f} ms")
    
    with open(f"{SAVE_DIR}/credit_single_process_results.json", "w") as f:
        json.dump({"results": all_results, "geo_mean_full": geo_full, "geo_mean_rest": geo_rest}, f, indent=2)
