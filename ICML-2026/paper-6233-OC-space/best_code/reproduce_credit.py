#!/usr/bin/env python3
"""Full reproduction: enumerate Credit Pareto models + OC-Tree verification"""

import json, os, sys, time, math, subprocess
from pathlib import Path

COMPRESSED_FILE = "/repo/data/raw/OC-space_paper_compression.txt"
SAVE_DIR = "/autosota_cache/oc_space"
MODELS_FILE = "/repo/data/raw/OC-space_paper_compression.txt"

os.environ["PRADA_DATA_DIR"] = "/datasets/prada"
os.makedirs("/datasets/prada", exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Step 1: Generate enumeration commands
print("=== Step 1: Listing Pareto front models for Credit ===")
import subprocess
result = subprocess.run(
    ["uv", "run", "python3", "experiments/oc_space_experiment.py", "list_pareto_models",
     COMPRESSED_FILE, SAVE_DIR, "--seed", "5823"],
    capture_output=True, text=True, cwd="/repo"
)
lines = result.stdout.strip().split("\n")
credit_lines = [l for l in lines if "Credit" in l and l.startswith("python3")]
print(f"Found {len(credit_lines)} Credit models to enumerate")

# Step 2: Enumerate each model
enum_results = []
print("\n=== Step 2: Enumerating OC-spaces ===")
for i, cmd_line in enumerate(credit_lines):
    # Parse the command line to extract args
    parts = cmd_line.split()
    dname = parts[3]
    
    # Extract options
    args = {}
    j = 4
    while j < len(parts):
        if parts[j].startswith("--"):
            key = parts[j][2:]
            if j + 1 < len(parts) and not parts[j+1].startswith("--"):
                args[key] = parts[j+1]
                j += 2
            else:
                args[key] = "true"
                j += 1
        else:
            j += 1
    
    fold = int(args.get("fold", 0))
    lr = float(args.get("learning-rate", 0.1))
    md = int(args.get("max-depth", 4))
    ne = int(args.get("n-estimators", 10))
    
    print(f"\n[{i+1}/{len(credit_lines)}] Credit fold={fold} lr={lr} depth={md} trees={ne}")
    
    result = subprocess.run(
        ["uv", "run", "python3", "experiments/oc_space_experiment.py", "enumerate_model", dname,
         "--models_file", MODELS_FILE, "--save_directory", SAVE_DIR,
         "--seed", "5823", "--fold", str(fold),
         "--learning-rate", str(lr), "--max-depth", str(md), "--n-estimators", str(ne),
         "--timeout", "3600"],
        capture_output=True, text=True, cwd="/repo"
    )
    
    # Parse enumeration output
    for line in result.stdout.strip().split("\n"):
        if line.startswith("{") and "enumeration" in line:
            data = json.loads(line)
            enum_info = data["enumeration"]
            enum_results.append({
                "params": {"fold": fold, "lr": lr, "depth": md, "trees": ne},
                "oc_file": enum_info["filename"],
                "oc_space": enum_info["oc_space"],
                "failed": enum_info["failed"],
                "elapsed": enum_info["elapsed"],
            })
            status = "OK" if not enum_info["failed"] else "FAILED"
            print(f"  -> OC-space={enum_info['oc_space']}, time={enum_info['elapsed']:.1f}s [{status}]")
            break

# Save enumeration results
with open(f"{SAVE_DIR}/credit_enum_results.json", "w") as f:
    json.dump(enum_results, f, indent=2)
print(f"\nEnumerated {len(enum_results)} models, saved to {SAVE_DIR}/credit_enum_results.json")

# Step 3: Run OC-Tree verification on each model
print("\n=== Step 3: OC-Tree verification ===")
verify_results = []

for i, er in enumerate(enum_results):
    if er["failed"]:
        print(f"[{i+1}/{len(enum_results)}] SKIP (enumeration failed)")
        continue
    
    p = er["params"]
    print(f"[{i+1}/{len(enum_results)}] Verifying fold={p['fold']} lr={p['lr']} depth={p['depth']} trees={p['trees']} (|O|={er['oc_space']})")
    
    result = subprocess.run(
        ["uv", "run", "python3", "experiments/oc_space_experiment.py", "verify_model",
         er["oc_file"],
         "--models_file", MODELS_FILE,
         "--dname", "Credit",
         "--fold", str(p["fold"]),
         "--learning-rate", str(p["lr"]),
         "--max-depth", str(p["depth"]),
         "--n-estimators", str(p["trees"]),
         "--seed", "5823",
         "--timeout", "3600"],
        capture_output=True, text=True, cwd="/repo", timeout=3600
    )
    
    # Parse verification output
    for line in result.stdout.strip().split("\n"):
        if line.startswith("{") and "verification" in line:
            data = json.loads(line)
            v = data["verification"]
            if "octree_index" in v:
                octree = v["octree_index"]
                n = octree.get("emp_rob_n", 0)
                if n > 0:
                    avg_time_ms = octree["emp_rob_time"] / n * 1000
                else:
                    avg_time_ms = float("inf")
                verify_results.append({
                    **er,
                    "avg_time_ms": avg_time_ms,
                    "total_time_s": octree["emp_rob_time"],
                    "n_instances": n,
                    "index_build_s": octree.get("index_building_time", 0),
                })
                print(f"  -> avg={avg_time_ms:.6f}ms per instance, index_build={octree.get('index_building_time', 0):.2f}s")
            break

# Save verification results
with open(f"{SAVE_DIR}/credit_verify_results.json", "w") as f:
    json.dump(verify_results, f, indent=2)

# Step 4: Compute geometric mean
print("\n=== Step 4: Results ===")
if verify_results:
    times = [r["avg_time_ms"] for r in verify_results]
    log_sum = sum(math.log(t) for t in times if t > 0)
    geo_mean = math.exp(log_sum / len(times))
    
    print(f"Models verified: {len(verify_results)}")
    print(f"Individual avg times (ms): {[f'{t:.6f}' for t in times[:10]]}...")
    print(f"Geometric mean time (ms): {geo_mean:.6f}")
    print(f"Paper value: 0.027 ms")
    
    # Also compute mean and median
    mean_time = sum(times) / len(times)
    sorted_times = sorted(times)
    median_time = sorted_times[len(sorted_times)//2]
    print(f"Mean time (ms): {mean_time:.6f}")
    print(f"Median time (ms): {median_time:.6f}")
    
    # Final output
    print(f"\nREPRODUCTION RESULT: geometric_mean={geo_mean:.6f}ms, paper=0.027ms")
else:
    print("ERROR: No verification results")
