#!/usr/bin/env python3
import subprocess
import re
import os
import sys
import numpy as np
import yaml
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

results = []
results_lock = threading.Lock()
total_runs = 100

def run_single(run_id):
    offset = run_id
    seed = run_id
    gpu_id = run_id % 2  # Alternate between GPU 0 and 1

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}

    # Build command with updated config inline via env-style override
    # Simpler: write config file per run (with lock to avoid race)
    config_path = f"configs/poisson_inv_run_{run_id}.yaml"
    with open("configs/poisson_inverse_u500.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["data"]["offset"] = offset
    config["generate"]["seed"] = seed
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    try:
        result = subprocess.run(
            ["python3", "generate_pde.py", "--config", config_path],
            capture_output=True, text=True, cwd="/repo",
            env=env, timeout=600
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {"run": run_id, "error": "timeout"}
    finally:
        try:
            os.remove(config_path)
        except:
            pass

    # Parse metrics
    rel_err_a_match = re.search(r"relative_error_a:([\d\.e\+\-]+)", output)
    pde_res_match = re.search(r"L_pde:([\d\.e\+\-]+)", output)
    loss_fd_match = re.search(r"loss_fd:([\d\.e\+\-]+)", output)

    if rel_err_a_match and pde_res_match:
        rel_err_a = float(rel_err_a_match.group(1))
        pde_res = float(pde_res_match.group(1))
        loss_fd = float(loss_fd_match.group(1)) if loss_fd_match else 0.0
        rel_err_a_pct = rel_err_a * 100

        # Clean up output .mat files
        for f in os.listdir("/repo"):
            if f.startswith("poisson_results_") and f.endswith(".mat"):
                try:
                    os.remove(os.path.join("/repo", f))
                except:
                    pass

        return {
            "run": run_id,
            "offset": offset,
            "seed": seed,
            "gpu": gpu_id,
            "rel_err_a_pct": rel_err_a_pct,
            "pde_res": pde_res,
            "loss_fd": loss_fd
        }
    else:
        return {"run": run_id, "error": f"parse_failed: {output[-200:]}"}

print(f"Starting {total_runs} evaluations on 2 GPUs...")
sys.stdout.flush()

# Run 2 at a time (one per GPU)
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {executor.submit(run_single, i): i for i in range(total_runs)}
    for future in as_completed(futures):
        res = future.result()
        with results_lock:
            if "error" in res:
                print(f"Run {res['run']:3d}/{total_runs}: FAILED - {res['error']}")
            else:
                results.append(res)
                print(f"Run {res['run']:3d}/{total_runs} [GPU{res['gpu']}]: Rel.err(a)={res['rel_err_a_pct']:.4f}%, PDE res.={res['pde_res']:.6f}, FD={res['loss_fd']:.6f}")
            sys.stdout.flush()

# Sort results by run ID
results.sort(key=lambda r: r["run"])

# Compute statistics
if results:
    rel_errs_a = np.array([r["rel_err_a_pct"] for r in results])
    pde_reses = np.array([r["pde_res"] for r in results])
    loss_fds = np.array([r["loss_fd"] for r in results])

    stats = {
        "n_runs": len(results),
        "rel_err_a_pct_mean": float(rel_errs_a.mean()),
        "rel_err_a_pct_std": float(rel_errs_a.std()),
        "rel_err_a_pct_min": float(rel_errs_a.min()),
        "rel_err_a_pct_max": float(rel_errs_a.max()),
        "pde_res_mean": float(pde_reses.mean()),
        "pde_res_std": float(pde_reses.std()),
        "pde_res_fd_mean": float(loss_fds.mean()),
        "pde_res_fd_std": float(loss_fds.std()),
        "paper_rel_err": 13.81,
        "paper_pde_res": 0.45,
        "repo_ci_lower": 10.70,
        "repo_ci_upper": 16.92,
    }

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({len(results)} runs)")
    print(f"{'='*60}")
    print(f"Rel. err(a) %:  mean={rel_errs_a.mean():.4f}, std={rel_errs_a.std():.4f}")
    print(f"                min={rel_errs_a.min():.4f}, max={rel_errs_a.max():.4f}")
    print(f"PDE res.:       mean={pde_reses.mean():.6f}, std={pde_reses.std():.6f}")
    print(f"PDE res. (FD):  mean={loss_fds.mean():.6f}, std={loss_fds.std():.6f}")
    print(f"\nPaper targets:")
    print(f"  Rel.err(a) = {stats['paper_rel_err']}%")
    print(f"  PDE res. = {stats['paper_pde_res']}")
    print(f"  Repro CI: [{stats['repo_ci_lower']}%, {stats['repo_ci_upper']}%]")
    print(f"\nWithin CI: {stats['repo_ci_lower'] <= rel_errs_a.mean() <= stats['repo_ci_upper']}")
    sys.stdout.flush()

    with open("eval_results_100.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved results to eval_results_100.json")
    sys.stdout.flush()
else:
    print("No successful runs!")
    sys.stdout.flush()
