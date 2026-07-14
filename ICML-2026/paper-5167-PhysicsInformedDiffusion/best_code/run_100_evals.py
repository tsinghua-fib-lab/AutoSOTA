import subprocess
import re
import sys
import os
import numpy as np
import yaml
import json

results = []
total_runs = 100

for run in range(total_runs):
    offset = run
    seed = run

    # Update config for this run
    with open("configs/poisson_inverse_u500.yaml", "r") as f:
        config = yaml.safe_load(f)
    config["data"]["offset"] = offset
    config["generate"]["seed"] = seed
    with open("configs/poisson_inverse_u500_run.yaml", "w") as f:
        yaml.dump(config, f)

    result = subprocess.run(
        ["python3", "generate_pde.py", "--config", "configs/poisson_inverse_u500_run.yaml"],
        capture_output=True, text=True, cwd="/repo",
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
    )

    output = result.stdout + result.stderr

    # Parse metrics - for inverse problem, primary metric is relative_error_a
    rel_err_u_match = re.search(r"relative_error_u:([\d\.e\+\-]+)", output)
    rel_err_a_match = re.search(r"relative_error_a:([\d\.e\+\-]+)", output)
    pde_res_match = re.search(r"L_pde:([\d\.e\+\-]+)", output)
    loss_fd_match = re.search(r"loss_fd:([\d\.e\+\-]+)", output)

    if rel_err_a_match and pde_res_match:
        rel_err_a = float(rel_err_a_match.group(1))
        rel_err_u = float(rel_err_u_match.group(1)) if rel_err_u_match else 0.0
        pde_res = float(pde_res_match.group(1))
        loss_fd = float(loss_fd_match.group(1)) if loss_fd_match else 0.0

        rel_err_a_pct = rel_err_a * 100
        rel_err_u_pct = rel_err_u * 100

        results.append({
            "run": run,
            "offset": offset,
            "seed": seed,
            "rel_err_a": rel_err_a,
            "rel_err_a_pct": rel_err_a_pct,
            "rel_err_u": rel_err_u,
            "rel_err_u_pct": rel_err_u_pct,
            "pde_res": pde_res,
            "loss_fd": loss_fd
        })
        print(f"Run {run:3d}/{total_runs}: Rel.err(a)={rel_err_a_pct:.4f}%, Rel.err(u)={rel_err_u_pct:.4f}%, PDE res.={pde_res:.6f}, FD={loss_fd:.6f}")
    else:
        print(f"Run {run:3d}/{total_runs}: FAILED to parse output")
        if result.returncode != 0:
            print(f"  Return code: {result.returncode}")
            print(f"  Last 300 chars: {output[-300:]}")

# Compute statistics
if results:
    rel_errs_a = np.array([r["rel_err_a_pct"] for r in results])
    rel_errs_u = np.array([r["rel_err_u_pct"] for r in results])
    pde_reses = np.array([r["pde_res"] for r in results])
    loss_fds = np.array([r["loss_fd"] for r in results])

    stats = {
        "n_runs": len(results),
        "rel_err_a_pct_mean": float(rel_errs_a.mean()),
        "rel_err_a_pct_std": float(rel_errs_a.std()),
        "rel_err_a_pct_min": float(rel_errs_a.min()),
        "rel_err_a_pct_max": float(rel_errs_a.max()),
        "rel_err_u_pct_mean": float(rel_errs_u.mean()),
        "rel_err_u_pct_std": float(rel_errs_u.std()),
        "pde_res_l2_mean": float(pde_reses.mean()),
        "pde_res_l2_std": float(pde_reses.std()),
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
    print(f"Rel. err(a) %%: mean={rel_errs_a.mean():.4f}, std={rel_errs_a.std():.4f}")
    print(f"                min={rel_errs_a.min():.4f}, max={rel_errs_a.max():.4f}")
    print(f"Rel. err(u) %%: mean={rel_errs_u.mean():.4f}, std={rel_errs_u.std():.4f}")
    print(f"PDE res. (L_pde): mean={pde_reses.mean():.6f}, std={pde_reses.std():.6f}")
    print(f"PDE res. (loss_fd): mean={loss_fds.mean():.6f}, std={loss_fds.std():.6f}")
    print(f"\nPaper targets:")
    print(f"  Rel.err(a) = {stats['paper_rel_err']}%")
    print(f"  PDE res. = {stats['paper_pde_res']}")
    print(f"  Repro CI: [{stats['repo_ci_lower']}%, {stats['repo_ci_upper']}%]")

    # Save stats
    with open("eval_results_100.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved results to eval_results_100.json")
else:
    print("No successful runs!")
