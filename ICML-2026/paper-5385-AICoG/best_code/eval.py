#!/usr/bin/env python3
"""Evaluation wrapper for AICoG Cora D=8 link prediction.
Runs 5 independent trials and reports average AUC-ROC and PR-AUC.
Matches paper settings: K=9 (D=8), Adam lr=0.01, 5000 iterations, Helmert ILR basis.
"""
import os
import subprocess
import sys
import re

# Fix MKL threading conflict in this container
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

N_RUNS = 5
EXTRA_ARGS = sys.argv[1:]  # Forward extra args to main.py

def run_one(seed=None):
    """Run one trial and return (roc_auc, pr_auc)."""
    env = os.environ.copy()
    env["MKL_SERVICE_FORCE_INTEL"] = "1"
    cmd = [
        sys.executable, "main.py",
        "--K", "9",
        "--dataset", "cora",
        "--LP", "True",
        "--clas", "False",
        "--epochs", "5000",
        "--lr", "0.01",
    ] + EXTRA_ARGS
    if seed is not None:
        cmd += ["--seed", str(seed)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/repo",
        env=env,
    )
    out = result.stdout + "\n" + result.stderr
    roc_match = re.search(r"ROC:\s+([\d.]+)", out)
    pr_match = re.search(r"PR:\s+([\d.]+)", out)
    if not roc_match or not pr_match:
        raise RuntimeError(f"Failed to parse output. stdout+stderr:\n{out}")
    return float(roc_match.group(1)), float(pr_match.group(1))

def main():
    rocs = []
    prs = []
    for i in range(N_RUNS):
        print(f"--- Run {i+1}/{N_RUNS} ---")
        roc, pr = run_one(seed=42 + i)
        rocs.append(roc)
        prs.append(pr)
        print(f"  AUC-ROC: {roc:.6f}  PR-AUC: {pr:.6f}")

    import numpy as np
    mean_roc = float(np.mean(rocs))
    mean_pr = float(np.mean(prs))
    std_roc = float(np.std(rocs))
    std_pr = float(np.std(prs))
    print(f"\n=== 5-run averages ===")
    print(f"AUC-ROC: {mean_roc:.6f} ± {std_roc:.6f}")
    print(f"PR-AUC:  {mean_pr:.6f} ± {std_pr:.6f}")
    print(f"\nSUMMARY: AUC-ROC={mean_roc:.4f} PR-AUC={mean_pr:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
