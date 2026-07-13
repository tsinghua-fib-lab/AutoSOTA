#!/usr/bin/env python3
"""Sweep learning rates with 3 seeds each."""
import os, sys, subprocess, re, numpy as np
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")

LRS = [0.005, 0.01, 0.02, 0.05, 0.001]
N_SEEDS = 3
NEG_RATIO = "3.0"

def run_one(lr, seed):
    env = os.environ.copy()
    env["MKL_SERVICE_FORCE_INTEL"] = "1"
    result = subprocess.run(
        [sys.executable, "main.py", "--K", "9", "--dataset", "cora",
         "--LP", "True", "--clas", "False", "--epochs", "5000",
         "--lr", str(lr), "--neg_ratio", NEG_RATIO, "--seed", str(seed)],
        capture_output=True, text=True, cwd="/repo", env=env,
    )
    out = result.stdout + "\n" + result.stderr
    roc = re.search(r"ROC:\s+([\d.]+)", out)
    pr = re.search(r"PR:\s+([\d.]+)", out)
    if not roc or not pr:
        return None, None
    return float(roc.group(1)), float(pr.group(1))

def main():
    results = {}
    for lr in LRS:
        rocs, prs = [], []
        for s in range(N_SEEDS):
            seed = 100 + int(lr*1000) + s
            print(f"lr={lr} seed={s+1}/{N_SEEDS} ...", flush=True)
            roc, pr = run_one(lr, seed)
            if roc is not None:
                rocs.append(roc); prs.append(pr)
                print(f"  ROC={roc:.6f} PR={pr:.6f}")
        if rocs:
            results[lr] = {"roc": np.mean(rocs), "roc_std": np.std(rocs),
                          "pr": np.mean(prs), "pr_std": np.std(prs)}
            print(f"lr={lr}: AUC-ROC={np.mean(rocs):.6f}±{np.std(rocs):.6f} PR={np.mean(prs):.6f}")
    
    print("\n=== LR SWEEP SUMMARY ===")
    for lr in sorted(results.keys()):
        r = results[lr]
        print(f"lr={lr:.4f}: AUC-ROC={r[roc]:.6f} PR-AUC={r[pr]:.6f}")
    best_lr = max(results, key=lambda x: results[x]["roc"])
    print(f"\nBest lr: {best_lr} (AUC-ROC={results[best_lr][roc]:.6f})")

if __name__ == "__main__":
    main()
