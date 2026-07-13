#!/usr/bin/env python3
"""Sweep negative sampling ratios with 3 seeds each, pick best AUC-ROC."""
import os, sys, subprocess, re, numpy as np

os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
RATIOS = [3.0, 7.0, 10.0, 15.0]
N_SEEDS = 3

def run_one(ratio):
    env = os.environ.copy()
    env["MKL_SERVICE_FORCE_INTEL"] = "1"
    result = subprocess.run(
        [sys.executable, "main.py", "--K", "9", "--dataset", "cora",
         "--LP", "True", "--clas", "False", "--epochs", "5000", "--lr", "0.01",
         "--neg_ratio", str(ratio)],
        capture_output=True, text=True, cwd="/repo", env=env,
    )
    out = result.stdout + "\n" + result.stderr
    roc_match = re.search(r"ROC:\s+([\d.]+)", out)
    pr_match = re.search(r"PR:\s+([\d.]+)", out)
    if not roc_match or not pr_match:
        print(f"  PARSE FAILED for ratio={ratio}")
        return None, None
    return float(roc_match.group(1)), float(pr_match.group(1))

def main():
    results = {}
    for ratio in RATIOS:
        rocs, prs = [], []
        for seed in range(N_SEEDS):
            print(f"Ratio={ratio} seed={seed+1}/{N_SEEDS} ...", flush=True)
            roc, pr = run_one(ratio)
            if roc is not None:
                rocs.append(roc); prs.append(pr)
                print(f"  ROC={roc:.6f} PR={pr:.6f}")
        if rocs:
            results[ratio] = {
                "roc_mean": np.mean(rocs), "roc_std": np.std(rocs),
                "pr_mean": np.mean(prs), "pr_std": np.std(prs),
            }
            print(f"Ratio={ratio}: AUC-ROC={np.mean(rocs):.6f}±{np.std(rocs):.6f} PR-AUC={np.mean(prs):.6f}±{np.std(prs):.6f}")
    
    print("\n=== SWEEP SUMMARY ===")
    best_ratio, best_roc = None, -1
    for ratio, r in sorted(results.items()):
        print(f"ratio={ratio:.1f}: AUC-ROC={r[roc_mean]:.6f} PR-AUC={r[pr_mean]:.6f}")
        if r["roc_mean"] > best_roc:
            best_roc = r["roc_mean"]
            best_ratio = ratio
    print(f"\nBest ratio: {best_ratio} (AUC-ROC={best_roc:.6f})")
    # Also include baseline ratio=5 for comparison
    print(f"Baseline ratio=5: AUC-ROC=0.8407")

if __name__ == "__main__":
    main()
