#!/usr/bin/env python3
"""Quickly evaluate an existing result JSON file and print metrics JSON."""
import sys, json, numpy as np
from scipy.stats import norm

def eval_file(filepath, alpha=0.05):
    with open(filepath) as f:
        data = json.load(f)
    summary = data["experiment_summary"]
    z_scores = [r["watermarked"]["summary"]["z_score"] for r in data["results"]]
    z_critical = norm.ppf(1 - alpha)
    mean_kl = summary["mean_kl"]
    neg_log_kl = -np.log(max(mean_kl, 1e-12))
    tpr = np.mean([z > z_critical for z in z_scores])
    return {
        "-log(DKL)": round(float(neg_log_kl), 4),
        "TPR": round(float(tpr), 4),
        "mean_kl": round(float(mean_kl), 6),
        "mean_z_score": round(float(summary.get("mean_z_score", 0)), 4),
        "n_samples": len(z_scores),
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 eval_result_file.py <result.json> [alpha]")
        sys.exit(1)
    filepath = sys.argv[1]
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    metrics = eval_file(filepath, alpha)
    print(json.dumps(metrics, indent=2))
