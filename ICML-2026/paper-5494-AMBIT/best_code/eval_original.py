#!/usr/bin/env python3
"""Evaluation script for wa-dCoBET reproduction (paper 5494).
Usage: python3 eval.py  (from /repo)
Reproduces Type I Error (b=0.0) and Power (b=0.2) for LogQuad, d=10, n=500.
"""
import time, json
from cobet import aggregated_weights_power

CONFIG = dict(
    n=500, theta=2, K=4, transform_key="logquad",
    R_eval=500, alpha=0.05, d_coords=10,
    unbiased_plugin=True, reuse_J=True, seed=123,
)

def main():
    results = {}
    for b_val, label in [(0.0, "Type_I_Error"), (0.2, "Power")]:
        print(f"Running wa-dCoBET b={b_val}...", flush=True)
        t0 = time.time()
        r = aggregated_weights_power(b=b_val, **CONFIG)
        elapsed = time.time() - t0
        val = r["power_aggregated"]
        results[label] = val
        results[f"{label}_runtime_s"] = round(elapsed, 1)
        print(f"  {label} = {val:.4f}  ({elapsed:.1f}s)", flush=True)

    print(f"\n=== RESULTS ===")
    print(f"Type_I_Error: {results['Type_I_Error']:.4f}  (Paper: 0.056)")
    print(f"Power:        {results['Power']:.4f}  (Paper: 0.981)")
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to eval_results.json")

if __name__ == "__main__":
    main()
