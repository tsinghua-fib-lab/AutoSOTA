#!/usr/bin/env python3
"""Reproduction script for paper 5494: wa-dCoBET on LogQuad d=10 n=500."""
import time, json, sys, os
import numpy as np
from cobet import aggregated_weights_power

CONFIG = dict(
    n=500, theta=2, K=4, transform_key="logquad",
    R_eval=500, alpha=0.05, d_coords=10,
    unbiased_plugin=True, reuse_J=True,
)

def run():
    results = {}
    # Type I Error (b=0.0)
    print("Running Type I Error (b=0.0)...", flush=True)
    t0 = time.time()
    r0 = aggregated_weights_power(b=0.0, seed=123, **CONFIG)
    t1 = time.time()
    results["Type_I_Error"] = r0["power_aggregated"]
    results["Type_I_runtime_s"] = round(t1 - t0, 1)
    print(f"  Type I Error = {r0['power_aggregated']:.4f}  ({t1-t0:.1f}s)", flush=True)

    # Power (b=0.2)
    print("Running Power (b=0.2)...", flush=True)
    t0 = time.time()
    r2 = aggregated_weights_power(b=0.2, seed=123, **CONFIG)
    t1 = time.time()
    results["Power"] = r2["power_aggregated"]
    results["Power_runtime_s"] = round(t1 - t0, 1)
    print(f"  Power = {r2['power_aggregated']:.4f}  ({t1-t0:.1f}s)", flush=True)

    # Summary
    print("\n" + "=" * 60)
    print("REPRODUCTION RESULTS")
    print("=" * 60)
    print(f"  Type I Error: {results['Type_I_Error']:.4f}  (Paper: 0.056)")
    print(f"  Power:        {results['Power']:.4f}  (Paper: 0.981)")

    # Write JSON
    with open("/repo/reproduce_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /repo/reproduce_results.json")

if __name__ == "__main__":
    run()
