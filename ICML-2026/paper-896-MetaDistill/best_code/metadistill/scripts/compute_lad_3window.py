#!/usr/bin/env python3
"""Compute 3-window average LAD from three summary JSONs."""
import json
import sys
import numpy as np

def compute_lad_from_summary(data):
    for dim_key, dim_data in data.items():
        if dim_key == "meta":
            continue
        per_fid = dim_data["per_fid_best"]
        lads = {}
        for fid_str, fid_data in per_fid.items():
            variants = fid_data["variants"]
            baseline_val = None
            md_vals = {}
            for v in variants:
                if v["variant"] == "baseline":
                    baseline_val = v["final_mean"]
                else:
                    md_vals[v["variant"]] = v["final_mean"]
            if baseline_val is None:
                continue
            best_variant = min(md_vals.items(), key=lambda x: x[1])
            def safe_log10(x):
                x = float(x)
                if x <= 0:
                    return np.log10(max(x, 1e-30))
                return np.log10(x)
            lad = safe_log10(baseline_val) - safe_log10(best_variant[1])
            lads[int(fid_str)] = lad
        return lads

if len(sys.argv) < 4:
    print("Usage: python3 scripts/compute_lad_3window.py <win0.json> <win1.json> <win2.json>")
    sys.exit(1)

all_lads = []
for path in sys.argv[1:4]:
    with open(path, "r") as f:
        data = json.load(f)
    lads = compute_lad_from_summary(data)
    all_lads.append(lads)
    avg = np.mean(list(lads.values()))
    print("  {}: LAD = {:.4f}".format(path.split("/")[-1][:40], avg))

# Per-function average across 3 windows
print("\nPer-function LAD (averaged across 3 windows):")
per_fid_avg = {}
for fid in sorted(all_lads[0].keys()):
    vals = [w[fid] for w in all_lads if fid in w]
    avg = np.mean(vals)
    per_fid_avg[fid] = avg
    print("  f{}: {:.4f}".format(fid, avg))

final_lad = np.mean(list(per_fid_avg.values()))
final_std = np.std(list(per_fid_avg.values()))
print("\n[FINAL RESULT] 3-window average LAD = {:.4f}".format(final_lad))
print("Std across functions = {:.4f}".format(final_std))
