#!/usr/bin/env python3
"""Compute LAD with per-function positive shift for negative values (matching paper approach).
Outputs machine-parseable JSON at the end with key __LAD_RESULT__.
"""
import json
import sys
import numpy as np

def compute_lad_shifted(summary_paths):
    all_per_fid_lads = []
    
    for path in summary_paths:
        with open(path, "r") as f:
            data = json.load(f)
        
        for dim_key, dim_data in data.items():
            if dim_key == "meta":
                continue
            per_fid = dim_data["per_fid_best"]
            per_fid_lad = {}
            
            for fid_str, fid_data in per_fid.items():
                variants = fid_data["variants"]
                all_vals = {}
                for v in variants:
                    all_vals[v["variant"]] = v["final_mean"]
                
                baseline_val = all_vals["baseline"]
                md_vals = {k: v for k, v in all_vals.items() if k != "baseline"}
                best_vname, best_val = min(md_vals.items(), key=lambda x: x[1])
                
                all_vals_list = list(all_vals.values())
                min_val = min(all_vals_list)
                
                if min_val <= 0:
                    shift = -min_val + 1e-12
                else:
                    shift = 1e-12
                
                baselog = np.log10(baseline_val + shift)
                mdlog = np.log10(best_val + shift)
                lad = baselog - mdlog
                per_fid_lad[int(fid_str)] = lad
            
            all_per_fid_lads.append(per_fid_lad)
    
    per_fid_avg = {}
    fids = sorted(all_per_fid_lads[0].keys())
    for fid in fids:
        vals = [w[fid] for w in all_per_fid_lads if fid in w]
        avg = np.mean(vals)
        per_fid_avg[fid] = avg
        print("  f{}: {:.4f}".format(fid, avg))
    
    final_lad = np.mean(list(per_fid_avg.values()))
    final_std = np.std(list(per_fid_avg.values()))
    print("\n[FINAL RESULT] 3-window average LAD = {:.4f}".format(final_lad))
    print("Std across functions = {:.4f}".format(final_std))
    return final_lad


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 scripts/compute_lad_shifted.py <win0.json> <win1.json> <win2.json>")
        sys.exit(1)
    result = compute_lad_shifted(sys.argv[1:4])
    print("\n__LAD_RESULT__")
    print(json.dumps({"LAD": round(float(result), 4)}))
