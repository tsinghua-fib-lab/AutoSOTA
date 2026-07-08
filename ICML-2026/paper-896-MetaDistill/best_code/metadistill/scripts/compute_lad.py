#!/usr/bin/env python3
"""Compute LAD from eval_compare_frameworks.py summary JSON."""
import json
import sys
import numpy as np

def compute_lad(summary_path):
    with open(summary_path, "r") as f:
        data = json.load(f)
    
    for dim_key, dim_data in data.items():
        if dim_key == "meta":
            continue
        
        per_fid = dim_data["per_fid_best"]
        lads = []
        
        for fid_str, fid_data in sorted(per_fid.items(), key=lambda x: int(x[0])):
            variants = fid_data["variants"]
            
            # Find baseline value
            baseline_val = None
            md_vals = {}
            for v in variants:
                if v["variant"] == "baseline":
                    baseline_val = v["final_mean"]
                else:
                    md_vals[v["variant"]] = v["final_mean"]
            
            if baseline_val is None:
                print(f"WARNING: No baseline for fid={fid_str}")
                continue
            
            # Best MetaDistill variant (lowest final mean)
            best_variant = min(md_vals.items(), key=lambda x: x[1])
            
            # LAD = log10(baseline) - log10(metadistill)
            # Using safe log10
            def safe_log10(x):
                x = float(x)
                if x <= 0:
                    return np.log10(max(x, 1e-30))
                return np.log10(x)
            
            lad = safe_log10(baseline_val) - safe_log10(best_variant[1])
            lads.append(lad)
            
            print(f"  f{fid_str}: baseline={baseline_val:.4f}, best_md={best_variant[0]}={best_variant[1]:.6f}, LAD={lad:.4f}")
        
        if lads:
            avg_lad = np.mean(lads)
            std_lad = np.std(lads)
            lad_str = ", ".join([f"{x:.4f}" for x in lads])
            print(f"\n  Dimension {dim_key}:")
            print(f"    LAD per function: [{lad_str}]")
            print(f"    Mean LAD: {avg_lad:.4f}")
            print(f"    Std LAD:  {std_lad:.4f}")
            return avg_lad, lads
    
    return None, []

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/compute_lad.py <summary.json>")
        sys.exit(1)
    
    avg, lads = compute_lad(sys.argv[1])
    if avg is not None:
        print(f"\n[RESULT] LAD = {avg:.4f}")
