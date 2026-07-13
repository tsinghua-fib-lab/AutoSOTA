#!/usr/bin/env python3
"""Fast evaluation: tighter MWU with existing SCC results."""

import subprocess, os, sys, json, glob

DATASET_CC = "/datasets/bitcoinotc_dedup.cc"
DATASET_GRAPH = "/datasets/bitcoinotc_from_cc.graph"
ISAR_BIN = "/repo/build/ISAR"

def run_isar_all_eps():
    result = subprocess.run(
        [ISAR_BIN, DATASET_CC, "-p", "CC"],
        capture_output=True, text=True, timeout=600
    )
    in_disagreement = False
    mwu_bounds = {}
    for line in result.stdout.split("\n"):
        if "DISAGREEMENT" in line:
            in_disagreement = True
            continue
        if "AGREEMENT" in line:
            in_disagreement = False
            continue
        if in_disagreement and "CERT: MWU_eps=" in line and "Single" not in line:
            parts = line.strip().split()
            eps_str = parts[1].replace("MWU_eps=", "")
            bound = int(parts[-1])
            mwu_bounds[eps_str] = bound
    return mwu_bounds

def count_disagreements(part_file):
    with open(part_file) as f:
        clusters = [int(line.strip()) for line in f]
    disagreements = 0
    with open(DATASET_CC) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 3: continue
            u, v, s = int(p[0]), int(p[1]), int(p[2])
            same = (clusters[u-1] == clusters[v-1])
            if (same and s == -1) or (not same and s == 1):
                disagreements += 1
    return disagreements

def main():
    print("=== IDEA-07: Tighter MWU Lower Bound (Fast) ===\n")
    
    # Run ISAR
    print("Running ISAR...")
    mwu_bounds = run_isar_all_eps()
    if not mwu_bounds:
        print("ERROR: Could not parse MWU bounds")
        sys.exit(1)
    
    for eps in sorted(mwu_bounds.keys(), key=float):
        print("  MWU eps={}: {}".format(eps, mwu_bounds[eps]))
    
    best_eps = max(mwu_bounds.keys(), key=lambda e: mwu_bounds[e])
    mwu_best = mwu_bounds[best_eps]
    mwu_005 = mwu_bounds.get("0.05", mwu_best)
    
    # Find best SCC result from existing partitions
    part_files = glob.glob("/tmp/scc_evo_*.txt")
    best_d = float("inf")
    best_file = None
    for pf in part_files:
        d = count_disagreements(pf)
        if d < best_d:
            best_d = d
            best_file = pf
    
    if best_file is None:
        print("No partition files found!")
        sys.exit(1)
    
    ratio_new = best_d / mwu_best
    ratio_old = best_d / mwu_005
    
    output = {
        "MWU_lower_bound_eps_0_05": mwu_005,
        "MWU_lower_bound_best": mwu_best,
        "MWU_best_epsilon": float(best_eps),
        "SCC_upper_bound": best_d,
        "approximation_ratio_eps_0_05": round(ratio_old, 4),
        "approximation_ratio": round(ratio_new, 4),
        "improvement_pct": round((ratio_old - ratio_new) / ratio_old * 100, 2),
        "idea": "IDEA-07: tighter MWU lower bound via smaller epsilon"
    }
    
    print("\n=============== FINAL RESULTS ===============")
    print("MWU lower bound (eps=0.05): {}".format(mwu_005))
    print("MWU lower bound (eps={}):   {}".format(best_eps, mwu_best))
    print("SCC upper bound:                 {}".format(best_d))
    print("Ratio (eps=0.05):                {:.4f}".format(ratio_old))
    print("Ratio (eps={}):                {:.4f}".format(best_eps, ratio_new))
    print("Improvement:                     {:.2f}%".format((ratio_old - ratio_new) / ratio_old * 100))
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
