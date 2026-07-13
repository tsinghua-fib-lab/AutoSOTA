#!/usr/bin/env python3
"""IDEA-07: Tighter MWU lower bound via smaller epsilon (0.025, 0.01).
Uses modified ISAR binary that computes MWU at multiple epsilon values.
Takes the best (highest) valid MWU lower bound to minimize the ratio.
"""

import subprocess, os, sys, json

DATASET_CC = "/datasets/bitcoinotc_dedup.cc"
DATASET_GRAPH = "/datasets/bitcoinotc_from_cc.graph"
ISAR_BIN = "/repo/build/ISAR"
SCC_EVO_BIN = "/autosota_cache/ScalableCorrelationClustering/build/scc_evolutionary_int"

def run_isar_all_eps():
    """Run ISAR and parse all MWU lower bounds at every epsilon."""
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

def run_scc_evolutionary(time_limit, seed):
    part_file = "/tmp/scc_evo_{}.txt".format(seed)
    env = os.environ.copy()
    env["OMPI_MCA_mca_base_component_show_load_errors"] = "0"
    subprocess.run(
        [SCC_EVO_BIN, DATASET_GRAPH,
         "--seed={}".format(seed),
         "--time_limit={}".format(time_limit),
         "--output_filename={}".format(part_file)],
        capture_output=True, text=True, timeout=time_limit + 120, env=env
    )
    return part_file

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
    return len(set(clusters)), disagreements

def main():
    print("=== IDEA-07: Tighter MWU Lower Bound ===\n")
    
    # Run ISAR for all MWU lower bounds
    print("Running ISAR (MWU at multiple epsilon values)...")
    mwu_bounds = run_isar_all_eps()
    if not mwu_bounds:
        print("ERROR: Could not parse any MWU lower bounds")
        sys.exit(1)
    
    print("MWU lower bounds by epsilon:")
    for eps in sorted(mwu_bounds.keys(), key=float):
        print("  eps={}: {}".format(eps, mwu_bounds[eps]))
    
    # Use the best (highest) lower bound
    best_eps = max(mwu_bounds.keys(), key=lambda e: mwu_bounds[e])
    mwu_lower = mwu_bounds[best_eps]
    mwu_005 = mwu_bounds.get("0.05", mwu_lower)
    
    print("\nBest MWU bound: {} at eps={}".format(mwu_lower, best_eps))
    print("Baseline MWU bound (eps=0.05): {}".format(mwu_005))
    
    # Run SCC with multiple seeds
    best_disagreements = float("inf")
    best_seed = -1
    seeds = [42, 123, 456, 789, 1313, 2020, 3333, 23, 5,
             777, 1111, 2222, 4444, 5555, 6666, 7777, 8888, 9999]
    time_limit = 60
    
    print("\nRunning SCC with {} seeds, {}s time limit...".format(len(seeds), time_limit))
    for seed in seeds:
        part_file = run_scc_evolutionary(time_limit=time_limit, seed=seed)
        n_clusters, disagreements = count_disagreements(part_file)
        ratio = disagreements / mwu_lower
        ratio_005 = disagreements / mwu_005
        print("  Seed {}: {} clusters, {} disagreements, ratio(eps=0.05)={:.4f}, ratio(eps={})={:.4f}".format(
            seed, n_clusters, disagreements, ratio_005, best_eps, ratio))
        if disagreements < best_disagreements:
            best_disagreements = disagreements
            best_seed = seed
    
    ratio_best = best_disagreements / mwu_lower
    ratio_baseline = best_disagreements / mwu_005
    
    output = {
        "problem": "correlation_clustering_disagreement",
        "dataset": "bitcoinOTC (deduplicated)",
        "nodes": 5881,
        "edges": 21492,
        "MWU_lower_bound_eps_0.05": mwu_005,
        "MWU_lower_bound_best": mwu_lower,
        "MWU_best_epsilon": float(best_eps),
        "SCC_upper_bound": best_disagreements,
        "approximation_ratio_baseline_eps005": round(ratio_baseline, 4),
        "approximation_ratio": round(ratio_best, 4),
        "best_seed": best_seed,
        "idea": "IDEA-07: tighter MWU lower bound via smaller epsilon",
    }
    
    print("\n=============== FINAL RESULTS ===============")
    print("Idea: IDEA-07 — Tighter MWU lower bound via smaller epsilon")
    print("MWU lower bound (eps=0.05, baseline): {}".format(mwu_005))
    print("MWU lower bound (eps={}, best):     {}".format(best_eps, mwu_lower))
    print("Best SCC upper bound:                 {} (seed={})".format(best_disagreements, best_seed))
    print("Ratio (baseline eps=0.05):            {:.4f}".format(ratio_baseline))
    print("Ratio (tight eps={}):               {:.4f}".format(best_eps, ratio_best))
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
