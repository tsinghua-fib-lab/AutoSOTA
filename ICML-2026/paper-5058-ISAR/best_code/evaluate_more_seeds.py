#!/usr/bin/env python3
"""IDEA-08: More SCC seeds (18) and longer time limit (60s) per seed.
Based on evaluate.py baseline with expanded search budget.
"""

import subprocess, os, sys, json

DATASET_CC = "/datasets/bitcoinotc_dedup.cc"
DATASET_GRAPH = "/datasets/bitcoinotc_from_cc.graph"
ISAR_BIN = "/repo/build/ISAR"
SCC_EVO_BIN = "/autosota_cache/ScalableCorrelationClustering/build/scc_evolutionary_int"

def run_isar():
    result = subprocess.run(
        [ISAR_BIN, DATASET_CC, "-p", "CC"],
        capture_output=True, text=True, timeout=300
    )
    in_disagreement = False
    mwu_lower = None
    for line in result.stdout.split("\n"):
        if "DISAGREEMENT" in line:
            in_disagreement = True
            continue
        if "AGREEMENT" in line:
            in_disagreement = False
            continue
        if in_disagreement and "CERT: MWU_eps=0.05" in line and "Single" not in line:
            parts = line.strip().split()
            mwu_lower = int(parts[-1])
            break
    return mwu_lower

def run_scc_evolutionary(time_limit, seed):
    part_file = f"/tmp/scc_evo_{seed}.txt"
    env = os.environ.copy()
    env["OMPI_MCA_mca_base_component_show_load_errors"] = "0"
    subprocess.run(
        [SCC_EVO_BIN, DATASET_GRAPH,
         f"--seed={seed}",
         f"--time_limit={time_limit}",
         f"--output_filename={part_file}"],
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
    mwu_lower = run_isar()
    if mwu_lower is None:
        print("ERROR: Could not parse MWU lower bound")
        sys.exit(1)

    best_disagreements = float("inf")
    best_seed = -1
    seeds = [42, 123, 456, 789, 1313, 2020, 3333, 23, 5,
             777, 1111, 2222, 4444, 5555, 6666, 7777, 8888, 9999]
    time_limit = 60

    for seed in seeds:
        part_file = run_scc_evolutionary(time_limit=time_limit, seed=seed)
        n_clusters, disagreements = count_disagreements(part_file)
        ratio = disagreements / mwu_lower
        print(f"Seed {seed}: {n_clusters} clusters, {disagreements} disagreements, ratio={ratio:.4f}")
        if disagreements < best_disagreements:
            best_disagreements = disagreements
            best_seed = seed

    ratio = best_disagreements / mwu_lower
    output = {
        "problem": "correlation_clustering_disagreement",
        "dataset": "bitcoinOTC (deduplicated)",
        "nodes": 5881,
        "edges": 21492,
        "MWU_lower_bound_eps_0.05": mwu_lower,
        "SCC_upper_bound": best_disagreements,
        "approximation_ratio": round(ratio, 4),
        "best_seed": best_seed,
        "cc_solver": "SCMLEvo (scc_evolutionary_int)",
        "num_seeds": len(seeds),
        "time_limit_per_seed": time_limit,
        "idea": "IDEA-08: more seeds + longer time limit",
    }

    print(f"\n=============== FINAL RESULTS ===============")
    print(f"Idea: IDEA-08 — More seeds ({len(seeds)}) + longer time limit ({time_limit}s)")
    print(f"Problem: Correlation Clustering Disagreement")
    print(f"MWU lower bound (eps=0.05):   {mwu_lower}")
    print(f"Best SCC upper bound:          {best_disagreements} (seed={best_seed})")
    print(f"Instance-Specific Approximation Ratio: {ratio:.4f}")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
