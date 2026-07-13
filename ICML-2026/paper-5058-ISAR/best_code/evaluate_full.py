#!/usr/bin/env python3
"""Full evaluation for ISAR paper reproduction on bitcoinOTC."""

import subprocess, os, sys, json, glob

DATASET_CC = "/datasets/bitcoinotc_dedup.cc"
DATASET_GRAPH = "/datasets/bitcoinotc.graph"
ISAR_BIN = "/repo/build/ISAR"
SCC_BIN = "/autosota_cache/ScalableCorrelationClustering/build/scc_int"
SCC_EVO_BIN = "/autosota_cache/ScalableCorrelationClustering/build/scc_evolutionary_int"

def run_isar():
    """Run ISAR to get MWU lower bound."""
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

def run_scc_multilevel(time_limit, seed):
    """Run multilevel SCC solver."""
    part_file = f"/tmp/scc_partition_ml_{seed}.txt"
    result = subprocess.run(
        [SCC_BIN, DATASET_GRAPH,
         f"--seed={seed}",
         f"--time_limit={time_limit}",
         f"--output_filename={part_file}"],
        capture_output=True, text=True, timeout=time_limit + 120
    )
    return part_file

def run_scc_evolutionary(time_limit, seed):
    """Run evolutionary SCC solver via MPI."""
    part_file = f"/tmp/scc_partition_evo_{seed}.txt"
    result = subprocess.run(
        ["mpirun", "--oversubscribe", "-n", "1",
         SCC_EVO_BIN, DATASET_GRAPH,
         f"--seed={seed}",
         f"--time_limit={time_limit}",
         f"--output_filename={part_file}"],
        capture_output=True, text=True, timeout=time_limit + 120
    )
    return part_file

def main():
    mwu_lower = run_isar()
    if mwu_lower is None:
        print("ERROR: Could not parse MWU lower bound")
        sys.exit(1)
    print(f"MWU lower bound (eps=0.05): {mwu_lower}")

    # Try multilevel SCC with longer time limits and many seeds
    best_disagreements = float("inf")
    best_info = ""

    seeds = [42, 123, 456, 789, 1313, 2020, 3333, 23, 5, 777, 999, 111, 222, 444, 888]
    time_limits = [120]  # 2 minutes per run

    print(f"Running multilevel SCC with {len(seeds)} seeds, time_limits={time_limits}...")
    for tl in time_limits:
        for seed in seeds:
            part_file = run_scc_multilevel(tl, seed)
            d = count_disagreements(part_file)
            print(f"  seed={seed}, time_limit={tl}s: disagreements={d}")
            if d < best_disagreements:
                best_disagreements = d
                best_info = f"seed={seed}, time_limit={tl}s"
            os.unlink(part_file)

    # Try evolutionary SCC
    print("Running evolutionary SCC...")
    for seed in [42, 123, 456, 789]:
        try:
            part_file = run_scc_evolutionary(120, seed)
            d = count_disagreements(part_file)
            print(f"  evo seed={seed}: disagreements={d}")
            if d < best_disagreements:
                best_disagreements = d
                best_info = f"evo seed={seed}, time_limit=120s"
            os.unlink(part_file)
        except Exception as e:
            print(f"  evo seed={seed}: FAILED - {e}")

    ratio = best_disagreements / mwu_lower
    print(f"\n=============== FINAL RESULTS ===============")
    print(f"Problem: Correlation Clustering Disagreement")
    print(f"MWU lower bound (eps=0.05):  {mwu_lower}")
    print(f"Best SCC upper bound:         {best_disagreements} ({best_info})")
    print(f"Instance-Specific Approximation Ratio: {ratio:.4f}")
    print(json.dumps({
        "ISAR_lower_bound": mwu_lower,
        "SCC_disagreements": best_disagreements,
        "approximation_ratio": round(ratio, 4),
        "best_info": best_info
    }))

if __name__ == "__main__":
    main()
