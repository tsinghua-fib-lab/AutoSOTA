#!/usr/bin/env python3
"""Corrected evaluation for ISAR paper reproduction on bitcoinOTC.
Uses .graph file generated from .cc file with correct signs."""

import subprocess, os, sys, json

DATASET_CC = "/datasets/bitcoinotc_dedup.cc"
DATASET_GRAPH = "/datasets/bitcoinotc_from_cc.graph"
ISAR_BIN = "/repo/build/ISAR"
SCC_BIN = "/autosota_cache/ScalableCorrelationClustering/build/scc_int"
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

def run_scc(binary, time_limit, seed, label):
    part_file = f"/tmp/scc_{label}_{seed}.txt"
    env = os.environ.copy()
    env["OMPI_MCA_mca_base_component_show_load_errors"] = "0"
    try:
        subprocess.run(
            [binary, DATASET_GRAPH,
             f"--seed={seed}",
             f"--time_limit={time_limit}",
             f"--output_filename={part_file}"],
            capture_output=True, text=True, timeout=time_limit + 120, env=env
        )
        return count_disagreements(part_file)
    except Exception as e:
        print(f"  {label} seed={seed}: FAILED - {e}")
        return None

def main():
    mwu_lower = run_isar()
    print(f"MWU lower bound (eps=0.05): {mwu_lower}")

    best_disagreements = float("inf")
    best_info = ""

    # Test with multilevel SCC
    print("\nMultilevel SCC (scc_int):")
    for seed in [42, 123, 456, 789, 1313, 2020, 3333, 23, 5]:
        result = run_scc(SCC_BIN, 5, seed, f"ml_s{seed}")
        if result:
            n_clust, d = result
            print(f"  seed={seed}: {n_clust} clusters, {d} disagreements, ratio={d/mwu_lower:.4f}")
            if d < best_disagreements:
                best_disagreements = d
                best_info = f"ml seed={seed}"

    # Test with evolutionary SCC
    print("\nEvolutionary SCC (scc_evolutionary_int):")
    for seed in [42, 123, 456, 789]:
        result = run_scc(SCC_EVO_BIN, 30, seed, f"evo_s{seed}")
        if result:
            n_clust, d = result
            print(f"  seed={seed}: {n_clust} clusters, {d} disagreements, ratio={d/mwu_lower:.4f}")
            if d < best_disagreements:
                best_disagreements = d
                best_info = f"evo seed={seed}"

    ratio = best_disagreements / mwu_lower
    print(f"\n=============== FINAL RESULTS ===============")
    print(f"Problem: Correlation Clustering Disagreement")
    print(f"Dataset: bitcoinOTC (deduplicated, 5881 nodes, 21492 edges)")
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
