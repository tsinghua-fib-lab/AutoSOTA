#!/usr/bin/env python3
"""Batch SCC evaluation with many seeds."""
import subprocess, os, sys

DATASET_CC = "/datasets/bitcoinotc_dedup.cc"
DATASET_GRAPH = "/datasets/bitcoinotc.graph"
SCC_BIN = "/autosota_cache/ScalableCorrelationClustering/build/scc_int"
MWU_LOWER = 672

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

best_disagreements = float("inf")
best_info = ""

env = os.environ.copy()
env["OMPI_MCA_mca_base_component_show_load_errors"] = "0"

for seed in range(1, 201):
    part_file = f"/tmp/scc_batch_{seed}.txt"
    try:
        result = subprocess.run(
            [SCC_BIN, DATASET_GRAPH,
             f"--seed={seed}",
             f"--time_limit=3",
             f"--output_filename={part_file}"],
            capture_output=True, text=True, timeout=30, env=env
        )
        d = count_disagreements(part_file)
        if d < best_disagreements:
            best_disagreements = d
            best_info = f"seed={seed}"
            print(f"NEW BEST: seed={seed}, disagreements={d}, ratio={d/MWU_LOWER:.4f}")
        os.unlink(part_file)
    except Exception as e:
        pass

print(f"\nFINAL BEST: {best_info}, disagreements={best_disagreements}, ratio={best_disagreements/MWU_LOWER:.4f}")
print(f"MWU lower bound: {MWU_LOWER}")
