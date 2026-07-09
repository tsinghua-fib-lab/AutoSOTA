#!/usr/bin/env python3
"""Reproduction script for paper 2462: FIEDLER on ca-GrQc, k=10."""
import sys
import os
import time
import resource

sys.path.insert(0, "/repo")

import helper_functions
import sparsifier
import treeDP

GRAPH_PATH = "/datasets/ca-GrQc.txt"
K = 10
BISECT_ALGO = "spectral"  # FIEDLER

print(f"Loading graph from {GRAPH_PATH}...", flush=True)
G = helper_functions.read_graph(GRAPH_PATH)
n = G.number_of_nodes()
m = G.number_of_edges()
print(f"Graph loaded: {n} nodes, {m} edges", flush=True)

# Measure sparsifier construction
print(f"Building sparsifier with {BISECT_ALGO} (FIEDLER)...", flush=True)
t0_sp = time.perf_counter()
s = sparsifier.Sparsifier(G, alg=BISECT_ALGO)
t1_sp = time.perf_counter()
sp_time = t1_sp - t0_sp
print(f"Sparsifier built in {sp_time:.2f}s (wall), tree has {len(s.tree)} vertices", flush=True)

# Measure DP
print(f"Running tree DP for k={K}...", flush=True)
t0_dp = time.perf_counter()
L = treeDP.solveGivenK(s.tree, K, root=s.root)
Lmapped = [res for key, res in s.mapping.items() if key in L]
t1_dp = time.perf_counter()
dp_time = t1_dp - t0_dp
print(f"DP solved in {dp_time:.2f}s (wall), selected {len(Lmapped)} vertices", flush=True)

# Cut verification
t0_cut = time.perf_counter()
tau, cut = helper_functions.cut_set(G, Lmapped)
t1_cut = time.perf_counter()
cut_time = t1_cut - t0_cut
print(f"Cut verification in {cut_time:.2f}s, tau={tau:.6f}", flush=True)

# Total wall time
total_wall = t1_cut - t0_sp

# Resource usage (covers this process and all children)
usage_self = resource.getrusage(resource.RUSAGE_SELF)
usage_children = resource.getrusage(resource.RUSAGE_CHILDREN)
user_time = usage_self.ru_utime + usage_children.ru_utime
sys_time = usage_self.ru_stime + usage_children.ru_stime

print(f"\n===== REPRODUCTION RESULTS =====", flush=True)
print(f"Algorithm: FIEDLER (spectral)", flush=True)
print(f"Dataset: ca-GrQc", flush=True)
print(f"Budget k: {K}", flush=True)
print(f"Real time (wall-clock total): {total_wall:.1f}s", flush=True)
print(f"User time (CPU user mode):   {user_time:.1f}s", flush=True)
print(f"System time (CPU kernel):     {sys_time:.1f}s", flush=True)
print(f"  - Sparsifier: {sp_time:.1f}s wall", flush=True)
print(f"  - DP:         {dp_time:.1f}s wall", flush=True)
print(f"  - Cut verify: {cut_time:.1f}s wall", flush=True)
print(f"Quality (tau):  {tau:.6f}", flush=True)
print(f"Selected:       {len(Lmapped)}/{n} vertices", flush=True)

# Save metrics
import json
metrics = {
    "real_time": round(total_wall, 1),
    "user_time": round(user_time, 1),
    "sys_time": round(sys_time, 1),
    "sparsifier_time": round(sp_time, 1),
    "dp_time": round(dp_time, 1),
    "cut_verify_time": round(cut_time, 1),
    "quality_tau": tau,
    "selected_vertices": len(Lmapped),
    "graph": "ca-GrQc",
    "graph_nodes": n,
    "graph_edges": m,
    "k": K,
    "algorithm": "FIEDLER",
}
os.makedirs("/repo/outputs", exist_ok=True)
with open("/repo/outputs/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to /repo/outputs/metrics.json", flush=True)
