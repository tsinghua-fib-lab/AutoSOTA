#!/usr/bin/env python3
"""IDEA-01: Greedy local search post-processing for SCC clusterings.
After SCC produces a clustering, iteratively moves single nodes to other clusters
if doing so reduces total disagreements. Monotonic descent only (no worsening moves).
"""

import subprocess, os, sys, json, copy, random
from collections import defaultdict

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

def run_scc_evolutionary(time_limit, seed, input_partition=None):
    part_file = f"/tmp/scc_evo_{seed}.txt"
    env = os.environ.copy()
    env["OMPI_MCA_mca_base_component_show_load_errors"] = "0"
    cmd = [SCC_EVO_BIN, DATASET_GRAPH,
         f"--seed={seed}",
         f"--time_limit={time_limit}",
         f"--output_filename={part_file}"]
    if input_partition:
        cmd.append(f"--input_partition={input_partition}")
    subprocess.run(
        cmd,
        capture_output=True, text=True, timeout=time_limit + 120, env=env
    )
    return part_file

def load_graph():
    """Load .cc file as adjacency lists with edge signs: (neighbor, sign)."""
    edges = []
    n_nodes = 0
    with open(DATASET_CC) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 3: continue
            u, v, s = int(p[0]), int(p[1]), int(p[2])
            edges.append((u-1, v-1, s))
            n_nodes = max(n_nodes, u, v)
    return n_nodes, edges

def count_disagreements_from_clusters(clusters, edges):
    """Count disagreements given a cluster assignment list and edges list."""
    disagreements = 0
    for u, v, s in edges:
        same = (clusters[u] == clusters[v])
        if (same and s == -1) or (not same and s == 1):
            disagreements += 1
    return disagreements

def count_disagreements(part_file, edges=None):
    with open(part_file) as f:
        clusters = [int(line.strip()) for line in f]
    if edges:
        n_clusters = len(set(clusters))
        disagreements = count_disagreements_from_clusters(clusters, edges)
        return n_clusters, disagreements
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

def build_node_edge_index(edges, n_nodes):
    """Build adjacency index: node -> [(neighbor, sign, edge_idx)]"""
    adj = [[] for _ in range(n_nodes)]
    for idx, (u, v, s) in enumerate(edges):
        adj[u].append((v, s, idx))
        adj[v].append((u, s, idx))
    return adj

def compute_node_disagreements(clusters, adj):
    """Compute per-node disagreement counts."""
    n = len(clusters)
    node_d = [0] * n
    for u in range(n):
        for v, s, _ in adj[u]:
            same = (clusters[u] == clusters[v])
            if (same and s == -1) or (not same and s == 1):
                node_d[u] += 1
    return node_d

def local_search(clusters, adj, edges, n_nodes, max_passes=20):
    """
    Greedy local search: try moving each node to each other cluster.
    Accept the best improving move. Repeat until no improvement.
    Returns improved clusters and total disagreements.
    """
    current = list(clusters)
    current_disagreements = count_disagreements_from_clusters(current, edges)
    
    # Get unique cluster IDs
    cluster_ids = list(set(current))
    
    improved = True
    n_passes = 0
    total_moves = 0
    
    while improved and n_passes < max_passes:
        improved = False
        n_passes += 1
        
        # Random node order for better exploration
        node_order = list(range(n_nodes))
        random.shuffle(node_order)
        
        for u in node_order:
            best_move_cluster = None
            best_delta = 0
            
            # Compute current disagreement contribution of node u
            old_cluster = current[u]
            current_u_d = 0
            for v, s, _ in adj[u]:
                same = (old_cluster == current[v])
                if (same and s == -1) or (not same and s == 1):
                    current_u_d += 1
            
            # Try moving to each other cluster
            for c in cluster_ids:
                if c == old_cluster:
                    continue
                # Compute new disagreement contribution
                new_u_d = 0
                for v, s, _ in adj[u]:
                    same = (c == current[v])
                    if (same and s == -1) or (not same and s == 1):
                        new_u_d += 1
                
                delta = new_u_d - current_u_d
                if delta < best_delta:
                    best_delta = delta
                    best_move_cluster = c
            
            if best_delta < 0:
                current[u] = best_move_cluster
                current_disagreements += best_delta
                total_moves += 1
                improved = True
                
                # Update cluster_ids if we emptied a cluster
                if old_cluster not in current:
                    cluster_ids.remove(old_cluster)
        
        if improved:
            print(f"  Local search pass {n_passes}: {total_moves} total moves, disagreements={current_disagreements}")
    
    return current, current_disagreements

def main():
    print("=== IDEA-01: Greedy Local Search Post-Processing ===\n")
    
    # Load graph once
    print("Loading graph...")
    n_nodes, edges = load_graph()
    adj = build_node_edge_index(edges, n_nodes)
    print(f"Graph: {n_nodes} nodes, {len(edges)} edges\n")
    
    # Run ISAR for MWU lower bound
    print("Running ISAR (MWU lower bound)...")
    mwu_lower = run_isar()
    if mwu_lower is None:
        print("ERROR: Could not parse MWU lower bound")
        sys.exit(1)
    print(f"MWU lower bound: {mwu_lower}\n")
    
    # Run SCC with baseline seeds
    best_disagreements = float("inf")
    best_seed = -1
    best_clusters = None
    seeds = [42, 123, 456, 789, 1313, 2020, 3333, 23, 5,
             777, 1111, 2222, 4444, 5555, 6666, 7777, 8888, 9999]
    time_limit = 60
    
    print(f"Running SCC with {len(seeds)} seeds, {time_limit}s time limit...")
    for seed in seeds:
        part_file = run_scc_evolutionary(time_limit=time_limit, seed=seed)
        n_clusters, disagreements = count_disagreements(part_file, edges=edges)
        ratio = disagreements / mwu_lower
        print(f"  Seed {seed}: {n_clusters} clusters, {disagreements} disagreements, ratio={ratio:.4f}")
        
        if disagreements < best_disagreements:
            best_disagreements = disagreements
            best_seed = seed
            # Read best clusters
            with open(part_file) as f:
                best_clusters = [int(line.strip()) for line in f]
    
    ratio_before = best_disagreements / mwu_lower
    print(f"\nBefore local search: {best_disagreements} disagreements, ratio={ratio_before:.4f} (seed={best_seed})")
    
    # Apply local search to best clustering
    print(f"\nApplying greedy local search to best clustering (seed={best_seed})...")
    improved_clusters, improved_disagreements = local_search(
        best_clusters, adj, edges, n_nodes, max_passes=20
    )
    
    ratio_after = improved_disagreements / mwu_lower
    improvement = best_disagreements - improved_disagreements
    pct_improvement = (improvement / best_disagreements) * 100
    
    print(f"\nAfter local search:  {improved_disagreements} disagreements, ratio={ratio_after:.4f}")
    print(f"Improvement:         -{improvement} disagreements ({pct_improvement:.2f}%)")
    print(f"Ratio reduction:     {ratio_before:.4f} -> {ratio_after:.4f}")
    
    output = {
        "problem": "correlation_clustering_disagreement",
        "dataset": "bitcoinOTC (deduplicated)",
        "nodes": n_nodes,
        "edges": len(edges),
        "MWU_lower_bound_eps_0.05": mwu_lower,
        "SCC_upper_bound_before_ls": best_disagreements,
        "SCC_upper_bound_after_ls": improved_disagreements,
        "approximation_ratio": round(ratio_after, 4),
        "ratio_before_ls": round(ratio_before, 4),
        "improvement_pct": round(pct_improvement, 2),
        "best_seed": best_seed,
        "cc_solver": "SCMLEvo + GreedyLocalSearch",
        "idea": "IDEA-01: greedy local search post-processing",
    }
    
    print(f"\n=============== FINAL RESULTS ===============")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
