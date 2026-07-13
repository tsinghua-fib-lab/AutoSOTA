#!/usr/bin/env python3
"""Apply local search to existing SCC partition files and report best result."""

import subprocess, os, sys, json, random, glob
from collections import defaultdict

DATASET_CC = "/datasets/bitcoinotc_dedup.cc"
ISAR_BIN = "/repo/build/ISAR"

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

def load_graph():
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
    disagreements = 0
    for u, v, s in edges:
        same = (clusters[u] == clusters[v])
        if (same and s == -1) or (not same and s == 1):
            disagreements += 1
    return disagreements

def read_partition(part_file):
    with open(part_file) as f:
        return [int(line.strip()) for line in f]

def build_node_edge_index(edges, n_nodes):
    adj = [[] for _ in range(n_nodes)]
    for idx, (u, v, s) in enumerate(edges):
        adj[u].append((v, s))
        adj[v].append((u, s))
    return adj

def local_search(clusters, adj, edges, n_nodes, max_passes=20):
    current = list(clusters)
    current_disagreements = count_disagreements_from_clusters(current, edges)
    cluster_ids = list(set(current))
    improved = True
    n_passes = 0
    total_moves = 0
    while improved and n_passes < max_passes:
        improved = False
        n_passes += 1
        node_order = list(range(n_nodes))
        random.shuffle(node_order)
        for u in node_order:
            old_cluster = current[u]
            current_u_d = 0
            for v, s in adj[u]:
                same = (old_cluster == current[v])
                if (same and s == -1) or (not same and s == 1):
                    current_u_d += 1
            best_move_cluster = None
            best_delta = 0
            for c in cluster_ids:
                if c == old_cluster:
                    continue
                new_u_d = 0
                for v, s in adj[u]:
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
                if old_cluster not in current:
                    cluster_ids.remove(old_cluster)
        if improved:
            print("  LS pass {}: {} moves, disagreements={}".format(n_passes, total_moves, current_disagreements))
    return current, current_disagreements

def main():
    print("=== IDEA-01: Local Search on Existing SCC Partitions ===\n")
    n_nodes, edges = load_graph()
    adj = build_node_edge_index(edges, n_nodes)
    print("Graph: {} nodes, {} edges\n".format(n_nodes, len(edges)))
    
    print("Running ISAR...")
    mwu_lower = run_isar()
    if mwu_lower is None:
        print("ERROR: Could not parse MWU lower bound")
        sys.exit(1)
    print("MWU lower bound: {}\n".format(mwu_lower))
    
    # Find all SCC partition files
    part_files = glob.glob("/tmp/scc_evo_*.txt")
    if not part_files:
        print("No SCC partition files found!")
        sys.exit(1)
    
    print("Evaluating {} SCC partition files...".format(len(part_files)))
    best_disagreements = float("inf")
    best_file = None
    results = []
    
    for pf in sorted(part_files):
        clusters = read_partition(pf)
        disagreements = count_disagreements_from_clusters(clusters, edges)
        seed_str = os.path.basename(pf).replace("scc_evo_", "").replace(".txt", "")
        ratio = disagreements / mwu_lower
        results.append((pf, seed_str, disagreements, ratio, len(set(clusters))))
        if disagreements < best_disagreements:
            best_disagreements = disagreements
            best_file = pf
    
    # Sort by disagreements
    results.sort(key=lambda x: x[2])
    
    print("Top 5 SCC results before local search:")
    for pf, seed, d, r, k in results[:5]:
        print("  seed={}: {} clusters, {} disagreements, ratio={:.4f}".format(seed, k, d, r))
    
    best_clusters = read_partition(best_file)
    best_seed = os.path.basename(best_file).replace("scc_evo_", "").replace(".txt", "")
    ratio_before = best_disagreements / mwu_lower
    
    print("\nBest: seed={} with {} disagreements, ratio={:.4f}".format(best_seed, best_disagreements, ratio_before))
    print("\nApplying greedy local search...")
    
    improved_clusters, improved_disagreements = local_search(best_clusters, adj, edges, n_nodes, max_passes=20)
    
    ratio_after = improved_disagreements / mwu_lower
    improvement = best_disagreements - improved_disagreements
    pct_improvement = (improvement / best_disagreements) * 100 if best_disagreements > 0 else 0
    
    print("\n=============== FINAL RESULTS ===============")
    print("Before local search: {} disagreements, ratio={:.4f}".format(best_disagreements, ratio_before))
    print("After local search:  {} disagreements, ratio={:.4f}".format(improved_disagreements, ratio_after))
    print("Improvement:         -{} disagreements ({:.2f}%)".format(improvement, pct_improvement))
    print("Ratio reduction:     {:.4f} -> {:.4f}".format(ratio_before, ratio_after))
    
    output = {
        "MWU_lower_bound": mwu_lower,
        "SCC_upper_bound_before_ls": best_disagreements,
        "SCC_upper_bound_after_ls": improved_disagreements,
        "approximation_ratio_before_ls": round(ratio_before, 4),
        "approximation_ratio": round(ratio_after, 4),
        "improvement_pct": round(pct_improvement, 2),
        "best_scc_seed": best_seed,
        "num_scc_partitions_evaluated": len(part_files),
        "idea": "IDEA-01: greedy local search post-processing"
    }
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
