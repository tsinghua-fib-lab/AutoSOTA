#!/usr/bin/env python3
"""
Reproduction evaluation script for FairRARI on PolBooks.
Computes TV (Total Variation) and Kendall Tau metrics
for phi-sum-fairness with 2 groups.
"""

import sys
import argparse
import numpy as np
import networkx as nx
from scipy.stats import kendalltau
import torch

# Import project modules
from init_graph import init_graph
import fairPageRank


def compute_tv(orig_scores, fair_scores):
    """Total Variation distance: TV(po, pF) = 0.5 * ||po - pF||_1"""
    return 0.5 * np.sum(np.abs(np.array(orig_scores) - np.array(fair_scores)))


def compute_kendall_tau(orig_scores, fair_scores):
    """Kendall Tau rank correlation between original and fair PageRank rankings."""
    # Higher scores = higher rank (rank 1 = highest score)
    # kendalltau uses the order of elements, so we pass the score vectors directly
    tau, p_value = kendalltau(orig_scores, fair_scores)
    return tau


def main():
    parser = argparse.ArgumentParser(description='FairRARI Reproduction Evaluation')
    parser.add_argument('--dataset-name', type=str, default='polbooks', help='Dataset name')
    parser.add_argument('--phi', type=float, default=0.8, help='Target fairness level')
    parser.add_argument('--gamma', type=float, default=0.15, help='Teleportation probability')
    parser.add_argument('--max-iters', type=int, default=10000, help='Maximum iterations')
    parser.add_argument('--edge-reweight', type=float, default=None, help='Edge reweight factor (>1 up-weights within-group edges)')
    parser.add_argument('--source-path', type=str, default='datasets/', help='Path to datasets')

    args = parser.parse_args()

    dataset_name = args.dataset_name
    phi = args.phi
    gamma = args.gamma
    max_iters = args.max_iters
    source_path = args.source_path
    edge_reweight = args.edge_reweight

    alpha_val = 1.0 - gamma  # alpha = 1 - gamma

    print("=" * 60)
    print("FairRARI Reproduction: phi-sum-fairness")
    print("=" * 60)
    print(f"Dataset:       {dataset_name}")
    print(f"phi:           {phi}")
    print(f"gamma:         {gamma}")
    print(f"alpha:         {alpha_val}")
    print(f"max_iters:     {max_iters}")
    print(f"source_path:   {source_path}")
    print()

    # Load graph
    G, protected_nodes, blue_nodes, red_nodes = init_graph(dataset_name, source_path)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    print(f"Graph loaded: {n} vertices, {m} edges")
    print(f"Protected nodes: {len(protected_nodes)}")
    print(f"Graph type: {'directed' if G.is_directed() else 'undirected'}")

    # Create group indicator vectors
    S_p = torch.zeros(n).int()
    S_p[protected_nodes] = 1
    S_up = torch.ones(n).int()
    S_up[protected_nodes] = 0

    # Compute original PageRank
    print("\nComputing original PageRank...")
    opr = nx.pagerank(G)
    opr_scores = torch.FloatTensor(list(opr.values()))

    # Original phi (fairness level of original PR)
    opr_S_p = opr_scores[S_p == 1]
    phi_opr = torch.sum(opr_S_p).item()
    print(f"Original PR phi (mass on protected group): {phi_opr:.6f}")

    # If phi == 0, use original phi
    if phi == 0.0:
        phi = phi_opr
        print(f"Using original phi: {phi:.6f}")

    # Run FairRARI
    print(f"\nRunning FairRARI with phi={phi:.4f}, alpha={alpha_val:.4f}, max_iter={max_iters}...")
    fair_nx_pr, nx_x_diff, nx_loss = fairPageRank.sum_fair_FairRARI(
        G, S_p, S_up, phi, alpha=alpha_val, max_iter=max_iters, personalization=opr, nstart=opr, edge_reweight_factor=edge_reweight
    )

    # Extract fair PageRank scores
    fair_nx_pr_scores = torch.FloatTensor(list(fair_nx_pr.values()))

    # Check achieved fairness
    fair_S_p = fair_nx_pr_scores[S_p == 1]
    achieved_fairness = torch.sum(fair_S_p).item()
    print(f"\nAchieved fairness (mass on protected group): {achieved_fairness:.6f}")
    print(f"Target fairness:                           {phi:.6f}")

    # Compute metrics
    opr_np = opr_scores.numpy()
    fair_np = fair_nx_pr_scores.numpy()

    tv = compute_tv(opr_np, fair_np)
    kt = compute_kendall_tau(opr_np, fair_np)

    print("\n" + "=" * 60)
    print("REPRODUCTION RESULTS")
    print("=" * 60)
    print(f"TV (Total Variation):           {tv:.6f}")
    print(f"Kendall Tau:                    {kt:.6f}")
    print(f"Final convergence err (l1):     {nx_x_diff[-1]:.10f}")
    print(f"Achieved fairness:              {achieved_fairness:.6f}")
    print()

    # Compare with rubric bounds
    print("RUBRIC COMPARISON")
    print("-" * 60)
    print(f"TV:           paper=0.37, bounds=[0.357, 0.50], ours={tv:.4f}")
    print(f"Kendall Tau:  paper=0.68, bounds=[0.48, 0.70],  ours={kt:.4f}")

    tv_in_bounds = 0.357 <= tv <= 0.50
    kt_in_bounds = 0.48 <= kt <= 0.70

    print(f"\nTV within CI bounds:       {tv_in_bounds}")
    print(f"Kendall Tau within CI bounds: {kt_in_bounds}")

    # Print final result line for parsing
    print()
    print("FINAL_METRICS:" + f" TV={tv:.6f} KendallTau={kt:.6f} AchievedFairness={achieved_fairness:.6f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
