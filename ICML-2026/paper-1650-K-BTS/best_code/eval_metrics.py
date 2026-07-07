#!/usr/bin/env python3
"""Evaluation script for K-BTS results. Computes docking, QED, SA, and diversity metrics."""
import pandas as pd
import numpy as np
import os
import sys
from utils.rdkit_tools import internal_diversity
from rdkit import RDLogger
import json

RDLogger.logger().setLevel(RDLogger.CRITICAL)

def evaluate_results(dir_path, n_targets=100, n_seeds=1, n_ligands=100):
    """Evaluate K-BTS optimization results.

    Args:
        dir_path: Path to results directory (e.g., 'results/rand')
        n_targets: Number of targets to evaluate
        n_seeds: Number of random seeds per target
        n_ligands: Expected number of ligands per target

    Returns:
        dict with all computed metrics
    """
    bo_diversity_list = []
    init_diversity_list = []
    init_qed_list, init_sa_list, init_logp_list, init_mw_list = [], [], [], []
    init_qed_median_list, init_sa_median_list, init_logp_median_list, init_mw_median_list = [], [], [], []
    bo_qed_list, bo_sa_list, bo_logp_list, bo_mw_list = [], [], [], []
    bo_qed_median_list, bo_sa_median_list, bo_logp_median_list, bo_mw_median_list = [], [], [], []

    init_smiles_list = []
    bo_smiles_list = []
    topk_result = {}

    for i in range(n_targets):
        name_protein = os.path.join(dir_path, str(i))
        result_path = os.path.join(name_protein, "init_score.csv")
        result_df = pd.read_csv(result_path)

        init_docking_scores = result_df["docking_scores"].tolist()
        init_qed = result_df["QED"].tolist()
        init_sa = result_df["SAS"].tolist()
        init_logp = result_df["logp"].tolist()
        init_mw = result_df["mw"].tolist()

        init_qed_list.append(np.mean(init_qed))
        init_sa_list.append(np.mean(init_sa))
        init_logp_list.append(np.mean(init_logp))
        init_mw_list.append(np.mean(init_mw))

        init_qed_median_list.append(np.median(init_qed))
        init_sa_median_list.append(np.median(init_sa))
        init_logp_median_list.append(np.median(init_logp))
        init_mw_median_list.append(np.median(init_mw))

        init_smiles = result_df["smile"].tolist()
        init_smiles_list.append(init_smiles)

        init_diversity = internal_diversity(init_smiles)
        init_diversity_list.append(init_diversity)
        init_docking_scores = np.sort(init_docking_scores)

        bo_dockings = []
        bo_smiles = []
        bo_qed = []
        bo_sa = []
        bo_logp = []
        bo_mw = []

        for seed in range(1, n_seeds + 1):
            bo_name = os.path.join(name_protein, str(seed) + "_result.csv")
            df = pd.read_csv(bo_name)
            bo_dockings.extend(df["docking_score"].tolist())
            bo_smiles.extend(df["SMILES"].tolist())
            bo_qed.extend(df["qed"].tolist())
            bo_sa.extend(df["sa"].tolist())
            bo_logp.extend(df["logp"].tolist())
            bo_mw.extend(df["mw"].tolist())

        if len(bo_dockings) != n_ligands:
            print("WARNING: target {}: {} ligands (expected {})".format(i, len(bo_dockings), n_ligands))

        bo_smiles_list.append(bo_smiles)
        bo_dockings = np.sort(bo_dockings).tolist()
        bo_qed_list.append(np.mean(bo_qed))
        bo_sa_list.append(np.mean(bo_sa))
        bo_logp_list.append(np.mean(bo_logp))
        bo_mw_list.append(np.mean(bo_mw))

        bo_qed_median_list.append(np.median(bo_qed))
        bo_sa_median_list.append(np.median(bo_sa))
        bo_logp_median_list.append(np.median(bo_logp))
        bo_mw_median_list.append(np.median(bo_mw))

        bo_diversity = internal_diversity(bo_smiles)
        bo_diversity_list.append(bo_diversity)

        for k in [1, 5, 10, 20]:
            mean_init = np.mean(init_docking_scores[:k])
            median_init = np.median(init_docking_scores[:k])
            mean_bo = np.mean(bo_dockings[:k])
            median_bo = np.median(bo_dockings[:k])

            topk_result.setdefault("init_mean_list_" + str(k), []).append(mean_init)
            topk_result.setdefault("init_median_list_" + str(k), []).append(median_init)
            topk_result.setdefault("bo_mean_list_" + str(k), []).append(mean_bo)
            topk_result.setdefault("bo_median_list_" + str(k), []).append(median_bo)

    result = {}
    for k in [1, 5, 10, 20]:
        ks = str(k)
        result["init_mean_list_" + ks] = topk_result["init_mean_list_" + ks]
        result["init_median_list_" + ks] = topk_result["init_median_list_" + ks]
        result["bo_mean_list_" + ks] = topk_result["bo_mean_list_" + ks]
        result["bo_median_list_" + ks] = topk_result["bo_median_list_" + ks]
        result["init_top" + ks + "_mean"] = np.mean(result["init_mean_list_" + ks])
        result["init_top" + ks + "_median"] = np.mean(result["init_median_list_" + ks])
        result["bo_top" + ks + "_mean"] = np.mean(result["bo_mean_list_" + ks])
        result["bo_top" + ks + "_median"] = np.mean(result["bo_median_list_" + ks])

    result["init_mean_qed"] = np.mean(init_qed_list)
    result["init_mean_sa"] = np.mean(init_sa_list)
    result["init_mean_diversity"] = np.mean(init_diversity_list)
    result["init_median_qed"] = np.mean(init_qed_median_list)
    result["init_median_sa"] = np.mean(init_sa_median_list)
    result["init_median_diversity"] = np.median(init_diversity_list)

    result["bo_mean_qed"] = np.mean(bo_qed_list)
    result["bo_mean_sa"] = np.mean(bo_sa_list)
    result["bo_mean_diversity"] = np.mean(bo_diversity_list)
    result["bo_median_qed"] = np.mean(bo_qed_median_list)
    result["bo_median_sa"] = np.mean(bo_sa_median_list)
    result["bo_median_diversity"] = np.median(bo_diversity_list)

    return result


if __name__ == "__main__":
    dir_path = sys.argv[1] if len(sys.argv) > 1 else "results/rand"
    n_targets = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    print("Evaluating {} with {} targets...".format(dir_path, n_targets))
    result = evaluate_results(dir_path, n_targets=n_targets)

    # Print key metrics
    print("\n" + "="*60)
    print("K-BTS REPRODUCTION RESULTS")
    print("="*60)
    for k in [1, 5, 10, 20]:
        print("Top{} Dock Avg: {:.2f}".format(k, result["bo_top" + str(k) + "_mean"]))
        print("Top{} Dock Med: {:.2f}".format(k, result["bo_top" + str(k) + "_median"]))
    print("QED Avg: {:.2f}".format(result["bo_mean_qed"]))
    print("QED Med: {:.2f}".format(result["bo_median_qed"]))
    print("SA Avg: {:.2f}".format(result["bo_mean_sa"]))
    print("SA Med: {:.2f}".format(result["bo_median_sa"]))
    print("Diversity Avg: {:.2f}".format(result["bo_mean_diversity"]))
    print("Diversity Med: {:.2f}".format(result["bo_median_diversity"]))
    print("="*60)

    # Also save as JSON for programmatic access
    metrics_json = {
        "Top1_Dock_Avg": round(float(result["bo_top1_mean"]), 2),
        "Top1_Dock_Med": round(float(result["bo_top1_median"]), 2),
        "Top5_Dock_Avg": round(float(result["bo_top5_mean"]), 2),
        "Top5_Dock_Med": round(float(result["bo_top5_median"]), 2),
        "Top10_Dock_Avg": round(float(result["bo_top10_mean"]), 2),
        "Top10_Dock_Med": round(float(result["bo_top10_median"]), 2),
        "Top20_Dock_Avg": round(float(result["bo_top20_mean"]), 2),
        "Top20_Dock_Med": round(float(result["bo_top20_median"]), 2),
        "QED_Avg": round(float(result["bo_mean_qed"]), 2),
        "QED_Med": round(float(result["bo_median_qed"]), 2),
        "SA_Avg": round(float(result["bo_mean_sa"]), 2),
        "SA_Med": round(float(result["bo_median_sa"]), 2),
        "Diversity_Avg": round(float(result["bo_mean_diversity"]), 2),
        "Diversity_Med": round(float(result["bo_median_diversity"]), 2),
    }

    json_path = os.path.join(dir_path, "reproduction_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print("\nMetrics saved to {}".format(json_path))
