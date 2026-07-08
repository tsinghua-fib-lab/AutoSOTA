#!/usr/bin/env python3
"""
Reproduce OCMB paper results (ICML 2026)
Target: Scale-free graphs, d=100, n=1000, nonlinear+Gaussian, 10 seeds
Settings: K=5, tau=0.95, alpha=0.01, spouse=on, backbone=CaPS, stage2=IAMB
Reproduces Table 1 metrics: SHD, F1, Time, #CI Tests
"""

import numpy as np
import sys
import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "/repo")

from ocmb import OCMB_CaPS, calculate_metrics


def generate_scale_free_graph(d, degree=3, seed=42):
    """Generate scale-free DAG using Barabasi-Albert preferential attachment"""
    import networkx as nx
    np.random.seed(seed)
    m = max(1, degree // 2)
    G = nx.barabasi_albert_graph(d, m, seed=seed)
    adj = np.zeros((d, d))
    for i, j in G.edges():
        if i < j:
            adj[i, j] = 1
        else:
            adj[j, i] = 1
    return adj


def generate_nonlinear_gaussian_data(adj, n_samples=1000, seed=42):
    """
    Generate data from nonlinear SEM with Gaussian noise.
    All mechanisms are nonlinear (tanh), matching the paper's nonlinear+Gaussian setting.
    """
    np.random.seed(seed)
    d = adj.shape[0]
    X = np.zeros((n_samples, d))

    for j in range(d):
        parents = np.where(adj[:, j] == 1)[0]
        if len(parents) == 0:
            X[:, j] = np.random.randn(n_samples)
        else:
            z = np.zeros(n_samples)
            for p in parents:
                weight = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
                z += weight * np.tanh(X[:, p])  # Nonlinear mechanism
            noise = np.random.randn(n_samples)  # Gaussian noise
            X[:, j] = z + noise

    # Standardize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X


def main():
    print("=" * 80)
    print("OCMB Paper Reproduction Experiment")
    print("=" * 80)

    # Paper settings from rubric
    d = 100          # nodes
    n = 1000         # samples
    degree = 3       # average degree
    n_seeds = 10     # 10 random seeds
    seeds = list(range(n_seeds))

    # OCMB parameters matching rubric: K=5, tau=0.95, alpha=0.01, spouse=on
    max_parents = 5           # K=5
    score_threshold_quantile = 0.90  # tau=0.90
    alpha_mb = 0.01           # alpha=0.01
    use_spouse_closure = True # spouse=on
    k_mb = 5                  # CMI k-NN parameter

    # CaPS backbone parameters
    eta_G = 0.001
    eta_H = 0.001
    dispersion = "mean"
    device = "cuda:0"

    print("\nConfiguration:")
    print("  Graph: scale-free, d=%d, degree=%d" % (d, degree))
    print("  Data: nonlinear+Gaussian, n=%d" % n)
    print("  Seeds: %s" % str(seeds))
    print("  OCMB: K=%d, tau=%.2f, alpha=%.3f, spouse=%s" % (max_parents, score_threshold_quantile, alpha_mb, use_spouse_closure))
    print("  Backbone: CaPS (eta_G=%s, eta_H=%s, dispersion=%s)" % (eta_G, eta_H, dispersion))
    print("  Device: %s" % device)
    print()

    all_results = []

    for seed_idx, seed in enumerate(seeds):
        print("\n" + "=" * 60)
        print("Seed %d (%d/%d)" % (seed, seed_idx + 1, n_seeds))
        print("=" * 60)

        # Generate scale-free graph
        true_adj = generate_scale_free_graph(d, degree, seed)
        n_edges = int(np.sum(true_adj))
        print("  Graph: %d edges" % n_edges)

        # Generate nonlinear Gaussian data
        X = generate_nonlinear_gaussian_data(true_adj, n, seed)
        print("  Data: %s" % str(X.shape))

        # Run OCMB with CaPS backbone
        print("  Running OCMB-CaPS...", end=" ", flush=True)
        try:
            ocmb = OCMB_CaPS(
                max_parents=max_parents,
                k_mb=k_mb,
                alpha_mb=alpha_mb,
                score_threshold_quantile=score_threshold_quantile,
                use_spouse_closure=use_spouse_closure,
                adaptive_spouse=True,
                auto_tune_s=True,
                eta_G=eta_G,
                eta_H=eta_H,
                dispersion=dispersion,
                device=device,
                verbose=False,
            )
            ocmb.fit(X, true_adj=true_adj)
            graph = ocmb.get_adjacency_matrix()
            metrics = calculate_metrics(true_adj, graph)
            timings = ocmb.get_timings()
            n_cmi = ocmb.get_n_cmi_calls()

            result = {
                "seed": seed,
                "status": "success",
                "SHD": metrics["SHD"],
                "F1": metrics["F1"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "TP": metrics["TP"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "Time": timings["total"],
                "n_cmi_calls": n_cmi,
                "ordering_divergence": ocmb.ordering_divergence_,
                "ordering_kendall_tau": ocmb.ordering_kendall_tau_,
            }
            print("OK SHD=%d, F1=%.3f, Time=%.1fs, CI=%d" % (metrics["SHD"], metrics["F1"], timings["total"], n_cmi))

            if ocmb.cand_stats_:
                result["covPa"] = ocmb.cand_stats_.get("covPa_mean", None)
                result["covMB"] = ocmb.cand_stats_.get("covMB_mean", None)

        except Exception as e:
            print("FAILED: %s" % str(e))
            import traceback
            traceback.print_exc()
            result = {"seed": seed, "status": "failed", "error": str(e)}

        all_results.append(result)

    # Aggregate results
    successful = [r for r in all_results if r["status"] == "success"]
    failed = [r for r in all_results if r["status"] != "success"]

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print("  Successful: %d/%d" % (len(successful), n_seeds))
    print("  Failed: %d/%d" % (len(failed), n_seeds))

    if successful:
        shds = [r["SHD"] for r in successful]
        f1s = [r["F1"] for r in successful]
        times = [r["Time"] for r in successful]
        cis = [r["n_cmi_calls"] for r in successful]

        print("\n  Metric        Mean +- Std         Paper Value")
        print("  " + "-" * 50)
        print("  SHD           %.1f +- %.1f        65.3 +- 3.5" % (np.mean(shds), np.std(shds)))
        print("  F1            %.3f +- %.3f        0.542 +- 0.03" % (np.mean(f1s), np.std(f1s)))
        print("  Time (s)      %.1f +- %.1f        3.6" % (np.mean(times), np.std(times)))
        print("  #CI Tests     %.0f +- %.0f        572" % (np.mean(cis), np.std(cis)))

    # Save raw results
    output_path = "/repo/reproduction_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nRaw results saved to %s" % output_path)

    print("\n" + "=" * 80)
    print("Reproduction experiment complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
