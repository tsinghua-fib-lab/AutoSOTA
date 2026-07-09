#!/usr/bin/env python3
"""Evaluation script for paper 2370: Online Conformal Prediction via Universal Portfolio Algorithms."""
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], "../"))
import numpy as np
import pickle
import argparse
from plotting_utils import longest_true_sequence

def compute_metrics(results_file, method="UP", lr=None, T_burnin=100):
    with open(results_file, "rb") as f:
        all_results = pickle.load(f)
    model_name = list(all_results.keys())[0]
    results = all_results[model_name]
    method_results = results[method]
    lr_results = method_results[lr]
    c = np.array(lr_results["coverages"])
    q = np.array([np.array(x) for x in lr_results["q"]])
    c_burnin = c[T_burnin:]
    q_burnin = q[T_burnin:]
    if q.ndim == 1:
        set_sizes = 2 * q_burnin
    else:
        set_sizes = q_burnin[:, 1] - q_burnin[:, 0]
    metrics = {
        "Marginal_Coverage": float(np.mean(c_burnin)),
        "Longest_Err_Seq": int(longest_true_sequence((1 - c_burnin).astype(bool))),
        "Avg_Set_Size": float(np.mean(set_sizes)),
        "Median_Set_Size": float(np.median(set_sizes)),
        "75pct_Quantile_Size": float(np.percentile(set_sizes, 75)),
        "90pct_Quantile_Size": float(np.percentile(set_sizes, 90)),
        "95pct_Quantile_Size": float(np.percentile(set_sizes, 95)),
    }
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="./results/AXP.pkl")
    parser.add_argument("--method", default="UP")
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--T_burnin", default=100, type=int)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        with open(args.results, "rb") as f:
            all_results = pickle.load(f)
        model_name = list(all_results.keys())[0]
        results = all_results[model_name]
        for method_key in results:
            if method_key in ["scores", "alpha", "T_burnin", "quantiles_given",
                            "multiple_series", "real_data", "score_function_name",
                            "asymmetric", "forecasts", "data"]:
                continue
            method_results = results[method_key]
            if not isinstance(method_results, dict):
                continue
            for lr, lr_results in method_results.items():
                if not isinstance(lr_results, dict) or "coverages" not in lr_results:
                    continue
                metrics = compute_metrics(args.results, method_key, lr, args.T_burnin)
                lr_str = "lr={}".format(lr) if lr is not None else "lr=None"
                print("")
                print("{} ({}):".format(method_key, lr_str))
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print("  {}: {:.4f}".format(k, v))
                    else:
                        print("  {}: {}".format(k, v))
    else:
        metrics = compute_metrics(args.results, args.method, args.lr, args.T_burnin)
        print("")
        print("{} (lr={}, T_burnin={}):".format(args.method, args.lr, args.T_burnin))
        for k, v in metrics.items():
            if isinstance(v, float):
                print("  {}: {:.4f}".format(k, v))
            else:
                print("  {}: {}".format(k, v))
