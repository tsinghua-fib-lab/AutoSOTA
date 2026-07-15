import json
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind

import argparse

def analyze_all(results_file):
    print(f"Loading {results_file}...")
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {results_file} not found.")
        return

    valid_data = data["valid"]
    invalid_data = data["invalid"]
    
    metrics = ["fiedler_value", "hfer", "smoothness", "entropy", "energy"]
    num_layers = len(valid_data[0]["trajectory"])
    
    results = []

    print(f"Scanning {num_layers} layers and {len(metrics)} metrics...")
    
    for layer in range(num_layers):
        for metric in metrics:
            # Extract values
            v_vals = [x["trajectory"][layer][metric] for x in valid_data if x["trajectory"][layer][metric] is not None]
            i_vals = [x["trajectory"][layer][metric] for x in invalid_data if x["trajectory"][layer][metric] is not None]
            
            if len(v_vals) < 2 or len(i_vals) < 2:
                continue
                
            # Mann-Whitney U Test (Non-parametric)
            try:
                u_stat, p_val = mannwhitneyu(v_vals, i_vals, alternative='two-sided')
            except ValueError:
                p_val = 1.0
            
            # Cohen's d
            n1, n2 = len(v_vals), len(i_vals)
            var1, var2 = np.var(v_vals, ddof=1), np.var(i_vals, ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            if pooled_std > 0:
                d = (np.mean(v_vals) - np.mean(i_vals)) / pooled_std
            else:
                d = 0.0

            results.append({
                "layer": layer,
                "metric": metric,
                "p_value": p_val,
                "cohen_d": d,
                "mean_valid": np.mean(v_vals),
                "mean_invalid": np.mean(i_vals)
            })

    # Sort by p-value
    results.sort(key=lambda x: x["p_value"])
    
    print("\n" + "="*80)
    print(f"{'Metric':<15} | {'Layer':<5} | {'p-value':<12} | {'Cohen d':<8} | {'Valid µ':<8} | {'Invalid µ':<8}")
    print("-" * 80)
    
    for res in results[:20]: # Top 20
        p_str = f"{res['p_value']:.2e}" if res['p_value'] < 0.001 else f"{res['p_value']:.4f}"
        star = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
        print(f"{res['metric']:<15} | {res['layer']:<5} | {p_str:<8} {star:<3} | {res['cohen_d']:+.2f}    | {res['mean_valid']:.3f}    | {res['mean_invalid']:.3f}")

    # Check for HFER specifically
    print("\n--- Deep Dive: HFER across all layers ---")
    hfer_results = [r for r in results if r["metric"] == "hfer"]
    hfer_results.sort(key=lambda x: x["layer"])
    for res in hfer_results:
         p_str = f"{res['p_value']:.2e}" if res['p_value'] < 0.001 else f"{res['p_value']:.4f}"
         star = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
         print(f"Layer {res['layer']:<2}: p={p_str:<8} {star:<3} | d={res['cohen_d']:+.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="experiment_results_Llama-3.2-1B-Instruct.json", help="Path to results json")
    args = parser.parse_args()
    analyze_all(args.file)
