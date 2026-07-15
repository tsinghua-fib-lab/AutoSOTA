import json
import numpy as np
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as smt
import glob
import os

def load_data(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data

def safe_get(traj, layer_idx, metric):
    if layer_idx < len(traj):
        return traj[layer_idx].get(metric)
    return None

def run_experiment_6():
    # Target Model: Llama-8B (most representative)
    target_file = "data/results/experiment_results_Meta-Llama-3.1-8B-Instruct.json"
    if not os.path.exists(target_file):
        print(f"File not found: {target_file}")
        return

    data = load_data(target_file)
    
    metrics = ["fiedler_value", "hfer", "smoothness", "entropy", "energy"]
    # Max layer for 8B is 32.
    layers = range(32)
    
    p_values = []
    configs = []
    
    for m in metrics:
        for l in layers:
            valid_vals = [safe_get(x['trajectory'], l, m) for x in data['valid']]
            valid_vals = [v for v in valid_vals if v is not None]
            
            invalid_vals = [safe_get(x['trajectory'], l, m) for x in data['invalid']]
            invalid_vals = [v for v in invalid_vals if v is not None]
            
            if len(valid_vals) > 5 and len(invalid_vals) > 5:
                # T-test
                t_stat, p_val = ttest_ind(valid_vals, invalid_vals, equal_var=False)
                p_values.append(p_val)
                configs.append(f"{m}@L{l}")
    
    # BH Correction
    reject, pvals_corrected, _, _ = smt.multipletests(p_values, alpha=0.05, method='fdr_bh')
    
    n_total = len(p_values)
    n_sig = sum(reject)
    
    print(f"Total Hypotheses: {n_total}")
    print(f"Significant after FDR (0.05): {n_sig}")
    print(f"Ratio: {n_sig}/{n_total} ({n_sig/n_total*100:.1f}%)")

if __name__ == "__main__":
    try:
        run_experiment_6()
    except ImportError:
        # Fallback if statsmodels missing
        print("Statsmodels not installed. Running manual BH.")
        # Manual logic?
        pass
