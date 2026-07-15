import json
import pandas as pd
import os

def load_data(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data

def safe_get(traj, layer_idx, metric):
    if layer_idx < len(traj):
        return traj[layer_idx].get(metric)
    return None

def run_experiment_5_prep():
    # Target: Llama-1B
    target_file = "data/results/experiment_results_Llama-3.2-1B-Instruct.json"
    
    # Threshold from Exp 3
    THRESHOLD = 0.05
    METRIC = "hfer"
    LAYER = 11
    
    data = load_data(target_file)
    
    discrepancies = []
    
    # We want Compiler=Invalid (Label 0) but Spectral=Valid (Score < Threshold)
    for item in data['invalid']:
        val = safe_get(item['trajectory'], LAYER, METRIC)
        if val is not None:
             # Check spectral validity
             if val < THRESHOLD:
                 # Discrepancy!
                 discrepancies.append({
                     "problem_file": item['file'],
                     "hfer_L11": val,
                     "threshold": THRESHOLD,
                     "discrepancy_type": "Spectral_Valid_But_Compiler_Invalid"
                 })
                 
    df = pd.DataFrame(discrepancies)
    output_path = "data/paper_figures/exp5_discrepancies.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Found {len(df)} discrepancies.")
    print(f"Saved list to {output_path}")

if __name__ == "__main__":
    run_experiment_5_prep()
