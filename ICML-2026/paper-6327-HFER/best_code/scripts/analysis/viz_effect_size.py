
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(results_file, list_b_file=None):
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    reclaimed_files = set()
    if list_b_file and os.path.exists(list_b_file):
        with open(list_b_file, 'r') as f:
            raw = json.load(f)
            if raw and isinstance(raw[0], dict):
                reclaimed_files = set(item['file'] for item in raw)
            else:
                reclaimed_files = set(raw)
            
    samples = []
    for item in data['valid']:
        samples.append({'label': 1, 'features': item['summary'], 'trajectory': item['trajectory']})
        
    for item in data['invalid']:
        label = 0
        if item['file'] in reclaimed_files:
            label = 1
        samples.append({'label': label, 'features': item['summary'], 'trajectory': item['trajectory']})
    
    return samples

def plot_separation(samples, metric, layer, model_name):
    # Extract values
    valid_vals = []
    invalid_vals = []
    
    for s in samples:
        traj = s['trajectory']
        idx = layer if layer < len(traj) else -1
        val = traj[idx].get(metric)
        if val is not None:
            if s['label'] == 1:
                valid_vals.append(val)
            else:
                invalid_vals.append(val)

    # Calc Stats
    mu_v, std_v = np.mean(valid_vals), np.std(valid_vals)
    mu_i, std_i = np.mean(invalid_vals), np.std(invalid_vals)
    
    n_v, n_i = len(valid_vals), len(invalid_vals)
    pooled_std = np.sqrt(((n_v - 1)*std_v**2 + (n_i - 1)*std_i**2) / (n_v + n_i - 2))
    d = (mu_v - mu_i) / pooled_std

    # Calculate p-value (Welch's t-test)
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(valid_vals, invalid_vals, equal_var=False)
    
    print(f"[{model_name}] {metric} @ L{layer}")
    print(f"Valid:   mu={mu_v:.4f}, std={std_v:.4f} (N={n_v})")
    print(f"Invalid: mu={mu_i:.4f}, std={std_i:.4f} (N={n_i})")
    print(f"Cohen's d: {d:.4f}")
    print(f"p-value:   {p_val:.4e}")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(valid_vals, fill=True, color='blue', label='Valid', alpha=0.3)
    sns.kdeplot(invalid_vals, fill=True, color='red', label='Invalid', alpha=0.3)
    
    # Add rug plot for individual points
    sns.rugplot(valid_vals, color='blue', height=0.1)
    sns.rugplot(invalid_vals, color='red', height=0.1)
    
    plt.title(f"Spectral Separation: {model_name} ({metric} @ L{layer})\nCohen's d = {d:.2f}, p < {p_val:.1e}")
    plt.xlabel(f"{metric} Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists("output"):
        os.makedirs("output")
    fname = os.path.join("output", f"separation_plot_{model_name}_{metric}_L{layer}.png")
    plt.savefig(fname)
    print(f"Saved {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--list-b", required=True)
    parser.add_argument("--metric", default="hfer")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--model", required=True)
    
    args = parser.parse_args()
    
    samples = load_data(args.file, args.list_b)
    plot_separation(samples, args.metric, args.layer, args.model)
