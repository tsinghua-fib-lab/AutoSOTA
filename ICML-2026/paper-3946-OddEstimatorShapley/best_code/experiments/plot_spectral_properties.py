"""
Plot spectral properties from the generated JSON files:
1. R^2 vs Sparsity (top-k Fourier coefficients)
2. Spectral Energy by Degree
"""

import sys
import os
import json
import glob
import re
from collections import defaultdict
import ast

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_NAMES = {
    "CaliforniaHousingLocalXAI": "Housing ($d=8$)",
    "ViT3by3Patches": "ViT9 ($d=9$)",
    "wine_quality": "Wine ($d=11$)",
    "BikeSharingLocalXAI": "Bike ($d=12$)",
    "ForestFiresLocalXAI": "Forest ($d=13$)",
    "AdultCensusLocalXAI": "Adult ($d=14$)",
    "ResNet18w14Superpixel": "ResNet18 ($d=14$)",
    "SentimentIMDBDistilBERT14": "DistilBERT ($d=14$)",
    "RealEstateLocalXAI": "Estate ($d=15$)",
    "ViT4by4Patches": "ViT16 ($d=16$)",
    "BreastCancerLocalXAI": "Cancer ($d=30$)",
    "Corrgroups60LocalXAI": "CG60 ($d=60$)",
    "IndependentLinear60LocalXAI": "IL60 ($d=60$)",
    "NHANESILocalXAI": "NHANES ($d=79$)",
    "CommunitiesAndCrimeLocalXAI": "Crime ($d=101$)",
    "SOUM": "soum",
}

def parse_ft_keys(ft_dict):
    """Parse string representations of tuples back to tuples."""
    parsed = {}
    
    for k_str, v in ft_dict.items():
        try:
            # Fast parsing without ast.literal_eval overhead
            clean_str = k_str.replace('np.int64(', '').replace(')', '')
            # Strip outer parens, split by comma, parse to int
            inner = clean_str.strip('()')
            if not inner:
                k = ()
            else:
                k = tuple(int(x.strip()) for x in inner.split(',') if x.strip())
            parsed[k] = v
        except Exception as e:
            print(f"Failed to parse key {k_str}: {e}")
    return parsed

def compute_sparsity_r2(ft, max_k):
    """Compute R^2 for top K coefficients."""
    # Exclude degree 0 (mean) for variance explained
    coeffs = [v for k, v in ft.items() if sum(k) > 0]
    sq_coeffs = np.array(coeffs) ** 2
    sq_coeffs.sort()
    sq_coeffs = sq_coeffs[::-1] # Descending
    
    tot_energy = np.sum(sq_coeffs)
    if tot_energy == 0:
        return np.zeros(max_k)
        
    cum_energy = np.cumsum(sq_coeffs)
    
    # Pad to max_k
    r2_curve = np.zeros(max_k)
    n_actual = len(cum_energy)
    
    r2 = cum_energy / tot_energy
    
    if max_k <= n_actual:
        r2_curve[:] = r2[:max_k]
    else:
        r2_curve[:n_actual] = r2
        r2_curve[n_actual:] = r2[-1]
        
    return r2_curve

def compute_degree_energy(ft):
    """Compute energy per degree."""
    energy_by_degree = defaultdict(float)
    tot_energy = 0.0
    
    for k, v in ft.items():
        deg = sum(k)
        if deg == 0:
            continue
        energy = v ** 2
        tot_energy += energy
        
        if deg >= 8:
            energy_by_degree["8+"] += energy
        else:
            energy_by_degree[deg] += energy
            
    if tot_energy == 0:
        tot_energy = 1.0
        
    return {d: energy_by_degree[d] / tot_energy for d in [1, 2, 3, 4, 5, 6, 7, "8+"]}

def main():
    json_dir = "approximations/spectral_properties"
    plot_dir = "plots/spectral_properties"
    os.makedirs(plot_dir, exist_ok=True)
    
    file_pattern = os.path.join(json_dir, "*_spex_*.json")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"No JSON files found in {json_dir}")
        return
        
    # Group by game identifier
    results_by_game = defaultdict(list)
    
    for fpath in files:
        with open(fpath, "r") as f:
            data = json.load(f)
            
        game_id = data.get("game_identifier", "Unknown")
        ft_dict = data.get("fourier_transform", {})
        
        if not ft_dict:
            continue
            
        ft_parsed = parse_ft_keys(ft_dict)
        results_by_game[game_id].append(ft_parsed)

    print(f"Found {len(files)} files spanning {len(results_by_game)} game types.")

    def get_d_val(game_id):
        base = game_id.split("_")[0]
        name_str = DATA_NAMES.get(base, "")
        match = re.search(r'd=(\d+)', name_str)
        if match:
            return int(match.group(1))
        return 9999

    # Sort the game IDs by increasing d value
    all_game_ids = sorted(list(results_by_game.keys()), key=get_d_val)
    
    # 1. Plot Sparsity vs R2 for each dataset in a 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, game_id in enumerate(all_game_ids):
        if i >= 8:
            break
        
        ax = axes[i]
        ft_list = results_by_game[game_id]
        
        # Find max coefficients across all instances to set a dynamic max_k if needed
        local_max = max((len([v for k,v in ft.items() if sum(k)>0]) for ft in ft_list), default=0)
        
        if local_max == 0:
            print(f"Skipping {game_id} because no non-zero degree coefficients were found.")
            ax.set_visible(False)
            continue
            
        k_eval = min(1000, local_max) # Cap at 1000 for plotting
        
        all_r2_curves = []
        for ft in ft_list:
            r2_curve = compute_sparsity_r2(ft, k_eval)
            all_r2_curves.append(r2_curve)
            
        r2_mean = np.mean(all_r2_curves, axis=0)
        r2_std = np.std(all_r2_curves, axis=0)
        
        x_vals = np.arange(1, k_eval + 1)
        
        if len(x_vals) == 0:
            print(f"Skipping {game_id} because x_vals is empty.")
            ax.set_visible(False)
            continue
            
        # Plot mean
        ax.plot(x_vals, r2_mean, label=f"Mean (n={len(ft_list)})", linewidth=2, color='tab:blue')
        # Plot std shading
        ax.fill_between(x_vals, r2_mean - r2_std, r2_mean + r2_std, alpha=0.2, color='tab:blue')
        
        try:
            ax.set_xscale('log')
        except Exception as e:
            print(f"Failed to log-scale X-axis for {game_id}: {e}")
            
        # We only need x/y labels on the outer plots for cleanness, 
        # but let's add them to all or specific ones
        if i >= 4:
            ax.set_xlabel("Sparsity (Top-K Fourier Coefficients)", fontsize=12)
        if i % 4 == 0:
            ax.set_ylabel("Explained Variance ($R^2$)", fontsize=12)
            
        base_game_id = game_id.split("_")[0]
        title_str = DATA_NAMES.get(base_game_id, base_game_id)
        ax.set_title(title_str, fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    # Hide any unused subplots
    for i in range(len(all_game_ids), 8):
        axes[i].set_visible(False)

    plt.tight_layout()
    plot1_path = os.path.join(plot_dir, "spectral_sparsity_r2_combined.pdf")
    plt.savefig(plot1_path)
    print(f"Saved {plot1_path}")
    plt.close()

    # 2. Plot Spectral Energy by Degree in a 2x4 grid
    labels = [1, 2, 3, 4, 5, 6, 7, "8+"]
    x = np.arange(len(labels))
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, game_id in enumerate(all_game_ids):
        if i >= 8:
            break
            
        ax = axes[i]
        ft_list = results_by_game[game_id]
        
        all_degree_energies = []
        for ft in ft_list:
            de = compute_degree_energy(ft)
            all_degree_energies.append([de[l] for l in labels])
            
        mean_energies = np.mean(all_degree_energies, axis=0)
        std_energies = np.std(all_degree_energies, axis=0)
        
        bars = ax.bar(x, mean_energies, yerr=std_energies, color='tab:orange', edgecolor='black', alpha=0.8, capsize=5)

        # Add text labels on top of the bars
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            err_val = std_energies[idx] if not np.isnan(std_energies[idx]) else 0
            ax.text(bar.get_x() + bar.get_width()/2., height + err_val + 0.02,
                     f'{height*100:.1f}%',
                     ha='center', va='bottom', fontsize=9)

        if i >= 4:
            ax.set_xlabel("Interaction Degree", fontsize=12)
        if i % 4 == 0:
            ax.set_ylabel("Fraction of Spectral Energy", fontsize=12)
            
        base_game_id = game_id.split("_")[0]
        title_str = DATA_NAMES.get(base_game_id, base_game_id)
        ax.set_title(title_str, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in labels])
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)

    # Hide any unused subplots
    for i in range(len(all_game_ids), 8):
        axes[i].set_visible(False)

    plt.tight_layout()
    plot2_path = os.path.join(plot_dir, "spectral_energy_by_degree_combined.pdf")
    plt.savefig(plot2_path)
    print(f"Saved {plot2_path}")
    plt.close()

if __name__ == "__main__":
    main()
