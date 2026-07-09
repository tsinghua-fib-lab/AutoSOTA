#!/usr/bin/env python3
"""
Combined Plotting Script
Generates:
performance_summary.pdf (1x2): Family F1 scores | Function Spearman correlations
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import MultipleLocator
from scipy import stats
from pathlib import Path

# Configuration
FAMILY_DIR = "../family_circuit/families"
FUNCTION_DIR = "../function_circuit/functions"
OUTPUT_DIR = "plots"

# Models and Colors
CLT_MODEL = "CLT_sequential"
PLT_MODEL = "PLT"
METHOD_MAP = {CLT_MODEL: "ProtoMech", PLT_MODEL: "PLT"}
COLOR_MAP = {CLT_MODEL: '#1b75bb', PLT_MODEL: '#f6921e'}
clean_color = 'gray'

# Thresholds
MIN_CLEAN_SPEARMAN = 0.01
MIN_SEQUENCES = 50

# Sizing & Styling 
font_path = '../circuit_utils/Helvetica.ttf'
try:
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    font_name = font_prop.get_name()
    plt.rcParams['font.family'] = font_name
except:
    plt.rcParams['font.family'] = 'sans-serif'
font_size = 8
plt.rcParams['font.size'] = font_size
mm = 1/72  
linewidth = 0.25 
plt.rcParams['axes.linewidth'] = linewidth
plt.rcParams['lines.linewidth'] = linewidth
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
REF_FIG_WIDTH = 280 * mm
REF_FIG_HEIGHT = 115  * mm

# Data loading
def load_family_data(data_dir=FAMILY_DIR, method_map=METHOD_MAP, min_seq=None, min_clean=None):
    """
    Universal Family Data Loader.
    Args:
        data_dir (str): Path to family directory.
        method_map (dict): Mapping of {folder_name: readable_name}.
        min_seq (int, optional): Filter rows with n_sequences < min_seq.
        min_clean (float, optional): Filter rows with clean_f1 < min_clean.
    """
    records = []
    base_path = Path(data_dir)
    if not base_path.exists():
        print(f"Warning: {data_dir} not found.")
        return pd.DataFrame()

    for folder_name, readable_name in method_map.items():
        search_path = base_path / folder_name / "*.json"
        files = glob.glob(str(search_path))
        
        for fpath in files:
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                
                n_seq = data.get("n_train", data.get("n_sequences", 0))
                
                row = {
                    "family": data.get("family", "Unknown"),
                    "model": folder_name,       # Raw internal name
                    "Model": readable_name,     # Readable name for plots
                    "n_sequences": n_seq,
                    "k": data.get("k", 0),
                    "clean": data.get("clean_f1", data.get("test_f1", 0)),
                    "max": data.get("max_f1", 0),
                    "recovered": data.get("recovered_f1", 0),
                }
                
                nodes_dict = data.get("nodes", {})
                for l in range(6): 
                    layer_nodes = nodes_dict.get(str(l), [])
                    row[f"Layer {l+1}"] = len(layer_nodes)
                
                records.append(row)
            except Exception: 
                pass
            
    df = pd.DataFrame(records)
    if df.empty: return df

    # Apply Filtering
    if min_seq is not None:
        df = df[df["n_sequences"] >= min_seq]
    if min_clean is not None:
        df = df[df["clean"] >= min_clean]
        
    return df.copy()

def load_function_data(data_dir=FUNCTION_DIR, method_map=METHOD_MAP, min_seq=None, min_clean=None):
    """
    Universal Function Data Loader.
    Parses Type (single/multiples) and Split (random/contiguous/modulo).
    """
    records = []
    base_path = Path(data_dir)
    if not base_path.exists():
        print(f"Warning: {data_dir} not found.")
        return pd.DataFrame()

    for folder_name, readable_name in method_map.items():
        # Recursive glob to find nested files
        files = glob.glob(str(base_path / folder_name / "**" / "*.json"), recursive=True)
        
        for fpath in files:
            try:
                path_obj = Path(fpath)
                with open(fpath, "r") as f:
                    data = json.load(f)
                
                # Extract Metadata from Path/Filename
                dtype = "single" if "single" in path_obj.parts else "multiples" if "multiples" in path_obj.parts else "unknown"
                fname = path_obj.stem
                
                split = "random" 
                if "contiguous" in fname: split = "contiguous"
                elif "modulo" in fname: split = "modulo"
                
                clean_sp = max(0.0, data.get("clean_spearman", 0))
                dms_name = data.get("DMS", "Unknown")
                
                row = {
                    "unique_id": f"{dms_name}_{fname}", 
                    "DMS": dms_name,
                    "model": folder_name,       
                    "Model": readable_name,     
                    "Type": dtype,
                    "Split": split,
                    "n_sequences": data.get("n_train", data.get("n_sequences", 0)),
                    "k": data.get("k", 0),
                    "clean": clean_sp,
                    "max": max(0.0, data.get("max_spearman", 0)),
                    "recovered": max(0.0, data.get("recovered_spearman", 0)),
                }
                
                nodes_dict = data.get("nodes", {})
                for l in range(6): 
                    layer_nodes = nodes_dict.get(str(l), [])
                    row[f"Layer {l+1}"] = len(layer_nodes)
                
                records.append(row)
            except Exception: 
                pass
            
    df = pd.DataFrame(records)
    if df.empty: return df

    # Apply Filtering
    if min_seq is not None:
        df = df[df["n_sequences"] >= min_seq]
    if min_clean is not None:
        df = df[df["clean"] >= min_clean]
        
    return df.copy()

# Helper functions
def draw_significance(ax, x1, x2, y_max, p_val):
    if p_val >= 0.05: return
    h = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    if p_val < 0.001: sig_symbol = "***"
    elif p_val < 0.01: sig_symbol = "**"
    else: sig_symbol = "*"
    ax.text((x1+x2)*.5, y_max+h, sig_symbol, ha='center', va='bottom', color='k', fontsize=font_size)

def plot_family_performance(ax, df):
    """Bar chart: ESM2 vs All Latents vs Circuit F1"""
    clean_vals = df.groupby("family")["clean"].mean()
    clean_mean = clean_vals.mean()
    clean_std = clean_vals.std()
    print(f"ESM2: {clean_mean:.3f} ± {clean_std:.3f}")

    stats_max = df.groupby("model")["max"].agg(['mean', 'std'])
    stats_rec = df.groupby("model")["recovered"].agg(['mean', 'std'])

    bar_width = 0.25
    pos_orig, pos_all, pos_circ = 0, 1.0, 2.0
    offsets = [-bar_width/2, bar_width/2]

    # ESM2
    ax.bar(pos_orig, clean_mean, yerr=clean_std, width=bar_width,
           color=clean_color, edgecolor='black', linewidth=linewidth,
           capsize=2, error_kw={'linewidth': linewidth}, label='ESM2')

    # All Latents & Circuit
    max_y = clean_mean + clean_std
    # Iterate over METHOD_MAP keys (CLT_MODEL, PLT_MODEL)
    for i, model_key in enumerate(METHOD_MAP.keys()):
        if model_key not in stats_max.index: continue
        
        # Max
        m, s = stats_max.loc[model_key, 'mean'], stats_max.loc[model_key, 'std']
        print(f"All Latents ({METHOD_MAP[model_key]}): {m:.3f} ± {s:.3f}")
        ax.bar(pos_all + offsets[i], m, yerr=s, width=bar_width,
               color=COLOR_MAP[model_key], edgecolor='black', linewidth=linewidth, 
               capsize=2, error_kw={'linewidth': linewidth})
        max_y = max(max_y, m + s)

        # Recovered
        m, s = stats_rec.loc[model_key, 'mean'], stats_rec.loc[model_key, 'std']
        print(f"Circuit ({METHOD_MAP[model_key]}): {m:.3f} ± {s:.3f}")
        ax.bar(pos_circ + offsets[i], m, yerr=s, width=bar_width,
               color=COLOR_MAP[model_key], edgecolor='black', linewidth=linewidth, 
               capsize=2, error_kw={'linewidth': linewidth}, label=METHOD_MAP[model_key])
        max_y = max(max_y, m + s)

    # Stats
    piv_max = df.pivot(index="family", columns="model", values="max").dropna()
    piv_rec = df.pivot(index="family", columns="model", values="recovered").dropna()
    
    if not piv_max.empty and len(piv_max) > 1:
        _, p_max = stats.ttest_rel(piv_max[CLT_MODEL], piv_max[PLT_MODEL])
        draw_significance(ax, pos_all + offsets[0], pos_all + offsets[1], max_y, p_max)
    
    if not piv_rec.empty and len(piv_rec) > 1:
        _, p_rec = stats.ttest_rel(piv_rec[CLT_MODEL], piv_rec[PLT_MODEL])
        rec_high = max(stats_rec.loc[CLT_MODEL, 'mean']+stats_rec.loc[CLT_MODEL, 'std'], 
                       stats_rec.loc[PLT_MODEL, 'mean']+stats_rec.loc[PLT_MODEL, 'std'])
        draw_significance(ax, pos_circ + offsets[0], pos_circ + offsets[1], rec_high, p_rec)

    ax.set_ylabel("Protein family F1 score")
    ax.set_xticks([pos_orig, pos_all, pos_circ])
    ax.set_xticklabels(["ESM2", "All latents", "Circuit"])
    ax.set_ylim(0, 1.15)
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=font_size-1)

def plot_function_performance(ax, df):
    """Grouped Bar chart for Function: ESM2 vs All Latents vs Circuit Spearman."""
    clean_vals = df.groupby("unique_id")["clean"].mean()
    clean_mean = clean_vals.mean()
    clean_std = clean_vals.std()
    print(f"ESM2: {clean_mean:.3f} ± {clean_std:.3f}")

    stats_max = df.groupby("model")["max"].agg(['mean', 'std'])
    stats_rec = df.groupby("model")["recovered"].agg(['mean', 'std'])

    bar_width = 0.25
    pos_orig, pos_all, pos_circ = 0, 1.0, 2.0
    offsets = [-bar_width/2, bar_width/2]
    
    # ESM2
    ax.bar(pos_orig, clean_mean, yerr=clean_std, width=bar_width,
           color=clean_color, edgecolor='black', linewidth=linewidth,
           capsize=3, error_kw={'linewidth': linewidth}, label='ESM2')

    # All Latents & Circuit
    max_y_global = clean_mean + clean_std
    for i, model_key in enumerate(METHOD_MAP.keys()):
        if model_key not in stats_max.index: continue
        
        # Max
        m, s = stats_max.loc[model_key, 'mean'], stats_max.loc[model_key, 'std']
        print(f"All Latents ({METHOD_MAP[model_key]}): {m:.3f} ± {s:.3f}")
        ax.bar(pos_all + offsets[i], m, yerr=s, width=bar_width,
               color=COLOR_MAP[model_key], edgecolor='black', linewidth=linewidth, 
               capsize=3, error_kw={'linewidth': linewidth})
        max_y_global = max(max_y_global, m + s)

        # Recovered
        m, s = stats_rec.loc[model_key, 'mean'], stats_rec.loc[model_key, 'std']
        print(f"Circuit ({METHOD_MAP[model_key]}): {m:.3f} ± {s:.3f}")
        ax.bar(pos_circ + offsets[i], m, yerr=s, width=bar_width,
               color=COLOR_MAP[model_key], edgecolor='black', linewidth=linewidth, 
               capsize=3, error_kw={'linewidth': linewidth}, label=METHOD_MAP[model_key])
        max_y_global = max(max_y_global, m + s)

    # Statistics (Paired T-test)
    piv_max = df.pivot(index="unique_id", columns="model", values="max").dropna()
    piv_rec = df.pivot(index="unique_id", columns="model", values="recovered").dropna()

    if not piv_max.empty and len(piv_max) > 1:
        _, p_max = stats.ttest_rel(piv_max[CLT_MODEL], piv_max[PLT_MODEL])
        local_h = max(stats_max.loc[CLT_MODEL, 'mean'] + stats_max.loc[CLT_MODEL, 'std'],
                      stats_max.loc[PLT_MODEL, 'mean'] + stats_max.loc[PLT_MODEL, 'std'])
        draw_significance(ax, pos_all + offsets[0], pos_all + offsets[1], local_h, p_max)

    if not piv_rec.empty and len(piv_rec) > 1:
        _, p_rec = stats.ttest_rel(piv_rec[CLT_MODEL], piv_rec[PLT_MODEL])
        local_h = max(stats_rec.loc[CLT_MODEL, 'mean'] + stats_rec.loc[CLT_MODEL, 'std'],
                      stats_rec.loc[PLT_MODEL, 'mean'] + stats_rec.loc[PLT_MODEL, 'std'])
        draw_significance(ax, pos_circ + offsets[0], pos_circ + offsets[1], local_h, p_rec)

    ax.set_ylabel("Function Spearman ρ")
    ax.set_xticks([pos_orig, pos_all, pos_circ])
    ax.set_xticklabels(["ESM2", "All latents", "Circuit"])
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_ylim(0, 0.85) 

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df_fam = load_family_data(min_seq=MIN_SEQUENCES)
    df_func = load_function_data(min_clean=MIN_CLEAN_SPEARMAN)
    
    print(f"Loaded {len(df_fam)} family records and {len(df_func)} function records.")

    print("Generating performance summary...")
    fig1, axes1 = plt.subplots(1, 2, figsize=(REF_FIG_WIDTH, REF_FIG_HEIGHT))
    
    if not df_fam.empty:
        print('Plotting family performance...')
        plot_family_performance(axes1[0], df_fam)
    
    if not df_func.empty:
        print('Plotting function performance...')
        plot_function_performance(axes1[1], df_func)
    
    plt.tight_layout()
    fig1.savefig(f"{OUTPUT_DIR}/performance_summary.pdf", dpi=300)
    plt.close(fig1)

    print(f"Done. Plots saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()