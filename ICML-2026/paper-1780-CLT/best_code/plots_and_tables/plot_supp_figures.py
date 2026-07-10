#!/usr/bin/env python3
"""
Combined Circuit Plots
Generates 4 multi-panel figures comparing ProtoMech/PLT performance.
Fig 1: Performance (F1 / Spearman)
Fig 2: Node Distribution per Layer [Fixed Data Loading]
Fig 3: Function Splits Performance
Fig 4: Low F1 Families Performance
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# --- Configuration & Style ---
# # ESM2-8M
# FAMILY_DIR = "../family_circuit/families"
# FUNCTION_DIR = "../function_circuit/functions"
# PLOTS_DIR = "plots"
# LAYERS = 6
# ESM2-35M
FAMILY_DIR = "../family_circuit/families_35M"
FUNCTION_DIR = "../function_circuit/functions_35M"
PLOTS_DIR = "plots_35M"
LAYERS = 12

# Thresholds
MIN_CLEAN_SPEARMAN = 0.01 
MIN_SEQUENCES = 50
MIN_LOW_F1 = 0.01
MAX_LOW_F1 = 0.55

# Style Setup
font_path = '../circuit_utils/Helvetica.ttf'
try:
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
except:
    print(f"Warning: Font not found at {font_path}. Using default sans-serif.")
    plt.rcParams['font.family'] = 'sans-serif'

font_size = 8
plt.rcParams['font.size'] = font_size
mm = 1/25.4  # mm in inches
linewidth = 0.5 
plt.rcParams['axes.linewidth'] = linewidth
plt.rcParams['lines.linewidth'] = linewidth

# Methods and ordering
METHOD_MAP = {
    "CLT_direct": "ProtoMech (direct)",
    "CLT_sequential": "ProtoMech (sequential)",
    "PLT": "PLT (sequential)",
    "CLT_sequential_no_frozen": "ProtoMech (full replacement)",
    "PLT_no_frozen": "PLT (full replacement)"
}

ORDERED_METHODS = [
    "ProtoMech (direct)", 
    "ProtoMech (sequential)", 
    "PLT (sequential)", 
    "ProtoMech (full replacement)", 
    "PLT (full replacement)"
]
COLORS = ['#1b75bb', '#af588a', '#f6921e', '#00A087', '#DC0000']
COLOR_MAP = dict(zip(ORDERED_METHODS, COLORS))

METHOD_MAP_FIG5 = {
    "BlockCLT_sequential": "ProtoMech (windowed)",
    "CLT_sequential": "ProtoMech (sequential)",
    "PLT": "PLT (sequential)",
}

ORDERED_METHODS_FIG5 = [
    "ProtoMech (sequential)",
    "ProtoMech (windowed)", 
    "PLT (sequential)"
]
COLORS_FIG5 = ['#af588a', '#7a5195', '#f6921e']
COLOR_MAP_FIG5 = dict(zip(ORDERED_METHODS_FIG5, COLORS_FIG5))

def load_family_data(data_dir=FAMILY_DIR, method_map=METHOD_MAP, min_seq=None, min_clean=None, max_clean=None):
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
                for l in range(LAYERS): 
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
    if max_clean is not None:
        df = df[df["clean"] <= max_clean]
        
    return df.copy()

def load_function_data(data_dir=FUNCTION_DIR, method_map=METHOD_MAP, min_seq=None, min_clean=None, max_clean=None):
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
                for l in range(LAYERS): 
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
    if max_clean is not None:
        df = df[df["clean"] <= max_clean]
        
    return df.copy()

def draw_grouped_bars(ax, df, metric_key="f1", show_legend=False, ylabel="Score", show_xticks=True, ylim=None, yticks=None, methods=None, color_map=None, legend_include_esm2=True):
    """Standard grouped bar plot for performance metrics.
    
    Args:
        methods: optional list of method names to use for spacing/ordering. Defaults to ORDERED_METHODS.
    """
    if df.empty: return
    if methods is None:
        methods = ORDERED_METHODS
    if color_map is None:
        color_map = COLOR_MAP

    group_col = "family" if "family" in df.columns else "unique_id"
    
    fam_clean = df.groupby(group_col)["clean"].mean()
    clean_mean = fam_clean.mean()
    clean_std = fam_clean.std()
    
    col_max = "max"
    col_rec = "recovered"
    
    xticklabels = ["ESM2", "All latents", "Circuit"]
    
    method_stats = df.groupby("Model")[[col_max, col_rec]].agg(['mean', 'std'])
    
    bar_width = 0.12
    pos_original = 0
    group_1_center = 0.85 
    group_2_center = 1.85
    centers = [pos_original, group_1_center, group_2_center]
    x_lims = (-0.5, 2.5)

    indices = np.arange(len(methods))
    offsets = (indices - (len(methods)-1)/2) * (bar_width * 1.1)

    # A. ESM2
    clean_lower_err = min(clean_mean, clean_std)
    clean_yerr = [[clean_lower_err], [clean_std]]
    print(f'ESM2 ({ylabel}): {clean_mean:.3f} +/- {clean_std:.3f}')
    
    ax.bar(pos_original, clean_mean, yerr=clean_yerr, width=bar_width, 
           color='gray', edgecolor='black', linewidth=linewidth,
           capsize=2, error_kw={'linewidth': linewidth}, label='ESM2')

    # B. All Latents (Max)
    for i, method in enumerate(methods):
        if method not in method_stats.index: continue
        mean_val = method_stats.loc[method, (col_max, 'mean')]
        std_val = method_stats.loc[method, (col_max, 'std')]
        print(f"{method} All latents: {mean_val:.3f} +/- {std_val:.3f}")
        
        lower_err = min(mean_val, std_val)
        yerr_asym = [[lower_err], [std_val]]
        
        ax.bar(group_1_center + offsets[i], mean_val, yerr=yerr_asym, 
             width=bar_width, color=color_map[method], 
               edgecolor='black', linewidth=linewidth,
               capsize=2, error_kw={'linewidth': linewidth})

    # C. Circuit (Recovered)
    for i, method in enumerate(methods):
        if method not in method_stats.index: continue
        mean_val = method_stats.loc[method, (col_rec, 'mean')]
        std_val = method_stats.loc[method, (col_rec, 'std')]
        print(f"{method} Circuit: {mean_val:.3f} +/- {std_val:.3f}")
        
        lower_err = min(mean_val, std_val)
        yerr_asym = [[lower_err], [std_val]]
        
        ax.bar(group_2_center + offsets[i], mean_val, yerr=yerr_asym, 
             width=bar_width, color=color_map[method], 
               edgecolor='black', linewidth=linewidth,
               capsize=2, error_kw={'linewidth': linewidth}, label=method)

    ax.set_ylabel(ylabel)
    ax.set_xlim(x_lims)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    
    if show_xticks:
        ax.set_xticks(centers)
        ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticks(centers)
        ax.set_xticklabels([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        final_handles = []
        final_labels = []
        if legend_include_esm2 and "ESM2" in by_label:
            final_handles.append(by_label["ESM2"])
            final_labels.append("ESM2")
        for m in methods:
            if m in by_label:
                final_handles.append(by_label[m])
                final_labels.append(m)
        ax.legend(final_handles, final_labels, fontsize=font_size, loc='lower center', 
                  bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)

def plot_nodes_per_layer(ax, df, title_label="", show_legend=False, xaxis=None):
    """
    Plots a grouped bar chart of Average Node Count per Layer (1-LAYERS).
    """
    print(f"\n--- {title_label} Node Statistics ---")
    layer_indices = range(LAYERS) # Layers 1-LAYERS (Index 0-LAYERS-1)
    x = np.arange(len(layer_indices))
    num_methods = len(ORDERED_METHODS)
    total_width = 0.8
    bar_width = total_width / num_methods
    offsets = (np.arange(num_methods) - (num_methods - 1) / 2) * bar_width
    
    for i, method in enumerate(ORDERED_METHODS):
        subset = df[df['Model'] == method]
        if subset.empty:
            continue
            
        means, stds = [], []
        # Calculate per-layer stats
        for l in layer_indices:
            col = f"Layer {l+1}"
            if col in subset.columns:
                means.append(subset[col].mean())
                stds.append(subset[col].std())
            else:
                means.append(0)
                stds.append(0)
        
        # Calculate Total Nodes for print logging
        layer_cols = [f"Layer {l+1}" for l in layer_indices if f"Layer {l+1}" in subset.columns]
        if layer_cols:
            total_nodes = subset[layer_cols].sum(axis=1)
            print(f"[{method}] Avg Total Nodes: {total_nodes.mean():.2f} ± {total_nodes.std():.2f}")

        lower_errs = [min(m, s) for m, s in zip(means, stds)]
        yerr_asym = [lower_errs, stds]

        ax.bar(x + offsets[i], means, yerr=yerr_asym, width=bar_width,
               color=COLOR_MAP[method], edgecolor='black', linewidth=linewidth,
               capsize=2, error_kw={'linewidth': linewidth},
               label=method)

    ax.set_ylabel("Average Latent Count")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l+1}" for l in layer_indices]) 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if xaxis:
        ax.set_xlabel(xaxis, fontsize=font_size)
    
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        final_handles = []
        final_labels = []
        for m in ORDERED_METHODS:
            if m in by_label:
                final_handles.append(by_label[m])
                final_labels.append(m)
        ax.legend(final_handles, final_labels, fontsize=font_size, loc='lower center', 
                  bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)

def make_fig1_performance(df_fam, df_func):
    """2x1: Family F1 (Top) + Function Spearman (Bottom), excluding windowed"""
    fig_width = 183 * mm
    fig_height = 150 * mm 
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    
    # Filter out windowed for this plot
    df_fam_filtered = df_fam[df_fam['Model'] != "ProtoMech (windowed)"].copy()
    df_func_filtered = df_func[df_func['Model'] != "ProtoMech (windowed)"].copy() if df_func is not None else None
    
    print('---')
    print('Fig 1: Family F1 Score')
    draw_grouped_bars(axes[0], df_fam_filtered, metric_key="f1", show_legend=True, 
                      ylabel="Protein family F1 score",
                      ylim=(0, 1.2), yticks=[0, 0.5, 1.0])
    
    print('---')
    print('Fig 1: Function Spearman')
    if df_func_filtered is not None and not df_func_filtered.empty:
        draw_grouped_bars(axes[1], df_func_filtered, metric_key="f1", show_legend=False, 
                          ylabel="Function Spearman ρ",
                          ylim=(0, 0.8), yticks=[0, 0.4, 0.8])
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4) 
    plt.savefig(f"{PLOTS_DIR}/supp_circuit_performance.pdf")
    plt.close()

def make_fig2_nodes(df_fam, df_func):
    """2x1: Family Nodes (Top) + Function Nodes (Bottom)"""
    fig_width = 183 * mm
    fig_height = 120 * mm
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    plot_nodes_per_layer(axes[0], df_fam, title_label="Protein Family", show_legend=True)
    plot_nodes_per_layer(axes[1], df_func, title_label="Function Prediction", show_legend=False, xaxis='Layer')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(f"{PLOTS_DIR}/supp_nodes_distribution.pdf")
    plt.close()

def make_fig3_splits(df_func):
    """4x1: Function splits (Multiples, Random, Contiguous, Modulo)"""
    fig_width = 183 * mm
    fig_height = 200 * mm
    fig, axes = plt.subplots(4, 1, figsize=(fig_width, fig_height))
    
    func_ylim = (0, 0.8)
    func_yticks = [0, 0.4, 0.8]

    print('---')
    print('Fig 3: Multiples')
    df_multi = df_func[df_func["Type"] == "multiples"]
    draw_grouped_bars(axes[0], df_multi, metric_key="f1", show_legend=True, 
                      ylabel="Multiple Spearman ρ",
                      ylim=func_ylim, yticks=func_yticks)
    print('---')

    print('Fig 3: Single Random')
    df_rand = df_func[(df_func["Type"] == "single") & (df_func["Split"] == "random")]
    draw_grouped_bars(axes[1], df_rand, metric_key="f1", show_legend=False, 
                      ylabel="Single (random) Spearman ρ",
                      ylim=func_ylim, yticks=func_yticks)
    print('---')

    print('Fig 3: Single Contiguous')
    df_cont = df_func[(df_func["Type"] == "single") & (df_func["Split"] == "contiguous")]
    draw_grouped_bars(axes[2], df_cont, metric_key="f1", show_legend=False, 
                      ylabel="Single (contiguous) Spearman ρ",
                      ylim=func_ylim, yticks=func_yticks)
    print('---')

    print('Fig 3: Single Modulo')
    df_mod = df_func[(df_func["Type"] == "single") & (df_func["Split"] == "modulo")]
    draw_grouped_bars(axes[3], df_mod, metric_key="f1", show_legend=False, 
                      ylabel="Single (modulo) Spearman ρ",
                      ylim=func_ylim, yticks=func_yticks)
    print('---')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(f"{PLOTS_DIR}/supp_splits_performance.pdf")
    plt.close()

def make_fig4_low_f1(df_fam):
    """
    Fig 4: Family F1, but ONLY for families where original ESM2 F1 < MAX_LOW_F1 (excluding windowed).
    """
    fam_means = df_fam.groupby("family")["clean"].mean()
    low_fams = fam_means[fam_means < MAX_LOW_F1].index

    df_subset = df_fam[df_fam["family"].isin(low_fams)].copy()
    # Filter out windowed for this plot
    df_subset = df_subset[df_subset['Model'] != "ProtoMech (windowed)"].copy()

    if df_subset.empty:
        print(f"Warning: No families found with Original F1 < {MAX_LOW_F1}. Skipping Fig 4.")
        return

    print(f"Found {len(low_fams)} families with Original F1 < {MAX_LOW_F1}")

    fig_width = 183 * mm
    fig_height = 60 * mm 
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    print('---')
    print(f'Fig 4: Family F1 (Low Original < {MAX_LOW_F1})')
    
    draw_grouped_bars(ax, df_subset, metric_key="f1", show_legend=True,
                      ylabel=f"Protein family F1 Score",
                      ylim=(0, 0.8), 
                      yticks=[0, 0.2, 0.4, 0.6, 0.8])

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/supp_low_f1_performance.pdf")
    plt.close()

def make_fig5_windowed():
    """
    Fig 5: Compare ProtoMech (sequential), ProtoMech (windowed), and PLT (sequential).
    Includes ESM2 baseline performance.
    """
    df_fam_filtered = load_family_data(method_map=METHOD_MAP_FIG5, min_seq=MIN_SEQUENCES)
    
    if df_fam_filtered.empty:
        print("Warning: No data found for windowed comparison. Skipping Fig 5.")
        return
    
    fig_width = 183 * mm
    fig_height = 60 * mm
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    print('---')
    print('Fig 5: Windowed Performance Comparison')
    
    draw_grouped_bars(ax, df_fam_filtered, metric_key="f1", show_legend=True,
                      ylabel="Protein family F1 score",
                      ylim=(0, 1.2), 
                      yticks=[0, 0.5, 1.0],
                      methods=ORDERED_METHODS_FIG5,
                      color_map=COLOR_MAP_FIG5,
                      legend_include_esm2=False)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/supp_windowed_performance.pdf")
    plt.close()

def main():
    Path(PLOTS_DIR).mkdir(exist_ok=True)
    
    print("Loading Family Data...")
    df_fam = load_family_data(min_seq=MIN_SEQUENCES)
    print(f"Loaded {len(df_fam)} family records.")
    
    print("Loading Function Data...")
    df_func = load_function_data(min_clean=MIN_CLEAN_SPEARMAN)
    print(f"Loaded {len(df_func)} function records.")
    
    if df_fam.empty and df_func.empty:
        print("Error: Missing all data. Check directories.")
        return

    # Generate Figures
    make_fig1_performance(df_fam, df_func)
    make_fig5_windowed()
    make_fig2_nodes(df_fam, df_func) 
    make_fig3_splits(df_func)
    
    df_fam_low = load_family_data(min_clean=MIN_LOW_F1, max_clean=MAX_LOW_F1)
    print('Low F1 Family Data:')
    print(len(df_fam_low))
    make_fig4_low_f1(df_fam_low)
    
    low_clean = df_func[(df_func["clean"] <= 0.2)]
    if not low_clean.empty:
        count_better = (low_clean["recovered"] > low_clean["clean"]).sum()
        total_low = len(low_clean)
        print(f"\nIn function data, {count_better} out of {total_low} cases with clean <= 0.2 had recovered > clean.")    
    
    print("All plots generated.")

if __name__ == "__main__":
    main()