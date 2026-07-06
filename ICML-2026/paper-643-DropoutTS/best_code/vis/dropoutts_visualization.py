import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

"""
Visualization script for comparing model performance before and after applying DropoutTS.
Example usage:
    python dropoutts_visualization.py --before <before_dir> --after <after_dir>
"""
# ==========================================
# 1. Academic Style Configuration
# ==========================================
def set_academic_style():
    """Configure ICML academic style (unified format)"""
    sns.set_theme(style="whitegrid", font_scale=1.0, rc={
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.edgecolor": ".15",
        "grid.linestyle": "--",
        "axes.linewidth": 1.2,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.title_fontsize": 10,
        "figure.dpi": 300,
        "savefig.bbox": "tight"
    })

# ==========================================
# 2. Data Loading Logic
# ==========================================
def load_test_results(result_dir):
    result_dir = Path(result_dir)
    test_results_dir = result_dir / "test_results"
    
    cfg_file = result_dir / "cfg.json"
    input_len, output_len, num_features = 96, 96, 7
    if cfg_file.exists():
        with open(cfg_file, 'r') as f:
            cfg = json.load(f)
            model_cfg = cfg.get("model_config", {})
            input_len = model_cfg.get("input_len", 96)
            output_len = model_cfg.get("output_len", 96)
            num_features = model_cfg.get("num_features", 7)
    
    def load_npy(name):
        path = test_results_dir / f"{name}.npy"
        try:
            data = np.load(path)
            # Ensure correct shape (B, L, C)
            if len(data.shape) == 2:  # May be flattened in some cases
                L = input_len if name == "inputs" else output_len
                B = len(data) // (L * num_features)
                data = data.reshape(B, L, num_features)
            return data
        except:
            data_mm = np.memmap(path, mode='r', dtype='float32')
            L = input_len if name == "inputs" else output_len
            B = len(data_mm) // (L * num_features)
            return data_mm.reshape(B, L, num_features)

    return load_npy("inputs"), load_npy("prediction"), load_npy("targets")

# ==========================================
# 3. Core Plotting Function (with filtering logic for Imp > 30%)
# ==========================================
def plot_high_imp_comparison(before_dir, after_dir, output_path, feat_idx=0, threshold=0.3):
    set_academic_style()
    
    print("Loading data...")
    in_b, pred_b, tar_b = load_test_results(before_dir)
    in_a, pred_a, tar_a = load_test_results(after_dir)
    
    # --- Filtering logic starts ---
    print(f"Filtering samples with Improvement > {threshold*100:.0f}%...")
    
    # Calculate MAE for all samples on this feature
    # Shape: (B, L_out) -> (B,)
    all_mae_base = np.mean(np.abs(pred_b[:, :, feat_idx] - tar_b[:, :, feat_idx]), axis=1)
    all_mae_ours = np.mean(np.abs(pred_a[:, :, feat_idx] - tar_a[:, :, feat_idx]), axis=1)
    
    # Calculate improvement ratio (avoid division by zero)
    all_imps = (all_mae_base - all_mae_ours) / (all_mae_base + 1e-9)
    
    # Find indices that meet the criteria
    qualified_indices = np.where(all_imps > threshold)[0]
    
    if len(qualified_indices) < 4:
        print(f"Warning: Only {len(qualified_indices)} samples found. Relaxing criteria to top 4.")
        indices = np.argsort(all_imps)[-4:]  # If insufficient, take top 4 best performers
    else:
        indices = np.random.choice(qualified_indices, 4, replace=False)
    # --- Filtering logic ends ---

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.0), sharex=True, constrained_layout=True)
    axes = axes.flatten()
    
    colors = {'in': '#7F8C8D', 'gt': '#27AE60', 'base': '#D62728', 'ours': '#1F77B4'}
    L_in, L_out = in_b.shape[1], tar_b.shape[1]
    t_in = np.arange(L_in)
    t_out = np.arange(L_in, L_in + L_out)

    for i, idx in enumerate(indices):
        ax = axes[i]
        seq_in, seq_gt = in_b[idx, :, feat_idx], tar_b[idx, :, feat_idx]
        seq_base, seq_ours = pred_b[idx, :, feat_idx], pred_a[idx, :, feat_idx]
        
        # Plotting
        ax.plot(t_in, seq_in, color=colors['in'], label='Input', lw=1.5, alpha=0.5)
        ax.plot(t_out, seq_gt, color=colors['gt'], label='Ground Truth', lw=2.0)
        ax.plot(t_out, seq_base, color=colors['base'], label='Baseline', lw=2.0, ls='--')
        ax.plot(t_out, seq_ours, color=colors['ours'], label='Ours (+DropoutTS)', lw=2.0, ls='-')
        
        ax.axvline(x=L_in-1, color='#34495E', linestyle=':', lw=1.2)
        ax.set_title(f"({chr(97+i)}) Sample #{idx} | Imp: {all_imps[idx]*100:+.1f}%", fontweight='bold', fontsize=11, loc='center')
        ax.grid(True, linestyle='--', alpha=0.25)
        
        if i >= 2: ax.set_xlabel("Time Steps")
        if i % 2 == 0: ax.set_ylabel("Normalized Value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.06), 
               framealpha=0.9, edgecolor='gray')

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / 'dropoutts_visualization.pdf', bbox_inches='tight', dpi=300)
    print(f"Success! Visualized samples: {indices}")
    print(f"Results saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=str, required=True)
    parser.add_argument('--after', type=str, required=True)
    parser.add_argument('--output', type=str, default='.')
    parser.add_argument('--feat', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.3, help='Min improvement threshold (default 0.3)')
    
    args = parser.parse_args()
    plot_high_imp_comparison(args.before, args.after, args.output, args.feat, args.threshold)