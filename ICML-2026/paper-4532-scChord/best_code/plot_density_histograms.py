"""
Plot density histograms of ground truth and predicted values for each protein across different methods.
Unified transformation to the ours method's log-normalized space for comparison.
Reference: log_normalize scheme from view_h5ad.py: log(count / total * 1e4 + 1)
"""
import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gmean, gaussian_kde, ks_2samp
import seaborn as sns

# Set font
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings("ignore")

# ============================================
# Data Preprocessing Functions (Reference: ComputeMetrics.ipynb)
# ============================================

def log_normalize(data: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """
    Log normalization: log(count / total * target_sum + 1)
    Used for ours method.
    """
    data = np.asarray(data)
    row_sums = np.sum(data, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-8
    return np.log(data / row_sums * target_sum + 1.)


def clr_transform(data):
    """
    CLR transform (Centered Log-Ratio)
    Used for cTp_net and Seurat methods.
    """
    data = np.array(data, dtype=np.float64)
    gm = gmean(data + 1.0, axis=0)
    return np.log((data + 1.0) / gm)


def scale_like_scanpy(data):
    """
    Simulate scanpy's standardization pipeline: normalize_total + log1p + scale
    Used for sciPENN ground truth data processing.
    """
    arr = np.array(data, dtype=np.float32)
    # normalize_total (per-cell)
    row_sums = np.sum(arr, axis=1, keepdims=True)
    arr = arr / (row_sums + 1e-8) * 1e4
    # log1p
    np.log1p(arr, out=arr)
    # scale (feature-wise z-score)
    means = np.mean(arr, axis=0, keepdims=True)
    stds = np.std(arr, axis=0, keepdims=True)
    arr = (arr - means) / (stds + 1e-8)
    return arr


def scale_only(data):
    """
    Only perform feature-wise z-score standardization.
    Used for sciPENN prediction data processing.
    """
    arr = np.array(data, dtype=np.float32)
    means = np.mean(arr, axis=0, keepdims=True)
    stds = np.std(arr, axis=0, keepdims=True)
    return (arr - means) / (stds + 1e-8)


def scvaeit_transform(data):
    """
    scVAEIT transform: log(Y/sum*1e4+1)
    """
    Y = np.array(data, dtype=np.float64)
    return np.log(Y / np.sum(Y, axis=1, keepdims=True) * 1e4 + 1.)


def convert_to_ours_space(data, method_name):
    """
    Convert prediction data from different methods to ours method's log-normalized space.
    Reference: standardization scheme from ComputeMetrics.ipynb
    
    According to ComputeMetrics.ipynb:
    - ours: prediction data is already in log-normalized space log(count/total*1e4+1)
    - scVAEIT: prediction data is already in log-normalized space (same transform as ours)
    - cTp_net, Seurat: prediction data is in CLR space, needs reverse conversion
    - sciPENN: prediction data is in z-score standardized space, needs reverse conversion
    - others (default): prediction data is raw counts, directly apply log normalization
    
    Args:
        data: Prediction data [n_cells, n_proteins]
        method_name: Method name
    
    Returns:
        Converted data (in log-normalized space)
    """
    data = np.asarray(data, dtype=np.float64)
    
    if method_name == 'ours':
        # ours method: prediction data is already in log-normalized space
        # Check if already in log space (max value typically <20)
        if np.max(data) < 20 and np.min(data) >= 0:
            return data.astype(np.float32)
        else:
            # If not, assume raw counts and apply log normalization
            return log_normalize(data, target_sum=1e4).astype(np.float32)
    
    elif method_name == 'scVAEIT':
        # scVAEIT: According to ComputeMetrics.ipynb, prediction data is already in log-normalized space
        # scvaeit_transform = log(Y/sum*1e4+1), same as log_normalize
        # Use directly (if appears to be in log space)
        if np.max(data) < 20 and np.min(data) >= 0:
            return data.astype(np.float32)
        else:
            # If not, apply log normalization
            data = np.maximum(data, 0)
            return log_normalize(data, target_sum=1e4).astype(np.float32)
    
    # elif method_name in ['cTp_net', 'Seurat']:
    elif method_name == 'cTp_net':
        # cTp_net and Seurat: prediction data is in CLR space (may have negative values)
        # CLR space: log((x+1)/gm), needs reverse conversion
        # Since gm is unknown, we try: if negative values exist, exp then log normalize
        if np.any(data < 0):
            # CLR space: try reverse conversion (not completely accurate, but usable for visualization)
            # exp(CLR) = (x+1)/gm, so (x+1) = exp(CLR) * gm
            # Since gm is unknown, we assume exp(CLR) represents relative counts
            data_exp = np.exp(data)
            # Re-normalize to raw counts space (assumed)
            data_exp = np.maximum(data_exp, 1e-8)
            # Apply log normalization
            return log_normalize(data_exp, target_sum=1e4).astype(np.float32)
        else:
            # If all positive, assume raw counts or already close to log space
            data = np.maximum(data, 0)
            if np.max(data) < 20:
                return data.astype(np.float32)
            else:
                return log_normalize(data, target_sum=1e4).astype(np.float32)
    
    elif method_name == 'sciPENN' or method_name == 'Seurat':
        # sciPENN: prediction data is in z-score standardized space (may have negative values, mean=0, std=1)
        # Needs reverse conversion, but z-score cannot be fully reversed (missing original mean and std)
        # Assume prediction data represents standardized relative values, shift to non-negative then log normalize
        if np.any(data < 0):
            # z-score standardized data, shift to non-negative
            data = data - np.min(data) + 1e-8
        else:
            data = np.maximum(data, 1e-8)
        # Apply log normalization
        return log_normalize(data, target_sum=1e4).astype(np.float32)
    
    else:
        # default: totalVI, scArches, Dengkw, Liger
        # Prediction data from these methods are typically raw counts, directly apply log normalization
        data = np.maximum(data, 0)
        if np.all(data < 1e-8):
            data = data + 1e-8
        return log_normalize(data, target_sum=1e4).astype(np.float32)


def preprocess_for_visualization(pred, true, method_name):
    """
    Unify all methods' data to ours method's log-normalized space.
    Reference: scheme from view_h5ad.py
    """
    # Ground truth: uniformly use log normalization (ours method's preprocessing)
    true_proc = log_normalize(true, target_sum=1e4).astype(np.float32)
    
    # Prediction data: convert to ours' log-normalized space
    pred_proc = convert_to_ours_space(pred, method_name)
    
    return pred_proc, true_proc


# ============================================
# Main Program
# ============================================

def main():
    # Data path
    data_path = "./results/"
    save_path = os.path.join(data_path, "figures")
    os.makedirs(save_path, exist_ok=True)
    
    # Load ground truth data
    print("Loading data...")
    true_data = np.load(os.path.join(data_path, "true_data_dataset1.npy"))
    
    # Load prediction data from each method
    pred_dict = {
        'ours': np.load(os.path.join(data_path, "pred_data_ours_dataset1.npy")),
        'totalVI': np.load(os.path.join(data_path, "pred_data_totalvi_dataset1.npy")),
        'scArches': np.load(os.path.join(data_path, "pred_data_scarches_dataset1.npy")),
        'Dengkw': np.load(os.path.join(data_path, "pred_data_Dengkw_dataset1.npy")),
        'cTp_net': np.load(os.path.join(data_path, "pred_data_ctpnet_dataset1.npy")),
        'Liger': np.load(os.path.join(data_path, "pred_data_liger_dataset1.npy")),
        'sciPENN': np.load(os.path.join(data_path, "pred_data_sciPENN_dataset1.npy")),
        'scVAEIT': np.load(os.path.join(data_path, "pred_data_scvaeit_dataset1.npy")),
        'Seurat': np.load(os.path.join(data_path, "pred_data_seurat_dataset1.npy")),

    }
    
    method_names = list(pred_dict.keys())
    n_proteins = true_data.shape[1]
    
    # Protein name list
    protein_names = ['CD3', 'CD4', 'CD8', 'CD45RA', 'CD56', 'CD16', 'CD10', 'CD11c', 
                     'CD14', 'CD19', 'CD34', 'CCR5', 'CCR7']
    
    # Ensure protein name count matches data
    if len(protein_names) != n_proteins:
        print(f"Warning: protein name count ({len(protein_names)}) does not match data protein count ({n_proteins})!")
        print(f"Using default names: Protein_1, Protein_2, ...")
        protein_names = [f'Protein_{i+1}' for i in range(n_proteins)]
    
    print(f"Data shape: {true_data.shape}")
    print(f"Number of proteins: {n_proteins}")
    print(f"Number of methods: {len(method_names)}")
    print(f"Protein names: {protein_names}")
    
    # Color configuration
    my_pal = {
        'ours': '#FF6B6B',
        'totalVI': '#E6D885',
        'scArches': '#F1C67F',
        'Dengkw': '#E7A365',
        'cTp_net': '#7CC38A',
        'Liger': '#65AADD',
        'sciPENN': '#84a4e8',
        'scVAEIT': "#bff5d6",
        'Seurat': '#7DC9C4'
        
    }
    
    # Plot density histograms for each protein
    print("\nGenerating density histograms...")
    for protein_idx in range(n_proteins):
        protein_name = protein_names[protein_idx]
        print(f"  Processing protein {protein_idx + 1}/{n_proteins}: {protein_name}...")
        
        # Create 3x3 subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        # Plot subplot for each method
        for idx, method_name in enumerate(method_names):
            ax = axes[idx]
            
            # Get prediction data and preprocess
            pred = pred_dict[method_name]
            pred_proc, true_proc = preprocess_for_visualization(pred, true_data, method_name)
            
            # Extract current protein data
            pred_protein = pred_proc[:, protein_idx]
            true_protein = true_proc[:, protein_idx]
            
            # Remove NaN and infinite values
            pred_protein = pred_protein[np.isfinite(pred_protein)]
            true_protein = true_protein[np.isfinite(true_protein)]
            
            # Calculate KS distance (Kolmogorov-Smirnov statistic)
            # KS statistic measures the maximum difference between two cumulative distribution functions
            if len(pred_protein) > 0 and len(true_protein) > 0:
                ks_statistic, _ = ks_2samp(true_protein, pred_protein)
            else:
                ks_statistic = np.nan
            
            # Determine x-axis range (using range of all data)
            x_min = min(np.min(pred_protein), np.min(true_protein))
            x_max = max(np.max(pred_protein), np.max(true_protein))
            x_range = x_max - x_min
            bins = 50
            
            # Plot density histogram
            # Use density=False (counts) instead of density=True, consistent with view_h5ad.py
            # Then use log scale for better visualization
            counts_true, bins_true, _ = ax.hist(true_protein, bins=bins, alpha=0.5, color='gray', 
                   label='True', density=False, edgecolor='black', linewidth=0.3)
            counts_pred, bins_pred, _ = ax.hist(pred_protein, bins=bins, alpha=0.5, color=my_pal[method_name],
                   label='Predicted', density=False, edgecolor='black', linewidth=0.3)
            
            # Overlay KDE curves for clearer distribution visualization
            # Convert KDE density values to counts for same-scale comparison with histogram
            try:
                bin_width = (x_max - x_min) / bins
                
                if len(true_protein) > 10:
                    kde_true = gaussian_kde(true_protein)
                    x_plot = np.linspace(x_min, x_max, 200)
                    kde_density = kde_true(x_plot)
                    # Convert density to counts: density * sample_count * bin_width
                    kde_counts = kde_density * len(true_protein) * bin_width
                    # Avoid 0 or negative values (log scale requirement)
                    kde_counts = np.maximum(kde_counts, 1e-3)
                    ax.plot(x_plot, kde_counts, color='black', linewidth=2, 
                           linestyle='--', alpha=0.8, label='True KDE')
                
                if len(pred_protein) > 10:
                    kde_pred = gaussian_kde(pred_protein)
                    x_plot = np.linspace(x_min, x_max, 200)
                    kde_density = kde_pred(x_plot)
                    kde_counts = kde_density * len(pred_protein) * bin_width
                    kde_counts = np.maximum(kde_counts, 1e-3)
                    ax.plot(x_plot, kde_counts, color=my_pal[method_name], 
                           linewidth=2, linestyle='-', alpha=0.8, label='Pred KDE')
            except:
                pass  # If KDE fails, only show histogram
            
            # Set y-axis to log scale (reference: view_h5ad.py)
            ax.set_yscale('log')
            
            # Set title and labels (including KS distance)
            ks_str = f'KS={ks_statistic:.3f}' if not np.isnan(ks_statistic) else 'KS=NaN'
            ax.set_title(f'{method_name} ({ks_str})', fontsize=18, fontweight='bold')
            ax.set_xlabel('Value(lognormalized)', fontsize=16)
            ax.set_ylabel('Count (log scale)', fontsize=16)
            ax.legend(fontsize=12, loc='upper left')
            ax.grid(True, alpha=0.3, linestyle='--', which='both')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Ensure y-axis minimum is greater than 0 (log scale requirement)
            # Find minimum non-zero count value
            nonzero_counts_true = counts_true[counts_true > 0]
            nonzero_counts_pred = counts_pred[counts_pred > 0]
            if len(nonzero_counts_true) > 0 and len(nonzero_counts_pred) > 0:
                y_min = min(np.min(nonzero_counts_true), np.min(nonzero_counts_pred))
                ax.set_ylim(bottom=max(y_min * 0.5, 0.1))
            elif len(nonzero_counts_true) > 0:
                y_min = np.min(nonzero_counts_true)
                ax.set_ylim(bottom=max(y_min * 0.5, 0.1))
            elif len(nonzero_counts_pred) > 0:
                y_min = np.min(nonzero_counts_pred)
                ax.set_ylim(bottom=max(y_min * 0.5, 0.1))
            else:
                ax.set_ylim(bottom=0.1)
            
            # Auto-adjust if data range is too small
            if x_range < 1e-10:
                ax.set_xlim(x_min - 0.1, x_max + 0.1)
        
        # Set overall title (using protein name)
        fig.suptitle(f'{protein_name} - Distribution Comparison', 
                    fontsize=20, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save figure (filename includes protein name and index)
        save_file = os.path.join(save_path, f'density_histogram_{protein_name}_({protein_idx + 1}).png')
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"\nDone! Figures saved to: {save_path}")
    print(f"Generated density histograms for {n_proteins} proteins")


if __name__ == '__main__':
    main()
