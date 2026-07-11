import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import os

def analyze_roi_vs_nonroi(sequences, scores, spans, name):
    """Analyze conservation patterns inside and outside ROIs."""
    roi_pred_values = []
    roi_true_values = []
    nonroi_pred_values = []
    nonroi_true_values = []
    
    for i in range(len(spans)):
        start, end = spans[i]
        
        # Get true scores with proper length
        true_scores = scores[i]
        if len(true_scores) > len(sequences[i]):
            true_scores = true_scores[1:-1]
            
        if len(true_scores) != len(sequences[i]):
            continue
            
        # Create mask for ROI
        mask = np.zeros(len(sequences[i]), dtype=bool)
        mask[start:end] = True
        
        # Collect ROI values
        roi_pred_values.extend(sequences[i][mask])
        roi_true_values.extend(true_scores[mask])
        
        # Collect non-ROI values
        nonroi_pred_values.extend(sequences[i][~mask])
        nonroi_true_values.extend(true_scores[~mask])
    
    # Convert to arrays
    roi_pred = np.array(roi_pred_values)
    roi_true = np.array(roi_true_values)
    nonroi_pred = np.array(nonroi_pred_values)
    nonroi_true = np.array(nonroi_true_values)
    
    print(f"\nAnalysis for {name}:")
    print("ROI regions:")
    print(f"  Predicted mean: {np.mean(roi_pred):.3f} ± {np.std(roi_pred):.3f}")
    print(f"  True mean: {np.mean(roi_true):.3f} ± {np.std(roi_true):.3f}")
    print(f"  Correlation: {stats.pearsonr(roi_pred, roi_true)[0]:.3f}")
    
    print("\nNon-ROI regions:")
    print(f"  Predicted mean: {np.mean(nonroi_pred):.3f} ± {np.std(nonroi_pred):.3f}")
    print(f"  True mean: {np.mean(nonroi_true):.3f} ± {np.std(nonroi_true):.3f}")
    print(f"  Correlation: {stats.pearsonr(nonroi_pred, nonroi_true)[0]:.3f}")
    
    # Calculate conservation enrichment
    pred_enrichment = np.mean(roi_pred) - np.mean(nonroi_pred)
    true_enrichment = np.mean(roi_true) - np.mean(nonroi_true)
    print(f"\nConservation enrichment (ROI - nonROI):")
    print(f"  Predicted: {pred_enrichment:.3f}")
    print(f"  True: {true_enrichment:.3f}")
    
    return (roi_pred, roi_true, nonroi_pred, nonroi_true)

def plot_roi_comparison(roi_data1, roi_data2, output_path, name1, name2):
    """Plot comparison of ROI vs non-ROI conservation patterns."""
    roi_pred1, roi_true1, nonroi_pred1, nonroi_true1 = roi_data1
    roi_pred2, roi_true2, nonroi_pred2, nonroi_true2 = roi_data2
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Conservation levels
    data = {
        f'{name1} ROI': np.mean(roi_pred1),
        f'{name1} non-ROI': np.mean(nonroi_pred1),
        f'{name2} ROI': np.mean(roi_pred2),
        f'{name2} non-ROI': np.mean(nonroi_pred2)
    }
    true_data = {
        f'{name1} ROI': np.mean(roi_true1),
        f'{name1} non-ROI': np.mean(nonroi_true1),
        f'{name2} ROI': np.mean(roi_true2),
        f'{name2} non-ROI': np.mean(nonroi_true2)
    }
    
    x = np.arange(len(data))
    width = 0.35
    
    axes[0].bar(x - width/2, data.values(), width, label='Predicted')
    axes[0].bar(x + width/2, true_data.values(), width, label='True')
    
    axes[0].set_ylabel('Mean Conservation Score')
    axes[0].set_title('Conservation Levels')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(data.keys(), rotation=45)
    axes[0].legend()
    
    # Plot 2: Correlation comparison
    corrs = {
        f'{name1} ROI': stats.pearsonr(roi_pred1, roi_true1)[0],
        f'{name1} non-ROI': stats.pearsonr(nonroi_pred1, nonroi_true1)[0],
        f'{name2} ROI': stats.pearsonr(roi_pred2, roi_true2)[0],
        f'{name2} non-ROI': stats.pearsonr(nonroi_pred2, nonroi_true2)[0]
    }
    
    axes[1].bar(corrs.keys(), corrs.values())
    axes[1].set_ylabel('Correlation Coefficient')
    axes[1].set_title('Prediction-Truth Correlations')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_sequence_average_correlations(sequences, scores, name, output_dir):
    """Create scatter plot with marginal distributions of average sequence conservation."""
    # Calculate average conservation per sequence
    seq_pred_means = []
    seq_true_means = []
    
    for i in range(len(sequences)):
        true_scores = scores[i]
        if len(true_scores) > len(sequences[i]):
            true_scores = true_scores[1:-1]
            
        if len(true_scores) != len(sequences[i]):
            continue
        
        seq_pred_means.append(np.mean(sequences[i]))
        seq_true_means.append(np.mean(true_scores))
    
    # Convert to numpy arrays
    seq_pred_means = np.array(seq_pred_means)
    seq_true_means = np.array(seq_true_means)
    
    # Calculate correlation
    corr, p_val = stats.pearsonr(seq_true_means, seq_pred_means)
    
    # Create figure
    plt.figure(figsize=(12, 12))
    
    # Create grid for plots
    gs = plt.GridSpec(3, 3)
    ax_scatter = plt.subplot(gs[1:, :-1])
    ax_hist_x = plt.subplot(gs[0, :-1])
    ax_hist_y = plt.subplot(gs[1:, -1])
    
    # Scatter plot
    ax_scatter.scatter(seq_true_means, seq_pred_means, alpha=0.6)
    
    # Add regression line
    z = np.polyfit(seq_true_means, seq_pred_means, 1)
    p = np.poly1d(z)
    x_range = np.linspace(min(seq_true_means), max(seq_true_means), 100)
    ax_scatter.plot(x_range, p(x_range), "r--", alpha=0.8, 
                   label=f'Regression line\ny = {z[0]:.3f}x + {z[1]:.3f}')
    
    # Add y=x line
    lims = [
        np.min([ax_scatter.get_xlim()[0], ax_scatter.get_ylim()[0]]),
        np.max([ax_scatter.get_xlim()[1], ax_scatter.get_ylim()[1]]),
    ]
    ax_scatter.plot(lims, lims, 'k--', alpha=0.5, label='y = x')
    
    # Add correlation coefficient
    ax_scatter.text(0.05, 0.95, 
                   f'r = {corr:.3f}\np = {p_val:.2e}\nN = {len(seq_true_means):,}', 
                   transform=ax_scatter.transAxes,
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Histograms with fewer bins
    ax_hist_x.hist(seq_true_means, bins=30, alpha=0.7)
    ax_hist_y.hist(seq_pred_means, bins=30, orientation='horizontal', alpha=0.7)
    
    # Labels
    ax_scatter.set_xlabel('Mean True Conservation Score')
    ax_scatter.set_ylabel('Mean Predicted Conservation Score')
    ax_hist_x.set_ylabel('Count')
    ax_hist_y.set_xlabel('Count')
    
    # Remove unnecessary labels
    ax_hist_x.set_xticklabels([])
    ax_hist_y.set_yticklabels([])
    
    # Add title
    plt.suptitle(f'Sequence-Average Conservation Correlation - {name}', y=0.95)
    
    # Add legend
    ax_scatter.legend()
    
    # Add grid
    ax_scatter.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sequence_average_correlation_{name}.png'))
    plt.close()
    
    return corr, p_val, len(seq_true_means)


def plot_combined_sequence_correlations(npz_files, names, output_dir):
    """
    Create combined scatter plot with marginal distributions for multiple datasets.
    
    Args:
        npz_files: List of paths to npz files
        names: List of dataset names (UCNE, Flanking, Random)
        output_dir: Directory to save the plot
    """
    colors = {'UCNE': '#1f77b4', 'Flanking': '#2ca02c', 'Random': '#ff7f0e'}
    
    # Create figure with gridspec
    fig = plt.figure(figsize=(12, 12))
    gs = plt.GridSpec(3, 3)
    ax_scatter = plt.subplot(gs[1:, :-1])
    ax_hist_x = plt.subplot(gs[0, :-1])
    ax_hist_y = plt.subplot(gs[1:, -1])
    
    # Storage for all data to determine common limits
    all_true_means = []
    all_pred_means = []
    
    # Process each dataset
    for npz_file, name in zip(npz_files, names):
        # Load and process data
        data = np.load(npz_file, allow_pickle=True)
        seq_pred_means = []
        seq_true_means = []
        
        for i in range(len(data['sequences'])):
            true_scores = data['scores'][i]
            if len(true_scores) > len(data['sequences'][i]):
                true_scores = true_scores[1:-1]
                
            if len(true_scores) != len(data['sequences'][i]):
                continue
            
            seq_pred_means.append(np.mean(data['sequences'][i]))
            seq_true_means.append(np.mean(true_scores))
        
        seq_pred_means = np.array(seq_pred_means)
        seq_true_means = np.array(seq_true_means)
        
        all_true_means.extend(seq_true_means)
        all_pred_means.extend(seq_pred_means)
        
        # Calculate correlation
        corr, p_val = stats.pearsonr(seq_true_means, seq_pred_means)
        
        # Scatter plot
        scatter = ax_scatter.scatter(seq_true_means, seq_pred_means, 
                                   c=colors[name], alpha=0.6, 
                                   label=f'{name} (r={corr:.3f})')
        
        # Add histograms on margins
        ax_hist_x.hist(seq_true_means, bins=50, alpha=0.6, color=colors[name],
                      density=True, label=name)
        ax_hist_y.hist(seq_pred_means, bins=50, alpha=0.6, color=colors[name],
                      density=True, orientation='horizontal', label=name)
        
        # Add regression line
        z = np.polyfit(seq_true_means, seq_pred_means, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(seq_true_means), max(seq_true_means), 100)
        ax_scatter.plot(x_range, p(x_range), "--", color=colors[name], alpha=0.8)
    
    # Set common limits based on all data
    all_min = min(min(all_true_means), min(all_pred_means))
    all_max = max(max(all_true_means), max(all_pred_means))
    ax_scatter.set_xlim(all_min, all_max)
    ax_scatter.set_ylim(all_min, all_max)
    
    # Add y=x line
    ax_scatter.plot([all_min, all_max], [all_min, all_max], 
                   'k--', alpha=0.5, label='y = x')
    
    # Labels and titles
    ax_scatter.set_xlabel('Mean True Conservation Score')
    ax_scatter.set_ylabel('Mean Predicted Conservation Score')
    ax_hist_x.set_ylabel('Density')
    ax_hist_y.set_xlabel('Density')
    
    # Remove unnecessary labels
    ax_hist_x.set_xticklabels([])
    ax_hist_y.set_yticklabels([])
    
    # Add title
    plt.suptitle('Sequence-Average Conservation Correlation Comparison', y=0.95)
    
    # Add legend
    ax_scatter.legend(bbox_to_anchor=(1.15, 1.1))
    ax_hist_x.legend()
    
    # Add grid
    ax_scatter.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_sequence_correlations.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_file1', default='/home/mica/gamba/data_processing/data/conserved_elements/representations_unseen_hg38UCNE_coordinates.npz', help='Path to first NPZ file')
    parser.add_argument('--npz_file2', default='/home/mica/gamba/data_processing/data/conserved_elements/representations_filteredunseen_hg38UCNE_coordinates_flanking.npz', help='Path to second NPZ file')
    parser.add_argument('--npz_file3', default ='/home/mica/gamba/data_processing/data/conserved_elements/representations_filteredunseen_hg38UCNE_coordinates_random.npz', help='Path to third NPZ file (optional)')
    parser.add_argument('--output_dir', default='/home/mica/gamba/data_processing/data/conserved_elements/', help='Output directory')
    parser.add_argument('--name1', default='UCNE', help='Name of first dataset')
    parser.add_argument('--name2', default='Flanking', help='Name of second dataset')
    parser.add_argument('--name3', default=None, help='Name of third dataset')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    #check if npz file contains "random" or "flanking" in title, then take the appropriate name
    if not args.name2:
        if "random" in args.npz_file2:
            args.name2 = "Random"
        elif "flanking" in args.npz_file2:
            args.name2 = "Flanking"

    if args.npz_file3:
        if not args.name3:
            if "random" in args.npz_file3:
                args.name3 = "Random"
            elif "flanking" in args.npz_file3:
                args.name3 = "Flanking"
    
    npz_files = [args.npz_file1, args.npz_file2]
    names = [args.name1, args.name2]
    
    if args.npz_file3:
        npz_files.append(args.npz_file3)
        names.append(args.name3)  # Or determine from filename
        plot_combined_sequence_correlations(npz_files, names, args.output_dir)
        return
        
    
    # Process both datasets
    roi_data = {}
    results = {}
    for npz_file, name in [(args.npz_file1, args.name1), (args.npz_file2, args.name2)]:
        print(f"\nProcessing {name}...")
        data = np.load(npz_file, allow_pickle=True)
        roi_data[name] = analyze_roi_vs_nonroi(data['sequences'], data['scores'], 
                                             data['spans'], name)
        corr, p_val, n = plot_sequence_average_correlations(
            data['sequences'], data['scores'], name, args.output_dir
        )
        results[name] = {'correlation': corr, 'p_value': p_val, 'n': n}

    # Print summary
    print("\nFull Sequence Correlation Results:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Correlation: {result['correlation']:.3f}")
        print(f"  P-value: {result['p_value']:.2e}")
        print(f"  Number of points: {result['n']:,}")
    
    # Create comparison plot
    plot_path = os.path.join(args.output_dir, 'roi_comparison.png')
    plot_roi_comparison(roi_data[args.name1], roi_data[args.name2], plot_path, 
                       args.name1, args.name2)

if __name__ == '__main__':
    main()