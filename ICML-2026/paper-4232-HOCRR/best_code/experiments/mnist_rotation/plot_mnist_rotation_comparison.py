#!/usr/bin/env python3
"""
Plot MNIST Rotation Comparison: (E, C)+M, (E, C, G)+M, and Alpha-Trimming

Creates standard randomized smoothing paper plots:
1. CDF comparison (cumulative distribution of certified radii)
2. Scatter plot comparisons (per-sample)
3. Box plot comparison (distribution summary)
4. Win rate statistics

Usage:
    python experiments/mnist_rotation/plot_mnist_rotation_comparison.py \
        --comparison comparison_vm_vs_wg_mnist_sigma0.5_eps10.0deg_20260112_042119.json \
        --alpha_trimming outputs/mnist_alpha/mnist_alpha_trimming_rotated_n100_sigma0.5_alpha0.49_20260111_232258.json \
        --output mnist_rotation_comparison_sigma0.5.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))


def load_json(json_path: str) -> Dict:
    """Load JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_radii_from_comparison(data: Dict) -> Tuple[List[float], List[float], List[int]]:
    """Extract radii from comparison JSON (has both variance_mean and with_gradient)."""
    variance_mean_radii = []
    with_gradient_radii = []
    test_indices = []
    
    for result in data.get('results', []):
        vm_r = result.get('radius_variance_mean')
        wg_r = result.get('radius_with_gradient')
        idx = result.get('test_dataset_idx')
        
        if vm_r is not None:
            variance_mean_radii.append(float(vm_r))
        if wg_r is not None:
            with_gradient_radii.append(float(wg_r))
        if idx is not None:
            test_indices.append(int(idx))
    
    return variance_mean_radii, with_gradient_radii, test_indices


def extract_radii_from_alpha_trimming(data: Dict) -> Tuple[List[float], Optional[float], Optional[float]]:
    """Extract radii from alpha-trimming JSON, also return P and alpha values."""
    radii = []
    P_value = None
    alpha_value = None
    
    # Extract P and alpha from parameters
    params = data.get('parameters', {})
    P_value = params.get('P', None)
    alpha_value = params.get('alpha', None)
    
    # Try different possible structures
    if 'samples' in data:
        for sample in data.get('samples', []):
            r = sample.get('certified_radius', 0.0)
            radii.append(float(r))
    elif 'certified_radii' in data:
        for item in data.get('certified_radii', []):
            r = item.get('certified_radius', 0.0)
            radii.append(float(r))
    else:
        # Try to find any field with 'radius' in it
        for key in data.keys():
            if 'radius' in key.lower() and isinstance(data[key], list):
                radii = [float(r) for r in data[key]]
                break
    
    return radii, P_value, alpha_value


def match_samples_by_index(
    comparison_indices: List[int],
    alpha_radii: List[float],
    alpha_data: Dict
) -> List[float]:
    """Match alpha-trimming radii to comparison samples by test_dataset_idx."""
    matched_radii = []
    
    # Try to extract indices from alpha data
    alpha_indices = []
    if 'samples' in alpha_data:
        for sample in alpha_data.get('samples', []):
            idx = sample.get('test_dataset_idx', sample.get('image_idx', None))
            alpha_indices.append(idx)
    elif 'test_indices' in alpha_data.get('parameters', {}):
        alpha_indices = alpha_data['parameters']['test_indices']
    
    # Create mapping
    if len(alpha_indices) == len(alpha_radii):
        alpha_map = {idx: r for idx, r in zip(alpha_indices, alpha_radii)}
    else:
        # If indices don't match, assume same order
        alpha_map = {i: r for i, r in enumerate(alpha_radii)}
    
    # Match by comparison indices
    for comp_idx in comparison_indices:
        if comp_idx in alpha_map:
            matched_radii.append(alpha_map[comp_idx])
        else:
            # Try to find by position if exact match fails
            matched_radii.append(0.0)  # Default to 0 if not found
    
    return matched_radii


def create_comparison_plots(
    variance_mean_radii: List[float],
    with_gradient_radii: List[float],
    alpha_radii: List[float],
    sigma: float,
    eps_y_deg: float,
    output_path: str,
    title_suffix: str = "",
    alpha_P: Optional[float] = None,
    alpha_alpha: Optional[float] = None,
    best_sigma_vm: Optional[float] = None,
    best_sigma_wg: Optional[float] = None,
    best_sigma_alpha: Optional[float] = None,
    show_statistics: bool = True
):
    """Create comprehensive comparison plots."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    vm_radii = np.array(variance_mean_radii)
    wg_radii = np.array(with_gradient_radii)
    alpha_radii_arr = np.array(alpha_radii)
    
    # Filter out zeros for better visualization (but keep track)
    vm_nonzero = vm_radii[vm_radii > 0]
    wg_nonzero = wg_radii[wg_radii > 0]
    alpha_nonzero = alpha_radii_arr[alpha_radii_arr > 0]
    
    # Plot 1: CDF Comparison (Main plot - standard in randomized smoothing papers)
    ax1 = fig.add_subplot(gs[0, 0])  # Top left, same size as others
    
    # Sort for CDF
    vm_sorted = np.sort(vm_radii)
    wg_sorted = np.sort(wg_radii)
    alpha_sorted = np.sort(alpha_radii_arr)
    
    n = len(vm_radii)
    cdf_y = np.arange(1, n + 1) / n
    
    # Build labels with best sigma indicators
    vm_label = '(E, C) + M'
    if best_sigma_vm is not None and abs(sigma - best_sigma_vm) < 1e-6:
        vm_label += f' (Best σ={sigma})'
    elif best_sigma_vm is not None:
        vm_label += f' (σ={sigma}, Best={best_sigma_vm})'
    else:
        vm_label += f' (σ={sigma})'
    
    wg_label = '(E, C, G) + M'
    if best_sigma_wg is not None and abs(sigma - best_sigma_wg) < 1e-6:
        wg_label += f' (Best σ={sigma})'
    elif best_sigma_wg is not None:
        wg_label += f' (σ={sigma}, Best={best_sigma_wg})'
    else:
        wg_label += f' (σ={sigma})'
    
    alpha_label = 'α-Trimming'
    if alpha_P is not None:
        alpha_label += f' (P={alpha_P:.2f})'
    if alpha_alpha is not None:
        alpha_label += f', α={alpha_alpha:.2f}'
    if best_sigma_alpha is not None and abs(sigma - best_sigma_alpha) < 1e-6:
        alpha_label += f' (Best σ={sigma})'
    elif best_sigma_alpha is not None:
        alpha_label += f' (σ={sigma}, Best={best_sigma_alpha})'
    else:
        alpha_label += f' (σ={sigma})'
    
    ax1.plot(vm_sorted, cdf_y, label=vm_label, linewidth=2.5, color='#2E86AB', linestyle='-')
    ax1.plot(wg_sorted, cdf_y, label=wg_label, linewidth=2.5, color='#A23B72', linestyle='-')
    ax1.plot(alpha_sorted, cdf_y, label=alpha_label, linewidth=2.5, color='#F18F01', linestyle='--')
    
    ax1.set_xlabel('Certified Radius (pixels)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Fraction', fontsize=13, fontweight='bold')
    ax1.set_title(f'CDF of Certified Radii (σ = {sigma}, ε_y = {eps_y_deg}°)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim([0, 1.05])
    
    # Add median lines
    vm_median = np.median(vm_radii)
    wg_median = np.median(wg_radii)
    alpha_median = np.median(alpha_radii_arr) if len(alpha_nonzero) > 0 else 0
    
    ax1.axvline(vm_median, color='#2E86AB', linestyle=':', alpha=0.7, linewidth=1.5)
    ax1.axvline(wg_median, color='#A23B72', linestyle=':', alpha=0.7, linewidth=1.5)
    if alpha_median > 0:
        ax1.axvline(alpha_median, color='#F18F01', linestyle=':', alpha=0.7, linewidth=1.5)
    
    # Plot 2: Scatter: (E, C, G) vs (E, C)
    ax2 = fig.add_subplot(gs[0, 1])  # Top right
    
    ax2.scatter(vm_radii, wg_radii, alpha=0.6, s=40, color='#A23B72')
    max_val = max(np.max(vm_radii), np.max(wg_radii))
    ax2.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='y=x')
    ax2.set_xlabel('(E, C) + M Radius', fontsize=11)
    ax2.set_ylabel('(E, C, G) + M Radius', fontsize=11)
    ax2.set_title('Gradient Improvement', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # Plot 3: Scatter: (E, C, G) vs α-Trimming
    ax3 = fig.add_subplot(gs[1, 0])  # Bottom left
    
    # Only plot non-zero pairs
    valid_mask = (wg_radii > 0) & (alpha_radii_arr > 0)
    if np.sum(valid_mask) > 0:
        ax3.scatter(alpha_radii_arr[valid_mask], wg_radii[valid_mask], 
                   alpha=0.6, s=40, color='#F18F01')
        max_val = max(np.max(wg_radii[valid_mask]), np.max(alpha_radii_arr[valid_mask]))
        ax3.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='y=x')
        ax3.set_xlabel('α-Trimming Radius', fontsize=11)
        ax3.set_ylabel('(E, C, G) + M Radius', fontsize=11)
        ax3.set_title('vs Baseline', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal', adjustable='box')
    else:
        ax3.text(0.5, 0.5, 'No valid\ncomparisons', 
                ha='center', va='center', fontsize=12)
        ax3.set_title('vs Baseline (No Data)', fontsize=12)
    
    # Plot 4: Box Plot Comparison
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom right
    
    data_to_plot = [vm_radii, wg_radii, alpha_radii_arr]
    labels = ['(E, C)\n+ M', '(E, C, G)\n+ M', 'α-Trimming']
    
    bp = ax4.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, 
                     widths=0.6, showmeans=True, meanline=True)
    
    # Color the boxes
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('Certified Radius (pixels)', fontsize=11)
    ax4.set_title('Distribution Summary', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(bottom=0)
    
    # Statistics (integrated into box plot if requested)
    if show_statistics:
        # Add compact statistics as text annotation to box plot
        # Compute statistics
        vm_mean = np.mean(vm_radii)
        vm_median = np.median(vm_radii)
        wg_mean = np.mean(wg_radii)
        wg_median = np.median(wg_radii)
        alpha_mean = np.mean(alpha_radii_arr)
        alpha_median = np.median(alpha_radii_arr)
        
        improvement = wg_radii - vm_radii
        improvement_pct = 100 * np.mean(improvement) / (np.mean(vm_radii) + 1e-10)
        
        # Compact statistics text (integrated into box plot)
        stats_text = f"Mean: {vm_mean:.3f} | {wg_mean:.3f} | {alpha_mean:.3f}\n"
        stats_text += f"Med: {vm_median:.3f} | {wg_median:.3f} | {alpha_median:.3f}"
        
        # Add to top-left of box plot (compact, not too messy)
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
                fontsize=8, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, 
                         edgecolor='gray', linewidth=0.5),
                zorder=10)
    
    # Overall title
    fig.suptitle(f'MNIST Rotation Certification Comparison{title_suffix}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_path}")
    plt.close()
    
    # Print summary
    print(f"\nSummary Statistics (σ = {sigma}, ε_y = {eps_y_deg}°):")
    print(f"  (E, C) + M:      Mean = {np.mean(vm_radii):.4f}, Median = {np.median(vm_radii):.4f}")
    print(f"  (E, C, G) + M:   Mean = {np.mean(wg_radii):.4f}, Median = {np.median(wg_radii):.4f}")
    print(f"  α-Trimming:      Mean = {np.mean(alpha_radii_arr):.4f}, Median = {np.median(alpha_radii_arr):.4f}")
    print(f"  Gradient wins:    {np.sum(wg_radii > vm_radii)}/{len(vm_radii)} samples")
    print(f"  vs α-Trimming:    {np.sum(wg_radii > alpha_radii_arr)}/{len(wg_radii)} samples")


def main():
    parser = argparse.ArgumentParser(
        description='Plot MNIST rotation comparison: (E, C)+M, (E, C, G)+M, and α-Trimming'
    )
    parser.add_argument(
        '--comparison',
        type=str,
        required=True,
        help='Comparison JSON file (contains both variance_mean and with_gradient radii)'
    )
    parser.add_argument(
        '--alpha_trimming',
        type=str,
        required=True,
        help='Alpha-trimming JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output plot file (auto-generated if not specified)'
    )
    parser.add_argument(
        '--title_suffix',
        type=str,
        default='',
        help='Additional text for plot title'
    )
    parser.add_argument(
        '--best_sigma_vm',
        type=float,
        default=None,
        help='Best sigma for (E, C) + M method'
    )
    parser.add_argument(
        '--best_sigma_wg',
        type=float,
        default=None,
        help='Best sigma for (E, C, G) + M method'
    )
    parser.add_argument(
        '--best_sigma_alpha',
        type=float,
        default=None,
        help='Best sigma for alpha-trimming method'
    )
    parser.add_argument(
        '--hide_statistics',
        action='store_true',
        help='Hide statistics summary (remove text box)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading comparison data: {args.comparison}")
    comparison_data = load_json(args.comparison)
    vm_radii, wg_radii, test_indices = extract_radii_from_comparison(comparison_data)
    print(f"✓ Found {len(vm_radii)} samples")
    
    print(f"Loading alpha-trimming data: {args.alpha_trimming}")
    alpha_data = load_json(args.alpha_trimming)
    alpha_radii_raw, alpha_P, alpha_alpha = extract_radii_from_alpha_trimming(alpha_data)
    print(f"✓ Found {len(alpha_radii_raw)} samples")
    if alpha_P is not None:
        print(f"  P value: {alpha_P}")
    if alpha_alpha is not None:
        print(f"  Alpha value: {alpha_alpha}")
    
    # Match samples
    if len(alpha_radii_raw) != len(vm_radii):
        print(f"⚠ Warning: Sample count mismatch. Matching by index...")
        alpha_radii = match_samples_by_index(test_indices, alpha_radii_raw, alpha_data)
    else:
        alpha_radii = alpha_radii_raw
    
    # Get parameters
    sigma = comparison_data.get('parameters', {}).get('sigma', 0.5)
    eps_y_deg = comparison_data.get('parameters', {}).get('eps_y_deg', 10.0)
    
    # Generate output path
    if args.output is None:
        output_path = f"mnist_rotation_comparison_sigma{sigma}_eps{eps_y_deg}deg.png"
    else:
        output_path = args.output
    
    # Create plots
    create_comparison_plots(
        vm_radii, wg_radii, alpha_radii,
        sigma, eps_y_deg, output_path, args.title_suffix,
        alpha_P, alpha_alpha,
        args.best_sigma_vm, args.best_sigma_wg, args.best_sigma_alpha,
        show_statistics=not args.hide_statistics
    )


if __name__ == '__main__':
    main()

