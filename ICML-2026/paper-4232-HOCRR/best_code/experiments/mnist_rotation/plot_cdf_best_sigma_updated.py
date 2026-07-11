#!/usr/bin/env python3
"""
Plot CDF comparison using updated results with success probability 0.9.

This script creates a single CDF plot comparing:
1. (E, C)+M: success prob 0.9, sigma=0.06
2. (E, C, G)+M: success prob 0.9, sigma=0.75
3. Alpha-smoothing: P=0.9, alpha=0.49, sigma=0.06
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from plot_mnist_rotation_comparison import (
    extract_radii_from_alpha_trimming,
    match_samples_by_index,
    load_json
)


def extract_radii_from_ec_file(ec_file: str) -> tuple[List[float], List[int], Dict]:
    """Extract radii from (E, C)+M results file."""
    with open(ec_file, 'r') as f:
        data = json.load(f)
    
    radii = []
    indices = []
    
    for result in data.get('results', []):
        r = result.get('radius')
        idx = result.get('test_dataset_idx', result.get('sample_idx'))
        
        if r is not None:
            radii.append(float(r))
        if idx is not None:
            indices.append(int(idx))
    
    return radii, indices, data


def extract_radii_from_ecg_file(ecg_file: str) -> tuple[List[float], List[int], Dict]:
    """Extract radii from (E, C, G)+M results file."""
    with open(ecg_file, 'r') as f:
        data = json.load(f)
    
    radii = []
    indices = []
    
    for result in data.get('results', []):
        r = result.get('radius')
        idx = result.get('test_dataset_idx', result.get('sample_idx'))
        
        if r is not None:
            radii.append(float(r))
        if idx is not None:
            indices.append(int(idx))
    
    return radii, indices, data


def find_alpha_file(alpha_dir: str, sigma: float, alpha: float = 0.49) -> Optional[str]:
    """Find alpha-smoothing file for given sigma and alpha."""
    files = glob.glob(f"{alpha_dir}/mnist_alpha_trimming_rotated_n100_sigma{sigma}_alpha{alpha}_*.json")
    if files:
        return sorted(files)[-1]  # Use most recent
    return None


def create_cdf_only_plot(
    ec_radii: List[float],
    ecg_radii: List[float],
    alpha_radii: List[float],
    ec_sigma: float,
    ecg_sigma: float,
    alpha_sigma: float,
    eps_y_deg: float,
    output_path: str,
    alpha_P: Optional[float] = None,
    alpha_alpha: Optional[float] = None
):
    """Create CDF-only plot comparing methods at their best sigma values with success prob 0.9."""
    
    # Use compact figure size
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    
    # Reduce margins around the plot area
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.90)
    
    ec_radii_arr = np.array(ec_radii)
    ecg_radii_arr = np.array(ecg_radii)
    alpha_radii_arr = np.array(alpha_radii)
    
    # Cap radii at 0.3 for better visualization (and to handle invalid outliers)
    ec_radii_arr = np.minimum(ec_radii_arr, 0.3)
    ecg_radii_arr = np.minimum(ecg_radii_arr, 0.3)
    alpha_radii_arr = np.minimum(alpha_radii_arr, 0.3)
    
    # Sort for CDF
    ec_sorted = np.sort(ec_radii_arr)
    ecg_sorted = np.sort(ecg_radii_arr)
    alpha_sorted = np.sort(alpha_radii_arr)
    
    n = len(ec_radii_arr)
    cdf_y = np.arange(1, n + 1) / n
    
    # Build labels with success probability and sigma information
    ec_label = f'$(E, C) + M$ (success prob. $= 0.9$, $\\sigma = {ec_sigma}$)'
    ecg_label = f'$(E, C, G) + M$ (success prob. $= 0.9$, $\\sigma = {ecg_sigma}$)'
    
    alpha_label = '$\\alpha$-smoothing'
    if alpha_P is not None and alpha_alpha is not None:
        alpha_label += f' ($P = {alpha_P:.1f}, \\alpha = {alpha_alpha:.2f}, \\sigma = {alpha_sigma}$)'
    elif alpha_P is not None:
        alpha_label += f' ($P = {alpha_P:.1f}, \\sigma = {alpha_sigma}$)'
    elif alpha_alpha is not None:
        alpha_label += f' ($\\alpha = {alpha_alpha:.2f}, \\sigma = {alpha_sigma}$)'
    else:
        alpha_label += f' ($\\sigma = {alpha_sigma}$)'
    
    # Plot CDFs with thicker lines for better visibility when scaled
    ax.plot(ec_sorted, cdf_y, label=ec_label, linewidth=3.0, color='#2E86AB', linestyle='-')
    ax.plot(ecg_sorted, cdf_y, label=ecg_label, linewidth=3.0, color='#A23B72', linestyle='-')
    ax.plot(alpha_sorted, cdf_y, label=alpha_label, linewidth=3.0, color='#F18F01', linestyle='--')
    
    # Add median lines (thinner, more subtle)
    ec_median = np.median(ec_radii_arr)
    ecg_median = np.median(ecg_radii_arr)
    alpha_median = np.median(alpha_radii_arr) if len(alpha_radii_arr[alpha_radii_arr > 0]) > 0 else 0
    
    ax.axvline(ec_median, color='#2E86AB', linestyle=':', alpha=0.5, linewidth=1.0)
    ax.axvline(ecg_median, color='#A23B72', linestyle=':', alpha=0.5, linewidth=1.0)
    if alpha_median > 0:
        ax.axvline(alpha_median, color='#F18F01', linestyle=':', alpha=0.5, linewidth=1.0)
    
    # Labels and title - smaller fonts for compactness
    ax.set_xlabel('Certified Radius (pixels)', fontsize=10)
    ax.set_ylabel('Cumulative Fraction', fontsize=10)
    ax.set_title('CDF of Certified Radii (Best $\\sigma$ per Method)', 
                 fontsize=11, fontweight='bold', pad=8)
    
    # Legend with smaller font, more compact
    ax.legend(fontsize=8, loc='lower right', handlelength=1.5, framealpha=0.9, 
              columnspacing=0.5, handletextpad=0.3)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Light gridlines
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    
    # Reduce tick count for cleaner x-axis
    max_radius = max(np.max(ec_radii_arr), np.max(ecg_radii_arr), np.max(alpha_radii_arr))
    # Set ticks at 0.0, 0.1, 0.2, 0.3 (or appropriate values)
    if max_radius > 0.3:
        # If max is larger, adjust spacing
        if max_radius <= 0.4:
            x_ticks = [0.0, 0.1, 0.2, 0.3, 0.4]
        else:
            x_ticks = [0.0, 0.1, 0.2, 0.3]
    else:
        x_ticks = [0.0, 0.1, 0.2, 0.3]
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', labelsize=9)
    
    # Y-axis ticks: 0.0, 0.5, 1.0
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis='y', labelsize=9)
    
    ax.set_xlim(left=0)
    ax.set_ylim([0, 1.05])
    
    # Very tight layout - minimal padding
    plt.tight_layout(pad=0.3)
    
    # Save as PDF with minimal padding
    if not output_path.endswith('.pdf'):
        output_path = output_path.replace('.png', '.pdf')
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05, format='pdf')
    print(f"✓ Saved CDF-only plot: {output_path}")
    plt.close()
    
    # Print summary (using original uncapped values for statistics)
    ec_radii_orig = np.array(ec_radii)
    ecg_radii_orig = np.array(ecg_radii)
    alpha_radii_orig = np.array(alpha_radii)
    
    # Count how many were capped
    ec_capped = np.sum(ec_radii_orig > 0.3)
    ecg_capped = np.sum(ecg_radii_orig > 0.3)
    alpha_capped = np.sum(alpha_radii_orig > 0.3)
    
    print(f"\nSummary Statistics (Best σ per Method, ε_y = {eps_y_deg}°):")
    print(f"  (E, C) + M (σ={ec_sigma}, success prob=0.9):      Mean = {np.mean(ec_radii_arr):.4f}, Median = {np.median(ec_radii_arr):.4f}")
    if ec_capped > 0:
        print(f"    (Note: {ec_capped} sample(s) capped at 0.3 for visualization)")
    print(f"  (E, C, G) + M (σ={ecg_sigma}, success prob=0.9):   Mean = {np.mean(ecg_radii_arr):.4f}, Median = {np.median(ecg_radii_arr):.4f}")
    if ecg_capped > 0:
        print(f"    (Note: {ecg_capped} sample(s) capped at 0.3 for visualization)")
    print(f"  α-smoothing (σ={alpha_sigma}, P=0.9):   Mean = {np.mean(alpha_radii_arr):.4f}, Median = {np.median(alpha_radii_arr):.4f}")
    if alpha_capped > 0:
        print(f"    (Note: {alpha_capped} sample(s) capped at 0.3 for visualization)")


def main():
    parser = argparse.ArgumentParser(
        description='Plot CDF-only comparison using updated results with success prob 0.9'
    )
    parser.add_argument(
        '--ec_file',
        type=str,
        default='ec_radii_sigma0.06_eps10.0deg_20260126_140257.json',
        help='(E, C)+M results file with success prob 0.9, sigma=0.06'
    )
    parser.add_argument(
        '--ecg_file',
        type=str,
        default='ecg_radii_sigma0.75_eps10.0deg_20260126_174354.json',
        help='(E, C, G)+M results file with success prob 0.9, sigma=0.75'
    )
    parser.add_argument(
        '--alpha_dir',
        type=str,
        default='outputs/mnist_alpha',
        help='Directory containing alpha-smoothing JSON files'
    )
    parser.add_argument(
        '--alpha_sigma',
        type=float,
        default=0.06,
        help='Sigma for alpha-smoothing'
    )
    parser.add_argument(
        '--alpha_alpha',
        type=float,
        default=0.49,
        help='Alpha parameter for alpha-smoothing'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='figures/mnist_rotation_cdf_best_sigma_updated.pdf',
        help='Output plot file (PDF recommended for better text clarity)'
    )
    parser.add_argument(
        '--eps_y_deg',
        type=float,
        default=10.0,
        help='Output tolerance in degrees'
    )
    
    args = parser.parse_args()
    
    # Load (E, C)+M results
    print(f"Loading (E, C)+M results from: {args.ec_file}")
    ec_radii, ec_indices, ec_data = extract_radii_from_ec_file(args.ec_file)
    ec_sigma = ec_data.get('parameters', {}).get('sigma', 0.06)
    print(f"  Found {len(ec_radii)} samples, sigma={ec_sigma}")
    
    # Load (E, C, G)+M results
    print(f"\nLoading (E, C, G)+M results from: {args.ecg_file}")
    ecg_radii, ecg_indices, ecg_data = extract_radii_from_ecg_file(args.ecg_file)
    ecg_sigma = ecg_data.get('parameters', {}).get('sigma', 0.75)
    print(f"  Found {len(ecg_radii)} samples, sigma={ecg_sigma}")
    
    # Load alpha-smoothing data
    print(f"\nLoading alpha-smoothing results (sigma={args.alpha_sigma}, alpha={args.alpha_alpha})...")
    alpha_file = find_alpha_file(args.alpha_dir, args.alpha_sigma, args.alpha_alpha)
    
    if alpha_file is None:
        print(f"Error: Could not find alpha-smoothing file for sigma={args.alpha_sigma}, alpha={args.alpha_alpha}")
        return
    
    print(f"  Found: {alpha_file}")
    alpha_data = load_json(alpha_file)
    alpha_radii_raw, alpha_P, alpha_alpha_loaded = extract_radii_from_alpha_trimming(alpha_data)
    
    # Match samples by index (use ecg_indices as reference since it's the same dataset)
    alpha_radii = match_samples_by_index(ecg_indices, alpha_radii_raw, alpha_data)
    
    if alpha_alpha_loaded is not None:
        args.alpha_alpha = alpha_alpha_loaded
    
    print(f"  Found {len(alpha_radii)} matched samples")
    
    # Create plot
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    create_cdf_only_plot(
        ec_radii, ecg_radii, alpha_radii,
        ec_sigma, ecg_sigma, args.alpha_sigma,
        args.eps_y_deg, args.output,
        alpha_P=alpha_P, alpha_alpha=args.alpha_alpha
    )


if __name__ == '__main__':
    main()
