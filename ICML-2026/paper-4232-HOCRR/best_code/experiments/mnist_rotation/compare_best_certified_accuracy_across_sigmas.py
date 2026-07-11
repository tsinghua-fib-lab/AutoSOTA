#!/usr/bin/env python3
"""
Compare Best Certified Accuracy Across Sigma Values

For each certification method, finds the sigma value that gives the best
certified accuracy at each radius threshold, then compares the best
performances across methods.

Usage:
    python experiments/mnist_rotation/compare_best_certified_accuracy_across_sigmas.py \
        --tolerance 10.0 \
        --output best_certified_accuracy_comparison.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import glob


def load_table_json(json_path: str) -> Dict:
    """Load JSON table file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_sigma_from_filename(filename: str) -> Optional[float]:
    """Extract sigma value from filename like '..._sigma0.5_tolerance...'"""
    try:
        parts = Path(filename).stem.split('_')
        for part in parts:
            if part.startswith('sigma'):
                return float(part.replace('sigma', ''))
    except:
        pass
    return None


def find_best_sigma_per_method(
    table_files: List[str],
    tolerance: float
) -> Tuple[Dict[str, Dict[float, List[float]]], Dict[str, Dict[float, float]]]:
    """
    Find best sigma value for each method at each radius threshold.
    
    Returns:
        - method_best_data: Dict mapping method_name -> {R: [accuracies across sigmas]}
        - method_best_sigma: Dict mapping method_name -> {R: best_sigma}
    """
    # Structure: method_name -> sigma -> [accuracies at each R]
    method_sigma_data = defaultdict(lambda: defaultdict(list))
    
    # Load all table files
    for table_file in sorted(table_files):
        sigma = extract_sigma_from_filename(table_file)
        if sigma is None:
            continue
        
        data = load_table_json(table_file)
        R_values = data.get('R_values', [])
        certified_accuracy = data.get('certified_accuracy', {})
        
        for method_name, accuracies in certified_accuracy.items():
            # Normalize method name (remove sigma/alpha from name since we're comparing across sigmas)
            base_method = normalize_method_name(method_name)
            method_sigma_data[base_method][sigma] = accuracies
    
    # Find best sigma for each method at each R
    method_best_data = {}
    method_best_sigma = {}
    
    for method_name, sigma_data in method_sigma_data.items():
        # Get R_values from first sigma (they should all be the same)
        first_sigma = list(sigma_data.keys())[0]
        R_values = data.get('R_values', [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
        
        best_accuracies = []
        best_sigmas = []
        
        # For each R threshold, find best sigma
        for r_idx, R in enumerate(R_values):
            best_acc = -1
            best_sig = None
            
            for sigma, accuracies in sigma_data.items():
                if r_idx < len(accuracies):
                    acc = accuracies[r_idx]
                    if acc > best_acc:
                        best_acc = acc
                        best_sig = sigma
            
            best_accuracies.append(best_acc)
            best_sigmas.append(best_sig)
        
        method_best_data[method_name] = {
            'accuracies': best_accuracies,
            'sigmas': best_sigmas,
            'R_values': R_values
        }
        method_best_sigma[method_name] = dict(zip(R_values, best_sigmas))
    
    return method_best_data, method_best_sigma


def normalize_method_name(method_name: str) -> str:
    """Normalize method name by removing sigma/alpha info."""
    if '(E, C) + M' in method_name or method_name.startswith('(E, C) + M'):
        return '(E, C) + M'
    elif '(E, C, G) + M' in method_name or method_name.startswith('(E, C, G) + M'):
        return '(E, C, G) + M'
    elif 'Alpha-Trimming' in method_name or 'Alpha' in method_name:
        return 'Alpha-Trimming'
    else:
        return method_name


def plot_best_comparison(
    method_best_data: Dict[str, Dict],
    method_best_sigma: Dict[str, Dict[float, float]],
    R_values: List[float],
    output_path: str,
    tolerance: float,
    title: Optional[str] = None
):
    """Create comparison plot showing best performance from each method."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Curves with sigma labels
    for method_name, data in method_best_data.items():
        accuracies = data['accuracies']
        sigmas = data['sigmas']
        
        # Create label with sigma info
        # Show most common sigma or range
        unique_sigmas = list(set(sigmas))
        if len(unique_sigmas) == 1:
            label = f"{method_name} (σ={unique_sigmas[0]})"
        else:
            # Show range or most common
            sigma_counts = {}
            for s in sigmas:
                sigma_counts[s] = sigma_counts.get(s, 0) + 1
            most_common = max(sigma_counts.items(), key=lambda x: x[1])[0]
            label = f"{method_name} (σ={most_common}, varies by R)"
        
        ax1.plot(R_values, accuracies, marker='o', label=label, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Certified Radius R', fontsize=12)
    ax1.set_ylabel('Certified Accuracy (%)', fontsize=12)
    ax1.set_title('Best Certified Accuracy Across Sigma Values', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Plot 2: Bar chart with sigma annotations
    x = np.arange(len(R_values))
    width = 0.8 / len(method_best_data)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_best_data)))
    
    for i, (method_name, data) in enumerate(method_best_data.items()):
        accuracies = data['accuracies']
        sigmas = data['sigmas']
        offset = (i - len(method_best_data) / 2 + 0.5) * width
        
        bars = ax2.bar(x + offset, accuracies, width, label=method_name, alpha=0.8, color=colors[i])
        
        # Annotate bars with sigma values
        for j, (bar, sigma) in enumerate(zip(bars, sigmas)):
            if accuracies[j] > 0:  # Only annotate if there's a value
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'σ={sigma:.2f}',
                        ha='center', va='bottom', fontsize=7, rotation=90)
    
    ax2.set_xlabel('Radius Threshold', fontsize=12)
    ax2.set_ylabel('Certified Accuracy (%)', fontsize=12)
    ax2.set_title('Best Certified Accuracy by Radius (with σ labels)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{R:.2f}' for R in R_values], rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 105])
    
    if title is None:
        title = f"Best Certified Accuracy Comparison (tolerance={tolerance}°)"
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}\n")
    plt.close()


def print_best_comparison_table(
    method_best_data: Dict[str, Dict],
    method_best_sigma: Dict[str, Dict[float, float]],
    R_values: List[float],
    tolerance: float
):
    """Print comparison table showing best performance and sigma values."""
    print("\n" + "="*100)
    print("BEST CERTIFIED ACCURACY COMPARISON ACROSS SIGMA VALUES")
    print(f"Correctness Tolerance: {tolerance}°")
    print("="*100)
    
    # Header
    print(f"{'Method':<25}", end="")
    for R in R_values:
        print(f"  R={R:.2f}", end="")
    print()
    print("-"*100)
    
    # Data rows
    for method_name, data in method_best_data.items():
        accuracies = data['accuracies']
        sigmas = data['sigmas']
        
        print(f"{method_name:<25}", end="")
        for acc, sigma in zip(accuracies, sigmas):
            print(f"  {acc:5.1f}% (σ={sigma:.2f})", end="")
        print()
    
    print("="*100)
    print("\nNote: Values show certified accuracy (%) and the sigma value (σ) that achieved it.\n")


def save_best_comparison_json(
    method_best_data: Dict[str, Dict],
    method_best_sigma: Dict[str, Dict[float, float]],
    R_values: List[float],
    tolerance: float,
    output_path: str
):
    """Save comparison results to JSON."""
    results = {
        'tolerance_deg': tolerance,
        'R_values': R_values,
        'best_performance': {},
        'best_sigma_per_R': {}
    }
    
    for method_name, data in method_best_data.items():
        results['best_performance'][method_name] = {
            'accuracies': data['accuracies'],
            'sigmas': data['sigmas']
        }
        results['best_sigma_per_R'][method_name] = method_best_sigma[method_name]
    
    json_path = output_path.replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to: {json_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare best certified accuracy across sigma values"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="Correctness tolerance in degrees (must match the table files)"
    )
    parser.add_argument(
        "--table_dir",
        type=str,
        default=".",
        help="Directory containing the table JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot file (auto-generated if not specified)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title (auto-generated if not specified)"
    )
    
    args = parser.parse_args()
    
    # Find all table files with the specified tolerance
    pattern = f"certified_accuracy_rotated_mnist_sigma*_tolerance{args.tolerance}deg_table.json"
    table_files = glob.glob(str(Path(args.table_dir) / pattern))
    
    if not table_files:
        print(f"ERROR: No table files found matching pattern: {pattern}")
        print(f"  Searched in: {Path(args.table_dir).absolute()}")
        return
    
    print(f"Found {len(table_files)} table files:")
    for f in sorted(table_files):
        print(f"  {f}")
    print()
    
    # Find best sigma for each method
    method_best_data, method_best_sigma = find_best_sigma_per_method(table_files, args.tolerance)
    
    if not method_best_data:
        print("ERROR: No data found in table files")
        return
    
    # Get R_values from first method
    first_method = list(method_best_data.keys())[0]
    R_values = method_best_data[first_method]['R_values']
    
    # Print table
    print_best_comparison_table(method_best_data, method_best_sigma, R_values, args.tolerance)
    
    # Create plot
    if args.output is None:
        args.output = f"best_certified_accuracy_comparison_tolerance{args.tolerance}deg.png"
    
    plot_best_comparison(
        method_best_data,
        method_best_sigma,
        R_values,
        args.output,
        args.tolerance,
        args.title
    )
    
    # Save JSON
    save_best_comparison_json(
        method_best_data,
        method_best_sigma,
        R_values,
        args.tolerance,
        args.output
    )


if __name__ == "__main__":
    main()

