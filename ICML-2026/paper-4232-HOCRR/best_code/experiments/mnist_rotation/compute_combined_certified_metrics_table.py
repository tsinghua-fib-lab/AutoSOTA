#!/usr/bin/env python3
"""
Generate Combined Certified Metrics Table

Combines absolute certified accuracy, conditional certified accuracy, and certified mean distance
into a single table for section 7.5.

Usage:
    python experiments/mnist_rotation/compute_combined_certified_metrics_table.py \
        --comparison_dir . \
        --alpha_dir outputs/mnist_alpha \
        --tolerance 10.0 \
        --R_values 0.05 0.10 0.15 0.20 0.25 \
        --output certified_metrics_combined_table.tex
"""

import json
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Import functions from sibling experiment scripts.
sys.path.append(str(Path(__file__).resolve().parent))
from compute_certified_accuracy_best_sigma import (
    load_all_sigma_data, find_best_sigma_overall, compute_certified_accuracy_at_fixed_sigma
)
from compute_certified_mean_distance_best_sigma import (
    compute_certified_mean_distance, compute_mean_distance_at_fixed_sigma, find_best_sigma_overall as find_best_sigma_for_mean_distance
)
from plot_certified_accuracy_curves import compute_certified_accuracy, extract_radii_from_file


def load_accuracy_data(comparison_dir: str, alpha_dir: str, sigmas: List[float], tolerance: float, normalize_by_certified: bool):
    """Load accuracy data and find best sigma for each method."""
    bounded_data, alpha_data = load_all_sigma_data(comparison_dir, alpha_dir, sigmas, tolerance)
    
    methods_data = {}
    R_values = [0.05, 0.10, 0.15, 0.20, 0.25]  # Will be overridden by args
    
    # Process bounded methods
    for method_key, method_name in [('variance_mean', '(E, C) + M'), ('with_gradient', '(E, C, G) + M')]:
        if method_key in bounded_data and len(bounded_data[method_key]) > 0:
            best_sigma = find_best_sigma_overall(
                bounded_data[method_key], R_values, tolerance, normalize_by_certified
            )
            if best_sigma is not None:
                accuracies, _ = compute_certified_accuracy_at_fixed_sigma(
                    bounded_data[method_key], best_sigma, R_values, tolerance, normalize_by_certified
                )
                methods_data[method_key] = {
                    'accuracies': accuracies,
                    'sigma': best_sigma,
                    'name': method_name
                }
    
    # Process alpha-smoothing
    if len(alpha_data) > 0:
        best_sigma = find_best_sigma_overall(
            alpha_data, R_values, tolerance, normalize_by_certified
        )
        if best_sigma is not None:
            accuracies, _ = compute_certified_accuracy_at_fixed_sigma(
                alpha_data, best_sigma, R_values, tolerance, normalize_by_certified
            )
            methods_data['alpha_trimming'] = {
                'accuracies': accuracies,
                'sigma': best_sigma,
                'name': 'alpha-smoothing'
            }
    
    return methods_data


def load_mean_distance_data(comparison_dir: str, alpha_dir: str, sigmas: List[float], tolerance: float):
    """Load mean distance data and find best sigma for each method."""
    bounded_data, alpha_data = load_all_sigma_data(comparison_dir, alpha_dir, sigmas, tolerance)
    
    methods_data = {}
    R_values = [0.05, 0.10, 0.15, 0.20, 0.25]  # Will be overridden by args
    
    # Process bounded methods
    for method_key, method_name in [('variance_mean', '(E, C) + M'), ('with_gradient', '(E, C, G) + M')]:
        if method_key in bounded_data and len(bounded_data[method_key]) > 0:
            best_sigma = find_best_sigma_for_mean_distance(
                bounded_data[method_key], R_values
            )
            if best_sigma is not None:
                mean_dists, _ = compute_mean_distance_at_fixed_sigma(
                    bounded_data[method_key], best_sigma, R_values
                )
                methods_data[method_key] = {
                    'mean_distances': mean_dists,
                    'sigma': best_sigma,
                    'name': method_name
                }
    
    # Process alpha-smoothing
    if len(alpha_data) > 0:
        best_sigma = find_best_sigma_for_mean_distance(alpha_data, R_values)
        if best_sigma is not None:
            mean_dists, _ = compute_mean_distance_at_fixed_sigma(alpha_data, best_sigma, R_values)
            methods_data['alpha_trimming'] = {
                'mean_distances': mean_dists,
                'sigma': best_sigma,
                'name': 'alpha-smoothing'
            }
    
    return methods_data


def create_combined_latex_table(
    abs_acc_data: Dict,
    cond_acc_data: Dict,
    mean_dist_data: Dict,
    R_values: List[float],
    output_file: str
):
    """Create combined LaTeX table with all three metrics."""
    
    method_order = ['variance_mean', 'with_gradient', 'alpha_trimming']
    method_labels = {
        'variance_mean': '$(E, C) + M$',
        'with_gradient': '$(E, C, G) + M$',
        'alpha_trimming': '$\\alpha$-smoothing ($P=0.9$)'
    }
    
    lines = [
        "% Combined Certified Metrics Table",
        "% Auto-generated: Absolute Accuracy, Conditional Accuracy, and Mean Distance",
        "",
        "\\begin{table}[t]",
        "    \\centering",
        "    \\small",
        "    \\setlength{\\tabcolsep}{3pt}",
        "    \\renewcommand{\\arraystretch}{1.05}",
        "    \\caption{Certified metrics at different radius thresholds $R$ (in pixels) on MNIST rotation task. ",
        "    For each method, we select a single $\\sigma$ value that maximizes the sum of certified accuracies (for absolute and conditional accuracy) or minimizes the sum of mean distances (for mean distance) across all thresholds, then report metrics at that fixed $\\sigma$ for all $R$ values. ",
        "    Absolute accuracy normalizes by total samples $n$, conditional accuracy normalizes by certified samples $|\\mathcal{S}_R|$, and mean distance is computed only over certified samples. ",
        "    $(E, C) + M$ uses $\\sigma=0.06$ for accuracy and $\\sigma=0.06$ for distance; ",
        "    $(E, C, G) + M$ uses $\\sigma=0.75$ for accuracy and $\\sigma=0.75$ for distance; ",
        "    $\\alpha$-smoothing uses $\\sigma=0.06$ for absolute accuracy, $\\sigma=0.12$ for conditional accuracy, and $\\sigma=0.12$ for distance.}",
        "    \\label{tab:certified_metrics_mnist}",
        f"    \\begin{{tabular}}{{l {'c ' * len(R_values)}}}",
        "        \\toprule",
        "        Method & \\multicolumn{" + str(len(R_values)) + "}{c}{Radius Threshold $R$ (pixels)} \\\\",
        "        \\cmidrule(lr){2-" + str(len(R_values) + 1) + "}",
        "        & " + " & ".join([f"{R:.2f}" for R in R_values]) + " \\\\",
        "        \\midrule"
    ]
    
    # Add Absolute Accuracy section
    lines.append(f"        \\multicolumn{{{len(R_values) + 1}}}{{l}}{{\\textit{{Absolute Accuracy (\\%)}}}} \\\\")
    for method_key in method_order:
        if method_key not in abs_acc_data:
            continue
        method_label = method_labels[method_key]
        abs_acc = abs_acc_data[method_key]['accuracies']
        formatted_abs = [f"{acc:.0f}" for acc in abs_acc]
        lines.append(f"        {method_label} & {' & '.join(formatted_abs)} \\\\")
    
    # Add Conditional Accuracy section
    lines.append("        \\midrule")
    lines.append(f"        \\multicolumn{{{len(R_values) + 1}}}{{l}}{{\\textit{{Conditional Accuracy (\\%)}}}} \\\\")
    for method_key in method_order:
        if method_key not in cond_acc_data:
            continue
        method_label = method_labels[method_key]
        cond_acc = cond_acc_data[method_key]['accuracies']
        formatted_cond = [f"{acc:.0f}" for acc in cond_acc]
        lines.append(f"        {method_label} & {' & '.join(formatted_cond)} \\\\")
    
    # Add Mean Distance section
    lines.append("        \\midrule")
    lines.append(f"        \\multicolumn{{{len(R_values) + 1}}}{{l}}{{\\textit{{Mean Distance (degrees)}}}} \\\\")
    for method_key in method_order:
        if method_key not in mean_dist_data:
            continue
        method_label = method_labels[method_key]
        mean_dists = mean_dist_data[method_key]['mean_distances']
        formatted_dist = []
        for md in mean_dists:
            if np.isnan(md):
                formatted_dist.append("---")
            else:
                formatted_dist.append(f"{md:.2f}")
        lines.append(f"        {method_label} & {' & '.join(formatted_dist)} \\\\")
    
    lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}"
    ])
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Saved combined metrics table to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate combined certified metrics table")
    parser.add_argument(
        "--comparison_dir",
        type=str,
        default=".",
        help="Directory containing comparison JSON files"
    )
    parser.add_argument(
        "--alpha_dir",
        type=str,
        default="outputs/mnist_alpha",
        help="Directory containing alpha-smoothing JSON files"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=10.0,
        help="Correctness tolerance in degrees"
    )
    parser.add_argument(
        "--R_values",
        type=float,
        nargs='+',
        default=[0.05, 0.10, 0.15, 0.20, 0.25],
        help="Radius thresholds to evaluate"
    )
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs='+',
        default=[0.06, 0.12, 0.25, 0.5, 0.75],
        help="Sigma values to compare"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="certified_metrics_combined_table.tex",
        help="Output LaTeX table file"
    )
    
    args = parser.parse_args()
    
    print("Loading absolute accuracy data...")
    abs_acc_data = load_accuracy_data(args.comparison_dir, args.alpha_dir, args.sigmas, args.tolerance, normalize_by_certified=False)
    
    print("Loading conditional accuracy data...")
    cond_acc_data = load_accuracy_data(args.comparison_dir, args.alpha_dir, args.sigmas, args.tolerance, normalize_by_certified=True)
    
    print("Loading mean distance data...")
    mean_dist_data = load_mean_distance_data(args.comparison_dir, args.alpha_dir, args.sigmas, args.tolerance)
    
    print("\nCreating combined LaTeX table...")
    create_combined_latex_table(abs_acc_data, cond_acc_data, mean_dist_data, args.R_values, args.output)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
