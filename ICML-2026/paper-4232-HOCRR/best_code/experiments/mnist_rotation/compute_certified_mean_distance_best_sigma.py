#!/usr/bin/env python3
"""
Compute Certified Mean Distance using best sigma for each method at each threshold.

This script computes the mean prediction error (circular distance) for certified samples:
Certified Mean Distance(R) = (1/|S_R|) * sum_{i in S_R} d(hat y_i, y_i)

where S_R = {i : r_i >= R} is the set of samples with certified radius >= R.

For each method and each radius threshold R, we find the sigma value that minimizes
the mean distance (i.e., gives the best prediction quality for certified samples).

Usage:
    python experiments/mnist_rotation/compute_certified_mean_distance_best_sigma.py \
        --comparison_dir . \
        --alpha_dir outputs/mnist_alpha \
        --R_values 0.05 0.10 0.15 0.20 0.25 \
        --output certified_mean_distance_table_best_sigma.tex
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob
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


def extract_pred_errors_from_comparison(data: Dict, json_path: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """Extract prediction errors from comparison file or referenced estimation file."""
    # Check if estimation file is referenced
    params = data.get('parameters', {})
    est_file = params.get('variance_gradient_file')
    
    if not est_file:
        return None, None
    
    # Resolve path relative to comparison file
    json_dir = Path(json_path).parent
    est_path = json_dir / est_file
    if not est_path.exists():
        # Try absolute path or relative to workspace root
        workspace_root = Path(__file__).resolve().parents[2]
        est_path = workspace_root / est_file
        if not est_path.exists():
            return None, None
    
    try:
        est_data = load_json(str(est_path))
        samples = est_data.get('samples', [])
        
        # Create mapping from test_dataset_idx to true_angle_deg and clean_pred_deg
        est_map = {}
        for sample in samples:
            idx = sample.get('test_dataset_idx')
            if idx is not None:
                true_angle = sample.get('true_angle_deg')
                clean_pred = sample.get('clean_pred_deg')
                if true_angle is not None and clean_pred is not None:
                    # Compute circular error
                    error = abs(((clean_pred - true_angle + 180) % 360) - 180)
                    est_map[idx] = error
        
        # Match with comparison results by test_dataset_idx
        results = data.get('results', [])
        vm_errors = []
        wg_errors = []
        for r in results:
            idx = r.get('test_dataset_idx')
            if idx is not None and idx in est_map:
                error = est_map[idx]
                vm_errors.append(error)
                wg_errors.append(error)
            else:
                vm_errors.append(None)
                wg_errors.append(None)
        
        if all(e is None for e in vm_errors):
            return None, None
        
        vm_errors = [float(e) if e is not None else float('inf') for e in vm_errors]
        wg_errors = [float(e) if e is not None else float('inf') for e in wg_errors]
        return vm_errors, wg_errors
    except Exception as e:
        print(f"Warning: Could not extract prediction errors from {est_path}: {e}")
        return None, None


def extract_radii_and_errors_from_alpha_trimming(data: Dict) -> Tuple[List[float], Optional[List[float]]]:
    """Extract radii and prediction errors from alpha-trimming JSON."""
    radii = []
    pred_errors = []
    
    if 'samples' in data:
        for sample in data.get('samples', []):
            r = sample.get('certified_radius', 0.0)
            radii.append(float(r))
            
            # Try to get prediction error
            error = sample.get('pred_error_deg', None)
            if error is not None:
                pred_errors.append(float(error))
            else:
                # Try to compute from true_angle and clean_pred
                true_angle = sample.get('true_angle_deg', None)
                clean_pred = sample.get('clean_pred_deg', None)
                if true_angle is not None and clean_pred is not None:
                    # Compute circular distance
                    error = abs(((clean_pred - true_angle + 180) % 360) - 180)
                    pred_errors.append(float(error))
                else:
                    pred_errors.append(None)
    
    # If we got errors for all samples, return them; otherwise return None
    if all(e is not None for e in pred_errors):
        return radii, [float(e) for e in pred_errors]
    else:
        return radii, None


def compute_certified_mean_distance(
    radii: List[float],
    pred_errors: List[float],
    R_threshold: float
) -> float:
    """
    Compute Certified Mean Distance at radius threshold R.
    
    Certified Mean Distance(R) = (1/|S_R|) * sum_{i in S_R} d(hat y_i, y_i)
    
    where S_R = {i : r_i >= R} is the set of samples with certified radius >= R.
    
    Args:
        radii: List of certified radii
        pred_errors: List of prediction errors (circular distances)
        R_threshold: Radius threshold
    
    Returns:
        Mean distance for certified samples, or NaN if no certified samples
    """
    if len(radii) == 0 or len(pred_errors) == 0:
        return float('nan')
    
    if len(radii) != len(pred_errors):
        return float('nan')
    
    # Find certified samples: r_i >= R
    certified_indices = [i for i, r in enumerate(radii) if r >= R_threshold]
    
    if len(certified_indices) == 0:
        return float('nan')
    
    # Compute mean distance for certified samples (not squared)
    certified_errors = [pred_errors[i] for i in certified_indices]
    mean_dist = np.mean(certified_errors)
    
    return float(mean_dist)


def load_all_sigma_data(
    comparison_dir: str,
    alpha_dir: str,
    sigmas: List[float]
) -> Tuple[Dict[str, Dict[float, Tuple[List[float], List[float]]]], Dict[str, Dict[float, Tuple[List[float], List[float]]]]]:
    """Load radii and prediction errors for all sigmas."""
    
    vm_data = {}  # sigma -> (radii, pred_errors)
    wg_data = {}  # sigma -> (radii, pred_errors)
    alpha_data = {}  # sigma -> (radii, pred_errors)
    
    # Load comparison files
    for sigma in sigmas:
        comparison_files = glob.glob(f"{comparison_dir}/comparison_vm_vs_wg_mnist_sigma{sigma}_eps*.json")
        if not comparison_files:
            continue
        
        # Use most recent file if multiple
        comparison_file = sorted(comparison_files)[-1]
        comparison_data = load_json(comparison_file)
        
        vm_radii, wg_radii, test_indices = extract_radii_from_comparison(comparison_data)
        vm_errors, wg_errors = extract_pred_errors_from_comparison(comparison_data, comparison_file)
        
        if len(vm_radii) > 0 and vm_errors is not None:
            vm_data[sigma] = (vm_radii, vm_errors)
        
        if len(wg_radii) > 0 and wg_errors is not None:
            wg_data[sigma] = (wg_radii, wg_errors)
    
    # Load alpha-trimming files
    for sigma in sigmas:
        alpha_files = glob.glob(f"{alpha_dir}/mnist_alpha_trimming_rotated_n100_sigma{sigma}_alpha*.json")
        if not alpha_files:
            continue
        
        # Use most recent file if multiple
        alpha_file = sorted(alpha_files)[-1]
        alpha_data_json = load_json(alpha_file)
        
        alpha_radii, alpha_errors = extract_radii_and_errors_from_alpha_trimming(alpha_data_json)
        
        if len(alpha_radii) > 0 and alpha_errors is not None:
            alpha_data[sigma] = (alpha_radii, alpha_errors)
    
    return (vm_data, wg_data), alpha_data


def find_best_sigma_overall(
    all_sigma_data: Dict[float, Tuple[List[float], List[float]]],
    R_values: List[float],
    min_valid_R: int = None
) -> float:
    """
    Find the best sigma for a method based on sum of mean distances across all R thresholds.
    
    For each sigma, compute mean distance at each R threshold, then sum these values.
    Select the sigma with the smallest sum (i.e., best overall prediction quality).
    
    Only considers sigmas that certify at a minimum number of R values to ensure
    fair comparison (default: all R values must be valid).
    
    Args:
        all_sigma_data: Dict mapping sigma to (radii, pred_errors)
        R_values: List of R thresholds to evaluate (typically [0.05, 0.10, 0.15, 0.20, 0.25])
        min_valid_R: Minimum number of R values that must have valid mean distances (default: len(R_values))
    
    Returns:
        best_sigma (or None if no data)
    """
    if min_valid_R is None:
        min_valid_R = len(R_values)  # Require all R values by default
    
    best_sigma = None
    best_sum_distance = float('inf')
    
    for sigma, (radii, pred_errors) in all_sigma_data.items():
        # Compute mean distance at each R threshold
        mean_distances = []
        for R in R_values:
            mean_dist = compute_certified_mean_distance(radii, pred_errors, R)
            mean_distances.append(mean_dist)
        
        # Count valid (non-NaN) mean distances
        valid_distances = [md for md in mean_distances if not np.isnan(md)]
        
        # Only consider sigmas that certify at minimum number of R values
        if len(valid_distances) >= min_valid_R:
            sum_distance = sum(valid_distances)
            
            if sum_distance < best_sum_distance:
                best_sum_distance = sum_distance
                best_sigma = sigma
    
    return best_sigma


def compute_mean_distance_at_fixed_sigma(
    all_sigma_data: Dict[float, Tuple[List[float], List[float]]],
    fixed_sigma: float,
    R_values: List[float]
) -> Tuple[List[float], float]:
    """
    Compute mean distance at a fixed sigma for all R thresholds.
    
    This ensures we're comparing prediction quality at the same noise level
    across different radius thresholds, providing a fair comparison.
    
    Returns:
        (mean_distances, fixed_sigma)
    """
    if fixed_sigma is None or fixed_sigma not in all_sigma_data:
        return [float('nan')] * len(R_values), fixed_sigma
    
    radii, pred_errors = all_sigma_data[fixed_sigma]
    mean_distances = []
    
    for R in R_values:
        mean_dist = compute_certified_mean_distance(radii, pred_errors, R)
        mean_distances.append(mean_dist)
    
    return mean_distances, fixed_sigma


def create_latex_table(
    mean_dist_table: Dict[str, List[float]],
    best_sigma_table: Dict[str, Dict[float, float]],
    R_values: List[float],
    output_file: str
):
    """Create LaTeX table for certified mean distance."""
    
    method_labels = {
        '(E, C) + M': '$(E, C) + M$',
        '(E, C, G) + M': '$(E, C, G) + M$',
        'Alpha-Trimming': '$\\alpha$-smoothing ($P=0.9$)',
        'alpha_trimming': '$\\alpha$-smoothing ($P=0.9$)'
    }
    
    lines = [
        "% Table: Certified Mean Distance at Different Radius Thresholds",
        "% Auto-generated using best sigma for each method at each threshold",
        "",
        "\\begin{table}[t]",
        "    \\centering",
        "    \\small",
        "    \\setlength{\\tabcolsep}{3pt}",
        "    \\renewcommand{\\arraystretch}{1.05}",
        "    \\caption{Certified mean distance (mean circular distance in degrees) at different radius thresholds $R$ (in pixels) on MNIST rotation task. For each method, we select a single $\\sigma$ value that minimizes the sum of mean distances across all $R$ thresholds (0.05, 0.10, 0.15, 0.20, 0.25), then compute mean distance at that fixed $\\sigma$ for all $R$ thresholds. This ensures fair comparison of prediction quality across different radius thresholds at the same noise level. Mean distance is computed only over samples with certified radius $r_i \\geq R$.}",
        "    \\label{tab:certified_mean_distance}",
        "    \\begin{tabular}{l c c c c c}",
        "        \\toprule",
        "        Method & \\multicolumn{5}{c}{Radius Threshold $R$ (pixels)} \\\\",
        "        \\cmidrule(lr){2-6}",
        f"        & {' & '.join([str(R) for R in R_values])} \\\\",
        "        \\midrule"
    ]
    
    # Add rows for each method
    for method_name, mean_dists in mean_dist_table.items():
        method_label = method_labels.get(method_name, method_name)
        
        # Format values (handle NaN)
        formatted_values = []
        for mean_dist in mean_dists:
            if np.isnan(mean_dist):
                formatted_values.append("---")
            else:
                formatted_values.append(f"{mean_dist:.2f}")
        
        lines.append(f"        {method_label} & {' & '.join(formatted_values)} \\\\")
    
    lines.extend([
        "        \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}"
    ])
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Saved certified mean distance table to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute Certified Mean Distance using best sigma for each method at each threshold'
    )
    parser.add_argument(
        '--comparison_dir',
        type=str,
        default='.',
        help='Directory containing comparison JSON files'
    )
    parser.add_argument(
        '--alpha_dir',
        type=str,
        default='outputs/mnist_alpha',
        help='Directory containing alpha-trimming JSON files'
    )
    parser.add_argument(
        '--R_values',
        type=float,
        nargs='+',
        default=[0.05, 0.10, 0.15, 0.20, 0.25],
        help='Radius thresholds to evaluate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='certified_mean_distance_table_best_sigma.tex',
        help='Output LaTeX table file'
    )
    
    args = parser.parse_args()
    
    # Load all sigma data
    sigmas = [0.06, 0.12, 0.25, 0.5, 0.75]
    (vm_data, wg_data), alpha_data = load_all_sigma_data(
        args.comparison_dir,
        args.alpha_dir,
        sigmas
    )
    
    # Filter R values to only include [0.05, 0.10, 0.15, 0.20, 0.25] for sigma selection
    R_values_for_selection = [R for R in args.R_values if R <= 0.25]
    
    # Compute mean distance for each method using FIXED best sigma (selected based on sum of mean distances)
    mean_dist_table = {}
    best_sigma_table = {}
    
    # (E, C) + M
    method_name = '(E, C) + M'
    if len(vm_data) > 0:
        # Select ONE best sigma based on sum of mean distances across R thresholds
        best_sigma = find_best_sigma_overall(vm_data, R_values_for_selection)
        # Use that fixed sigma for all R values
        mean_dists, fixed_sigma = compute_mean_distance_at_fixed_sigma(vm_data, best_sigma, args.R_values)
        mean_dist_table[method_name] = mean_dists
        best_sigma_table[method_name] = {R: fixed_sigma for R in args.R_values}
    
    # (E, C, G) + M
    method_name = '(E, C, G) + M'
    if len(wg_data) > 0:
        # Select ONE best sigma based on sum of mean distances across R thresholds
        best_sigma = find_best_sigma_overall(wg_data, R_values_for_selection)
        # Use that fixed sigma for all R values
        mean_dists, fixed_sigma = compute_mean_distance_at_fixed_sigma(wg_data, best_sigma, args.R_values)
        mean_dist_table[method_name] = mean_dists
        best_sigma_table[method_name] = {R: fixed_sigma for R in args.R_values}
    
    # Alpha-Trimming
    method_name = 'Alpha-Trimming'
    if len(alpha_data) > 0:
        # Select ONE best sigma based on sum of mean distances across R thresholds
        best_sigma = find_best_sigma_overall(alpha_data, R_values_for_selection)
        # Use that fixed sigma for all R values
        mean_dists, fixed_sigma = compute_mean_distance_at_fixed_sigma(alpha_data, best_sigma, args.R_values)
        mean_dist_table[method_name] = mean_dists
        best_sigma_table[method_name] = {R: fixed_sigma for R in args.R_values}
    
    # Print summary
    print("\nCertified Mean Distance Summary (using fixed best sigma per method):")
    print("=" * 70)
    for method_name, mean_dists in mean_dist_table.items():
        print(f"\n{method_name}:")
        best_sigmas = best_sigma_table.get(method_name, {})
        # All R values use the same sigma, so get it from any R
        fixed_sigma = list(best_sigmas.values())[0] if best_sigmas else 'N/A'
        print(f"  Fixed sigma: {fixed_sigma} (selected based on minimizing sum of mean distances across R thresholds)")
        for R, mean_dist in zip(args.R_values, mean_dists):
            if np.isnan(mean_dist):
                print(f"    R={R:.2f}: No certified samples")
            else:
                print(f"    R={R:.2f}: Mean Distance = {mean_dist:.2f}°")
    
    # Create LaTeX table
    create_latex_table(mean_dist_table, best_sigma_table, args.R_values, args.output)
    
    # Also save JSON
    json_output = args.output.replace('.tex', '.json')
    with open(json_output, 'w') as f:
        json.dump({
            'R_values': args.R_values,
            'certified_mean_distance': mean_dist_table,
            'best_sigma_per_R': best_sigma_table
        }, f, indent=2)
    print(f"✓ Saved JSON data to: {json_output}")


if __name__ == '__main__':
    main()
