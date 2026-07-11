#!/usr/bin/env python3
"""
Compute Certified Accuracy Table with Best Sigma Selection

For each method, finds the sigma value that maximizes certified accuracy at each R threshold,
then creates a comparison table including all methods (E, C)+M, (E, C, G)+M, and alpha-smoothing.

Usage:
    python experiments/mnist_rotation/compute_certified_accuracy_best_sigma.py \
        --comparison_dir . \
        --alpha_dir outputs/mnist_alpha \
        --tolerance 10.0 \
        --R_values 0.05 0.10 0.15 0.20 0.25 \
        --output certified_accuracy_table_best_sigma.tex
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import glob
import sys

# Add path to import from plot_certified_accuracy_curves
sys.path.append(str(Path(__file__).resolve().parent))
from plot_certified_accuracy_curves import (
    load_json,
    extract_radii_from_file,
    compute_certified_accuracy
)


def extract_radii_from_comparison(data: Dict, json_path: str = None) -> Tuple[List[float], List[float], Optional[List[float]], Optional[List[float]]]:
    """
    Extract radii and prediction errors from comparison JSON file.
    
    Returns:
        (vm_radii, wg_radii, vm_pred_errors, wg_pred_errors)
    """
    vm_radii = []
    wg_radii = []
    vm_pred_errors = []
    wg_pred_errors = []
    
    # Try 'results' first (comparison files), then 'samples' (other formats)
    data_list = data.get('results', data.get('samples', []))
    for sample in data_list:
        # Extract (E, C) + M radius
        vm_radius = sample.get('radius_variance_mean', 0.0)
        if vm_radius is not None:
            vm_radii.append(float(vm_radius))
        
        # Extract (E, C, G) + M radius
        wg_radius = sample.get('radius_with_gradient', 0.0)
        if wg_radius is not None:
            wg_radii.append(float(wg_radius))
        
        # Extract prediction errors if available directly
        clean_error_deg = sample.get('clean_error_deg')
        if clean_error_deg is not None:
            vm_pred_errors.append(float(clean_error_deg))
            wg_pred_errors.append(float(clean_error_deg))
        else:
            vm_pred_errors.append(None)
            wg_pred_errors.append(None)
    
    # If prediction errors not found directly, try to extract from estimation file
    if all(e is None for e in vm_pred_errors) and json_path:
        vm_pred_errors, wg_pred_errors = _extract_pred_errors_from_estimation_file(data, json_path)
    
    # Filter out None values if any
    vm_pred_errors = [e for e in vm_pred_errors if e is not None] if any(e is not None for e in vm_pred_errors) else None
    wg_pred_errors = [e for e in wg_pred_errors if e is not None] if any(e is not None for e in wg_pred_errors) else None
    
    return vm_radii, wg_radii, vm_pred_errors, wg_pred_errors


def _extract_pred_errors_from_estimation_file(data: Dict, json_path: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Extract prediction errors from the estimation file referenced in comparison JSON.
    
    Returns:
        (vm_pred_errors, wg_pred_errors) or (None, None) if not available
    """
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
        results = data.get('results', data.get('samples', []))
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


def load_all_sigma_data(
    comparison_dir: str,
    alpha_dir: str,
    sigmas: List[float],
    tolerance: float
) -> Tuple[Dict[str, Dict[float, Tuple[List[float], Optional[List[float]]]]], Dict[str, Dict[float, Tuple[List[float], Optional[List[float]]]]]]:
    """
    Load radii and prediction errors for all methods across all sigma values.
    
    Returns:
        (bounded_data, alpha_data)
        where bounded_data = {'variance_mean': {sigma: (radii, pred_errors)}, 'with_gradient': {...}}
        and alpha_data = {sigma: (radii, pred_errors)}
    """
    bounded_data = {
        'variance_mean': {},
        'with_gradient': {}
    }
    alpha_data = {}
    
    # Load comparison files for bounded certifiers
    for sigma in sigmas:
        comparison_files = glob.glob(f"{comparison_dir}/comparison_vm_vs_wg_mnist_sigma{sigma}_eps*.json")
        if not comparison_files:
            continue
        
        # Use most recent file if multiple
        comparison_file = sorted(comparison_files)[-1]
        try:
            comparison_data = load_json(comparison_file)
            vm_radii, wg_radii, vm_pred_errors, wg_pred_errors = extract_radii_from_comparison(comparison_data, comparison_file)
            
            if len(vm_radii) > 0:
                bounded_data['variance_mean'][sigma] = (vm_radii, vm_pred_errors)
            if len(wg_radii) > 0:
                bounded_data['with_gradient'][sigma] = (wg_radii, wg_pred_errors)
        except Exception as e:
            print(f"Warning: Failed to load {comparison_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Load alpha-smoothing files
    for sigma in sigmas:
        alpha_files = glob.glob(f"{alpha_dir}/mnist_alpha_trimming_rotated_n100_sigma{sigma}_*.json")
        if not alpha_files:
            continue
        
        # Use most recent file if multiple
        alpha_file = sorted(alpha_files)[-1]
        try:
            radii, _, _, pred_errors = extract_radii_from_file(alpha_file, 'alpha_trimming')
            if len(radii) > 0:
                alpha_data[sigma] = (radii, pred_errors)
        except Exception as e:
            print(f"Warning: Failed to load {alpha_file}: {e}")
            continue
    
    return bounded_data, alpha_data


def find_best_sigma_overall(
    all_sigma_data: Dict[float, Tuple[List[float], Optional[List[float]]]],
    R_values: List[float],
    tolerance: float,
    normalize_by_certified: bool = False,
    min_valid_R: int = None
) -> float:
    """
    Find the best sigma for a method based on sum of certified accuracies across all R thresholds.
    
    For each sigma, compute certified accuracy at each R threshold, then sum these values.
    Select the sigma with the largest sum (i.e., best overall certified accuracy).
    
    Only considers sigmas that certify at a minimum number of R values to ensure
    fair comparison (default: all R values must be valid).
    
    Args:
        all_sigma_data: Dict mapping sigma to (radii, pred_errors)
        R_values: List of R thresholds to evaluate
        tolerance: Correctness tolerance in degrees
        normalize_by_certified: If True, use conditional definition (normalize by |S_R|)
        min_valid_R: Minimum number of R values that must have valid accuracies (default: len(R_values))
    
    Returns:
        best_sigma (or None if no data)
    """
    if min_valid_R is None:
        min_valid_R = len(R_values)  # Require all R values by default
    
    best_sigma = None
    best_sum_accuracy = -1
    
    for sigma, (radii, pred_errors) in all_sigma_data.items():
        # Compute certified accuracy at each R threshold
        accuracies = []
        for R in R_values:
            acc = compute_certified_accuracy(
                radii, R, pred_errors, tolerance, normalize_by_certified
            ) * 100  # Convert to percentage
            accuracies.append(acc)
        
        # Count valid (non-NaN) accuracies - 0% is valid (means no samples are correct AND certified)
        valid_accuracies = [acc for acc in accuracies if not np.isnan(acc)]
        
        # Only consider sigmas that have valid results at minimum number of R values
        # Note: 0% accuracy is valid (it means 0 samples are correct AND certified at that R)
        if len(valid_accuracies) >= min_valid_R:
            sum_accuracy = sum(accuracies)
            
            if sum_accuracy > best_sum_accuracy:
                best_sum_accuracy = sum_accuracy
                best_sigma = sigma
    
    return best_sigma


def compute_certified_accuracy_at_fixed_sigma(
    all_sigma_data: Dict[float, Tuple[List[float], Optional[List[float]]]],
    fixed_sigma: float,
    R_values: List[float],
    tolerance: float,
    normalize_by_certified: bool = False
) -> Tuple[List[float], float]:
    """
    Compute certified accuracy at a fixed sigma for all R thresholds.
    
    This ensures we're comparing certified accuracy at the same noise level
    across different radius thresholds, providing a fair comparison.
    
    Returns:
        (accuracies, fixed_sigma)
    """
    if fixed_sigma is None or fixed_sigma not in all_sigma_data:
        return [0.0] * len(R_values), fixed_sigma
    
    radii, pred_errors = all_sigma_data[fixed_sigma]
    accuracies = []
    
    for R in R_values:
        acc = compute_certified_accuracy(
            radii, R, pred_errors, tolerance, normalize_by_certified
        ) * 100  # Convert to percentage
        accuracies.append(acc)
    
    return accuracies, fixed_sigma


def create_latex_table(
    methods_data: Dict[str, Tuple[List[float], List[float]]],
    R_values: List[float],
    output_file: str,
    normalize_by_certified: bool = False
):
    """
    Create LaTeX table comparing certified accuracy at best sigma for each method.
    
    Args:
        methods_data: Dict mapping method_name -> (accuracies, sigmas)
        R_values: List of R thresholds
        output_file: Output LaTeX file path
    """
    with open(output_file, 'w') as f:
        f.write("% Auto-generated certified accuracy table (one fixed sigma per method)\n")
        f.write("\\begin{table}[t]\n")
        f.write("    \\centering\n")
        f.write("    \\small\n")
        f.write("    \\setlength{\\tabcolsep}{3pt}\n")
        f.write("    \\renewcommand{\\arraystretch}{1.05}\n")
        # Collect sigma values for caption
        sigma_info = []
        for method_name, (accuracies, sigmas) in methods_data.items():
            fixed_sigma = sigmas[0] if sigmas and sigmas[0] is not None else None
            if fixed_sigma is not None:
                if method_name == 'variance_mean':
                    sigma_info.append(f"$(E, C) + M$ uses $\\sigma={fixed_sigma:.2f}$")
                elif method_name == 'with_gradient':
                    sigma_info.append(f"$(E, C, G) + M$ uses $\\sigma={fixed_sigma:.2f}$")
                elif method_name == 'alpha_trimming':
                    sigma_info.append(f"$\\alpha$-smoothing uses $\\sigma={fixed_sigma:.2f}$")
        
        sigma_text = ", ".join(sigma_info) if sigma_info else ""
        norm_type = "conditional" if normalize_by_certified else "absolute"
        f.write("\\caption{Certified accuracy (\\%, " + norm_type + ") at different radius thresholds $R$ (in pixels) on MNIST rotation task. ")
        f.write("For each method, we select a single $\\sigma$ value that maximizes the sum of certified accuracies across all thresholds, then report accuracies at that fixed $\\sigma$ for all $R$ values. ")
        if sigma_text:
            f.write(sigma_text + ".}\n")
        else:
            f.write("}\n")
        f.write("    \\label{tab:certified_accuracy_mnist}\n")
        f.write("    \\begin{tabular}{l" + "c" * len(R_values) + "}\n")
        f.write("    \\toprule\n")
        
        # Header row
        f.write("    Method")
        for R in R_values:
            f.write(f" & $R={R:.2f}$")
        f.write(" \\\\\n")
        f.write("    \\midrule\n")
        
        # Data rows
        for method_name, (accuracies, sigmas) in methods_data.items():
            # Format method name
            if method_name == 'variance_mean':
                method_label = '$(E, C) + M$'
            elif method_name == 'with_gradient':
                method_label = '$(E, C, G) + M$'
            elif method_name == 'alpha_trimming':
                method_label = '$\\alpha$-smoothing ($P=0.9$)'
            else:
                method_label = method_name
            
            f.write(f"    {method_label}")
            # All R values use the same sigma (first one, since we use fixed sigma per method)
            for acc in accuracies:
                f.write(f" & {acc:.0f}\\%")
            f.write(" \\\\\n")
        
        f.write("    \\bottomrule\n")
        f.write("    \\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ Saved certified accuracy table to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compute certified accuracy table with best sigma selection")
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
        default=None,
        help="Output LaTeX table file (auto-generated if not specified based on normalize_by_certified flag)"
    )
    parser.add_argument(
        "--normalize_by_certified",
        action="store_true",
        default=False,
        help="Use conditional certified accuracy (normalize by |S_R|). Default: False (absolute, normalize by n)."
    )
    
    args = parser.parse_args()
    
    # Auto-generate output filename if not specified
    if args.output is None:
        if args.normalize_by_certified:
            args.output = "certified_accuracy_table_best_sigma_conditional.tex"
        else:
            args.output = "certified_accuracy_table_best_sigma_absolute.tex"
    
    print("Loading data for all sigma values...")
    bounded_data, alpha_data = load_all_sigma_data(
        args.comparison_dir, args.alpha_dir, args.sigmas, args.tolerance
    )
    
    print(f"\nLoaded data:")
    print(f"  (E, C) + M: {len(bounded_data['variance_mean'])} sigma values")
    print(f"  (E, C, G) + M: {len(bounded_data['with_gradient'])} sigma values")
    print(f"  Alpha-smoothing: {len(alpha_data)} sigma values")
    
    # Find best sigma for each method (one fixed sigma per method)
    methods_data = {}
    
    for method_key, method_name in [('variance_mean', '(E, C) + M'), ('with_gradient', '(E, C, G) + M')]:
        if method_key in bounded_data and len(bounded_data[method_key]) > 0:
            print(f"\nFinding best sigma for {method_name}...")
            best_sigma = find_best_sigma_overall(
                bounded_data[method_key], args.R_values, args.tolerance, args.normalize_by_certified
            )
            if best_sigma is not None:
                accuracies, _ = compute_certified_accuracy_at_fixed_sigma(
                    bounded_data[method_key], best_sigma, args.R_values, args.tolerance, args.normalize_by_certified
                )
                methods_data[method_key] = (accuracies, [best_sigma] * len(args.R_values))
                print(f"  Best sigma: {best_sigma}")
                print(f"  Accuracies: {[f'{a:.1f}%' for a in accuracies]}")
            else:
                print(f"  Warning: No valid sigma found for {method_name}")
    
    if len(alpha_data) > 0:
        print(f"\nFinding best sigma for Alpha-smoothing...")
        best_sigma = find_best_sigma_overall(
            alpha_data, args.R_values, args.tolerance, args.normalize_by_certified
        )
        if best_sigma is not None:
            accuracies, _ = compute_certified_accuracy_at_fixed_sigma(
                alpha_data, best_sigma, args.R_values, args.tolerance, args.normalize_by_certified
            )
            methods_data['alpha_trimming'] = (accuracies, [best_sigma] * len(args.R_values))
            print(f"  Best sigma: {best_sigma}")
            print(f"  Accuracies: {[f'{a:.1f}%' for a in accuracies]}")
        else:
            print(f"  Warning: No valid sigma found for Alpha-smoothing")
    
    # Create LaTeX table
    print(f"\nCreating LaTeX table...")
    create_latex_table(methods_data, args.R_values, args.output, args.normalize_by_certified)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
