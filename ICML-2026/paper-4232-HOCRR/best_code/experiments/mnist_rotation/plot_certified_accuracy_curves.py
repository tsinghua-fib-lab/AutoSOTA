#!/usr/bin/env python3
"""
Plot Certified Accuracy Curves

Creates the standard "certified accuracy" plot showing what fraction of test set
is certified at different radii. This is the standard plot in certification papers
(like Table 4 in Cohen et al. 2019).

Usage:
    python experiments/mnist_rotation/plot_certified_accuracy_curves.py \
        --bounded_radii comparison_vm_vs_wg_mnist_sigma0.06_eps10.0deg_20260111_165311.json \
        --alpha_trimming mnist_alpha_trimming_n100_20251106_173521.json \
        --output certified_accuracy_curves.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))


def load_json(json_path: str) -> Dict:
    """Load JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_radii_from_file(json_path: str, method: str = 'auto') -> Tuple[List[float], Optional[float], Optional[float], Optional[List[float]]]:
    """
    Extract certified radii from JSON file, along with sigma, alpha, and prediction errors if available.
    
    Args:
        json_path: Path to JSON file
        method: Which radius to extract:
            - 'auto': Auto-detect based on file structure
            - 'variance_mean': Extract radius_variance_mean
            - 'with_gradient': Extract radius_with_gradient
            - 'bounded': Extract radius (from bounded certifier)
            - 'alpha_trimming': Extract radius (from alpha-trimming)
    
    Returns:
        Tuple of (radii_list, sigma, alpha, pred_errors) where sigma, alpha, and pred_errors may be None
    """
    data = load_json(json_path)
    
    # Extract sigma and alpha from parameters if available
    params = data.get('parameters', {})
    sigma = params.get('sigma', None)
    alpha = params.get('alpha', None)
    
    # Auto-detect method based on file structure
    if method == 'auto':
        if 'results' in data:
            # Check first result to see what fields exist
            first_result = data['results'][0] if data['results'] else {}
            if 'radius_variance_mean' in first_result:
                method = 'variance_mean'
            elif 'radius_with_gradient' in first_result:
                method = 'with_gradient'
            elif 'radius' in first_result:
                method = 'bounded'
            else:
                raise ValueError(f"Could not auto-detect method from file: {json_path}")
        elif 'samples' in data or 'certified_radii' in data:
            method = 'alpha_trimming'
        else:
            raise ValueError(f"Unknown file structure: {json_path}")
    
    # Extract radii and prediction errors
    pred_errors = None
    if method == 'variance_mean':
        radii = [r.get('radius_variance_mean', 0.0) for r in data.get('results', [])]
        # Try to extract prediction errors from estimation file if available
        pred_errors = _extract_pred_errors_from_estimation_file(data, json_path, 'variance_mean')
    elif method == 'with_gradient':
        radii = [r.get('radius_with_gradient', 0.0) for r in data.get('results', [])]
        # Try to extract prediction errors from estimation file if available
        pred_errors = _extract_pred_errors_from_estimation_file(data, json_path, 'with_gradient')
    elif method == 'bounded':
        radii = [r.get('radius', 0.0) for r in data.get('results', [])]
    elif method == 'alpha_trimming':
        # Try 'samples' first (new format), then 'certified_radii' (old format)
        if 'samples' in data:
            radii = [r.get('certified_radius', 0.0) for r in data.get('samples', [])]
            # Extract prediction errors if available
            pred_errors = [r.get('pred_error_deg', None) for r in data.get('samples', [])]
            if all(e is None for e in pred_errors):
                pred_errors = None
            else:
                pred_errors = [float(e) if e is not None else float('inf') for e in pred_errors]
        else:
            radii = [r.get('certified_radius', 0.0) for r in data.get('certified_radii', [])]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return ([float(r) for r in radii if r is not None], sigma, alpha, pred_errors)


def _extract_pred_errors_from_estimation_file(data: Dict, json_path: str, method: str) -> Optional[List[float]]:
    """
    Extract prediction errors from the estimation file referenced in comparison JSON.
    
    Args:
        data: Loaded JSON data from comparison file
        json_path: Path to comparison JSON file (for resolving relative paths)
        method: 'variance_mean' or 'with_gradient'
    
    Returns:
        List of prediction errors in degrees, or None if not available
    """
    # Check if estimation file is referenced
    params = data.get('parameters', {})
    est_file = params.get('variance_gradient_file')
    
    if not est_file:
        return None
    
    # Resolve path relative to comparison file
    json_dir = Path(json_path).parent
    est_path = json_dir / est_file
    if not est_path.exists():
        # Try absolute path or relative to workspace root
        workspace_root = Path(__file__).resolve().parents[2]
        est_path = workspace_root / est_file
        if not est_path.exists():
            return None
    
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
        pred_errors = []
        for r in results:
            idx = r.get('test_dataset_idx')
            if idx is not None and idx in est_map:
                pred_errors.append(est_map[idx])
            else:
                # If we can't find it, use None (will be skipped)
                pred_errors.append(None)
        
        if all(e is None for e in pred_errors):
            return None
        
        return [float(e) if e is not None else float('inf') for e in pred_errors]
    except Exception as e:
        # If anything fails, return None (will use certified coverage instead)
        return None


def compute_certified_accuracy(
    radii: List[float], 
    R_threshold: float,
    pred_errors: Optional[List[float]] = None,
    tolerance: Optional[float] = None,
    normalize_by_certified: bool = False
) -> float:
    """
    Compute certified accuracy at radius threshold R.
    
    Two definitions are supported:
    1. Absolute (Cohen et al. 2019 style): cert_acc(R) = (1/n) * sum 1{correct AND certified}
    2. Conditional: cert_acc(R) = (1/|S_R|) * sum_{i in S_R} 1{correct}
    
    For regression: "correct" means prediction error <= tolerance.
    
    If pred_errors and tolerance are None, falls back to:
    fraction of samples with R_cert >= R_threshold (certified coverage).
    
    Args:
        radii: List of certified radii
        R_threshold: Radius threshold
        pred_errors: Optional list of prediction errors (for correctness check)
        tolerance: Optional tolerance for correctness (if None, skip correctness check)
        normalize_by_certified: If True, use conditional definition (normalize by |S_R|).
                                If False, use absolute definition (normalize by n). Default: False.
    
    Returns:
        Fraction of samples (absolute) or certified samples (conditional) that are correct
    """
    if len(radii) == 0:
        return 0.0
    
    if pred_errors is not None and tolerance is not None:
        # Count samples that are correct AND certified
        correct_certified = sum(1 for r, err in zip(radii, pred_errors) 
                               if err <= tolerance and r >= R_threshold)
        
        if normalize_by_certified:
            # Conditional definition: normalize by |S_R|
            certified_count = sum(1 for r in radii if r >= R_threshold)
            if certified_count == 0:
                return 0.0
            return correct_certified / certified_count
        else:
            # Absolute definition (Cohen et al. 2019): normalize by n
            return correct_certified / len(radii)
    else:
        # Fallback: just certified coverage
        certified = sum(1 for r in radii if r >= R_threshold)
        if normalize_by_certified:
            # Conditional: all certified samples are "correct" by default
            return 1.0 if certified > 0 else 0.0
        else:
            # Absolute: fraction of all samples that are certified
            return certified / len(radii)


def create_certified_accuracy_table(
    methods_data: Dict[str, List[float]],
    R_values: List[float],
    method_pred_errors: Optional[Dict[str, List[float]]] = None,
    method_tolerance: Optional[Dict[str, float]] = None,
    normalize_by_certified: bool = False
) -> Dict[str, List[float]]:
    """
    Create certified accuracy table.
    
    Args:
        methods_data: Dict mapping method name to list of radii
        R_values: List of radius thresholds
        method_pred_errors: Optional dict mapping method name to list of prediction errors
        method_tolerance: Optional dict mapping method name to tolerance for correctness
        normalize_by_certified: If True, use conditional definition (normalize by |S_R|).
                                If False, use absolute definition (normalize by n). Default: False.
        
    Returns:
        Dict mapping method name to list of certified accuracies
    """
    table = {}
    for method_name, radii in methods_data.items():
        pred_errors = method_pred_errors.get(method_name) if method_pred_errors else None
        tolerance = method_tolerance.get(method_name) if method_tolerance else None
        accuracies = [compute_certified_accuracy(radii, R, pred_errors, tolerance, normalize_by_certified) * 100 
                     for R in R_values]
        table[method_name] = accuracies
    return table


def plot_certified_accuracy_curves(
    methods_data: Dict[str, List[float]],
    R_values: List[float],
    output_path: str,
    title: str = "Certified Accuracy vs Radius",
    method_metadata: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
    method_pred_errors: Optional[Dict[str, List[float]]] = None,
    method_tolerance: Optional[Dict[str, float]] = None,
    normalize_by_certified: bool = False
):
    """
    Create certified accuracy curves plot.
    
    Args:
        methods_data: Dict mapping method name to list of radii
        R_values: List of radius thresholds
        output_path: Path to save plot
        title: Plot title
        method_metadata: Optional dict mapping method name to {'sigma': float, 'alpha': float}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Build labels with sigma and alpha if available
    def format_label(method_name: str) -> str:
        label = method_name
        if method_metadata and method_name in method_metadata:
            meta = method_metadata[method_name]
            parts = []
            if meta.get('sigma') is not None:
                parts.append(f"σ={meta['sigma']}")
            if meta.get('alpha') is not None:
                parts.append(f"α={meta['alpha']}")
            if parts:
                label += f" ({', '.join(parts)})"
        return label
    
    # Plot 1: Curves (CDF-like)
    for method_name, radii in methods_data.items():
        pred_errors = method_pred_errors.get(method_name) if method_pred_errors else None
        tolerance = method_tolerance.get(method_name) if method_tolerance else None
        accuracies = [compute_certified_accuracy(radii, R, pred_errors, tolerance, normalize_by_certified) * 100 for R in R_values]
        ax1.plot(R_values, accuracies, marker='o', label=format_label(method_name), linewidth=2, markersize=6)
    
    ax1.set_xlabel('Certified Radius R', fontsize=12)
    ax1.set_ylabel('Certified Accuracy (%)', fontsize=12)
    ax1.set_title('Certified Accuracy Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Plot 2: Table visualization (bar chart)
    table = create_certified_accuracy_table(methods_data, R_values, method_pred_errors, method_tolerance, normalize_by_certified)
    
    x = np.arange(len(R_values))
    width = 0.8 / len(methods_data)
    
    for i, (method_name, accuracies) in enumerate(table.items()):
        offset = (i - len(methods_data) / 2 + 0.5) * width
        ax2.bar(x + offset, accuracies, width, label=format_label(method_name), alpha=0.8)
    
    ax2.set_xlabel('Radius Threshold', fontsize=12)
    ax2.set_ylabel('Certified Accuracy (%)', fontsize=12)
    ax2.set_title('Certified Accuracy by Radius', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{R:.2f}' for R in R_values], rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 105])
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}\n")
    plt.close()


def print_certified_accuracy_table(
    methods_data: Dict[str, List[float]],
    R_values: List[float],
    method_metadata: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
    method_pred_errors: Optional[Dict[str, List[float]]] = None,
    method_tolerance: Optional[Dict[str, float]] = None,
    normalize_by_certified: bool = False
):
    """
    Print certified accuracy table (like Table 4 in Cohen et al. 2019).
    
    Args:
        methods_data: Dict mapping method name to list of radii
        R_values: List of radius thresholds
        method_metadata: Optional dict mapping method name to {'sigma': float, 'alpha': float}
        method_pred_errors: Optional dict mapping method name to list of prediction errors
        method_tolerance: Optional dict mapping method name to tolerance for correctness
        normalize_by_certified: If True, use conditional definition (normalize by |S_R|).
                                If False, use absolute definition (normalize by n). Default: False.
    """
    table = create_certified_accuracy_table(methods_data, R_values, method_pred_errors, method_tolerance, normalize_by_certified)
    
    def format_method_name(method_name: str) -> str:
        label = method_name
        if method_metadata and method_name in method_metadata:
            meta = method_metadata[method_name]
            parts = []
            if meta.get('sigma') is not None:
                parts.append(f"σ={meta['sigma']}")
            if meta.get('alpha') is not None:
                parts.append(f"α={meta['alpha']}")
            if parts:
                label += f" ({', '.join(parts)})"
        return label
    
    print("\n" + "="*80)
    print("CERTIFIED ACCURACY TABLE")
    print("="*80)
    print(f"{'Method':<50}", end="")
    for R in R_values:
        print(f"  R={R:.2f}", end="")
    print()
    print("-"*80)
    
    for method_name, accuracies in table.items():
        formatted_name = format_method_name(method_name)
        print(f"{formatted_name:<50}", end="")
        for acc in accuracies:
            print(f"  {acc:5.1f}%", end="")
        print()
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot certified accuracy curves")
    parser.add_argument(
        "--bounded_radii",
        type=str,
        nargs='+',
        default=[],
        help="JSON file(s) with bounded certifier radii (can specify multiple for different methods)"
    )
    parser.add_argument(
        "--variance_mean_radii",
        type=str,
        default=None,
        help="JSON file with (E, C) + M radii"
    )
    parser.add_argument(
        "--with_gradient_radii",
        type=str,
        default=None,
        help="JSON file with (E, C, G) + M radii"
    )
    parser.add_argument(
        "--alpha_trimming",
        type=str,
        nargs='+',
        default=[],
        help="JSON file(s) with alpha-trimming radii"
    )
    parser.add_argument(
        "--R_values",
        type=float,
        nargs='+',
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        help="Radius thresholds to evaluate"
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
    parser.add_argument(
        "--correctness_tolerance_deg",
        type=float,
        default=None,
        help="Tolerance for correctness check (degrees). If None, uses certified coverage instead of certified accuracy. Should be different from eps_y used in certification."
    )
    parser.add_argument(
        "--normalize_by_certified",
        action="store_true",
        default=False,
        help="If set, use conditional certified accuracy (normalize by |S_R|). Otherwise, use absolute certified accuracy (normalize by n, Cohen et al. 2019 style). Default: False (absolute)."
    )
    
    args = parser.parse_args()
    
    # Collect all methods and their metadata (sigma, alpha)
    methods_data = {}
    method_metadata = {}  # Maps method name to {'sigma': float, 'alpha': float}
    method_pred_errors = {}  # Maps method name to list of prediction errors
    
    # Load (E, C) + M radii
    if args.variance_mean_radii:
        radii, sigma, alpha, pred_errors = extract_radii_from_file(args.variance_mean_radii, 'variance_mean')
        methods_data['(E, C) + M'] = radii
        method_metadata['(E, C) + M'] = {'sigma': sigma, 'alpha': None}
        if pred_errors is not None:
            method_pred_errors['(E, C) + M'] = pred_errors
        print(f"Loaded (E, C) + M: {len(radii)} samples from {args.variance_mean_radii}")
        if sigma is not None:
            print(f"  σ={sigma}")
    
    # Load (E, C, G) + M radii
    if args.with_gradient_radii:
        radii, sigma, alpha, pred_errors = extract_radii_from_file(args.with_gradient_radii, 'with_gradient')
        methods_data['(E, C, G) + M'] = radii
        method_metadata['(E, C, G) + M'] = {'sigma': sigma, 'alpha': None}
        if pred_errors is not None:
            method_pred_errors['(E, C, G) + M'] = pred_errors
        print(f"Loaded (E, C, G) + M: {len(radii)} samples from {args.with_gradient_radii}")
        if sigma is not None:
            print(f"  σ={sigma}")
    
    # Load from comparison file (if single file with both methods)
    if args.bounded_radii:
        for file_path in args.bounded_radii:
            # Try to extract both methods
            try:
                radii_vm, sigma_vm, alpha_vm, pred_errors_vm = extract_radii_from_file(file_path, 'variance_mean')
                methods_data['(E, C) + M'] = radii_vm
                method_metadata['(E, C) + M'] = {'sigma': sigma_vm, 'alpha': None}
                if pred_errors_vm is not None:
                    method_pred_errors['(E, C) + M'] = pred_errors_vm
                print(f"Loaded (E, C) + M: {len(radii_vm)} samples from {file_path}")
                if sigma_vm is not None:
                    print(f"  σ={sigma_vm}")
                if pred_errors_vm is not None:
                    print(f"  ✓ Prediction errors available for correctness check")
            except (KeyError, ValueError):
                pass
            
            try:
                radii_wg, sigma_wg, alpha_wg, pred_errors_wg = extract_radii_from_file(file_path, 'with_gradient')
                methods_data['(E, C, G) + M'] = radii_wg
                method_metadata['(E, C, G) + M'] = {'sigma': sigma_wg, 'alpha': None}
                if pred_errors_wg is not None:
                    method_pred_errors['(E, C, G) + M'] = pred_errors_wg
                print(f"Loaded (E, C, G) + M: {len(radii_wg)} samples from {file_path}")
                if sigma_wg is not None:
                    print(f"  σ={sigma_wg}")
                if pred_errors_wg is not None:
                    print(f"  ✓ Prediction errors available for correctness check")
            except (KeyError, ValueError):
                pass
    
    # Load alpha-trimming
    if args.alpha_trimming:
        for file_path in args.alpha_trimming:
            radii, sigma, alpha, pred_errors = extract_radii_from_file(file_path, 'alpha_trimming')
            # Create unique method name with sigma and alpha
            method_name = 'Alpha-Trimming'
            if sigma is not None and alpha is not None:
                method_name = f'Alpha-Trimming (σ={sigma}, α={alpha})'
            elif sigma is not None:
                method_name = f'Alpha-Trimming (σ={sigma})'
            elif alpha is not None:
                method_name = f'Alpha-Trimming (α={alpha})'
            
            methods_data[method_name] = radii
            method_metadata[method_name] = {'sigma': sigma, 'alpha': alpha}
            if pred_errors is not None:
                method_pred_errors[method_name] = pred_errors
            print(f"Loaded {method_name}: {len(radii)} samples from {file_path}")
            if sigma is not None:
                print(f"  σ={sigma}")
            if alpha is not None:
                print(f"  α={alpha}")
            if pred_errors is not None:
                print(f"  ✓ Prediction errors available for correctness check")
    
    if len(methods_data) == 0:
        parser.error("No methods specified! Use --bounded_radii, --alpha_trimming, etc.")
    
    print(f"\nLoaded {len(methods_data)} method(s) with radii\n")
    
    # Set up tolerance for correctness check
    method_tolerance = {}
    if args.correctness_tolerance_deg is not None:
        print(f"Using correctness tolerance: {args.correctness_tolerance_deg}°")
        print("  (Computing Cohen-style certified accuracy: correct AND certified)")
        for method_name in methods_data.keys():
            method_tolerance[method_name] = args.correctness_tolerance_deg
    else:
        print("No correctness tolerance specified - using certified coverage")
        print("  (Computing: fraction with radius >= R, not checking correctness)")
    
    # Print table
    print_certified_accuracy_table(methods_data, args.R_values, method_metadata, method_pred_errors, method_tolerance, args.normalize_by_certified)
    
    # Create plot
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Build filename with sigma and alpha if available
        filename_parts = ["certified_accuracy_curves"]
        
        # Collect unique sigma values
        sigmas = set()
        alphas = set()
        for meta in method_metadata.values():
            if meta.get('sigma') is not None:
                sigmas.add(meta['sigma'])
            if meta.get('alpha') is not None:
                alphas.add(meta['alpha'])
        
        if len(sigmas) == 1:
            filename_parts.append(f"sigma{list(sigmas)[0]:.2f}".replace('.', 'p'))
        elif len(sigmas) > 1:
            filename_parts.append(f"sigma{min(sigmas):.2f}-{max(sigmas):.2f}".replace('.', 'p'))
        
        if len(alphas) == 1:
            filename_parts.append(f"alpha{list(alphas)[0]:.2f}".replace('.', 'p'))
        elif len(alphas) > 1:
            filename_parts.append(f"alpha{min(alphas):.2f}-{max(alphas):.2f}".replace('.', 'p'))
        
        filename_parts.append(timestamp)
        args.output = "_".join(filename_parts) + ".png"
    
    if args.title is None:
        # Build title with sigma and alpha if available
        title_parts = ["Certified Accuracy vs Radius"]
        sigmas = set()
        alphas = set()
        for meta in method_metadata.values():
            if meta.get('sigma') is not None:
                sigmas.add(meta['sigma'])
            if meta.get('alpha') is not None:
                alphas.add(meta['alpha'])
        
        if len(sigmas) == 1:
            title_parts.append(f"(σ={list(sigmas)[0]})")
        if len(alphas) == 1:
            title_parts.append(f"(α={list(alphas)[0]})")
        
        args.title = " ".join(title_parts) if len(title_parts) > 1 else "Certified Accuracy vs Radius"
    
    plot_certified_accuracy_curves(methods_data, args.R_values, args.output, args.title, method_metadata, method_pred_errors, method_tolerance, args.normalize_by_certified)
    
    # Save table as JSON
    table = create_certified_accuracy_table(methods_data, args.R_values, method_pred_errors, method_tolerance, args.normalize_by_certified)
    table_json = {
        'timestamp': datetime.now().isoformat(),
        'R_values': args.R_values,
        'certified_accuracy': table,
        'n_samples': {name: len(radii) for name, radii in methods_data.items()},
        'method_metadata': method_metadata
    }
    
    table_output = args.output.replace('.png', '_table.json')
    with open(table_output, 'w') as f:
        json.dump(table_json, f, indent=2)
    
    print(f"✓ Saved table to: {table_output}\n")


if __name__ == "__main__":
    main()

