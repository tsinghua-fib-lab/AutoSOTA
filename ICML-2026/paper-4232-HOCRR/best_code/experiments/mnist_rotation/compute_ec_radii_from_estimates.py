#!/usr/bin/env python3
"""
Compute (E, C)+M certified radii from 80 percent confidence estimation results.

This script computes certified radii using BoundedCertifierVarianceMean (E, C)+M
from estimation results with 80 percent confidence level.

Usage:
    # Local run
    python experiments/mnist_rotation/compute_ec_radii_from_estimates.py \
        --estimation_file estimation_results_80pct/mnist_rotation_full_cert_rotated_n100_sigma0.06_N10000_conf80pct.json \
        --eps_y_deg 10.0 \
        --N 10000 \
        --trial 0

    # On Hyak
    sbatch configs/slurm/<ec_radii_template>.sbatch \
        estimation_results_80pct/mnist_rotation_full_cert_rotated_n100_sigma0.06_N10000_conf80pct.json \
        10.0
"""

import json
import numpy as np
import argparse
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, List
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

# Add src path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from regression_certifiers.certify.bounded_fn_certifier_variance_mean import BoundedCertifierVarianceMean


def load_estimation_results(json_path: str) -> Dict:
    """Load estimation results JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def compute_ec_radii(
    estimation_data: Dict,
    eps_y_deg: float,
    N: int,
    trial_idx: int = 0,
    ci_type: str = 'analytical',
    confidence: float = 0.95,
    max_samples: int = None,
) -> List[Dict]:
    """
    Compute (E, C)+M certified radii from estimation results.
    
    Args:
        estimation_data: Data from mnist_rotation_full_certification.py
        eps_y_deg: Output tolerance in degrees
        N: Sample size to use (e.g., 10000)
        trial_idx: Which trial to use (default 0)
        ci_type: 'analytical' or 'bootstrap'
        confidence: Confidence level for certifier (default 0.95)
        
    Returns:
        List of dicts with certified radii and metadata
    """
    # Extract parameters
    sigma = estimation_data['parameters']['sigma']
    eps_y_rad = np.radians(eps_y_deg)
    
    # Initialize (E, C)+M certifier with M = π (angles in radians)
    M = np.pi
    certifier = BoundedCertifierVarianceMean(
        sigma=sigma,
        M=M,
        eps_y=eps_y_rad,
        confidence=confidence,
        quadrature_points=60
    )
    
    print(f"\n{'='*80}")
    print(f"Computing (E, C)+M Certified Radii")
    print(f"{'='*80}")
    print(f"Parameters:")
    print(f"  σ = {sigma}")
    print(f"  ε_y = {eps_y_deg}° = {eps_y_rad:.4f} rad")
    print(f"  M = π = {M:.4f} rad")
    print(f"  N = {N}")
    print(f"  Trial = {trial_idx}")
    print(f"  CI type = {ci_type}")
    print(f"  Confidence = {confidence}")
    print(f"  Estimation confidence = {estimation_data['parameters'].get('confidence', 'unknown')}")
    print(f"{'='*80}\n")
    
    results = []
    samples = estimation_data['samples']
    
    # Filter valid samples
    valid_samples = []
    for i, sample in enumerate(samples):
        if str(N) not in sample.get('results_by_N', {}):
            continue
        trials = sample['results_by_N'][str(N)]
        if trial_idx >= len(trials):
            continue
        valid_samples.append((i, sample))
    
    print(f"Found {len(valid_samples)} valid samples (with N={N}, trial={trial_idx})")
    
    # Limit samples if requested (for testing)
    if max_samples is not None and max_samples > 0:
        valid_samples = valid_samples[:max_samples]
        print(f"Processing first {len(valid_samples)} samples (--max_samples={max_samples})\n")
    else:
        print(f"Processing all {len(valid_samples)} samples\n")
    
    # Process samples with progress bar
    for i, sample in tqdm(valid_samples, desc="Computing (E, C)+M radii", unit="sample", ncols=100):
        estimates = sample['results_by_N'][str(N)][trial_idx]
        
        # Extract upper bounds for variance
        if ci_type == 'analytical':
            C_ucb = estimates['C_upper_analytical']
        else:
            C_ucb = estimates.get('C_upper_bootstrap', estimates['C_upper_analytical'])
        
        # Extract mean estimate
        E_est = estimates.get('g_z_hat', 0.0)
        
        # Compute certified radius
        try:
            radius = certifier.certify_point_from_estimates(C_ucb, E_est)
            error = None
        except Exception as e:
            radius = None
            error = str(e)
            print(f"\n⚠️  Sample {i}: Error computing radius: {e}")
        
        # Collect results
        result = {
            'sample_idx': sample.get('sample_idx', i),
            'test_dataset_idx': sample.get('test_dataset_idx', sample.get('image_idx', i)),
            'digit_label': sample.get('digit_label', None),
            'radius': float(radius) if radius is not None else None,
            'error': error,
            'C_hat': float(estimates['C_hat']),
            'C_ucb': float(C_ucb),
            'E_est': float(E_est),
            'clean_pred_deg': sample.get('clean_pred_deg', None),
            'N': N,
            'trial': trial_idx
        }
        
        results.append(result)
    
    print(f"\n✓ Computed {len(results)} certified radii")
    successful = sum(1 for r in results if r['radius'] is not None)
    print(f"  Successful: {successful}/{len(results)}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute (E, C)+M certified radii from 80 percent confidence estimation results'
    )
    parser.add_argument(
        '--estimation_file',
        type=str,
        required=True,
        help='Path to estimation results JSON file (80 percent confidence)'
    )
    parser.add_argument(
        '--eps_y_deg',
        type=float,
        default=10.0,
        help='Output tolerance in degrees (default: 10.0)'
    )
    parser.add_argument(
        '--N',
        type=int,
        default=10000,
        help='Sample size to use (default: 10000)'
    )
    parser.add_argument(
        '--trial',
        type=int,
        default=0,
        help='Trial index (default: 0)'
    )
    parser.add_argument(
        '--ci_type',
        type=str,
        choices=['analytical', 'bootstrap'],
        default='analytical',
        help='CI type to use (default: analytical)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.95,
        help='Confidence level for certifier (default: 0.95)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (auto-generated if not provided)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing, default: all)'
    )
    
    args = parser.parse_args()
    
    # Load estimation results
    print(f"Loading estimation results from: {args.estimation_file}")
    estimation_data = load_estimation_results(args.estimation_file)
    
    # Compute radii
    results = compute_ec_radii(
        estimation_data,
        args.eps_y_deg,
        args.N,
        args.trial,
        args.ci_type,
        args.confidence,
        args.max_samples
    )
    
    # Compute summary statistics
    radii = [r['radius'] for r in results if r['radius'] is not None]
    
    summary = {
        'n_samples': len(results),
        'n_successful': len(radii),
        'n_failed': len(results) - len(radii),
    }
    
    if radii:
        summary['mean_radius'] = float(np.mean(radii))
        summary['median_radius'] = float(np.median(radii))
        summary['std_radius'] = float(np.std(radii))
        summary['min_radius'] = float(np.min(radii))
        summary['max_radius'] = float(np.max(radii))
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Samples: {summary['n_samples']}")
    print(f"Successful: {summary['n_successful']}")
    print(f"Failed: {summary['n_failed']}")
    if radii:
        print(f"\nRadius statistics:")
        print(f"  Mean:   {summary['mean_radius']:.6f}")
        print(f"  Median: {summary['median_radius']:.6f}")
        print(f"  Std:    {summary['std_radius']:.6f}")
        print(f"  Range:  [{summary['min_radius']:.6f}, {summary['max_radius']:.6f}]")
    
    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sigma = estimation_data['parameters']['sigma']
        output_file = f"ec_radii_sigma{sigma}_eps{args.eps_y_deg}deg_{timestamp}.json"
    
    output_data = {
        'experiment_type': 'ec_radii_from_estimates',
        'certifier': '(E, C)+M (BoundedCertifierVarianceMean)',
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'estimation_file': args.estimation_file,
            'estimation_confidence': estimation_data['parameters'].get('confidence', 'unknown'),
            'sigma': estimation_data['parameters']['sigma'],
            'eps_y_deg': args.eps_y_deg,
            'eps_y_rad': np.radians(args.eps_y_deg),
            'M': np.pi,
            'N': args.N,
            'trial': args.trial,
            'ci_type': args.ci_type,
            'certifier_confidence': args.confidence,
            'note': 'Uses C from 80 percent confidence estimation, certifier uses 95 percent confidence'
        },
        'summary': summary,
        'results': results
    }
    
    output_path = Path(output_file)
    if output_path.parent != Path(""):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved results to: {output_file}")
    print("="*80)
    
    return output_data


if __name__ == "__main__":
    main()
