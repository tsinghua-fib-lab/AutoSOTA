"""
Test Unbounded Certifiers on Unbounded Synthetic Functions
==========================================================

This script tests the unbounded certifier (VarianceGradientCertifier) and alpha-smoothing
on unbounded synthetic functions. This is an alternative to testing bounded functions
with (E, C, G) + M, which may have infeasibility issues due to estimation errors.

The unbounded certifier uses C and G (no M constraint), making it more suitable for
controlled synthetic experiments where we can compute ground truth.

Usage:
    python experiments/synthetic/run_unbounded_synthetic.py \
        --function all_unbounded \
        --sigma 0.1 \
        --eps_y 0.5 \
        --N_samples 10000 \
        --n_test_points 10
"""

import numpy as np
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from regression_certifiers.certify.variance_gradient_certifier import VarianceGradientCertifier
from regression_certifiers.certify.alpha_trimming_certifier import AlphaTrimmingCertifier
from regression_certifiers.synthetic_functions import (
    synthetic_quadratic,
    synthetic_slice_function,
    synthetic_sandwich_function,
    create_test_points,
    compute_true_radius_analytical
)


def test_single_function(
    function_name: str,
    function_fn,
    function_params: Dict,
    test_points: List[np.ndarray],
    sigma: float,
    eps_y: float,
    N_samples: int = 10000,
    compute_true_radius: bool = True,
    alpha_trim: float = 0.35,
    P: float = 0.9,
    alpha_n_tr: int = 10000,
    alpha_n_sample: int = 16,
    alpha_cp_alpha: float = 1e-3
) -> Dict:
    """Test unbounded certifier and alpha-smoothing on a single function."""
    
    print(f"\n{'='*80}")
    print(f"Testing: {function_name}")
    print(f"{'='*80}")
    print(f"Parameters: {function_params}")
    print(f"Test points: {len(test_points)}")
    
    # Initialize unbounded certifier (uses C and G, no M)
    # Match alpha-smoothing confidence: P means success probability threshold
    # If P=0.9, that's 10% failure probability, so we use confidence=0.9
    # (10% failure for union bound on C and G estimates)
    confidence_unbounded = P  # Match failure probability budget with alpha-smoothing
    certifier_unbounded = VarianceGradientCertifier(
        sigma=sigma,
        eps_y=eps_y,
        confidence=confidence_unbounded
    )
    # Use canonical alpha-trimming certification instead of oracle helper.
    certifier_alpha = AlphaTrimmingCertifier(
        sigma=sigma,
        eps_y=eps_y,
        alpha=alpha_trim,
        n_tr=alpha_n_tr,
        n_sample=alpha_n_sample,
        confidence=1.0 - alpha_cp_alpha,
        P=P,
        center='pred',
        circular=False
    )
    
    results = {
        'function_name': function_name,
        'function_params': function_params,
        'sigma': sigma,
        'eps_y': eps_y,
        'N_samples': N_samples,
        'alpha_trim': alpha_trim,
        'P': P,
        'alpha_n_tr': alpha_n_tr,
        'alpha_n_sample': alpha_n_sample,
        'alpha_cp_alpha': alpha_cp_alpha,
        'confidence_unbounded': confidence_unbounded,
        'n_test_points': len(test_points),
        'results_by_point': []
    }
    
    for idx, z in enumerate(tqdm(test_points, desc=f"  {function_name}")):
        point_result = {
            'point_idx': idx,
            'z': z.tolist()
        }
        
        # Generate samples for statistical estimation
        rng = np.random.default_rng(42 + idx)
        eps_samples = rng.normal(0, sigma, size=(N_samples, 2))
        
        # Evaluate function at noisy points
        f_vals = np.array([
            function_fn(*(z + eps))
            for eps in eps_samples
        ])
        
        # Estimate statistics using unbounded certifier's estimators
        _, _, C_ucb = certifier_unbounded.u_statistic_variance_estimator_alpha_half(f_vals)
        _, _, G_ucb = certifier_unbounded.u_statistic_gradient_norm_estimator_alpha_half(f_vals, eps_samples)
        
        point_result['C_ucb'] = float(C_ucb)
        point_result['G_ucb'] = float(G_ucb)
        
        # Test Unbounded Certifier (C, G) - no M
        try:
            r_unbounded = certifier_unbounded.variance_gradient_certificate(C_ucb, G_ucb, eps_y)
            point_result['radius_unbounded'] = float(r_unbounded)
        except Exception as e:
            point_result['radius_unbounded'] = None
            point_result['error_unbounded'] = str(e)
        
        # Test Alpha-trimming (paper implementation)
        try:
            def model_fn_vec(x: np.ndarray) -> float:
                return float(function_fn(float(x[0]), float(x[1])))

            r_alpha = certifier_alpha.certify_point(
                z, model_fn=model_fn_vec, y_true=None, seed=42 + idx
            )
            point_result['radius_alpha'] = float(r_alpha)
        except Exception as e:
            point_result['radius_alpha'] = None
            point_result['error_alpha'] = str(e)
        
        # Compute true radius if requested
        # For unbounded functions, we use optimization-based approach directly
        # (not the bounded analytical methods, since these functions are truly unbounded)
        if compute_true_radius:
            try:
                from scipy.optimize import differential_evolution
                
                def compute_expectation(x: np.ndarray, n_samples: int = 50000) -> float:
                    """Compute E[f(x + ε)] using MC for the unbounded function."""
                    rng = np.random.default_rng(42 + idx)
                    e_samples = rng.normal(0.0, sigma, size=(n_samples, x.size))
                    f_vals = np.array([function_fn(*(x + e)) for e in e_samples])
                    return float(np.mean(f_vals))
                
                g_z = compute_expectation(z)
                
                def find_worst_case_at_radius(r: float) -> tuple:
                    """Find worst-case perturbation for radius r."""
                    if r == 0.0:
                        return 0.0, np.zeros(z.size)
                    
                    def objective_delta(delta_2d: np.ndarray) -> float:
                        delta = delta_2d.reshape(-1)
                        delta_norm = np.linalg.norm(delta)
                        if delta_norm > r:
                            delta = delta * (r / delta_norm)
                        g_perturbed = compute_expectation(z + delta, n_samples=50000)
                        return -abs(g_perturbed - g_z)  # Negative for maximization
                    
                    bounds = [(-r * 1.2, r * 1.2) for _ in range(z.size)]
                    result = differential_evolution(
                        objective_delta,
                        bounds,
                        seed=42 + idx,
                        maxiter=50,  # Reduced for faster execution
                        popsize=10,  # Reduced for faster execution
                        tol=1e-4  # Slightly relaxed tolerance
                    )
                    delta_star = result.x.reshape(-1)
                    delta_norm = np.linalg.norm(delta_star)
                    if delta_norm > r:
                        delta_star = delta_star * (r / delta_norm)
                    
                    g_perturbed = compute_expectation(z + delta_star, n_samples=100000)
                    max_change = abs(g_perturbed - g_z)
                    return max_change, delta_star
                
                # Binary search for true radius
                r_low, r_high = 0.0, 10.0 * sigma
                
                max_change_high, _ = find_worst_case_at_radius(r_high)
                if max_change_high <= eps_y:
                    point_result['radius_true'] = float(r_high)
                    point_result['true_radius_info'] = {'method': 'optimization', 'hit_upper_bound': True}
                else:
                    for _ in range(30):  # Reduced iterations for faster execution
                        r_mid = (r_low + r_high) / 2.0
                        max_change, _ = find_worst_case_at_radius(r_mid)
                        
                        if max_change <= eps_y:
                            r_low = r_mid
                        else:
                            r_high = r_mid
                        
                        if (r_high - r_low) < 1e-4:  # Slightly relaxed tolerance
                            break
                    
                    point_result['radius_true'] = float(r_low)
                    point_result['true_radius_info'] = {'method': 'optimization', 'hit_upper_bound': False}
            except Exception as e:
                point_result['radius_true'] = None
                point_result['error_true_radius'] = str(e)
        else:
            point_result['radius_true'] = None
        
        results['results_by_point'].append(point_result)
    
    # Compute summary statistics
    radii_unbounded = [r['radius_unbounded'] for r in results['results_by_point'] 
                       if r.get('radius_unbounded') is not None]
    radii_alpha = [r['radius_alpha'] for r in results['results_by_point'] 
                   if r.get('radius_alpha') is not None]
    radii_true = [r['radius_true'] for r in results['results_by_point'] 
                  if r.get('radius_true') is not None]
    
    if radii_unbounded:
        results['summary'] = {
            'mean_radius_unbounded': float(np.mean(radii_unbounded)),
            'mean_radius_alpha': float(np.mean(radii_alpha)) if radii_alpha else None,
            'mean_radius_true': float(np.mean(radii_true)) if radii_true else None,
            'n_valid_unbounded': len(radii_unbounded),
            'n_valid_alpha': len(radii_alpha),
            'n_valid_true': len(radii_true)
        }
        
        # Compute ratios if true radius is available
        if radii_true and len(radii_true) == len(radii_unbounded):
            ratios_unbounded = [u / t for u, t in zip(radii_unbounded, radii_true) if t > 0]
            ratios_alpha = [a / t for a, t in zip(radii_alpha, radii_true) 
                           if a is not None and t > 0]
            
            results['summary']['mean_ratio_unbounded'] = float(np.mean(ratios_unbounded)) if ratios_unbounded else None
            results['summary']['mean_ratio_alpha'] = float(np.mean(ratios_alpha)) if ratios_alpha else None
            
            # Soundness check (certified <= true)
            sound_unbounded = sum(1 for u, t in zip(radii_unbounded, radii_true) if u <= t)
            sound_alpha = sum(1 for a, t in zip(radii_alpha, radii_true) 
                             if a is not None and a <= t)
            
            results['summary']['soundness_unbounded'] = sound_unbounded / len(radii_true) if radii_true else None
            results['summary']['soundness_alpha'] = sound_alpha / len(radii_true) if radii_true and radii_alpha else None
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test unbounded certifiers on synthetic functions')
    parser.add_argument('--function', type=str, default='all_unbounded',
                       choices=['unbounded_quadratic', 'unbounded_slice', 'unbounded_sandwich', 'all_unbounded'],
                       help='Function to test')
    parser.add_argument('--sigma', type=float, default=0.1, help='Noise std dev')
    parser.add_argument('--eps_y', type=float, default=0.5, help='Output tolerance')
    parser.add_argument('--N_samples', type=int, default=10000, help='Number of samples for estimation')
    parser.add_argument('--n_test_points', type=int, default=10, help='Number of test points')
    parser.add_argument('--compute_true_radius', action='store_true', default=True,
                       help='Compute true radius for comparison')
    parser.add_argument('--skip_true_radius', action='store_false', dest='compute_true_radius',
                       help='Skip expensive true-radius optimization for smoke tests')
    parser.add_argument('--alpha_trim', type=float, default=0.35, help='Alpha-trimming parameter')
    parser.add_argument('--P', type=float, default=0.9, help='Success probability threshold for alpha-smoothing (default: 0.9 = 10%% failure)')
    parser.add_argument('--alpha_n_tr', type=int, default=10000,
                       help='Alpha-trimming MC samples for p_A estimation')
    parser.add_argument('--alpha_n_sample', type=int, default=16,
                       help='Alpha-trimming n_sample used in binomial q mapping')
    parser.add_argument('--alpha_cp_alpha', type=float, default=1e-3,
                       help='Clopper-Pearson significance level for alpha-trimming')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    
    args = parser.parse_args()
    
    print("="*80)
    print("UNBOUNDED CERTIFIER TEST: Synthetic Functions")
    print("="*80)
    print(f"\nTest parameters:")
    print(f"  sigma = {args.sigma}")
    print(f"  eps_y = {args.eps_y}")
    print(f"  N_samples = {args.N_samples}")
    print(f"  n_test_points = {args.n_test_points}")
    print(f"  compute_true_radius = {args.compute_true_radius}")
    print(f"  alpha_trim = {args.alpha_trim}")
    print(f"  P (alpha-smoothing success prob) = {args.P}")
    print(f"  confidence (unbounded certifier) = {args.P} (matching P for fair comparison)")
    
    # Function configurations
    FUNCTION_CONFIGS = {
        'unbounded_quadratic': {
            'fn': lambda x1, x2, **kwargs: synthetic_quadratic(x1, x2, **kwargs),
            'params': {'center': (0.0, 0.0), 'scale': 1.0}
        },
        'unbounded_slice': {
            'fn': lambda x1, x2, **kwargs: synthetic_slice_function(x1, x2, **kwargs),
            'params': {'threshold': 0.0}
        },
        'unbounded_sandwich': {
            'fn': lambda x1, x2, **kwargs: synthetic_sandwich_function(x1, x2, **kwargs),
            'params': {'width': 1.0}
        }
    }
    
    # Determine which functions to test
    if args.function == 'all_unbounded':
        functions_to_test = list(FUNCTION_CONFIGS.keys())
    else:
        functions_to_test = [args.function]
    
    print(f"\nFunctions to test: {functions_to_test}")
    
    # Create test points (shared across functions for consistency)
    test_points = create_test_points(
        n_points=args.n_test_points,
        domain=(-1.0, 1.0),
        seed=42
    )
    
    # Test each function
    all_results = {}
    
    for func_name in functions_to_test:
        config = FUNCTION_CONFIGS[func_name]
        results = test_single_function(
            function_name=func_name,
            function_fn=config['fn'],
            function_params=config['params'],
            test_points=test_points,
            sigma=args.sigma,
            eps_y=args.eps_y,
            N_samples=args.N_samples,
            compute_true_radius=args.compute_true_radius,
            alpha_trim=args.alpha_trim,
            P=args.P,
            alpha_n_tr=args.alpha_n_tr,
            alpha_n_sample=args.alpha_n_sample,
            alpha_cp_alpha=args.alpha_cp_alpha
        )
        all_results[func_name] = results
    
    # Create output data structure
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'sigma': args.sigma,
            'eps_y': args.eps_y,
            'N_samples': args.N_samples,
            'n_test_points': args.n_test_points,
            'alpha_trim': args.alpha_trim,
            'P': args.P,
            'alpha_n_tr': args.alpha_n_tr,
            'alpha_n_sample': args.alpha_n_sample,
            'alpha_cp_alpha': args.alpha_cp_alpha,
            'compute_true_radius': args.compute_true_radius
        },
        'results': all_results
    }
    
    # Save results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"unbounded_synthetic_sigma{args.sigma}_epsy{args.eps_y}_alpha{args.alpha_trim}_{timestamp}.json"
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for func_name, func_results in all_results.items():
        summary = func_results.get('summary', {})
        print(f"\n{func_name}:")
        if summary.get('mean_radius_unbounded') is not None:
            print(f"  Unbounded (C, G):     {summary['mean_radius_unbounded']:.6f}")
        if summary.get('mean_radius_alpha') is not None:
            print(f"  Alpha-trimming:       {summary['mean_radius_alpha']:.6f}")
        if summary.get('mean_radius_true') is not None:
            print(f"  True radius:          {summary['mean_radius_true']:.6f}")
        if summary.get('mean_ratio_unbounded') is not None:
            print(f"    Unbounded vs True:  {summary['mean_ratio_unbounded']:.1%}")
        if summary.get('mean_ratio_alpha') is not None:
            print(f"    Alpha vs True:      {summary['mean_ratio_alpha']:.1%}")
        if summary.get('soundness_unbounded') is not None:
            print(f"    Soundness (unbounded): {summary['soundness_unbounded']:.1%}")
        if summary.get('soundness_alpha') is not None:
            print(f"    Soundness (alpha):     {summary['soundness_alpha']:.1%}")
    
    print(f"\n✓ Saved results to: {output_path}")
    
    return output_data


if __name__ == '__main__':
    main()
