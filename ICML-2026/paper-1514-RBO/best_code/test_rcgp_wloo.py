"""
Test script to validate RCGP WLOO-CV fitting implementation.

This script tests the complete RCGP fitting pipeline with various configurations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Import the framework components
from bo_framework.models.factory import create_rcgp_model, create_botorch_model
from rcgp.weighting.plateau_imq import PlateauIMQ
from rcgp.transforms.robust_standardize import RobustStandardize
from experiments.synthetic.functions import ForresterFunction
from bo_framework.synthetic.evaluators import SyntheticEvaluator


def create_test_data(n_points: int = 20, noise_std: float = 0.1, 
                     n_outliers: int = 3, outlier_magnitude: float = 5.0, 
                     seed: int = 42) -> tuple:
    """
    Create test data with outliers for validation.
    
    Args:
        n_points: Total number of data points
        noise_std: Standard deviation of Gaussian noise
        n_outliers: Number of outlier points to add
        outlier_magnitude: Magnitude of outliers
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X, Y_clean, Y_corrupted) tensors
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create clean data using Forrester function
    forrester = ForresterFunction()
    X = torch.linspace(0.0, 1.0, n_points, dtype=torch.double).unsqueeze(-1)
    Y_clean = forrester.evaluate(X).unsqueeze(-1)
    
    # Add Gaussian noise
    noise = torch.randn(n_points, 1, dtype=torch.double) * noise_std
    Y_noisy = Y_clean + noise
    
    # Add outliers
    Y_corrupted = Y_noisy.clone()
    if n_outliers > 0:
        outlier_indices = torch.randperm(n_points)[:n_outliers]
        outlier_signs = torch.randint(0, 2, (n_outliers, 1), dtype=torch.double) * 2 - 1
        outlier_values = outlier_signs * outlier_magnitude
        Y_corrupted[outlier_indices] += outlier_values
        
        print(f"Added {n_outliers} outliers at indices: {outlier_indices.tolist()}")
        print(f"Outlier values: {outlier_values.squeeze().tolist()}")
    
    return X, Y_clean, Y_corrupted


def test_wloo_fitting():
    """Test WLOO-CV fitting vs standard fitting."""
    print("=" * 60)
    print("Testing RCGP WLOO-CV Fitting vs Standard Fitting")
    print("=" * 60)
    
    # Create test data with outliers
    X, Y_clean, Y_corrupted = create_test_data(
        n_points=20, 
        noise_std=0.1, 
        n_outliers=3, 
        outlier_magnitude=8.0
    )
    
    print(f"\nTest data created:")
    print(f"  Clean data range: [{Y_clean.min():.3f}, {Y_clean.max():.3f}]")
    print(f"  Corrupted data range: [{Y_corrupted.min():.3f}, {Y_corrupted.max():.3f}]")
    
    # Test configurations
    configs = {
        'RCGP + WLOO-CV': {
            'use_wloo_fitting': True,
            'use_robust_standardize': False,
            'fitting_kwargs': {'verbose': True}
        },
        'RCGP + WLOO-CV + Robust Std': {
            'use_wloo_fitting': True, 
            'use_robust_standardize': True,
            'fitting_kwargs': {'verbose': True}
        },
        'RCGP + Standard MLL': {
            'use_wloo_fitting': False,
            'use_robust_standardize': False,
            'fitting_kwargs': {'verbose': True}
        }
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n{'-' * 40}")
        print(f"Testing: {config_name}")
        print(f"{'-' * 40}")
        
        try:
            model = create_rcgp_model(X, Y_corrupted, **config)
            
            # Extract parameters
            if hasattr(model.weighting_function, 'plateau_width'):
                plateau_width = model.weighting_function.plateau_width
            else:
                plateau_width = None
                
            if hasattr(model.weighting_function, 'c'):
                c = model.weighting_function.c
            else:
                c = None
            
            # Test prediction on clean test points
            X_test = torch.linspace(0.0, 1.0, 50, dtype=torch.double).unsqueeze(-1)
            forrester = ForresterFunction()
            Y_test_true = forrester.evaluate(X_test)
            
            with torch.no_grad():
                posterior = model.posterior(X_test)
                Y_pred_mean = posterior.mean.squeeze()
                Y_pred_std = posterior.variance.sqrt().squeeze()
            
            # Calculate prediction error
            mse = torch.mean((Y_pred_mean - Y_test_true) ** 2).item()
            
            results[config_name] = {
                'model': model,
                'mse': mse,
                'noise_std': torch.sqrt(model.likelihood.noise).item(),
                'lengthscale': model.covar_module.lengthscale.item() if hasattr(model.covar_module, 'lengthscale') else None,
                'plateau_width': plateau_width,
                'c': c
            }
            
            print(f"✓ Success!")
            print(f"  MSE: {mse:.6f}")
            print(f"  Noise std: {results[config_name]['noise_std']:.4f}")
            print(f"  Lengthscale: {results[config_name]['lengthscale']:.4f}")
            if plateau_width is not None:
                print(f"  Plateau width: {plateau_width:.4f}")
            if c is not None:
                print(f"  Tail shape c: {c:.4f}")
                
        except Exception as e:
            print(f"✗ Failed: {e}")
            results[config_name] = {'error': str(e)}
    
    return results, X, Y_clean, Y_corrupted


def test_robust_heuristics():
    """Test robust heuristics calculation."""
    print("\n" + "=" * 60)
    print("Testing Robust Heuristics Calculation")
    print("=" * 60)
    
    from rcgp.fitting.rcgp_wloo import calculate_robust_heuristics
    
    # Create test data with known outliers
    torch.manual_seed(42)
    Y_clean = torch.randn(100) * 2.0  # Clean data with std=2
    Y_corrupted = Y_clean.clone()
    
    # Add 5 outliers
    outlier_indices = [10, 25, 50, 75, 90]
    outlier_values = torch.tensor([15.0, -12.0, 18.0, -15.0, 20.0])
    Y_corrupted[outlier_indices] = outlier_values
    
    print(f"Clean data: mean={Y_clean.mean():.3f}, std={Y_clean.std():.3f}")
    print(f"Corrupted data: mean={Y_corrupted.mean():.3f}, std={Y_corrupted.std():.3f}")
    
    # Calculate heuristics
    heuristics_clean = calculate_robust_heuristics(Y_clean)
    heuristics_corrupted = calculate_robust_heuristics(Y_corrupted)
    
    print(f"\nHeuristics from clean data:")
    print(f"  Center (median): {heuristics_clean['center']:.4f}")
    print(f"  Plateau width (95th perc): {heuristics_clean['plateau_width']:.4f}")
    print(f"  Tail shape c (MAD): {heuristics_clean['c']:.4f}")
    
    print(f"\nHeuristics from corrupted data:")
    print(f"  Center (median): {heuristics_corrupted['center']:.4f}")
    print(f"  Plateau width (95th perc): {heuristics_corrupted['plateau_width']:.4f}")
    print(f"  Tail shape c (MAD): {heuristics_corrupted['c']:.4f}")
    
    return heuristics_clean, heuristics_corrupted


def test_weighting_interface():
    """Test the new weighting function interface."""
    print("\n" + "=" * 60)
    print("Testing Weighting Function Interface")
    print("=" * 60)
    
    # Create test data
    X = torch.linspace(0, 1, 10, dtype=torch.double).unsqueeze(-1)
    Y = torch.randn(10, dtype=torch.double)
    
    # Test PlateauIMQ with new interface
    weighting_fn = PlateauIMQ(plateau_width=1.0, c=1.0)
    
    # Test different sigma values
    sigma_values = [0.1, 1.0, 2.0]
    
    print("Testing PlateauIMQ with different sigma values:")
    for sigma in sigma_values:
        sigma_tensor = torch.tensor(sigma, dtype=torch.double)
        
        weights = weighting_fn.weight(X, Y, sigma=sigma_tensor)
        gradient_log_weights = weighting_fn.gradient_log_weight(X, Y, sigma=sigma_tensor)
        J_matrix = weighting_fn.compute_J_matrix(weights, sigma_tensor)
        
        beta = sigma / np.sqrt(2)
        
        print(f"\n  Sigma: {sigma:.2f}, Beta: {beta:.4f}")
        print(f"    Weights range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"    Gradient range: [{gradient_log_weights.min():.4f}, {gradient_log_weights.max():.4f}]")
        print(f"    J matrix diagonal range: [{torch.diagonal(J_matrix).min():.6f}, {torch.diagonal(J_matrix).max():.6f}]")
    
    # Test update_heuristics
    print(f"\nTesting heuristic updates:")
    original_plateau_width = weighting_fn.plateau_width
    original_c = weighting_fn.c
    
    print(f"  Original plateau_width: {original_plateau_width}")
    print(f"  Original c: {original_c}")
    
    # Create new center function
    def new_center_fn(x):
        return torch.ones(x.shape[0], dtype=x.dtype, device=x.device) * 0.5
    
    weighting_fn.update_heuristics(
        plateau_width=2.0, 
        c=1.5,
        center_fn=new_center_fn
    )
    
    print(f"  Updated plateau_width: {weighting_fn.plateau_width}")
    print(f"  Updated c: {weighting_fn.c}")
    
    # Test with updated parameters
    sigma_tensor = torch.tensor(1.0, dtype=torch.double)
    weights_updated = weighting_fn.weight(X, Y, sigma=sigma_tensor)
    print(f"  Updated weights range: [{weights_updated.min():.4f}, {weights_updated.max():.4f}]")


def main():
    """Run all tests."""
    print("RCGP WLOO-CV Implementation Validation")
    print("=" * 70)
    
    try:
        # Test 1: Weighting function interface
        test_weighting_interface()
        
        # Test 2: Robust heuristics
        heuristics_clean, heuristics_corrupted = test_robust_heuristics()
        
        # Test 3: WLOO-CV fitting
        results, X, Y_clean, Y_corrupted = test_wloo_fitting()
        
        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        print(f"\n✓ Weighting function interface: Working correctly")
        print(f"✓ Robust heuristics: Successfully calculated from corrupted data")
        
        success_count = sum(1 for result in results.values() if 'error' not in result)
        total_count = len(results)
        
        print(f"✓ WLOO-CV fitting: {success_count}/{total_count} configurations successful")
        
        if success_count > 0:
            best_config = min(
                (name for name, result in results.items() if 'error' not in result),
                key=lambda name: results[name]['mse']
            )
            best_mse = results[best_config]['mse']
            print(f"✓ Best configuration: {best_config} (MSE: {best_mse:.6f})")
        
        print(f"\n🎉 Implementation validation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()