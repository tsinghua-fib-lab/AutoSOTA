#!/usr/bin/env python3
"""
Enhanced A2RCGP Debug Script with Incremental Fitting and Detailed Analysis

This script:
1. Fits models point by point from the 5th data point to the end
2. Tracks parameter evolution for RCGP, A2RCGP inner, and A2RCGP outer
3. Provides detailed final analysis including weights, J matrix, and centering functions
"""

import torch
import numpy as np
from bo_framework.models.factory import create_rcgp_model, create_a2rcgp_model

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Load the same data as the original debug script
X_full = torch.tensor([
    [0.1], [0.5], [0.9], [0.3], [0.7], [0.2], [0.8], [0.4], [0.6], [0.15],
    [0.85], [0.25], [0.75], [0.35], [0.65], [0.45], [0.55], [0.05], [0.95], [0.12],
    [0.88], [0.22], [0.78], [0.32], [0.68], [0.42], [0.58], [0.08], [0.92], [0.18],
    [0.82], [0.28], [0.72], [0.38], [0.62]
], dtype=torch.float64)

Y_full = torch.tensor([
    2.2584, 0.4618, 1.7345, -1.3231, 13.2402, -50.0, 50.0, 2.308, 2.6239, 2.4306,
    3.2092, 2.1705, 1.956, 2.1352, 1.5181, 2.4609, 2.5928, 2.0879, 3.1499, 2.2671,
    3.2818, 2.4003, 2.5326, 2.6649, 2.7972, 2.9295, 3.0618, 2.1941, 3.3264, 2.4587,
    3.5910, 2.7233, 2.8556, 2.9879, 3.1202
], dtype=torch.float64)

print("=== A2RCGP INCREMENTAL FITTING ANALYSIS ===")
print(f"Full data - X shape: {X_full.shape}, Y shape: {Y_full.shape}")
print(f"Y range: [{Y_full.min().item():.2f}, {Y_full.max().item():.2f}]")

# Model configuration
rcgp_kwargs = {
    'param_handling_dict': {
        'plateau_width': {'method': 'heuristics', 'value': 2.0},
        'c': {'method': 'manual', 'value': 1.0},
        'sigma': {'method': 'fit'},
        'mean': {'method': 'fit'}
    },
    'fitting_objective_type': 'wloo-cv',
    'optimizer_type': 'lbfgs',
    'standardize': True,
    'verbose': False
}

a2rcgp_kwargs = {
    'inner_param_handling_dict': {
        'plateau_width': {'method': 'heuristics', 'value': 2.0},
        'c': {'method': 'manual', 'value': 1.0},
        'sigma': {'method': 'fit'},
        'mean': {'method': 'fit'}
    },
    'outer_param_handling_dict': {
        'plateau_width': {'method': 'heuristics', 'value': 1.5},
        'c': {'method': 'manual', 'value': 1.0},
        'sigma': {'method': 'fit'},
        'mean': {'method': 'fit'}
    },
    'fitting_objective_type': 'wloo-cv',
    'optimizer_type': 'lbfgs',
    'standardize': True,
    'verbose': False
}

def get_model_parameters(model, model_name):
    """Extract key parameters from a model."""
    params = {}
    
    # Lengthscale
    if hasattr(model, 'covar_module'):
        if hasattr(model.covar_module, 'base_kernel'):
            # ScaleKernel case
            params['lengthscale'] = model.covar_module.base_kernel.lengthscale.item()
        else:
            # RBFKernel case
            params['lengthscale'] = model.covar_module.lengthscale.item()
    
    # Noise
    if hasattr(model, 'likelihood'):
        params['noise'] = model.likelihood.noise.item()
    
    # Mean constant
    if hasattr(model, 'mean_module'):
        params['mean_constant'] = model.mean_module.constant.item()
    
    # Weighting function parameters
    if hasattr(model, 'weighting_function'):
        params['plateau_width'] = model.weighting_function.plateau_width.item()
        params['c'] = model.weighting_function.c.item()
    
    return params

def print_parameter_evolution(rcgp_params_history, inner_params_history, outer_params_history):
    """Print parameter evolution in a formatted table."""
    print(f"\n{'='*80}")
    print("PARAMETER EVOLUTION")
    print(f"{'='*80}")
    
    # Get all iteration numbers
    iterations = sorted(set(rcgp_params_history.keys()) | set(inner_params_history.keys()) | set(outer_params_history.keys()))
    
    # Print header
    print(f"{'Iter':<4} {'Model':<8} {'Lengthscale':<12} {'Noise':<10} {'Mean':<10} {'Plateau':<10} {'C':<8}")
    print("-" * 80)
    
    for iter_num in iterations:
        # RCGP parameters
        if iter_num in rcgp_params_history:
            params = rcgp_params_history[iter_num]
            print(f"{iter_num:<4} {'RCGP':<8} {params.get('lengthscale', 0):<12.4f} {params.get('noise', 0):<10.4f} {params.get('mean_constant', 0):<10.4f} {params.get('plateau_width', 0):<10.4f} {params.get('c', 0):<8.4f}")
        
        # A2RCGP Inner parameters
        if iter_num in inner_params_history:
            params = inner_params_history[iter_num]
            print(f"{iter_num:<4} {'A2Inner':<8} {params.get('lengthscale', 0):<12.4f} {params.get('noise', 0):<10.4f} {params.get('mean_constant', 0):<10.4f} {params.get('plateau_width', 0):<10.4f} {params.get('c', 0):<8.4f}")
        
        # A2RCGP Outer parameters
        if iter_num in outer_params_history:
            params = outer_params_history[iter_num]
            print(f"{iter_num:<4} {'A2Outer':<8} {params.get('lengthscale', 0):<12.4f} {params.get('noise', 0):<10.4f} {params.get('mean_constant', 0):<10.4f} {params.get('plateau_width', 0):<10.4f} {params.get('c', 0):<8.4f}")
        
        print("-" * 80)

def analyze_final_models(rcgp_model, a2rcgp_model, X_final, Y_final):
    """Analyze the final models with detailed weights, J matrix, and centering functions."""
    print(f"\n{'='*80}")
    print("FINAL MODEL ANALYSIS")
    print(f"{'='*80}")
    
    # Get robust components for all models
    models = {
        'RCGP': rcgp_model,
        'A2RCGP Inner': a2rcgp_model.inner_rcgp,
        'A2RCGP Outer': a2rcgp_model
    }
    
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---")
        
        try:
            # Get robust components
            weights, J_matrix, log_gradients = model._get_robust_components(X_final, Y_final.unsqueeze(-1) if Y_final.dim() == 1 else Y_final)
            J_diagonal = torch.diag(J_matrix)
            
            print(f"Weights: {weights.squeeze().tolist()}")
            print(f"Weight range: [{weights.min().item():.4f}, {weights.max().item():.4f}]")
            print(f"J matrix diagonal: {J_diagonal.tolist()}")
            print(f"J diagonal range: [{J_diagonal.min().item():.4f}, {J_diagonal.max().item():.4f}]")
            
            # Centering function values
            if hasattr(model, 'weighting_function') and hasattr(model.weighting_function, 'center_fn'):
                center_values = model.weighting_function.center_fn(X_final)
                print(f"Centering function values: {center_values.squeeze().tolist()}")
                print(f"Center range: [{center_values.min().item():.4f}, {center_values.max().item():.4f}]")
                print(f"Are center values constant? {torch.allclose(center_values, center_values[0])}")
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
    
    # Compare inner posterior mean with outer centering function
    print(f"\n--- INNER POSTERIOR MEAN vs OUTER CENTERING COMPARISON ---")
    try:
        # Get inner posterior mean
        with torch.no_grad():
            a2rcgp_model.inner_rcgp.eval()
            inner_posterior_mean = a2rcgp_model.inner_rcgp.posterior(X_final).mean.squeeze(-1)
        
        # Get outer centering function values
        outer_center_values = a2rcgp_model.outer_weighting_function.center_fn(X_final)
        
        print(f"X values: {X_final.squeeze().tolist()}")
        print(f"Inner posterior mean: {inner_posterior_mean.tolist()}")
        print(f"Outer centering function: {outer_center_values.squeeze().tolist()}")
        print(f"Difference (inner - outer): {(inner_posterior_mean - outer_center_values.squeeze()).tolist()}")
        print(f"Max absolute difference: {torch.abs(inner_posterior_mean - outer_center_values.squeeze()).max().item():.6f}")
        print(f"Are they close? {torch.allclose(inner_posterior_mean, outer_center_values.squeeze(), atol=1e-4)}")
        
    except Exception as e:
        print(f"Error in comparison: {e}")

# Incremental fitting from 5th point to the end
start_point = 5
rcgp_params_history = {}
inner_params_history = {}
outer_params_history = {}

print(f"\n{'='*80}")
print(f"INCREMENTAL FITTING FROM POINT {start_point} TO {len(X_full)}")
print(f"{'='*80}")

for n_points in range(start_point, len(X_full) + 1):
    print(f"\n--- Fitting with {n_points} points ---")
    
    # Get subset of data
    X_subset = X_full[:n_points]
    Y_subset = Y_full[:n_points]
    
    try:
        # Fit RCGP
        print("Fitting RCGP...")
        rcgp_model = create_rcgp_model(X_subset, Y_subset, **rcgp_kwargs)
        rcgp_params = get_model_parameters(rcgp_model, "RCGP")
        rcgp_params_history[n_points] = rcgp_params
        
        # Fit A2RCGP
        print("Fitting A2RCGP...")
        a2rcgp_model = create_a2rcgp_model(X_subset, Y_subset, **a2rcgp_kwargs)
        
        # Get inner and outer parameters
        inner_params = get_model_parameters(a2rcgp_model.inner_rcgp, "A2RCGP Inner")
        outer_params = get_model_parameters(a2rcgp_model, "A2RCGP Outer")
        
        inner_params_history[n_points] = inner_params
        outer_params_history[n_points] = outer_params
        
        print(f"RCGP lengthscale: {rcgp_params.get('lengthscale', 0):.4f}")
        print(f"A2RCGP inner lengthscale: {inner_params.get('lengthscale', 0):.4f}")
        print(f"A2RCGP outer lengthscale: {outer_params.get('lengthscale', 0):.4f}")
        
    except Exception as e:
        print(f"Error fitting models with {n_points} points: {e}")
        continue

# Print parameter evolution
print_parameter_evolution(rcgp_params_history, inner_params_history, outer_params_history)

# Final detailed analysis
if len(rcgp_params_history) > 0:
    final_n_points = max(rcgp_params_history.keys())
    X_final = X_full[:final_n_points]
    Y_final = Y_full[:final_n_points]
    
    print(f"\n{'='*80}")
    print(f"FITTING FINAL MODELS FOR DETAILED ANALYSIS ({final_n_points} points)")
    print(f"{'='*80}")
    
    try:
        # Fit final models
        rcgp_model = create_rcgp_model(X_final, Y_final, **rcgp_kwargs)
        a2rcgp_model = create_a2rcgp_model(X_final, Y_final, **a2rcgp_kwargs)
        
        # Detailed analysis
        analyze_final_models(rcgp_model, a2rcgp_model, X_final, Y_final)
        
    except Exception as e:
        print(f"Error in final analysis: {e}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
