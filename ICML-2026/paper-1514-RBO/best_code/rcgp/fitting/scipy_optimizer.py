"""
Scipy-based optimization for GP models with custom objectives.

This module provides scipy L-BFGS-B optimization that is more compatible
with custom objectives like LOO-CV and WLOO-CV compared to fit_gpytorch_mll.
"""

import torch
import numpy as np
from scipy.optimize import minimize
from typing import Optional


def optimize_with_scipy_lbfgs(mll, model, max_iterations: int = 1000, verbose: bool = False):
    """Optimize using scipy's L-BFGS-B which is more compatible with custom objectives like LOO-CV.
    
    This function is particularly useful for LOO-CV and WLOO-CV objectives where
    fit_gpytorch_mll may not work properly due to incompatibilities.
    
    Args:
        mll: The marginal log likelihood object to optimize
        model: The GP model to optimize
        max_iterations: Maximum number of optimization iterations
        verbose: Whether to print optimization progress
        
    Returns:
        None (optimizes the model in-place)
    """
    
    # Get trainable parameters
    trainable_params = []
    param_names = []
    param_bounds = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            param_names.append(name)
            
            # Set reasonable bounds for different parameter types
            if 'lengthscale' in name:
                param_bounds.append((0.01, 10.0))  # Lengthscale bounds
            elif 'outputscale' in name:
                param_bounds.append((0.01, 10.0))  # Output scale bounds
            elif 'noise' in name or 'raw_noise' in name:
                param_bounds.append((0.001, 10.0))  # Noise bounds
            elif 'constant' in name or 'raw_constant' in name:
                param_bounds.append((-10.0, 10.0))  # Mean bounds
            else:
                param_bounds.append((None, None))  # No bounds
    
    if not trainable_params:
        if verbose:
            print("No trainable parameters found")
        return
    
    def objective(x):
        """Objective function for scipy optimization."""
        # Set parameters from x
        idx = 0
        for param in trainable_params:
            param_size = param.numel()
            param.data = torch.tensor(x[idx:idx+param_size], dtype=param.dtype, device=param.device).view(param.shape)
            idx += param_size
        
        # Compute loss
        try:
            output = model(model.train_inputs[0])
            loss = -mll(output, model.train_targets)
            return loss.item()
        except Exception as e:
            if verbose:
                print(f"Error in objective: {e}")
            return float('inf')
    
    def gradient(x):
        """Gradient function for scipy optimization."""
        # Set parameters from x
        idx = 0
        for param in trainable_params:
            param_size = param.numel()
            param.data = torch.tensor(x[idx:idx+param_size], dtype=param.dtype, device=param.device).view(param.shape)
            idx += param_size
        
        # Compute gradients
        try:
            output = model(model.train_inputs[0])
            loss = -mll(output, model.train_targets)
            loss.backward()
            
            # Collect gradients
            grads = []
            for param in trainable_params:
                if param.grad is not None:
                    grads.extend(param.grad.detach().cpu().numpy().flatten())
                else:
                    grads.extend(np.zeros(param.numel()))
            
            return np.array(grads)
        except Exception as e:
            if verbose:
                print(f"Error in gradient: {e}")
            return np.zeros(len(x))
    
    # Initial parameters
    x0 = []
    for param in trainable_params:
        x0.extend(param.detach().cpu().numpy().flatten())
    x0 = np.array(x0)
    
    # Optimize
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=gradient,
            bounds=param_bounds,
            options={'maxiter': max_iterations, 'disp': verbose}
        )
        
        if verbose:
            print(f"Optimization completed: {result.message}")
            print(f"Final loss: {result.fun:.6f}")
            
    except Exception as e:
        if verbose:
            print(f"Optimization failed: {e}")
