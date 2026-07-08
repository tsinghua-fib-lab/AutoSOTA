"""
RCGP-specific fitting using Weighted Leave-One-Out Cross-Validation (WLOO-CV).

This module implements the complete fitting procedure for RCGP models, including
robust heuristic calculation and WLOO-CV optimization.
"""

import torch
from typing import Dict, Optional, Callable
from botorch.fit import fit_gpytorch_mll

from ..models.robust_gp import RobustConjugateGP
from .wloo_mll import WeightedRobustLeaveOneOutMLL, RobustLeaveOneOutMLL


class ConstantCenterFunction:
    """A pickleable center function that returns a constant value."""
    
    def __init__(self, center_value: float):
        """
        Initialize constant center function.
        
        Args:
            center_value: The constant center value (typically median of standardized targets)
        """
        self.center_value = center_value
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return constant center values for all inputs."""
        return torch.full((x.shape[0],), self.center_value, dtype=x.dtype, device=x.device)


def create_constant_center_fn(center_value: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a center function that returns a constant value.
    
    Args:
        center_value: The constant center value (typically median of standardized targets)
        
    Returns:
        A pickleable function that maps inputs to constant center values
    """
    return ConstantCenterFunction(center_value)


def calculate_robust_heuristics(Y_std: torch.Tensor, quantile: float = 0.95) -> Dict[str, float]:
    """
    Calculate robust heuristics from standardized data.

    Args:
        Y_std: Standardized target values [n] or [n, 1]
        quantile: Quantile of absolute deviations to use for plateau_width (default: 0.95)

    Returns:
        Dictionary with heuristic values:
        - center: Median of Y_std
        - plateau_width: quantile-th percentile of absolute deviations (using kth element)
        - c: Median Absolute Deviation (MAD)
    """
    # Ensure 1D
    if Y_std.dim() > 1:
        Y_std = Y_std.squeeze(-1)
    
    n = Y_std.numel()
    
    # Calculate median
    median_y = torch.median(Y_std)
    
    # Calculate absolute deviations from median
    abs_dev = torch.abs(Y_std - median_y)
    
    # Plateau width: quantile-th percentile of deviations using kth element
    # For quantile-th percentile, k = ceil(quantile * n) - 1 (0-indexed)
    k_q = int(torch.ceil(torch.tensor(quantile * n)).item()) - 1
    k_q = max(0, min(k_q, n - 1))  # Ensure k is within bounds

    # Sort absolute deviations and take kth element
    sorted_abs_dev, _ = torch.sort(abs_dev)
    plateau_width = sorted_abs_dev[k_q]
    
    # MAD: median of absolute deviations
    mad = torch.median(abs_dev)
    
    # Ensure minimum values for numerical stability
    plateau_width = torch.clamp(plateau_width, min=0.1)
    mad = torch.clamp(mad, min=0.1)
    
    # Estimate noise using robust statistics
    # Use MAD as a robust estimate of noise level
    noise_estimate = mad * 1.4826  # Scale MAD to match standard deviation for Gaussian data
    
    return {
        'center': median_y.item(),
        'plateau_width': plateau_width.item(),
        'c': mad.item(),
        'noise_estimate': noise_estimate.item()
    }


def fit_rcgp_wloo(
    model: RobustConjugateGP,
    learning_rate: float = 0.01,
    max_iterations: int = 500,
    verbose: bool = True,
    options: Optional[Dict] = None,
    use_robust_heuristics: bool = True,
    weighted_loss: bool = True,
    fit_sigma: bool = True,
    fit_mean: bool = True
) -> Dict[str, float]:
    """
    Fit RCGP hyperparameters using WLOO-CV with optional robust heuristics.
    
    This function:
    1. Optionally calculates robust heuristics from standardized training data
    2. Updates the weighting function with heuristics (if enabled)
    3. Optimizes GP hyperparameters (theta, optionally sigma² and mean) via WLOO-CV
    
    Args:
        model: RobustConjugateGP model to fit (must have training data)
        learning_rate: Learning rate for optimizer
        max_iterations: Maximum optimization iterations
        verbose: Whether to print progress
        options: Additional optimizer options
        use_robust_heuristics: Whether to calculate and apply robust heuristics
        fit_sigma: Whether to fit the noise parameter (sigma²)
        fit_mean: Whether to fit the mean parameter
        options: Additional options for fit_gpytorch_mll
        
    Returns:
        Dictionary of optimized parameters including:
        - Kernel hyperparameters (lengthscale, outputscale if applicable)
        - Noise variance (sigma²)
        - Mean constant (if present)
        - Robust heuristics (plateau_width, c, center)
    """
    if options is None:
        options = {}
        
    # Step 1: Optionally calculate and apply robust heuristics
    heuristics = None
    if use_robust_heuristics:
        # Get standardized targets
        # Note: model.train_targets already contains standardized data if outcome_transform was used
        Y_std = model.train_targets.detach()
        
        # Calculate robust heuristics on standardized data
        heuristics = calculate_robust_heuristics(Y_std)
        
        if verbose:
            print("Calculated robust heuristics from standardized data:")
            print(f"  Center (median): {heuristics['center']:.4f}")
            print(f"  Plateau width (95th percentile): {heuristics['plateau_width']:.4f}")
            print(f"  Tail shape c (MAD): {heuristics['c']:.4f}")
            print(f"  Noise estimate (MAD*1.4826): {heuristics['noise_estimate']:.4f}")
        
        # Update weighting function with heuristics
        center_fn = create_constant_center_fn(heuristics['center'])
        model.weighting_function.update_heuristics(
            plateau_width=heuristics['plateau_width'],
            c=heuristics['c'],
            center_fn=center_fn
        )
    elif verbose:
        print("Using manual robust parameters (skipping heuristics calculation)")
    
    # Step 2: Set up WLOO-CV objective
    model.train()  # Ensure model is in training mode
    wloo_mll = WeightedRobustLeaveOneOutMLL(model.likelihood, model) if weighted_loss else RobustLeaveOneOutMLL(model.likelihood, model)
    
    # Step 3: Handle sigma fitting
    if not fit_sigma:
        # Freeze the noise parameter if not fitting
        model.likelihood.noise.requires_grad_(False)
        if verbose:
            current_sigma = torch.sqrt(model.likelihood.noise).item()
            print(f"Sigma parameter frozen at: {current_sigma:.4f}")
    elif verbose:
        print("Sigma parameter will be optimized")
    
    # Step 3b: Handle mean fitting
    if not fit_mean:
        # Freeze the mean parameter if not fitting
        if hasattr(model.mean_module, 'constant'):
            model.mean_module.constant.requires_grad_(False)
            if verbose:
                current_mean = model.mean_module.constant.item()
                print(f"Mean parameter frozen at: {current_mean:.4f}")
    elif verbose:
        print("Mean parameter will be optimized")
    
    # Step 4: Optimize using BoTorch's fit function
    if verbose:
        print("\nOptimizing GP hyperparameters via WLOO-CV...")
        
    # Configure optimizer options
    optimizer_options = {
        'lr': learning_rate,
        'max_iter': max_iterations,
        **options  # Allow additional options to be passed through
    }
    
    # Run optimization
    from botorch.fit import fit_gpytorch_mll
    fit_gpytorch_mll(wloo_mll, options=optimizer_options)
    
    # Step 6: Set model back to eval mode
    model.eval()
    
    # Step 5: Extract and return final parameters
    final_params = extract_parameters(model)
    
    # Add heuristics to the returned parameters (if calculated)
    if heuristics is not None:
        final_params.update({
            'heuristic_center': heuristics['center'],
            'heuristic_plateau_width': heuristics['plateau_width'],
            'heuristic_c': heuristics['c'],
            'heuristic_noise_estimate': heuristics['noise_estimate']
        })
    
    # Keep gradients disabled for noise parameter if it was frozen
    # (don't re-enable them as they were disabled intentionally)
    
    if verbose:
        print("\nOptimization completed. Final parameters:")
        print(f"  Noise std (sigma): {torch.sqrt(torch.tensor(final_params['noise'])):.4f}")
        print(f"  Lengthscale: {final_params.get('lengthscale', 'N/A')}")
        print(f"  Mean constant: {final_params.get('mean_constant', 'N/A')}")
        
    return final_params


def extract_parameters(model: RobustConjugateGP) -> Dict[str, float]:
    """
    Extract current parameter values from RCGP model.
    
    Args:
        model: RobustConjugateGP model
        
    Returns:
        Dictionary of parameter values
    """
    params = {}
    
    # Likelihood parameters
    params['noise'] = model.likelihood.noise.item()
    
    # Mean parameters
    if hasattr(model.mean_module, 'constant'):
        params['mean_constant'] = model.mean_module.constant.item()
    
    # Kernel parameters
    covar_module = model.covar_module
    
    # Handle both ScaleKernel wrapper and plain kernel
    if hasattr(covar_module, 'base_kernel'):
        # ScaleKernel case
        base_kernel = covar_module.base_kernel
        if hasattr(base_kernel, 'lengthscale'):
            lengthscale = base_kernel.lengthscale
            if lengthscale.numel() == 1:
                params['lengthscale'] = lengthscale.item()
            else:
                # ARD case - store as list
                params['lengthscale'] = lengthscale.detach().cpu().numpy().tolist()
        if hasattr(covar_module, 'outputscale'):
            params['outputscale'] = covar_module.outputscale.item()
    else:
        # Plain kernel case
        if hasattr(covar_module, 'lengthscale'):
            lengthscale = covar_module.lengthscale
            if lengthscale.numel() == 1:
                params['lengthscale'] = lengthscale.item()
            else:
                # ARD case - store as list
                params['lengthscale'] = lengthscale.detach().cpu().numpy().tolist()
    
    # Weighting function parameters (current values)
    if hasattr(model.weighting_function, 'plateau_width'):
        params['plateau_width'] = model.weighting_function.plateau_width
    if hasattr(model.weighting_function, 'c'):
        params['c'] = model.weighting_function.c
    
    return params