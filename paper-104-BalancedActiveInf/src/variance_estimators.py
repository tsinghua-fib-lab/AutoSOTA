"""
Variance Estimation Module

This module implements variance estimators for different sampling designs:
    1. Bernoulli/Poisson variance estimator
    2. Cube (balanced sampling) variance estimator

These estimators account for the sampling design to provide valid
variance estimates for population mean estimation.
"""

import numpy as np
from typing import Optional


def estimate_bernoulli_variance(
    y: np.ndarray,
    y_pred: np.ndarray,
    sampling_indicators: np.ndarray,
    inclusion_probs: np.ndarray
) -> float:
    """
    Estimate variance for Bernoulli/Poisson sampling designs.
    
    This estimator is appropriate for sampling designs where each unit
    is independently selected with potentially different inclusion
    probabilities (Poisson sampling).
    
    The variance estimator is:
        V̂ = (1/N²) * Σᵢ∈s [(yᵢ - ŷᵢ)² * (1 - πᵢ) / πᵢ²]
    
    where:
        - N is the population size
        - s is the set of sampled units
        - yᵢ is the true label for unit i
        - ŷᵢ is the predicted label for unit i
        - πᵢ is the inclusion probability for unit i
    
    Args:
        y: True labels of shape (N,)
        y_pred: Predicted labels of shape (N,)
        sampling_indicators: Binary indicators (1 if sampled) of shape (N,)
        inclusion_probs: Inclusion probabilities of shape (N,)
    
    Returns:
        variance_estimate: Estimated variance of the population mean estimator
    
    Example:
        >>> variance = estimate_bernoulli_variance(y, y_pred, xi, pi)
    """
    N = len(y)
    
    # Extract sampled units
    sampled_indices = np.where(sampling_indicators == 1)[0]
    
    # Compute squared residuals for sampled units
    residuals_squared = (y[sampled_indices] - y_pred[sampled_indices]) ** 2
    
    # Compute variance terms with design weights
    pi_sampled = inclusion_probs[sampled_indices]
    variance_terms = residuals_squared * (1.0 - pi_sampled) / (pi_sampled ** 2)
    
    # Sum and scale by population size
    variance_estimate = variance_terms.sum() / (N ** 2)
    
    return variance_estimate


def estimate_cube_variance(
    y: np.ndarray,
    X: np.ndarray,
    sampling_indicators: np.ndarray,
    inclusion_probs: np.ndarray,
    N: int,
    max_iter: int = 100,
    tol: float = 1e-6
) -> float:
    """
    Estimate variance for cube (balanced) sampling designs.
    
    This estimator is appropriate for balanced sampling designs where
    the sample is balanced on auxiliary variables. The estimator uses
    calibrated weights that ensure exact balance on the auxiliaries.
    
    The algorithm iteratively computes calibration weights c that minimize
    variance while maintaining balance constraints.
    
    Args:
        y: Response variable (residuals) of shape (N,)
        X: Auxiliary variables of shape (N, p) or (N, 1)
        sampling_indicators: Binary indicators (1 if sampled) of shape (N,)
        inclusion_probs: Inclusion probabilities of shape (N,)
        N: Population size
        max_iter: Maximum iterations for calibration (default: 100)
        tol: Convergence tolerance (default: 1e-6)
    
    Returns:
        variance_estimate: Estimated variance of the population mean estimator
    
    Reference:
        Deville, J.-C., & Tillé, Y. (2004). Efficient balanced sampling.
        Biometrika, 91(4), 893-912.
    
    Example:
        >>> variance = estimate_cube_variance(
        ...     y_residuals, X_auxiliary, xi, pi, N=5000
        ... )
    """
    # Extract sampled units
    sampled_indices = np.where(sampling_indicators == 1)[0]
    y_sample = y[sampled_indices]
    X_sample = X[sampled_indices, :]
    pi_sample = inclusion_probs[sampled_indices]
    
    n, p = len(y_sample), X_sample.shape[1]
    
    # Initialize calibration weights
    # Starting point: c = (n/(n-p)) * (1 - π)
    c = (n / (n - p)) * (1.0 - pi_sample)
    
    # Iterative calibration algorithm
    for iteration in range(max_iter):
        # Compute weighted cross-product matrix M = Σᵢ (cᵢ/πᵢ²) * xᵢxᵢᵀ
        M = np.zeros((p, p))
        for i in range(n):
            weight = c[i] / (pi_sample[i] ** 2)
            M += weight * np.outer(X_sample[i], X_sample[i])
        
        # Update calibration weights
        c_new = np.zeros(n)
        M_inv = np.linalg.inv(M)
        
        for k in range(n):
            x_k = X_sample[k]
            
            # Compute quadratic coefficient
            A = (x_k @ M_inv @ x_k) / pi_sample[k]
            
            # Solve quadratic equation for optimal weight
            discriminant = 1.0 + 4.0 * A * (1.0 - pi_sample[k])
            
            if discriminant < 0 or A == 0:
                # Fallback to simple weight if no valid solution
                c_new[k] = 1.0 - pi_sample[k]
            else:
                # Optimal calibration weight
                c_new[k] = (-1.0 + np.sqrt(discriminant)) / (2.0 * A)
        
        # Check convergence
        if np.max(np.abs(c_new - c)) < tol:
            break
        
        c = c_new
    
    # Compute calibrated weights w = c / π²
    weights = c / (pi_sample ** 2)
    
    # Perform weighted regression to compute fitted values
    # Solve: (XᵀWX)b = XᵀWy
    XtW = X_sample.T * weights  # Shape: (p, n)
    XtWX = XtW @ X_sample  # Shape: (p, p)
    XtWy = XtW @ y_sample  # Shape: (p,)
    
    # Solve for regression coefficients
    b = np.linalg.solve(XtWX, XtWy)
    
    # Compute residuals from weighted regression
    residuals = y_sample - X_sample @ b
    
    # Final variance estimate
    variance_estimate = (c * residuals ** 2 / pi_sample ** 2).sum() / (N ** 2)
    
    return variance_estimate


def compute_confidence_interval(
    point_estimate: float,
    variance_estimate: float,
    confidence_level: float = 0.95
) -> tuple:
    """
    Compute confidence interval for population mean.
    
    Args:
        point_estimate: Estimated population mean
        variance_estimate: Estimated variance of the estimator
        confidence_level: Confidence level (default: 0.95)
    
    Returns:
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        interval_width: Width of the confidence interval
    
    Example:
        >>> lb, ub, width = compute_confidence_interval(mean_est, var_est)
    """
    from scipy.stats import norm
    
    # Compute quantile for the given confidence level
    alpha = 1.0 - confidence_level
    z_quantile = norm.ppf(1.0 - alpha / 2.0)
    
    # Standard error
    std_error = np.sqrt(variance_estimate)
    
    # Margin of error
    margin = z_quantile * std_error
    
    # Confidence bounds
    lower_bound = point_estimate - margin
    upper_bound = point_estimate + margin
    interval_width = 2.0 * margin
    
    return lower_bound, upper_bound, interval_width
