#!/usr/bin/env python3
"""
BoundedCertifier Convergence Analysis - Two-Part Analysis
========================================================

This script implements convergence analysis for BoundedCertifier following the same
two-part validation structure as complete_samc_truth_for_bounded_fmple_size_validation.py:

PART 1: Verification of Estimators (Sanity Check)
- Show convergence of variance (C) and gradient norm (G) estimates
- Demonstrates that statistical estimators are working correctly

PART 2: Verification of Final Certificate (Main Result)  
- Show convergence of final certified radii (R)
- Most important validation - proves end-to-end system stability

Supports bounded synthetic functions:
1. Bounded Quadratic Function: Analytical ground truth available
2. Bounded Slice Function: Analytical ground truth available  
3. Bounded Sine Function: Analytical ground truth available (rotation-like)
4. Bounded Linear Function: Analytical ground truth available

Sample sizes tested: [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]

Usage:
    python bounded_certifier_convergence_analysis.py --function all
    python bounded_certifier_convergence_analysis.py --function bounded_quadratic
    python bounded_certifier_convergence_analysis.py --function bounded_sine --sigma 0.1 --eps_y 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import json
import math
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import brentq

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from regression_certifiers.certify.bounded_fn_certifier import BoundedCertifier

# Import bounded synthetic functions from the controlled test (optional - only needed for validation)
sys.path.append(str(Path(__file__).parent))
try:
    from test_bounded_certifier_controlled import (
        bounded_quadratic, bounded_quadratic_smoothed,
        bounded_slice_function, bounded_slice_smoothed,
        bounded_sine_function, bounded_sine_smoothed,
        bounded_linear_function, bounded_linear_smoothed
    )
    HAS_TEST_FUNCTIONS = True
except ImportError:
    # These are only needed for validation/testing, not for the U-statistic estimators
    HAS_TEST_FUNCTIONS = False
    print("Note: test_bounded_certifier_controlled not found - validation functions unavailable")


class BoundedCertifierConvergenceValidator:
    """
    Convergence validation framework for BoundedCertifier.
    """
    
    def __init__(self, sigma: float = 0.1, eps_y: float = 1.0, confidence: float = 0.999):
        self.sigma = sigma
        self.eps_y = eps_y
        self.confidence = confidence
        
        # Note: BoundedCertifier will be initialized per function with specific M value
    
    def u_statistic_variance_estimator_alpha_half(self, samples: np.ndarray) -> tuple:
        """U-statistic variance estimator with α/2 confidence interval for union bound."""
        from scipy.stats import norm
        
        n = len(samples)
        
        # U-statistic estimator: S² (unbiased sample variance)
        theta_hat = np.var(samples, ddof=1)  # This is S² directly
        
        # Sample estimates for asymptotic variance
        mean_val = np.mean(samples)
        fourth_moment = np.mean((samples - mean_val)**4)
        
        # Asymptotic variance: m̂₄ - (S²)²
        # Clamp for numerical safety (as suggested by collaborator)
        asymptotic_var = max(0.0, fourth_moment - theta_hat**2)
        
        # Use α/2 for union bound
        alpha_total = 1 - self.confidence
        alpha_split = alpha_total / 2.0
        # Use t-distribution to account for finite-sample uncertainty
        from scipy.stats import t
        t_critical = t.ppf(1 - alpha_split, df=n-1)  # One-sided UCB for certification
        se = np.sqrt(asymptotic_var / n)
        
        lower_bound = theta_hat - t_critical * se
        upper_bound = theta_hat + t_critical * se
        
        return theta_hat, lower_bound, upper_bound
    
    def u_statistic_variance_estimator_bootstrap(self, samples: np.ndarray, 
                                                 B: int = 2000,
                                                 rng: Optional[np.random.Generator] = None) -> tuple:
        """
        U-statistic variance estimator with BOOTSTRAP confidence interval.
        
        Addresses the collaborator's concern: The analytical CI uses m̂₄ - (S²)²
        for asymptotic variance, but (S²)² is a BIASED estimator of σ⁴ due to
        Jensen's inequality: E[(S²)²] > (E[S²])² = σ⁴.
        
        This causes the asymptotic variance to be underestimated, leading to
        deceptively narrow CIs with poor coverage.
        
        Solution: Use bootstrap percentile CI which is non-parametric and avoids
        the biased (S²)² term entirely.
        
        Algorithm:
        1. Keep the U-statistic point estimate: Ĉ = S² (unbiased)
        2. Bootstrap resample: Create B bootstrap samples by resampling with replacement
        3. For each bootstrap sample, compute S²_b
        4. CI = [percentile(2.5%), percentile(97.5%)] of bootstrap S² values
        
        Args:
            samples: Array of function values f(z + η_i)
            B: Number of bootstrap samples (default: 2000)
            rng: Random number generator
            
        Returns:
            (C_hat, C_lower, C_upper): Point estimate and bootstrap CI
        """
        if rng is None:
            rng = np.random.default_rng()
        
        n = len(samples)
        
        if n < 2:
            return 0.0, 0.0, 0.0
        
        # U-statistic point estimate (unbiased)
        theta_hat = np.var(samples, ddof=1)
        
        # Bootstrap procedure
        bootstrap_variances = []
        for b in range(B):
            # Resample with replacement
            bootstrap_sample = rng.choice(samples, size=n, replace=True)
            
            # Compute S² for this bootstrap sample
            if len(np.unique(bootstrap_sample)) > 1:  # Check for non-zero variance
                C_b = np.var(bootstrap_sample, ddof=1)
            else:
                C_b = 0.0
            
            bootstrap_variances.append(C_b)
        
        bootstrap_variances = np.array(bootstrap_variances)
        
        # Compute bootstrap percentile CI
        # Use α/2 for union bound (consistent with other methods)
        alpha_total = 1 - self.confidence
        alpha_split = alpha_total / 2.0
        
        lower_percentile = (alpha_split / 2) * 100
        upper_percentile = (1 - alpha_split / 2) * 100
        
        lower_bound = np.percentile(bootstrap_variances, lower_percentile)
        upper_bound = np.percentile(bootstrap_variances, upper_percentile)
        
        return theta_hat, lower_bound, upper_bound
    
    def mom_gradient_norm_estimator_with_bootstrap(self, f_values: np.ndarray, eta_samples: np.ndarray, 
                                                   K: Optional[int] = None, B: int = 1000, alpha: float = 0.05, 
                                                   rng: Optional[np.random.Generator] = None) -> tuple:
        """
        Median-of-Means (MoM) gradient norm estimator with bootstrap CI.
        
        Following collaborator's recommendation for heavy-tailed distributions.
        
        Algorithm:
        1. Compute W_i = (1/σ²) * ε_i * f(z + ε_i) for all samples
        2. Partition N samples into K blocks
        3. Compute block means: μ̂_k for each block k
        4. MoM estimator: μ̂_MoM = coordinate-wise median of block means
        5. Point estimate: ||μ̂_MoM||
        6. Bootstrap CI: Resample block means (not original samples) and compute quantiles
        
        Args:
            f_values: Function values at noisy points
            eta_samples: Noise samples
            K: Number of blocks. If None, uses K = ⌈√N⌉ (recommended dynamic choice)
            B: Number of bootstrap iterations (default 1000)
            alpha: Significance level (default 0.05 for 95% CI)
            rng: Random number generator
            
        Returns:
            G_hat_mom: MoM point estimate of ||G||
            G_lower: Lower CI bound
            G_upper: Upper CI bound
        """
        if rng is None:
            rng = np.random.default_rng()
        
        n = len(f_values)
        
        # Use dynamic K if not specified: K = ⌈√N⌉
        # This ensures block size m = N/K grows with N, allowing CLT to apply better
        if K is None:
            K = int(np.ceil(np.sqrt(n)))
            # Ensure K is at least 2 and at most N//2
            K = max(2, min(K, n // 2))
        
        # Construct W_i = (1/σ²) * ε_i * f(z + ε_i)
        W_samples = (1 / self.sigma**2) * eta_samples * f_values[:, np.newaxis]
        
        # Step 1: Partition into K blocks
        indices = rng.permutation(n)
        blocks = np.array_split(indices, K)
        
        # Step 2: Compute block means
        block_means = np.array([W_samples[block].mean(axis=0) for block in blocks])  # Shape: (K, d)
        
        # Step 3: MoM estimator = coordinate-wise median
        mu_mom = np.median(block_means, axis=0)  # Shape: (d,)
        
        # Step 4: Point estimate
        G_hat_mom = float(np.linalg.norm(mu_mom))
        
        # Step 5: Bootstrap CI on block means
        bootstrap_stats = []
        for _ in range(B):
            # Resample K block means with replacement
            boot_indices = rng.integers(0, K, size=K)
            boot_block_means = block_means[boot_indices]
            
            # Compute MoM for this bootstrap sample
            boot_mu_mom = np.median(boot_block_means, axis=0)
            boot_G = np.linalg.norm(boot_mu_mom)
            bootstrap_stats.append(boot_G)
        
        # Step 6: Percentile CI
        G_lower, G_upper = np.quantile(bootstrap_stats, [alpha/2, 1-alpha/2])
        
        return float(G_hat_mom), float(G_lower), float(G_upper)
    
    def mom_antithetic_gradient_norm_estimator(self, f_bounded, z: np.ndarray, sigma: float,
                                               N: int, K: Optional[int] = None, B: int = 1000,
                                               alpha: float = 0.05, rng: Optional[np.random.Generator] = None) -> tuple:
        """
        Median-of-Means (MoM) gradient norm estimator with ANTITHETIC SAMPLING.
        
        Following collaborator's recommendation to eliminate skewness by forcing symmetry.
        
        Key Innovation:
        - Uses antithetic pairs (ε, -ε) to create symmetrized W vectors
        - Distribution of W_symm is GUARANTEED symmetric by construction
        - This fixes the skewness issue that plagued standard MoM
        
        ⚠️ CRITICAL: This method CAN ONLY be used with MoM, NOT with U-statistics!
        - U-statistics require INDEPENDENT samples
        - Antithetic pairs (ε, -ε) are perfectly correlated
        - MoM only requires block means to be independent, so antithetic is OK within blocks
        
        Algorithm:
        1. Generate N/2 noise samples {ε₁, ..., ε_{N/2}}
        2. For each εᵢ, compute symmetrized W vector:
           W_symm,i = (1/2σ²) εᵢ · (f(z + εᵢ) - f(z - εᵢ))
        3. Now have N/2 symmetrized W vectors
        4. Partition into K blocks and apply standard MoM
        
        Args:
            f_bounded: Bounded scalar function
            z: Point to certify (flattened array)
            sigma: Noise standard deviation
            N: Total number of evaluations (will use N function calls)
            K: Number of blocks. If None, uses K = ⌈√(N/2)⌉
            B: Number of bootstrap iterations
            alpha: Significance level
            rng: Random number generator
            
        Returns:
            G_hat_mom_anti: MoM antithetic point estimate of ||G||
            G_lower: Lower CI bound
            G_upper: Upper CI bound
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Step 1: Generate N/2 noise samples
        n_half = N // 2
        eta_half = rng.normal(0.0, sigma, size=(n_half, z.size))
        
        # Step 2: Compute symmetrized W vectors
        # W_symm,i = (1/2σ²) εᵢ · (f(z + εᵢ) - f(z - εᵢ))
        W_symm_samples = []
        for i in range(n_half):
            eps_i = eta_half[i]
            
            # Evaluate function at z + εᵢ and z - εᵢ (antithetic pair)
            f_plus = f_bounded(z + eps_i)
            f_minus = f_bounded(z - eps_i)
            
            # Symmetrized W vector
            W_symm_i = (1.0 / (2.0 * sigma**2)) * eps_i * (f_plus - f_minus)
            W_symm_samples.append(W_symm_i)
        
        W_symm_samples = np.array(W_symm_samples)  # Shape: (N/2, d)
        
        # Step 3: Dynamic K for N/2 samples
        if K is None:
            K = int(np.ceil(np.sqrt(n_half)))
            K = max(2, min(K, n_half // 2))
        
        # Step 4: Partition symmetrized W vectors into K blocks
        indices = rng.permutation(n_half)
        blocks = np.array_split(indices, K)
        
        # Step 5: Compute block means
        block_means = np.array([W_symm_samples[block].mean(axis=0) for block in blocks])  # Shape: (K, d)
        
        # Step 6: MoM estimator = coordinate-wise median
        # Now the distribution is SYMMETRIC, so median = mean!
        mu_mom_anti = np.median(block_means, axis=0)  # Shape: (d,)
        
        # Step 7: Point estimate
        G_hat_mom_anti = float(np.linalg.norm(mu_mom_anti))
        
        # Step 8: Bootstrap CI on block means
        bootstrap_stats = []
        for _ in range(B):
            # Resample K block means with replacement
            boot_indices = rng.integers(0, K, size=K)
            boot_block_means = block_means[boot_indices]
            
            # Compute MoM for this bootstrap sample
            boot_mu_mom = np.median(boot_block_means, axis=0)
            boot_G = np.linalg.norm(boot_mu_mom)
            bootstrap_stats.append(boot_G)
        
        # Step 9: Percentile CI
        G_lower, G_upper = np.quantile(bootstrap_stats, [alpha/2, 1-alpha/2])
        
        return float(G_hat_mom_anti), float(G_lower), float(G_upper)
    
    def bias_corrected_gradient_norm_estimator(self, f_values: np.ndarray, eta_samples: np.ndarray) -> tuple:
        """
        Bias-corrected gradient norm estimator with Delta Method CI.
        
        Following collaborator's recommendation to correct the downward bias in √θ̂.
        
        Key insight: The estimator Ĝ = √θ̂ is biased low due to Jensen's inequality.
        
        Bias Formula (from Taylor expansion):
            Bias(√θ̂) ≈ -Var(θ̂) / (8θ^(3/2))
        
        Bias-Corrected Estimator:
            Ĝ_corrected = √θ̂ + Var(θ̂) / (8θ̂^(3/2))
        
        This estimator is approximately unbiased and provides tighter CIs.
        
        Returns:
            G_hat_corrected: Bias-corrected point estimate
            G_lower: Lower CI bound
            G_upper: Upper CI bound
        """
        from scipy.stats import norm
        
        n = len(f_values)
        
        # Construct W_i = (1/σ²) * ε_i * f(z + ε_i)
        W_samples = (1 / self.sigma**2) * eta_samples * f_values[:, np.newaxis]
        
        # Step 1: U-statistic for θ = ||G||²
        sum_W = np.sum(W_samples, axis=0)
        sum_W_sq_norm = np.dot(sum_W, sum_W)
        sum_sq_norm_W = np.sum(np.linalg.norm(W_samples, axis=1) ** 2)
        off_diagonal_sum = 0.5 * (sum_W_sq_norm - sum_sq_norm_W)
        num_pairs = n * (n - 1) / 2
        
        if num_pairs > 0:
            theta_hat_sq = off_diagonal_sum / num_pairs
        else:
            theta_hat_sq = 0.0
        
        # Handle negative estimates (numerical issues)
        if theta_hat_sq < 0:
            mu_hat = np.mean(W_samples, axis=0)
            theta_hat_sq = max(0.0, np.dot(mu_hat, mu_hat))
        
        # Step 2: Biased-low estimate (before correction)
        grad_norm_biased = np.sqrt(theta_hat_sq)
        
        # Step 3: Compute variance of θ̂ for bias correction
        mu_hat = np.mean(W_samples, axis=0)
        centered_W = W_samples - mu_hat
        Sigma_hat = centered_W.T @ centered_W / (n - 1)
        
        # Asymptotic variance of θ̂: Var_asym(θ̂) = 4μᵀΣμ
        asymptotic_var_theta = max(0.0, 4.0 * (mu_hat @ Sigma_hat @ mu_hat))
        var_theta_hat = asymptotic_var_theta / n
        
        # Step 4: BIAS CORRECTION
        # Bias ≈ -Var(θ̂) / (8θ^(3/2))
        # So: Ĝ_corrected = √θ̂ - Bias = √θ̂ + Var(θ̂) / (8θ̂^(3/2))
        if theta_hat_sq > 1e-9:  # Avoid division by zero
            bias_correction_term = var_theta_hat / (8.0 * (theta_hat_sq ** 1.5))
            grad_norm_corrected = grad_norm_biased + bias_correction_term
        else:
            grad_norm_corrected = grad_norm_biased
        
        # Step 5: Delta Method CI (using corrected center, same variance)
        # Var(Ĝ) = Var(√θ̂) ≈ (μᵀΣμ) / (n·θ)
        if theta_hat_sq > 1e-9:
            var_G = (mu_hat @ Sigma_hat @ mu_hat) / (n * theta_hat_sq)
            var_G = max(0.0, var_G)
            se_G = np.sqrt(var_G)
            
            # Use α/2 for union bound
            alpha_total = 1 - self.confidence
            alpha_split = alpha_total / 2.0
            # Use t-distribution to account for finite-sample uncertainty
            from scipy.stats import t
            t_critical = t.ppf(1 - alpha_split, df=n-1)  # One-sided UCB for certification
            
            # CI centered on bias-corrected estimate
            grad_norm_lower = max(0.0, grad_norm_corrected - t_critical * se_G)
            grad_norm_upper = grad_norm_corrected + t_critical * se_G
        else:
            grad_norm_lower = 0.0
            grad_norm_upper = 0.0
        
        return float(grad_norm_corrected), float(grad_norm_lower), float(grad_norm_upper)
    
    def transformation_of_endpoints_gradient_norm_estimator(self, f_values: np.ndarray, eta_samples: np.ndarray) -> tuple:
        """
        Transformation of Endpoints method for gradient norm CI.
        
        This is the SIMPLE, CONSERVATIVE, and GUARANTEED COVERAGE method:
        1. Compute CI for θ = ||G||² using U-statistic
        2. Take √(CI_lower) and √(CI_upper) as the CI for ||G||
        
        Advantages:
        - Guaranteed valid coverage if the CI for θ is valid
        - Simple and robust
        - No additional assumptions needed
        
        Disadvantages:
        - Statistically inefficient (wider CI than Delta Method)
        - Conservative (actual coverage may be > 95%)
        
        This method was previously removed but is now added back as a baseline.
        
        Args:
            f_values: Function values at noisy points
            eta_samples: Noise samples
            
        Returns:
            G_hat: Point estimate of ||G|| (same as Delta Method: √θ̂)
            G_lower: Lower CI bound = √(θ_lower)
            G_upper: Upper CI bound = √(θ_upper)
        """
        from scipy.stats import norm
        
        n = len(f_values)
        
        # Step 1: Compute W_i = (1/σ²) * ε_i * f(z + ε_i)
        W_samples = (1 / self.sigma**2) * eta_samples * f_values[:, np.newaxis]
        
        # Step 2: U-statistic for θ = ||G||²
        # CORRECT unbiased U-statistic
        sum_W = np.sum(W_samples, axis=0)
        sum_W_sq_norm = np.dot(sum_W, sum_W)
        sum_sq_norm_W = np.sum(np.linalg.norm(W_samples, axis=1) ** 2)
        off_diagonal_sum = 0.5 * (sum_W_sq_norm - sum_sq_norm_W)
        num_pairs = n * (n - 1) / 2

        if num_pairs > 0:
            theta_hat_sq = off_diagonal_sum / num_pairs
        else:
            theta_hat_sq = 0.0

        # FIXED: Compute mu_hat BEFORE using it (bug fix)
        mu_hat = np.mean(W_samples, axis=0)
        
        if theta_hat_sq < 0:
            # Fallback to sample mean approach
            theta_hat_sq = max(0.0, np.dot(mu_hat, mu_hat))
        
        # Step 3: Asymptotic variance for θ̂
        centered_W = W_samples - mu_hat[np.newaxis, :]
        Sigma_hat = centered_W.T @ centered_W / (n - 1)
        
        # Var_asym(θ̂) = 4μᵀΣμ
        asymptotic_var_theta = max(0.0, 4.0 * (mu_hat @ Sigma_hat @ mu_hat))
        var_theta_hat = asymptotic_var_theta / n
        
        # Step 4: CI for θ = ||G||² using t-distribution
        alpha_total = 1 - self.confidence
        alpha_split = alpha_total / 2.0
        # Use t-distribution to account for finite-sample uncertainty
        from scipy.stats import t
        t_critical = t.ppf(1 - alpha_split, df=n-1)  # One-sided UCB for certification
        
        se_theta = np.sqrt(var_theta_hat)
        theta_lower = max(0.0, theta_hat_sq - t_critical * se_theta)
        theta_upper = theta_hat_sq + t_critical * se_theta
        
        # Step 5: Transform endpoints by taking square root
        G_hat = np.sqrt(theta_hat_sq)
        G_lower = np.sqrt(theta_lower)  # Conservative: √(lower bound)
        G_upper = np.sqrt(theta_upper)  # Conservative: √(upper bound)
        
        return float(G_hat), float(G_lower), float(G_upper)
    
    def chebyshev_transformation_gradient_norm_estimator(self, f_values: np.ndarray, eta_samples: np.ndarray) -> tuple:
        """
        Chebyshev-based Transformation CI for gradient norm.
        
        Following collaborator's recommendation for a "twice-conservative" CI that should
        guarantee coverage without assuming normality.
        
        Key Innovation:
        - Uses Chebyshev's inequality instead of CLT for the CI on θ = ||G||²
        - Chebyshev's inequality holds for ANY distribution with finite variance
        - Much wider margin of error: √(1/α_split) instead of z_{α/2}
        
        Algorithm:
        1. Compute θ̂ = ||G||² using U-statistic (SAME as Delta Method)
        2. Compute SE(θ̂) using asymptotic variance
        3. Chebyshev CI for θ: θ̂ ± √(1/α_split) · SE(θ̂)
        4. Transform endpoints: √(CI_lower), √(CI_upper)
        
        For 95% CI with two-sided test (α_split = 0.025):
            Margin = √(1/0.025) ≈ 6.32 * SE(θ̂)
        
        Compare to normal approximation:
            Margin = z_{0.025} ≈ 1.96 * SE(θ̂)
        
        Chebyshev is ~3.2× wider, making this VERY conservative but non-asymptotic.
        
        Args:
            f_values: Function values at noisy points
            eta_samples: Noise samples
            
        Returns:
            G_hat: Point estimate of ||G|| (√θ̂, same as Delta Method)
            G_lower: Lower CI bound = √(θ_lower)
            G_upper: Upper CI bound = √(θ_upper)
        """
        n = len(f_values)
        
        # Step 1: Compute W_i = (1/σ²) * ε_i * f(z + ε_i)
        W_samples = (1 / self.sigma**2) * eta_samples * f_values[:, np.newaxis]
        
        # Step 2: U-statistic for θ = ||G||²
        # CORRECT unbiased U-statistic (same as other methods)
        sum_W = np.sum(W_samples, axis=0)
        sum_W_sq_norm = np.dot(sum_W, sum_W)
        sum_sq_norm_W = np.sum(np.linalg.norm(W_samples, axis=1) ** 2)
        off_diagonal_sum = 0.5 * (sum_W_sq_norm - sum_sq_norm_W)
        num_pairs = n * (n - 1) / 2

        if num_pairs > 0:
            theta_hat_sq = off_diagonal_sum / num_pairs
        else:
            theta_hat_sq = 0.0

        # Compute mu_hat for variance estimation
        mu_hat = np.mean(W_samples, axis=0)
        
        if theta_hat_sq < 0:
            # Fallback to sample mean approach
            theta_hat_sq = max(0.0, np.dot(mu_hat, mu_hat))
        
        # Step 3: Asymptotic variance for θ̂
        centered_W = W_samples - mu_hat[np.newaxis, :]
        Sigma_hat = centered_W.T @ centered_W / (n - 1)
        
        # Var_asym(θ̂) = 4μᵀΣμ
        asymptotic_var_theta = max(0.0, 4.0 * (mu_hat @ Sigma_hat @ mu_hat))
        var_theta_hat = asymptotic_var_theta / n
        
        # Step 4: CHEBYSHEV CI for θ = ||G||²
        alpha_total = 1 - self.confidence
        alpha_split = alpha_total / 2.0
        
        # Chebyshev margin: √(1/α_split) * SE(θ̂)
        # For 95% CI: α_split = 0.025, so margin = √(40) ≈ 6.32 * SE(θ̂)
        chebyshev_factor = np.sqrt(1.0 / alpha_split)
        
        se_theta = np.sqrt(var_theta_hat)
        margin_theta = chebyshev_factor * se_theta
        
        theta_lower = max(0.0, theta_hat_sq - margin_theta)
        theta_upper = theta_hat_sq + margin_theta
        
        # Step 5: Transform endpoints by taking square root
        G_hat = np.sqrt(theta_hat_sq)  # Same center as Delta Method
        G_lower = np.sqrt(theta_lower)  # Very conservative lower bound
        G_upper = np.sqrt(theta_upper)  # Very conservative upper bound
        
        return float(G_hat), float(G_lower), float(G_upper)
    
    def bootstrap_bias_corrected_gradient_norm_estimator(self, f_values: np.ndarray, eta_samples: np.ndarray,
                                                         B: int = 1000, rng: Optional[np.random.Generator] = None) -> tuple:
        """
        Bootstrap Bias-Corrected gradient norm estimator with BCa confidence interval.
        
        Following collaborator's recommendation for empirical bias estimation with
        Bias-Corrected and accelerated (BCa) bootstrap CI.
        
        Algorithm:
        1. Compute original estimate Ĝ from original data (W vectors)
        2. Bootstrap resample W vectors B times (with replacement)
        3. For each bootstrap sample, compute Ĝ*_b
        4. Estimate bias = mean(Ĝ*_b) - Ĝ
        5. Corrected estimator = Ĝ - bias = 2Ĝ - mean(Ĝ*_b)
        6. Construct BCa CI that accounts for bias and skewness
        
        BCa Method:
        - Computes bias-correction factor (ẑ₀) from bootstrap distribution
        - Computes acceleration factor (â) from jackknife
        - Adjusts percentile endpoints to account for both factors
        
        Args:
            f_values: Function values at noisy points
            eta_samples: Noise samples
            B: Number of bootstrap iterations (default 1000)
            rng: Random number generator
            
        Returns:
            G_hat_corrected_boot: Bootstrap bias-corrected estimate
            G_lower: Lower BCa CI bound
            G_upper: Upper BCa CI bound
        """
        if rng is None:
            rng = np.random.default_rng()
        
        from scipy.stats import norm
        
        n = len(f_values)
        
        # Step 1: Compute W_i = (1/σ²) * ε_i * f(z + ε_i)
        W_samples = (1 / self.sigma**2) * eta_samples * f_values[:, np.newaxis]
        
        # Step 2: Original estimate Ĝ = ||mean(W)||
        mu_hat_original = np.mean(W_samples, axis=0)
        G_hat_original = float(np.linalg.norm(mu_hat_original))
        
        # Step 3: Bootstrap resampling of W vectors
        bootstrap_stats = []
        for _ in range(B):
            # Resample indices with replacement
            boot_indices = rng.integers(0, n, size=n)
            boot_W = W_samples[boot_indices]
            
            # Compute bootstrap estimate
            boot_mu = np.mean(boot_W, axis=0)
            boot_G = np.linalg.norm(boot_mu)
            bootstrap_stats.append(boot_G)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Step 4: Estimate bias empirically
        mean_bootstrap_G = np.mean(bootstrap_stats)
        bias_boot = mean_bootstrap_G - G_hat_original
        
        # Step 5: Bias-corrected estimator
        G_hat_corrected_boot = G_hat_original - bias_boot  # = 2*Ĝ - mean(Ĝ*)
        G_hat_corrected_boot = max(0.0, G_hat_corrected_boot)  # Ensure non-negative
        
        # Step 6: BCa Confidence Interval
        # Following collaborator's detailed BCa procedure
        
        alpha_total = 1 - self.confidence
        
        # 6a. Bias-Correction Factor (ẑ₀)
        # Proportion of bootstrap estimates less than original estimate
        proportion_less = np.sum(bootstrap_stats < G_hat_original) / B
        # Avoid edge cases
        proportion_less = np.clip(proportion_less, 1e-6, 1 - 1e-6)
        z_0 = norm.ppf(proportion_less)
        
        # 6b. Acceleration Factor (â) via Jackknife
        # EFFICIENT O(N) implementation following collaborator's recommendation
        # Key insight: sum of leave-one-out = total_sum - element_left_out
        
        # Pre-compute total sum once - O(1) operation
        total_sum_W = np.sum(W_samples, axis=0)
        
        # Compute leave-one-out estimates efficiently - O(N) instead of O(N²)
        jackknife_estimates = []
        for i in range(n):
            # This is now an O(1) operation!
            jack_sum = total_sum_W - W_samples[i]
            jack_mu = jack_sum / (n - 1)
            jack_G = np.linalg.norm(jack_mu)
            jackknife_estimates.append(jack_G)
        
        jackknife_estimates = np.array(jackknife_estimates)
        jack_mean = np.mean(jackknife_estimates)
        
        # Acceleration factor formula
        numerator = np.sum((jack_mean - jackknife_estimates) ** 3)
        denominator = 6 * (np.sum((jack_mean - jackknife_estimates) ** 2) ** 1.5)
        
        if abs(denominator) > 1e-9:
            a_hat = numerator / denominator
        else:
            a_hat = 0.0
        
        # 6c. Adjusted Percentile Endpoints
        z_alpha_half = norm.ppf(alpha_total / 2)
        z_1_minus_alpha_half = norm.ppf(1 - alpha_total / 2)
        
        # α₁ = Φ(ẑ₀ + (ẑ₀ + z_{α/2}) / (1 - â(ẑ₀ + z_{α/2})))
        numerator_1 = z_0 + z_alpha_half
        denominator_1 = 1 - a_hat * (z_0 + z_alpha_half)
        if abs(denominator_1) > 1e-6:
            alpha_1 = norm.cdf(z_0 + numerator_1 / denominator_1)
        else:
            alpha_1 = alpha_total / 2  # Fallback
        
        # α₂ = Φ(ẑ₀ + (ẑ₀ - z_{α/2}) / (1 - â(ẑ₀ - z_{α/2})))
        numerator_2 = z_0 - z_1_minus_alpha_half
        denominator_2 = 1 - a_hat * (z_0 - z_1_minus_alpha_half)
        if abs(denominator_2) > 1e-6:
            alpha_2 = norm.cdf(z_0 + numerator_2 / denominator_2)
        else:
            alpha_2 = 1 - alpha_total / 2  # Fallback
        
        # Ensure α₁ < α₂ and within [0, 1]
        alpha_1 = np.clip(alpha_1, 0.001, 0.999)
        alpha_2 = np.clip(alpha_2, 0.001, 0.999)
        if alpha_1 >= alpha_2:
            # Fallback to standard percentiles
            alpha_1 = alpha_total / 2
            alpha_2 = 1 - alpha_total / 2
        
        # 6d. Extract CI from bootstrap distribution
        G_lower = max(0.0, float(np.quantile(bootstrap_stats, alpha_1)))
        G_upper = float(np.quantile(bootstrap_stats, alpha_2))
        
        return float(G_hat_corrected_boot), float(G_lower), float(G_upper)





    def finite_sample_gradient_norm_estimator(
        self,
        f_values: np.ndarray,
        eta_samples: np.ndarray,
        sigma: float, # Use self.sigma instead if integrating into the class fully
        M: float,     # Bound on f(x) output, e.g., np.pi
        confidence: float, # Use self.confidence instead if integrating
        d: int          # Input dimension (e.g., 784 for MNIST)
    ) -> tuple:
        """
        Finite-sample gradient norm estimator based on Bernstein-style bound.

        Derivation assumes f is bounded in [-M, M]. Provides a non-asymptotic
        confidence interval. The bound 't' depends explicitly on dimension 'd'.
        
        Based on Theorem 2 from Levine et al. (2020) "Tight Second-Order Certificates 
        for Randomized Smoothing" (arXiv:2010.10549).

        Args:
            f_values: (N,) array of function values f(z + ε_i).
            eta_samples: (N, d) array of noise samples ε_i.
            sigma: Noise standard deviation.
            M: Bound on function output f(x).
            confidence: Desired confidence level (e.g., 0.95).
            d: Input dimension (e.g., 784 for MNIST).
        
        Note:
            N in the condition checks refers to the number of input samples = len(f_values).

        Returns:
            G_hat: Point estimate sqrt(θ_hat_sq). Biased low.
            G_lower: Lower bound of the confidence interval for ||G||.
            G_upper: Upper bound of the confidence interval for ||G||.
        """
        N = len(f_values)
        if N < 2:
            # U-statistic requires at least 2 samples
            return 0.0, 0.0, 0.0

        # Pre-compute constants for efficiency and clarity
        sigma_sq = sigma**2
        sigma_quad = sigma**4
        M_sq = M**2
        alpha = 1.0 - confidence
        alpha_half = alpha / 2.0  # Renamed from 'eta' to avoid confusion with eta_samples

        if alpha_half <= 0 or alpha_half >= 0.5:
            # Avoid log(0) or invalid alpha_half; return degenerate interval
            return 0.0, 0.0, 0.0

        # --- Step 1: Calculate W_i vectors ---
        # W_i = (1/σ²) * ε_i * f(z + ε_i)
        W_samples = (1 / sigma_sq) * eta_samples * f_values[:, np.newaxis]

        # --- Step 2: Calculate V_tilde using the unbiased U-statistic for θ = ||G||² ---
        # θ̂_sq = (1 / C(N,2)) * Σ_{i<j} W_i^T W_j
        # Efficient calculation: θ̂_sq = [ ||Σ W_i||² - Σ ||W_i||² ] / (N(N-1))
        sum_W = np.sum(W_samples, axis=0)           # Shape (d,)
        sum_W_sq_norm = np.dot(sum_W, sum_W)        # Scalar ||Σ W_i||²
        sum_sq_norm_W = np.sum(np.linalg.norm(W_samples, axis=1) ** 2) # Scalar Σ ||W_i||²

        # N(N-1) is 2 * C(N, 2)
        if N * (N - 1) > 0:
            theta_hat_sq = (sum_W_sq_norm - sum_sq_norm_W) / (N * (N - 1))
        else:
            theta_hat_sq = 0.0

        # Clamp theta_hat_sq to be non-negative (can be slightly negative due to numerics)
        theta_hat_sq = max(0.0, theta_hat_sq)

        # V_tilde = σ⁴ * θ̂_sq
        V_tilde = sigma_quad * theta_hat_sq

        # Point estimate G_hat = sqrt(θ̂_sq)
        G_hat = np.sqrt(theta_hat_sq)

        # --- Step 3: Calculate the error bound t(α/2) ---
        log_alpha_half = math.log(alpha_half)  # This will be negative
        neg_2_log_alpha_half = -2.0 * log_alpha_half  # This will be positive

        # Determine which case of the bound applies
        # Note: N here refers to the number of input samples len(f_values)
        if neg_2_log_alpha_half <= d * N:
            # Case 1: t scales with sqrt(d*N) - dimension-dependent regime
            t_bound = 2.0 * sigma_sq * M_sq * math.sqrt((-d * log_alpha_half) / N)
        else:
            # Case 2: t scales linearly with N - large-sample regime
            t_bound = (-2.0 * math.sqrt(2.0) * sigma_sq * M_sq * log_alpha_half) / N

        # Ensure t_bound is non-negative
        t_bound = max(0.0, t_bound)

        # --- Step 4: Construct the CI for ||G|| ---
        # CI for V: [max(0, V_tilde - t_bound), V_tilde + t_bound]
        V_lower = max(0.0, V_tilde - t_bound)
        V_upper = V_tilde + t_bound  # V_tilde is non-negative, t_bound is non-negative

        # CI for ||G||²: [V_lower / σ⁴, V_upper / σ⁴]
        G_sq_lower = V_lower / sigma_quad
        G_sq_upper = V_upper / sigma_quad

        # CI for ||G||: [sqrt(G_sq_lower), sqrt(G_sq_upper)]
        G_lower = math.sqrt(G_sq_lower)
        G_upper = math.sqrt(G_sq_upper)

        return G_hat, G_lower, G_upper

    # ... (Keep other methods like geometric_median, etc.) ...

    # Example Usage (within the script or called externally):
    # Assuming `validator` is an instance of BoundedCertifierConvergenceValidator
    # and you have `f_values`, `eta_samples`, `sigma`, `M`, `confidence`, `d`

    # G_hat_fs, G_lower_fs, G_upper_fs = validator.finite_sample_gradient_norm_estimator(
    #     f_values, eta_samples, sigma, M, confidence, d
    # )
    # print(f"Finite Sample Est: {G_hat_fs:.6f} [{G_lower_fs:.6f}, {G_upper_fs:.6f}]")
    
    def u_statistic_gradient_norm_estimator_alpha_half(self, f_values: np.ndarray, eta_samples: np.ndarray) -> tuple:
        """
        U-statistic gradient norm estimator with DELTA METHOD confidence interval.
        
        Following collaborator's recommendation:
        - Estimate θ̂ = ||G||² using U-statistic
        - Apply Delta Method to get CI for Ĝ = √θ̂
        - This is the CORRECT and statistically sound approach
        
        Delta Method Formula:
            Var(Ĝ) ≈ (1/n) · Var_asym(θ̂) · [g'(θ)]²
        
        where:
            - Var_asym(θ̂) = 4μᵀΣμ
            - g(θ) = √θ, so g'(θ) = 1/(2√θ)
            - This gives: Var(Ĝ) = (1/n) · (μᵀΣμ) / θ̂
        
        CI: Ĝ ± z_{α/2} · √Var(Ĝ)
        """
        from scipy.stats import norm
        
        n = len(f_values)
        
        # Construct W_i = (1/σ²) * ε_i * f(z + ε_i) from pre-computed samples
        W_samples = (1 / self.sigma**2) * eta_samples * f_values[:, np.newaxis]

        # Efficient U-statistic for ||G||^2 using the formula:
        # θ̂ = (1/C(n,2)) * [||∑W_i||² - ∑||W_i||²] / 2
        
        sum_W = np.sum(W_samples, axis=0)
        sum_W_sq_norm = np.dot(sum_W, sum_W)
        sum_sq_norm_W = np.sum(np.linalg.norm(W_samples, axis=1) ** 2)
        off_diagonal_sum = 0.5 * (sum_W_sq_norm - sum_sq_norm_W)
        num_pairs = n * (n - 1) / 2
        
        if num_pairs > 0:
            theta_hat_sq = off_diagonal_sum / num_pairs
        else:
            theta_hat_sq = 0.0
        
        # The U-statistic can be negative due to numerical issues or high curvature
        # In such cases, use the sample mean approach as fallback
        if theta_hat_sq < 0:
            mu_hat = np.mean(W_samples, axis=0)
            theta_hat_sq = max(0.0, np.dot(mu_hat, mu_hat))
        
        # Point estimate: Ĝ = √θ̂
        grad_norm_estimate = np.sqrt(theta_hat_sq)
        
        # ===== DELTA METHOD CONFIDENCE INTERVAL =====
        # Compute μ̂ and Σ̂ for the asymptotic variance
        mu_hat = np.mean(W_samples, axis=0)
        centered_W = W_samples - mu_hat
        Sigma_hat = centered_W.T @ centered_W / (n - 1)
        
        # Asymptotic variance of θ̂: Var_asym(θ̂) = 4μᵀΣμ
        var_theta_asym = max(0.0, 4.0 * (mu_hat @ Sigma_hat @ mu_hat))
        
        # Delta Method: Var(√θ̂) = (1/n) · Var_asym(θ̂) · [g'(θ̂)]²
        # where g'(θ) = 1/(2√θ), so [g'(θ)]² = 1/(4θ)
        # Thus: Var(Ĝ) = (1/n) · 4μᵀΣμ · 1/(4θ̂) = (1/n) · (μᵀΣμ) / θ̂
        
        if theta_hat_sq > 0:
            # Variance of Ĝ using Delta Method
            var_G = (mu_hat @ Sigma_hat @ mu_hat) / (n * theta_hat_sq)
            var_G = max(0.0, var_G)  # Clamp for numerical safety
            
            # Standard error of Ĝ
            se_G = np.sqrt(var_G)
            
            # Use α/2 for union bound
            alpha_total = 1 - self.confidence
            alpha_split = alpha_total / 2.0
            # Use t-distribution to account for finite-sample uncertainty
            from scipy.stats import t
            t_critical = t.ppf(1 - alpha_split, df=n-1)  # One-sided UCB for certification
            
            # Confidence interval for Ĝ (clamped to non-negative)
            grad_norm_lower = max(0.0, grad_norm_estimate - t_critical * se_G)
            grad_norm_upper = grad_norm_estimate + t_critical * se_G
        else:
            # Degenerate case: θ̂ ≈ 0
            grad_norm_lower = 0.0
            grad_norm_upper = 0.0
        
        return grad_norm_estimate, grad_norm_lower, grad_norm_upper
    
    def geometric_median(self, points: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
        """
        Compute the Geometric Median (Fermat-Weber point) of a set of vectors.
        
        The geometric median minimizes the sum of Euclidean distances to all points:
            μ_geom = argmin_y Σ ||points[i] - y||₂
        
        This is the true multivariate generalization of the 1D median, unlike the
        coordinate-wise median which treats each dimension independently.
        
        Algorithm: Weiszfeld's algorithm (iterative reweighted least squares)
        
        Args:
            points: (n_points, d) array of vectors
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            The geometric median (d,) array
        """
        n_points, d = points.shape
        
        # Initialize at the coordinate-wise mean (good starting point)
        mu = np.mean(points, axis=0)
        
        for iteration in range(max_iter):
            # Compute distances from current estimate to all points
            dists = np.linalg.norm(points - mu, axis=1)  # (n_points,)
            
            # Avoid division by zero: if any distance is 0, that point is the geometric median
            if np.any(dists < 1e-10):
                # Return the point with zero distance
                idx = np.argmin(dists)
                return points[idx].copy()
            
            # Weiszfeld update: weighted mean with weights = 1/distance
            weights = 1.0 / dists  # (n_points,)
            mu_new = np.sum(points * weights[:, np.newaxis], axis=0) / np.sum(weights)
            
            # Check convergence
            if np.linalg.norm(mu_new - mu) < tol:
                return mu_new
            
            mu = mu_new
        
        # If not converged, return best estimate
        return mu
    
    def geometric_mom_gradient_norm_estimator(self, f_values: np.ndarray, eta_samples: np.ndarray,
                                             K: Optional[int] = None, B: int = 1000,
                                             rng: Optional[np.random.Generator] = None) -> tuple:
        """
        Geometric Median of Means (Geometric MoM) gradient norm estimator.
        
        This is a sophisticated robust estimator that addresses the pathological
        distribution of W vectors by:
        1. Using block structure (Median-of-Means framework) for variance reduction
        2. Using Geometric Median for true multivariate robustness
        3. Separating estimation (of the vector G) from transformation (taking ||G||)
        
        Key Innovation (from collaborator):
        "The difficulties (bias from Jensen's inequality, failure of the Delta Method)
        arose from trying to estimate a non-linear function (|| · ||) of a statistical
        expectation (E[W]) simultaneously. By focusing on getting a good estimate of
        E[W] first, you handle the core statistical challenge of the heavy-tailed,
        skewed distribution head-on."
        
        Algorithm:
        1. Generate N samples of W_i = (1/σ²) * η_i * f(z + η_i)
        2. Partition into K = ⌈√N⌉ blocks
        3. Compute block means: {μ̂₁, ..., μ̂_K}
        4. Compute Geometric Median: Ĝ_robust = GeometricMedian({μ̂₁, ..., μ̂_K})
        5. Point estimate: ||Ĝ_robust||
        6. Bootstrap the block means for CI
        
        Why Geometric Median?
        - Unlike coordinate-wise median, it's truly multivariate
        - Minimizes sum of Euclidean distances (rotation-invariant)
        - Robust to outliers in ANY direction in high-dimensional space
        - Perfect for our 784-dimensional W vectors
        
        Args:
            f_values: Function evaluations at noisy points
            eta_samples: Noise samples
            K: Number of blocks (default: ⌈√N⌉)
            B: Number of bootstrap iterations
            rng: Random number generator
            
        Returns:
            Tuple of (grad_norm_estimate, grad_norm_lower, grad_norm_upper)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        n = len(f_values)
        
        # Construct W_i = (1/σ²) * η_i * f(z + η_i)
        W_samples = (1 / self.sigma**2) * eta_samples * f_values[:, np.newaxis]
        
        # Step 1 & 2: Partition into K blocks
        if K is None:
            K = max(2, int(np.ceil(np.sqrt(n))))
        
        block_size = n // K
        if block_size < 1:
            block_size = 1
            K = n
        
        # Step 3: Compute block means
        block_means = []
        for k in range(K):
            start_idx = k * block_size
            end_idx = min((k + 1) * block_size, n)
            if start_idx >= n:
                break
            block = W_samples[start_idx:end_idx]
            block_mean = np.mean(block, axis=0)
            block_means.append(block_mean)
        
        block_means = np.array(block_means)  # (K, d)
        actual_K = len(block_means)
        
        # Step 4: Compute Geometric Median of block means
        G_robust = self.geometric_median(block_means)
        
        # Step 5: Point estimate
        grad_norm_estimate = np.linalg.norm(G_robust)
        
        # Step 6: Bootstrap CI on the block means
        bootstrap_norms = []
        
        for b in range(B):
            # Resample block means with replacement
            bootstrap_indices = rng.choice(actual_K, size=actual_K, replace=True)
            bootstrap_block_means = block_means[bootstrap_indices]
            
            # Compute geometric median of bootstrap sample
            G_bootstrap = self.geometric_median(bootstrap_block_means)
            
            # Take norm
            bootstrap_norm = np.linalg.norm(G_bootstrap)
            bootstrap_norms.append(bootstrap_norm)
        
        bootstrap_norms = np.array(bootstrap_norms)
        
        # Percentile CI (most direct and reliable for bootstrap)
        alpha_total = 1 - self.confidence
        lower_percentile = (alpha_total / 2) * 100
        upper_percentile = (1 - alpha_total / 2) * 100
        
        grad_norm_lower = np.percentile(bootstrap_norms, lower_percentile)
        grad_norm_upper = np.percentile(bootstrap_norms, upper_percentile)
        
        return float(grad_norm_estimate), float(grad_norm_lower), float(grad_norm_upper)
    
    def get_analytical_ground_truth(self, z: np.ndarray, function_type: str, 
                                  function_params: Dict, M: float) -> Tuple[float, float, float]:
        """
        Get analytical ground truth for bounded functions.
        
        Args:
            z: Test point
            function_type: Type of function ("bounded_quadratic", "bounded_slice", etc.)
            function_params: Function parameters
            M: Bound on function output
            
        Returns:
            Tuple of (g_z_true, C_true, G_norm_true)
        """
        if not HAS_TEST_FUNCTIONS:
            raise ImportError("Test functions not available. This method requires test_bounded_certifier_controlled.")
        
        if function_type == "bounded_quadratic":
            center = function_params.get("center", (0.0, 0.0))
            scale = function_params.get("scale", 1.0)
            return bounded_quadratic_smoothed(z, self.sigma, center=center, scale=scale, M=M)
            
        elif function_type == "bounded_slice":
            threshold = function_params.get("threshold", 0.0)
            return bounded_slice_smoothed(z, self.sigma, threshold=threshold, M=M)
            
        elif function_type == "bounded_sine":
            frequency = function_params.get("frequency", 1.0)
            return bounded_sine_smoothed(z, self.sigma, frequency=frequency, M=M)
            
        elif function_type == "bounded_linear":
            return bounded_linear_smoothed(z, self.sigma, M=M)
            
        else:
            raise ValueError(f"Unknown function type: {function_type}")
    
    def compute_theoretical_certified_radius(self, C_true: float, G_norm_true: float, M: float) -> float:
        """
        Compute theoretical certified radius using analytical ground truth.
        
        This uses the same formula as BoundedCertifier's _H_objective method.
        """
        def objective(R):
            """Objective function: harm(R) - eps_y = 0"""
            if R < 0:
                return -self.eps_y
            
            # Compute the harm function for bounded functions (same as BoundedCertifier)
            # FIXED: V_r formula was missing the - R²/σ² term
            V_r = np.exp(R**2 / self.sigma**2) - 1 - R**2 / self.sigma**2
            if V_r <= 0:
                harm = R * G_norm_true
            else:
                # FIXED: Need to subtract σ²||G||² from variance
                harm = np.sqrt(C_true - self.sigma**2 * G_norm_true**2) * np.sqrt(V_r) + R * G_norm_true
            
            return harm - self.eps_y
        
        try:
            # Find the radius where objective(R) = 0
            R_max = 10.0 * self.sigma
            
            # Check if there's a solution
            if objective(R_max) < 0:
                return 0.0
            
            # Find the root
            R_true = brentq(objective, 0.0, R_max, xtol=1e-8, rtol=1e-8)
            return max(0.0, R_true)
            
        except (ValueError, RuntimeError):
            return 0.0
    
    def run_part1_estimator_validation(self, 
                                     z: np.ndarray,
                                     function_type: str,
                                     function_params: Dict,
                                     M: float,
                                     N_values: List[int] = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
                                     n_trials: int = 30,
                                     seed: Optional[int] = 42) -> Dict:
        """
        PART 1: Verify convergence of variance (C) and gradient norm (G) estimators.
        
        This is the "sanity check" that shows our statistical estimators are working correctly.
        """
        if not HAS_TEST_FUNCTIONS:
            raise ImportError("Test functions not available. This method requires test_bounded_certifier_controlled.")
        
        print("="*80)
        print("PART 1: VERIFICATION OF ESTIMATORS (SANITY CHECK)")
        print("="*80)
        print(f"Function: {function_type}")
        print(f"Test point: z={z}")
        print(f"Bound M: {M}")
        print(f"Sample sizes: {N_values}")
        print(f"Trials per size: {n_trials}")
        
        # Define the function
        if function_type == "bounded_quadratic":
            center = function_params.get("center", (0.0, 0.0))
            scale = function_params.get("scale", 1.0)
            def model_fn(x: np.ndarray) -> float:
                return bounded_quadratic(x[0], x[1], center=center, scale=scale, M=M)
        elif function_type == "bounded_slice":
            threshold = function_params.get("threshold", 0.0)
            def model_fn(x: np.ndarray) -> float:
                return bounded_slice_function(x[0], x[1], threshold=threshold, M=M)
        elif function_type == "bounded_sine":
            frequency = function_params.get("frequency", 1.0)
            def model_fn(x: np.ndarray) -> float:
                return bounded_sine_function(x[0], x[1], frequency=frequency, M=M)
        elif function_type == "bounded_linear":
            def model_fn(x: np.ndarray) -> float:
                return bounded_linear_function(x[0], x[1], M=M)
        else:
            raise ValueError(f"Unknown function type: {function_type}")
        
        # Get analytical ground truth
        g_z_true, C_true, G_norm_true = self.get_analytical_ground_truth(z, function_type, function_params, M)
        
        print(f"Ground truth: g_z={g_z_true:.4f}, C={C_true:.4f}, ||G||={G_norm_true:.4f}")
        
        # Run experiments
        results = {
            'function_type': function_type,
            'function_params': function_params,
            'M': M,
            'z': z.tolist(),
            'ground_truth': {'g_z': g_z_true, 'C': C_true, 'G_norm': G_norm_true},
            'N_values': N_values,
            'n_trials': n_trials,
            'results_by_N': {N: [] for N in N_values}
        }
        
        trial_count = 0
        for N in N_values:
            print(f"\nTesting N={N}: ", end="", flush=True)
            for i in range(n_trials):
                trial_seed = seed + trial_count if seed is not None else None
                rng = np.random.default_rng(trial_seed)
                
                # Generate samples
                eta_samples = rng.normal(0.0, self.sigma, size=(N, 2))
                f_values = np.array([model_fn(z + eta) for eta in eta_samples])
                
                # Estimate variance using U-statistic (same as BoundedCertifier)
                # Use the actual U-statistic estimator from BoundedCertifier
                _, _, C_estimate = self.u_statistic_variance_estimator_alpha_half(f_values)
                
                # Estimate gradient norm using U-statistic (same as BoundedCertifier)
                # Use the actual U-statistic estimator from BoundedCertifier
                _, _, G_norm_estimate = self.u_statistic_gradient_norm_estimator_alpha_half(f_values, eta_samples)
                
                # Store results
                trial_result = {
                    'C_estimate': C_estimate,
                    'G_norm_estimate': G_norm_estimate,
                    'C_true': C_true,
                    'G_norm_true': G_norm_true,
                    'N_samples': N,
                    'trial': i
                }
                
                results['results_by_N'][N].append(trial_result)
                trial_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"{i+1} ", end="", flush=True)
            print(f"({n_trials} trials)")
        
        return results
    
    def run_part2_certificate_validation(self, 
                                       z: np.ndarray,
                                       function_type: str,
                                       function_params: Dict,
                                       M: float,
                                       N_values: List[int] = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
                                       n_trials: int = 30,
                                       seed: Optional[int] = 42) -> Dict:
        """
        PART 2: Verify convergence of final certified radii (R).
        
        This is the main result - showing that the final certified radii converge
        to their theoretical values as sample size increases.
        """
        if not HAS_TEST_FUNCTIONS:
            raise ImportError("Test functions not available. This method requires test_bounded_certifier_controlled.")
        
        print("="*80)
        print("PART 2: VERIFICATION OF FINAL CERTIFICATE (MAIN RESULT)")
        print("="*80)
        print(f"Function: {function_type}")
        print(f"Test point: z={z}")
        print(f"Bound M: {M}")
        print(f"Sample sizes: {N_values}")
        print(f"Trials per size: {n_trials}")
        print("NOTE: Empirical certificates use conservative upper bounds, theoretical use exact values.")
        print("Empirical certificates should be ≤ theoretical certificates (more conservative).")
        
        # Define the function
        if function_type == "bounded_quadratic":
            center = function_params.get("center", (0.0, 0.0))
            scale = function_params.get("scale", 1.0)
            def model_fn(x: np.ndarray) -> float:
                return bounded_quadratic(x[0], x[1], center=center, scale=scale, M=M)
        elif function_type == "bounded_slice":
            threshold = function_params.get("threshold", 0.0)
            def model_fn(x: np.ndarray) -> float:
                return bounded_slice_function(x[0], x[1], threshold=threshold, M=M)
        elif function_type == "bounded_sine":
            frequency = function_params.get("frequency", 1.0)
            def model_fn(x: np.ndarray) -> float:
                return bounded_sine_function(x[0], x[1], frequency=frequency, M=M)
        elif function_type == "bounded_linear":
            def model_fn(x: np.ndarray) -> float:
                return bounded_linear_function(x[0], x[1], M=M)
        else:
            raise ValueError(f"Unknown function type: {function_type}")
        
        # Get analytical ground truth
        g_z_true, C_true, G_norm_true = self.get_analytical_ground_truth(z, function_type, function_params, M)
        
        print(f"Ground truth: g_z={g_z_true:.4f}, C={C_true:.4f}, ||G||={G_norm_true:.4f}")
        
        # Compute theoretical certified radius
        r_theoretical = self.compute_theoretical_certified_radius(C_true, G_norm_true, M)
        print(f"Theoretical certified radius: {r_theoretical:.4f}")
        
        # Run experiments
        results = {
            'function_type': function_type,
            'function_params': function_params,
            'M': M,
            'z': z.tolist(),
            'ground_truth': {'g_z': g_z_true, 'C': C_true, 'G_norm': G_norm_true},
            'theoretical_radius': r_theoretical,
            'N_values': N_values,
            'n_trials': n_trials,
            'results_by_N': {N: [] for N in N_values}
        }
        
        trial_count = 0
        for N in N_values:
            print(f"\nTesting N={N}: ", end="", flush=True)
            for i in range(n_trials):
                trial_seed = seed + trial_count if seed is not None else None
                
                # Get empirical certificate from BoundedCertifier

                # Initialize certifier with specific M value for this function
                bounded_certifier = BoundedCertifier(
                    sigma=self.sigma, M=M, eps_y=self.eps_y, confidence=self.confidence
                )
                r_empirical = bounded_certifier.certify_point(
                    z, model_fn, N_samples_stats=N, N_samples_mc=min(N//2, 5000), seed=trial_seed
                )
                
                # Store results
                trial_result = {
                    'r_empirical': r_empirical,
                    'r_theoretical': r_theoretical,
                    'C_true': C_true,
                    'G_norm_true': G_norm_true,
                    'N_samples': N,
                    'trial': i
                }
                
                results['results_by_N'][N].append(trial_result)
                trial_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"{i+1} ", end="", flush=True)
            print(f"({n_trials} trials)")
        
        return results
    
    def plot_part1_results(self, results: Dict, save_prefix: str = "bounded_part1_estimators"):
        """
        Plot Part 1 results: convergence of variance and gradient norm estimators.
        
        FIXED: Now uses 95% CI of the mean (SEM-based) instead of standard deviation.
        This shows uncertainty of the mean estimate, not spread across trials.
        """
        N_values = results['N_values']
        C_true = results['ground_truth']['C']
        G_norm_true = results['ground_truth']['G_norm']
        n_trials = results['n_trials']
        
        # Prepare data
        C_estimates = {N: [] for N in N_values}
        G_norm_estimates = {N: [] for N in N_values}
        
        for N in N_values:
            for trial in results['results_by_N'][N]:
                C_estimates[N].append(trial['C_estimate'])
                G_norm_estimates[N].append(trial['G_norm_estimate'])
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Variance estimates
        ax1 = axes[0]
        C_means = [np.mean(C_estimates[N]) for N in N_values]
        # FIXED: Use 95% CI for the mean (z * SEM) instead of SD
        z = 1.96  # 95% CI
        C_yerr = [z * np.std(C_estimates[N], ddof=1) / np.sqrt(n_trials) for N in N_values]
        
        ax1.errorbar(N_values, C_means, yerr=C_yerr, 
                    marker='o', capsize=5, label='Empirical Estimate (95% CI)', linewidth=2)
        ax1.axhline(y=C_true, color='r', linestyle='--', 
                   label=f'True Value (C={C_true:.4f})', linewidth=2)
        
        ax1.set_xlabel('Number of Samples (N)')
        ax1.set_ylabel('Variance Estimate (C)')
        function_type = results.get('function_type', 'unknown')
        ax1.set_title(f'Variance Estimator Convergence\n(Part 1: Sanity Check - {function_type.title()} Function)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot 2: Gradient norm estimates
        ax2 = axes[1]
        G_means = [np.mean(G_norm_estimates[N]) for N in N_values]
        # FIXED: Use 95% CI for the mean (z * SEM) instead of SD
        G_yerr = [z * np.std(G_norm_estimates[N], ddof=1) / np.sqrt(n_trials) for N in N_values]
        
        ax2.errorbar(N_values, G_means, yerr=G_yerr, 
                    marker='o', capsize=5, label='Empirical Estimate (95% CI)', linewidth=2)
        ax2.axhline(y=G_norm_true, color='r', linestyle='--', 
                   label=f'True Value (||G||={G_norm_true:.4f})', linewidth=2)
        
        ax2.set_xlabel('Number of Samples (N)')
        ax2.set_ylabel('Gradient Norm Estimate (||G||)')
        ax2.set_title(f'Gradient Norm Estimator Convergence\n(Part 1: Sanity Check - {function_type.title()} Function)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        
        # Save plot
        save_path = f"{save_prefix}_{function_type}_convergence.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Part 1 plot: {save_path}")
        
        plt.show()
    
    def plot_part2_results(self, results: Dict, save_prefix: str = "bounded_part2_certificates"):
        """
        Plot Part 2 results: convergence of final certified radii.
        
        FIXED: Now uses 95% CI of the mean (SEM-based) instead of standard deviation.
        This shows uncertainty of the mean estimate, not spread across trials.
        """
        N_values = results['N_values']
        n_trials = results['n_trials']
        
        # Prepare data
        empirical_radii = {N: [] for N in N_values}
        theoretical_radius = results['theoretical_radius']
        
        for N in N_values:
            for trial in results['results_by_N'][N]:
                empirical_radii[N].append(trial['r_empirical'])
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        empirical_means = [np.mean(empirical_radii[N]) for N in N_values]
        # FIXED: Use 95% CI for the mean (z * SEM) instead of SD
        z = 1.96  # 95% CI
        empirical_yerr = [z * np.std(empirical_radii[N], ddof=1) / np.sqrt(n_trials) for N in N_values]
        
        ax.errorbar(N_values, empirical_means, yerr=empirical_yerr, 
                   marker='o', capsize=5, label='Empirical (BoundedCertifier, 95% CI)', linewidth=2)
        ax.axhline(y=theoretical_radius, color='r', linestyle='--', 
                  label=f'Theoretical (R={theoretical_radius:.4f})', linewidth=2)
        
        ax.set_xlabel('Number of Samples (N)')
        ax.set_ylabel('Certified Radius')
        function_type = results.get('function_type', 'unknown')
        M = results.get('M', 'unknown')
        ax.set_title(f'BoundedCertifier Convergence\n(Part 2: Main Result - {function_type.title()} Function, M={M})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        
        # Save plot
        save_path = f"{save_prefix}_{function_type}_convergence.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Part 2 plot: {save_path}")
        
        plt.show()
    
    def analyze_results(self, part1_results: Dict, part2_results: Dict):
        """
        Analyze and print results from both parts.
        """
        print("\n" + "="*80)
        print("BOUNDEDCERTIFIER CONVERGENCE ANALYSIS")
        print("="*80)
        
        # Part 1 Analysis
        print("\nPART 1: ESTIMATOR CONVERGENCE (SANITY CHECK)")
        print("-" * 50)
        
        N_values = part1_results['N_values']
        C_true = part1_results['ground_truth']['C']
        G_norm_true = part1_results['ground_truth']['G_norm']
        
        for N in N_values:
            C_estimates = [trial['C_estimate'] for trial in part1_results['results_by_N'][N]]
            G_estimates = [trial['G_norm_estimate'] for trial in part1_results['results_by_N'][N]]
            
            C_mean = np.mean(C_estimates)
            C_std = np.std(C_estimates)
            G_mean = np.mean(G_estimates)
            G_std = np.std(G_estimates)
            
            C_bias = abs(C_mean - C_true) / C_true if C_true > 0 else 0
            G_bias = abs(G_mean - G_norm_true) / G_norm_true if G_norm_true > 0 else 0
            
            print(f"N={N:5d}: C={C_mean:.4f}±{C_std:.4f} (bias={C_bias:.1%}), "
                  f"||G||={G_mean:.4f}±{G_std:.4f} (bias={G_bias:.1%})")
        
        # Part 2 Analysis
        print("\nPART 2: CERTIFICATE CONVERGENCE (MAIN RESULT)")
        print("-" * 50)
        
        theoretical_radius = part2_results['theoretical_radius']
        print(f"Theoretical certified radius: {theoretical_radius:.4f}")
        
        for N in N_values:
            empirical_radii = [trial['r_empirical'] for trial in part2_results['results_by_N'][N]]
            
            empirical_mean = np.mean(empirical_radii)
            empirical_std = np.std(empirical_radii)
            
            bias = empirical_mean - theoretical_radius
            relative_bias = abs(bias) / theoretical_radius if theoretical_radius > 0 else 0
            cv = empirical_std / empirical_mean if empirical_mean > 0 else 0
            
            print(f"N={N:5d}: Empirical={empirical_mean:.4f}±{empirical_std:.4f}, "
                  f"Theoretical={theoretical_radius:.4f}, "
                  f"Bias={relative_bias:.1%}, CV={cv:.3f}")
        
        # Sufficiency analysis
        print(f"\n{'='*80}")
        print("SAMPLE SIZE SUFFICIENCY ANALYSIS")
        print("="*80)
        
        N_target = 10000
        if N_target in N_values:
            # Part 1 sufficiency
            C_estimates = [trial['C_estimate'] for trial in part1_results['results_by_N'][N_target]]
            G_estimates = [trial['G_norm_estimate'] for trial in part1_results['results_by_N'][N_target]]
            
            C_bias = abs(np.mean(C_estimates) - C_true) / C_true if C_true > 0 else 0
            G_bias = abs(np.mean(G_estimates) - G_norm_true) / G_norm_true if G_norm_true > 0 else 0
            
            print(f"PART 1 - Estimators at N={N_target}:")
            print(f"  Variance bias: {C_bias:.1%} ({'✓' if C_bias < 0.05 else '✗'} < 5%)")
            print(f"  Gradient bias: {G_bias:.1%} ({'✓' if G_bias < 0.05 else '✗'} < 5%)")
            
            # Part 2 sufficiency
            empirical_radii = [trial['r_empirical'] for trial in part2_results['results_by_N'][N_target]]
            
            empirical_mean = np.mean(empirical_radii)
            empirical_std = np.std(empirical_radii)
            
            bias = abs(empirical_mean - theoretical_radius) / theoretical_radius if theoretical_radius > 0 else 0
            cv = empirical_std / empirical_mean if empirical_mean > 0 else 0
            
            bias_acceptable = bias < 0.05
            variance_acceptable = cv < 0.1
            
            print(f"PART 2 - BoundedCertifier at N={N_target}:")
            print(f"  Bias: {bias:.1%} ({'✓' if bias_acceptable else '✗'} < 5%)")
            print(f"  CV: {cv:.3f} ({'✓' if variance_acceptable else '✗'} < 0.1)")
            print(f"  Overall: {'SUFFICIENT' if bias_acceptable and variance_acceptable else 'INSUFFICIENT'}")
        
        print("="*80)


    # =============================================================================
    # NEW: COVERAGE VALIDATION METHODS (Preserves backwards compatibility)
    # =============================================================================
    
    @staticmethod
    def compute_theta_ustatistic(W_samples: np.ndarray) -> float:
        """
        Compute UNBIASED U-statistic for θ = ||G||².
        
        This is the unbiased estimator for the squared norm of the gradient.
        Unlike the plug-in estimator ||mean(W)||², this U-statistic has no bias.
        
        Formula: θ̂ = (1/C(n,2)) * Σ_{i<j} W_i^T W_j
        
        Efficient computation: θ̂ = [||Σ W_i||² - Σ ||W_i||²] / (n(n-1))
        
        Args:
            W_samples: (N, d) array of W vectors
            
        Returns:
            theta_hat: Unbiased estimate of θ = ||G||²
        """
        n = len(W_samples)
        if n < 2:
            return 0.0
        
        sum_W = np.sum(W_samples, axis=0)
        sum_W_sq_norm = np.dot(sum_W, sum_W)
        sum_sq_norm_W = np.sum(np.linalg.norm(W_samples, axis=1) ** 2)
        
        # U-statistic formula
        if n * (n - 1) > 0:
            theta_hat = (sum_W_sq_norm - sum_sq_norm_W) / (n * (n - 1))
        else:
            theta_hat = 0.0
        
        return max(0.0, theta_hat)
    
    def compute_theta_ci_with_z_critical(self, f_values: np.ndarray, eta_samples: np.ndarray, 
                                         confidence: float = 0.95) -> tuple:
        """
        Compute CI for θ = ||G||² using U-statistic with z-critical values.
        
        This uses ASYMPTOTIC normality assumption (z-critical), suitable when:
        - Sample size is large enough for CLT to apply
        - Theory is based on asymptotic analysis
        
        Uses union bound split: allocates half of the failure probability to θ (G),
        consistent with other estimators that split between C and G.
        
        Args:
            f_values: Function values at noisy points
            eta_samples: Noise samples
            confidence: Total confidence level (default 0.95). Half is allocated to θ via union bound.
            
        Returns:
            theta_hat: Point estimate (U-statistic)
            theta_lower: Lower CI bound
            theta_upper: Upper CI bound
        """
        from scipy.stats import norm
        
        n = len(f_values)
        if n < 2:
            return 0.0, 0.0, 0.0
        
        # Construct W_i = (1/σ²) * ε_i * f(z + ε_i)
        W_samples = (1 / self.sigma**2) * eta_samples * f_values[:, np.newaxis]
        
        # Step 1: U-statistic for θ = ||G||²
        theta_hat = self.compute_theta_ustatistic(W_samples)
        
        # Step 2: Asymptotic variance for θ̂
        mu_hat = np.mean(W_samples, axis=0)
        
        if n > 1:
            Sigma_hat = np.cov(W_samples.T)
        else:
            Sigma_hat = np.zeros((W_samples.shape[1], W_samples.shape[1]))
        
        # Var_asym(θ̂) = 4μᵀΣμ
        asymptotic_var_theta = max(0.0, 4.0 * (mu_hat @ Sigma_hat @ mu_hat))
        var_theta_hat = asymptotic_var_theta / n
        
        # Step 3: CI using z-critical (asymptotic normality)
        # Use α/2 for union bound (consistent with other methods)
        # This allocates half the failure probability to θ (G), half to C
        alpha_total = 1 - confidence
        alpha_split = alpha_total / 2.0
        z_critical = norm.ppf(1 - alpha_split)  # One-sided UCB for certification
        
        se_theta = np.sqrt(var_theta_hat)
        theta_lower = max(0.0, theta_hat - z_critical * se_theta)
        theta_upper = theta_hat + z_critical * se_theta
        
        return float(theta_hat), float(theta_lower), float(theta_upper)
    
    @staticmethod
    def mc_truth_for_bounded_f(f_bounded, z, sigma, n_big=5_000_000, rng=None):
        """
        Numerical ground truth using the SAME bounded function with large MC samples.
        
        CRITICAL FIX: Returns THREE values to avoid Jensen's inequality bias:
        1. C_true: Variance (plug-in, unbiased with large N)
        2. theta_true: Squared gradient norm θ = ||G||² (U-statistic, for Part 1 validation)
        3. G_norm_true: Gradient norm ||G|| (plug-in, for Part 2 radius calculation)
        
        Why THREE values?
        - Part 1 needs θ_true (U-statistic) to validate θ̂ (U-statistic)
        - Part 2 needs ||G||_true (plug-in) for radius calculation
        - √θ_true ≠ ||G||_true due to Jensen's inequality (√θ̂ is biased LOW)
        
        Args:
            f_bounded: Bounded function (expects array input)
            z: Test point (numpy array)
            sigma: Noise standard deviation
            n_big: Number of Monte Carlo samples (default 5M for high precision)
            rng: Random number generator
            
        Returns:
            C_true_mc: Population variance (plug-in, unbiased with large N)
            theta_true_mc: Squared gradient norm θ = ||G||² (U-statistic, unbiased)
            G_norm_true_mc: Gradient norm ||G|| (plug-in, unbiased with large N)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Generate noise samples
        e = rng.normal(0.0, sigma, size=(n_big, z.size))
        
        # Evaluate bounded function at all noisy points
        vals = np.array([f_bounded(z + e_i) for e_i in e])
        
        # 1. Variance: plug-in (essentially unbiased with large N)
        C_true_mc = float(np.var(vals, ddof=0))
        
        # Compute W vectors using Stein/score identity for Gaussian noise
        W = (e * vals[:, None]) / (sigma**2)
        mu_W = np.mean(W, axis=0)
        
        # 2. Gradient norm ||G||: plug-in (for Part 2 radius calculation)
        # With N=100K, plug-in ||mean(W)|| is essentially unbiased for ||G||
        G_norm_true_mc = float(np.linalg.norm(mu_W))
        
        # 3. Squared gradient norm θ = ||G||²: U-statistic (for Part 1 validation)
        # U-statistic is unbiased for θ, unlike ||mean(W)||² which is biased HIGH
        theta_true_mc = BoundedCertifierConvergenceValidator.compute_theta_ustatistic(W)
        
        return C_true_mc, theta_true_mc, G_norm_true_mc
    
    def run_part1_coverage_validation(self,
                                     z: np.ndarray,
                                     function_type: str,
                                     function_params: Dict,
                                     M: float,
                                     N_values: List[int],
                                     n_trials: int = 30,
                                     seed: Optional[int] = 42) -> Dict:
        """
        NEW METHOD: Run Part 1 with coverage validation.
        
        Stores point estimates + CI bounds to enable coverage analysis.
        Coverage = fraction of trials where per-trial CI contains truth.
        """
        if not HAS_TEST_FUNCTIONS:
            raise ImportError("Test functions not available. This method requires test_bounded_certifier_controlled.")
        
        print("="*80)
        print("PART 1: COVERAGE VALIDATION (validates CI construction)")
        print("="*80)
        print(f"Function: {function_type}")
        print(f"Test point: z={z}")
        print(f"Bound M: {M}")
        print(f"Sample sizes: {N_values}")
        print(f"Trials per size: {n_trials}")
        
        # Define the bounded function
        if function_type == "bounded_quadratic":
            center = function_params.get("center", (0.0, 0.0))
            scale = function_params.get("scale", 1.0)
            def model_fn(x: np.ndarray) -> float:
                return bounded_quadratic(x[0], x[1], center=center, scale=scale, M=M)
        elif function_type == "bounded_slice":
            threshold = function_params.get("threshold", 0.0)
            def model_fn(x: np.ndarray) -> float:
                return bounded_slice_function(x[0], x[1], threshold=threshold, M=M)
        elif function_type == "bounded_sine":
            frequency = function_params.get("frequency", 1.0)
            def model_fn(x: np.ndarray) -> float:
                return bounded_sine_function(x[0], x[1], frequency=frequency, M=M)
        elif function_type == "bounded_linear":
            def model_fn(x: np.ndarray) -> float:
                return bounded_linear_function(x[0], x[1], M=M)
        else:
            raise ValueError(f"Unknown function type: {function_type}")
        
        # Compute MC ground truth using the SAME bounded function
        print("\nComputing MC ground truth using the SAME bounded function...")
        C_true, G_norm_true = self.mc_truth_for_bounded_f(
            model_fn, z, self.sigma, n_big=5_000_000, rng=np.random.default_rng(42)
        )
        print(f"MC ground truth: C={C_true:.6f}, ||G||={G_norm_true:.6f}")
        
        # Run experiments
        results = {
            'function_type': function_type,
            'function_params': function_params,
            'M': M,
            'z': z.tolist(),
            'ground_truth': {'C': C_true, 'G_norm': G_norm_true},
            'N_values': N_values,
            'n_trials': n_trials,
            'results_by_N': {N: [] for N in N_values}
        }
        
        trial_count = 0
        for N in N_values:
            print(f"\nTesting N={N}: ", end="", flush=True)
            for i in range(n_trials):
                trial_seed = seed + trial_count if seed is not None else None
                rng = np.random.default_rng(trial_seed)
                
                # Generate samples
                eta_samples = rng.normal(0.0, self.sigma, size=(N, 2))
                f_values = np.array([model_fn(z + eta) for eta in eta_samples])
                
                # Get point estimates + CI bounds
                C_hat, C_lower, C_upper = self.u_statistic_variance_estimator_alpha_half(f_values)
                G_hat, G_lower, G_upper = self.u_statistic_gradient_norm_estimator_alpha_half(f_values, eta_samples)
                
                # Store ALL values for coverage analysis
                trial_result = {
                    # Point estimates
                    'C_hat': C_hat,
                    'G_hat': G_hat,
                    # CI bounds
                    'C_lower': C_lower,
                    'C_upper': C_upper,
                    'G_lower': G_lower,
                    'G_upper': G_upper,
                    # Ground truth
                    'C_true': C_true,
                    'G_norm_true': G_norm_true,
                    # Metadata
                    'N_samples': N,
                    'trial': i
                }
                
                results['results_by_N'][N].append(trial_result)
                trial_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"{i+1} ", end="", flush=True)
            print(f"({n_trials} trials)")
        
        return results
    
    def plot_coverage_results(self, results: Dict, save_prefix: str = "bounded_coverage"):
        """
        NEW METHOD: Plot convergence WITH coverage validation.
        
        Creates 2x2 plot:
        - Top row: Convergence plots (C and G) with CI of mean
        - Bottom row: Coverage plots (should approach nominal level)
        """
        N_values = results['N_values']
        C_true = results['ground_truth']['C']
        G_norm_true = results['ground_truth']['G_norm']
        n_trials = results['n_trials']
        
        # Compute coverage for each N
        C_coverage = []
        G_coverage = []
        
        # Compute means and CIs for convergence plots
        C_means = []
        C_yerr = []
        G_means = []
        G_yerr = []
        
        z = 1.96  # 95% CI
        
        for N in N_values:
            trials = results['results_by_N'][N]
            
            # Extract data
            C_hats = [t['C_hat'] for t in trials]
            C_lowers = [t['C_lower'] for t in trials]
            C_uppers = [t['C_upper'] for t in trials]
            G_hats = [t['G_hat'] for t in trials]
            G_lowers = [t['G_lower'] for t in trials]
            G_uppers = [t['G_upper'] for t in trials]
            
            # Coverage: fraction of trials where CI contains truth
            C_cov = np.mean([(C_lowers[i] <= C_true <= C_uppers[i]) for i in range(len(trials))])
            G_cov = np.mean([(G_lowers[i] <= G_norm_true <= G_uppers[i]) for i in range(len(trials))])
            
            C_coverage.append(C_cov)
            G_coverage.append(G_cov)
            
            # Means and CIs for convergence plots
            C_means.append(np.mean(C_hats))
            C_yerr.append(z * np.std(C_hats, ddof=1) / np.sqrt(n_trials))
            G_means.append(np.mean(G_hats))
            G_yerr.append(z * np.std(G_hats, ddof=1) / np.sqrt(n_trials))
        
        # Create 2x2 plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        function_type = results.get('function_type', 'unknown')
        M = results.get('M', 'unknown')
        fig.suptitle(f'Coverage Validation: {function_type.title()} (M={M})', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Variance convergence
        ax1 = axes[0, 0]
        ax1.errorbar(N_values, C_means, yerr=C_yerr,
                    marker='o', capsize=5, label='Point Estimate (95% CI of mean)', linewidth=2)
        ax1.axhline(y=C_true, color='r', linestyle='--',
                   label=f'MC Truth (C={C_true:.6f})', linewidth=2)
        ax1.set_xlabel('Number of Samples (N)', fontsize=12)
        ax1.set_ylabel('Variance Estimate (C)', fontsize=12)
        ax1.set_title('Variance Estimator Convergence', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot 2: Gradient convergence
        ax2 = axes[0, 1]
        ax2.errorbar(N_values, G_means, yerr=G_yerr,
                    marker='o', capsize=5, label='Point Estimate (95% CI of mean)', linewidth=2)
        ax2.axhline(y=G_norm_true, color='r', linestyle='--',
                   label=f'MC Truth (||G||={G_norm_true:.6f})', linewidth=2)
        ax2.set_xlabel('Number of Samples (N)', fontsize=12)
        ax2.set_ylabel('Gradient Norm Estimate (||G||)', fontsize=12)
        ax2.set_title('Gradient Norm Estimator Convergence', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # Plot 3: Variance COVERAGE
        ax3 = axes[1, 0]
        ax3.plot(N_values, C_coverage, marker='s', markersize=10, linewidth=2.5,
                label=f'Empirical Coverage', color='darkblue')
        ax3.axhline(y=self.confidence, color='r', linestyle='--', linewidth=2,
                   label=f'Nominal Level ({self.confidence:.1%})')
        ax3.set_xlabel('Number of Samples (N)', fontsize=12)
        ax3.set_ylabel('Coverage (Fraction of CIs containing truth)', fontsize=12)
        ax3.set_title('Variance CI Coverage Validation', fontsize=14, fontweight='bold')
        ax3.set_ylim([0.5, 1.0])
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # Plot 4: Gradient COVERAGE
        ax4 = axes[1, 1]
        ax4.plot(N_values, G_coverage, marker='s', markersize=10, linewidth=2.5,
                label=f'Empirical Coverage', color='darkgreen')
        ax4.axhline(y=self.confidence, color='r', linestyle='--', linewidth=2,
                   label=f'Nominal Level ({self.confidence:.1%})')
        ax4.set_xlabel('Number of Samples (N)', fontsize=12)
        ax4.set_ylabel('Coverage (Fraction of CIs containing truth)', fontsize=12)
        ax4.set_title('Gradient CI Coverage Validation', fontsize=14, fontweight='bold')
        ax4.set_ylim([0.5, 1.0])
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        
        plt.tight_layout()
        
        # Save plot
        save_path = f"{save_prefix}_{function_type}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved coverage plot: {save_path}")
        
        plt.show()
        
        # Print coverage analysis
        print("\n" + "="*80)
        print("COVERAGE ANALYSIS")
        print("="*80)
        print(f"Nominal Coverage: {self.confidence:.1%}")
        print("\n" + "-"*80)
        print(f"{'N':<8} {'C_coverage':<12} {'G_coverage':<12}")
        print("-"*80)
        for i, N in enumerate(N_values):
            print(f"{N:<8} {C_coverage[i]:<12.1%} {G_coverage[i]:<12.1%}")
        print("="*80)


def main():
    """
    Main function to run the BoundedCertifier convergence analysis.
    """
    if not HAS_TEST_FUNCTIONS:
        print("ERROR: test_bounded_certifier_controlled.py not found.")
        print("This script requires test functions for convergence analysis.")
        print("The U-statistic estimators can still be used by importing BoundedCertifierConvergenceValidator.")
        return
    
    import argparse
    
    parser = argparse.ArgumentParser(description="BoundedCertifier convergence analysis")
    parser.add_argument("--function", 
                       choices=["bounded_quadratic", "bounded_slice", "bounded_sine", "bounded_linear", "all"], 
                       default="all", help="Function type to use (or 'all' for all functions)")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise standard deviation")
    parser.add_argument("--eps_y", type=float, default=1.0, help="Output tolerance")
    parser.add_argument("--M", type=float, default=None, help="Bound on function output (auto-selected if not provided)")
    parser.add_argument("--N_values", nargs="+", type=int,
                       default=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
                       help="Sample sizes to test")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of trials per sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test_point", nargs=2, type=float, default=[1.0, 1.0], 
                       help="Test point coordinates (x1, x2)")
    args = parser.parse_args()
    
    # Determine which functions to run
    if args.function == "all":
        functions_to_run = ["bounded_quadratic", "bounded_slice", "bounded_sine", "bounded_linear"]
    else:
        functions_to_run = [args.function]
    
    print("="*80)
    print("BOUNDEDCERTIFIER CONVERGENCE ANALYSIS EXPERIMENTS")
    print("="*80)
    print(f"Functions: {functions_to_run}")
    print(f"Sample sizes: {args.N_values}")
    print(f"Trials per size: {args.n_trials}")
    print(f"Parameters: σ={args.sigma}, ε_y={args.eps_y}")
    print(f"Test point: z={args.test_point}")
    print("="*80)
    
    all_results = {}
    
    # Run validations for each function
    for function_type in functions_to_run:
        print(f"\n{'='*80}")
        print(f"RUNNING CONVERGENCE ANALYSIS FOR {function_type.upper()} FUNCTION")
        print("="*80)
        
        # Initialize validator
        validator = BoundedCertifierConvergenceValidator(sigma=args.sigma, eps_y=args.eps_y)
        
        # Set up function parameters and bounds
        if function_type == "bounded_quadratic":
            function_params = {"center": (0.0, 0.0), "scale": 1.0}
            M = args.M if args.M is not None else 10.0
        elif function_type == "bounded_slice":
            function_params = {"threshold": 0.0}
            M = args.M if args.M is not None else 5.0
        elif function_type == "bounded_sine":
            function_params = {"frequency": 1.0}
            M = args.M if args.M is not None else 180.0
        elif function_type == "bounded_linear":
            function_params = {}
            M = args.M if args.M is not None else 10.0
        
        test_point = np.array(args.test_point)
        
        print(f"Function: {function_type}")
        print(f"Function parameters: {function_params}")
        print(f"Bound M: {M}")
        print(f"Test point: {test_point}")
        print(f"Sample sizes: {args.N_values}")
        print(f"Trials per size: {args.n_trials}")
        
        # Run Part 1: Estimator validation
        part1_results = validator.run_part1_estimator_validation(
            z=test_point,
            function_type=function_type,
            function_params=function_params,
            M=M,
            N_values=args.N_values,
            n_trials=args.n_trials,
            seed=args.seed
        )
        
        # Run Part 2: Certificate validation
        part2_results = validator.run_part2_certificate_validation(
            z=test_point,
            function_type=function_type,
            function_params=function_params,
            M=M,
            N_values=args.N_values,
            n_trials=args.n_trials,
            seed=args.seed
        )
        
        # Save results
        part1_file = f"bounded_{function_type}_part1_estimators.json"
        part2_file = f"bounded_{function_type}_part2_certificates.json"
        
        with open(part1_file, 'w') as f:
            json.dump(part1_results, f, indent=2)
        with open(part2_file, 'w') as f:
            json.dump(part2_results, f, indent=2)
        
        print(f"Saved Part 1 results: {part1_file}")
        print(f"Saved Part 2 results: {part2_file}")
        
        # Plot results
        validator.plot_part1_results(part1_results, save_prefix=f"bounded_{function_type}_part1")
        validator.plot_part2_results(part2_results, save_prefix=f"bounded_{function_type}_part2")
        
        # Analyze results
        validator.analyze_results(part1_results, part2_results)
        
        all_results[function_type] = {
            'part1': part1_results,
            'part2': part2_results
        }
    
    # Summary across all functions
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS SUMMARY ACROSS ALL FUNCTIONS")
    print("="*80)
    
    for func_name, results in all_results.items():
        print(f"\n{func_name.upper()} FUNCTION:")
        
        # Part 1 summary
        part1 = results['part1']
        N_target = 10000
        if N_target in part1['N_values']:
            C_estimates = [trial['C_estimate'] for trial in part1['results_by_N'][N_target]]
            G_estimates = [trial['G_norm_estimate'] for trial in part1['results_by_N'][N_target]]
            
            C_bias = abs(np.mean(C_estimates) - part1['ground_truth']['C']) / part1['ground_truth']['C']
            G_bias = abs(np.mean(G_estimates) - part1['ground_truth']['G_norm']) / part1['ground_truth']['G_norm']
            
            print(f"  Part 1 (Estimators): C bias={C_bias:.1%}, ||G|| bias={G_bias:.1%}")
        
        # Part 2 summary
        part2 = results['part2']
        if N_target in part2['N_values']:
            empirical_radii = [trial['r_empirical'] for trial in part2['results_by_N'][N_target]]
            theoretical_radius = part2['theoretical_radius']
            
            empirical_mean = np.mean(empirical_radii)
            bias = abs(empirical_mean - theoretical_radius) / theoretical_radius if theoretical_radius > 0 else 0
            
            print(f"  Part 2 (BoundedCertifier): Bias={bias:.1%}")
    
    print("\n" + "="*80)
    print("All BoundedCertifier convergence analyses completed!")
    print("="*80)


if __name__ == "__main__":
    main()
