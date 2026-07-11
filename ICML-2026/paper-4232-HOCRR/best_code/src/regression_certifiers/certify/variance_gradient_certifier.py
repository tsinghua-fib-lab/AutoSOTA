"""
Variance + Gradient Certificate - Uses your exact theoretical formula
This implements Certificate B using U-statistics and your exact formula.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Callable, Optional, Tuple
from .base import BaseCertifier


class VarianceGradientCertifier(BaseCertifier):
    """
    Variance + gradient certificate using your exact formula.
    """
    
    def __init__(self, *, sigma: float, eps_y: float, confidence: float = 0.999):
        super().__init__(sigma=sigma)
        self.eps_y = eps_y
        self.confidence = confidence
        self.name = "Variance + Gradient Certificate"

    def u_statistic_variance_estimator_alpha_half(self, samples: np.ndarray) -> tuple:
        """U-statistic variance estimator with α/2 confidence interval for union bound."""
        n = len(samples)
        
        # Edge case: need at least 2 samples for variance
        if n < 2:
            return 0.0, 0.0, 0.0
        
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
        z_critical = norm.ppf(1 - alpha_split)  # One-sided UCB for certification
        se = np.sqrt(asymptotic_var / n)
        
        lower_bound = theta_hat - z_critical * se
        upper_bound = theta_hat + z_critical * se
        
        return theta_hat, lower_bound, upper_bound
    
    # CHANGED: This function now accepts samples directly for efficiency and correctness.
    def u_statistic_gradient_norm_estimator_alpha_half(
        self, 
        f_values: np.ndarray, 
        eta_samples: np.ndarray
    ) -> tuple:
        """U-statistic gradient norm estimator with α/2 confidence interval for union bound."""
        n = len(f_values)
        
        # Edge case: need at least 2 samples
        if n < 2:
            return 0.0, 0.0, 0.0
        
        # Construct W_i = (1/σ²) * ε_i * f(z + ε_i) from pre-computed samples
        W_samples = (1 / self.sigma**2) * eta_samples * f_values[:, np.newaxis]

        # --- Corrected U-statistic for ||G||^2 ---
        sum_W = np.sum(W_samples, axis=0)
        sum_W_sq_norm = np.dot(sum_W, sum_W)
        sum_sq_norm_W = np.sum(np.linalg.norm(W_samples, axis=1) ** 2)
        off_diagonal_sum = 0.5 * (sum_W_sq_norm - sum_sq_norm_W)
        num_pairs = n * (n - 1) / 2
        theta_hat_sq = off_diagonal_sum / num_pairs if num_pairs > 0 else 0.0
        # ----------------------------------------

        # Asymptotic variance for sqrt(n)( ||G_hat||^2 - ||G||^2 ) uses mu_hat and Sigma_hat
        mu_hat = np.mean(W_samples, axis=0)
        centered_W = W_samples - mu_hat
        Sigma_hat = np.cov(centered_W, rowvar=False, ddof=1)
        asymptotic_var = max(0.0, 4 * np.dot(mu_hat, np.dot(Sigma_hat, mu_hat)))
        
        # Use α/2 for union bound
        alpha_total = 1 - self.confidence
        alpha_split = alpha_total / 2.0
        z_critical = norm.ppf(1 - alpha_split)  # One-sided UCB for certification
        se = np.sqrt(asymptotic_var / n)
        
        # Confidence interval for ||∇g(z)||²
        theta_lower_sq = theta_hat_sq - z_critical * se
        theta_upper_sq = theta_hat_sq + z_critical * se
        
        # Convert to confidence interval for ||∇g(z)|| (norm)
        grad_norm_lower = np.sqrt(max(0, theta_lower_sq))
        grad_norm_upper = np.sqrt(max(0, theta_upper_sq))
        grad_norm_estimate = np.sqrt(max(0, theta_hat_sq))
        
        return grad_norm_estimate, grad_norm_lower, grad_norm_upper

    # In class VarianceGradientCertifier:

    def variance_gradient_certificate(self, C_ucb: float, G_ucb: float, eps_y: float) -> float:
        """
        Computes the certificate by finding the tightest upper bound on the harm function.
        This is more robust to noise than the simple check.
        """
        
        def get_max_harm_at_r(r: float) -> float:
            """
            For a given radius r, find the maximum possible harm by optimizing over g.
            maximize: sqrt(C_ucb - σ²g²) * sqrt(Vr) + r*g
            subject to: 0 <= g <= min(G_ucb, sqrt(C_ucb)/σ)
            """
            if r < 0: return -float('inf')
            
            # Pre-compute Vr term
            V_r_arg = r**2 / self.sigma**2
            V_r = np.exp(V_r_arg) - 1 - V_r_arg
            if V_r <= 0:
                # If Vr is zero or negative, the harm function is just r*g, maximized at g_max
                return r * G_ucb

            # Define the valid range for g
            g_max = min(G_ucb, np.sqrt(C_ucb) / self.sigma if C_ucb > 0 else 0.0)

            # The function is f(g) = a*sqrt(C - b*g^2) + c*g.
            # Its critical point g* (from df/dg = 0) is derived from:
            # d(harm)/dg = -σ²*g*sqrt(Vr) / sqrt(C_ucb - σ²g²) + r = 0
            # => g* = (r * sqrt(C_ucb)) / sqrt(σ^4 * Vr + σ² * r^2)
            
            # Calculate the unconstrained optimizer g_star
            numerator = r * np.sqrt(C_ucb)
            # CORRECTED: Added missing σ² factor on r² term
            denominator = np.sqrt(self.sigma**4 * V_r + self.sigma**2 * r**2)
            g_star = numerator / denominator if denominator > 1e-9 else 0.0
            
            # The true optimizer is the projection of g_star onto the valid range [0, g_max]
            g_optimizer = min(g_star, g_max)
            
            # The harm is maximized at this g_optimizer
            harm_val = (np.sqrt(max(0, C_ucb - self.sigma**2 * g_optimizer**2)) * np.sqrt(V_r)) + r * g_optimizer
            return harm_val

        try:
            # Bisection search to find R such that get_max_harm_at_r(R) = eps_y
            r_high = 20.0 * self.sigma
            if get_max_harm_at_r(r_high) < self.eps_y:
                return r_high
                
            R = brentq(lambda r: get_max_harm_at_r(r) - self.eps_y, 0.0, r_high, xtol=1e-7, rtol=1e-7)
            return max(0.0, R)
        except (ValueError, RuntimeError):
            return 0.0

    # CHANGED: Logic is now cleaner and more correct.
    def certify_point(self, z: np.ndarray, model_fn: Callable, N_samples: int = 10000,
                     seed: Optional[int] = None) -> float:
        """
        Compute U-statistic variance + gradient certificate.
        Uses ALL N samples for both estimates with a union bound.
        """
        rng = np.random.default_rng(seed)
        
        # 1. Generate all necessary random samples ONCE.
        eta_samples = rng.normal(0.0, self.sigma, size=(N_samples, z.shape[-1]))
        z = np.asarray(z)
        def eval_model(x: np.ndarray):
            try:
                return model_fn(x)
            except TypeError:
                return model_fn(x[0], x[1])
        f_values = np.array([eval_model(z + eta) for eta in eta_samples])
        
        # 2. Estimate variance with α/2 confidence interval.
        _, _, var_upper = self.u_statistic_variance_estimator_alpha_half(f_values)
        
        # 3. Estimate gradient norm with α/2 confidence interval using the SAME samples.
        _, _, grad_norm_upper = self.u_statistic_gradient_norm_estimator_alpha_half(
            f_values, eta_samples
        )
        
        # 4. Compute certificate with conservative upper bounds.
        R = self.variance_gradient_certificate(var_upper, grad_norm_upper, self.eps_y)
        
        return R