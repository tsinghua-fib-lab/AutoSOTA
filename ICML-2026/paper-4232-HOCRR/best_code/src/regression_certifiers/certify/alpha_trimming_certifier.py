#!/usr/bin/env python3
"""
Alpha-Trimming Certifier for Regression Tasks

Implements the certification method from:
"Certified Adversarial Robustness via Randomized α-Smoothing for Regression Models"
(Rekavandi et al., NeurIPS 2024)

This is adapted from their DSAC* camera relocalization code to work with 
high-dimensional regression tasks like MNIST rotation.
"""

import numpy as np
from typing import Callable, Optional
from scipy.stats import norm, binom
from scipy.optimize import brentq


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Compute Clopper-Pearson lower confidence bound for binomial proportion.
    
    Args:
        k: Number of successes
        n: Number of trials
        alpha: Significance level (returns 1-alpha confidence bound)
        
    Returns:
        Lower confidence bound on the true proportion
    """
    from scipy.stats import beta
    
    if n <= 0:
        return 0.0
    if k <= 0:
        return 0.0
    if k >= n:
        # ✅ FIX 1: one-sided lower bound for all-successes
        # NOT 1.0, which would blow up norm.ppf
        return float(beta.ppf(alpha, n, 1))  # ~ alpha**(1/n) < 1
    
    # Use beta distribution quantile (equivalent to Clopper-Pearson)
    return float(beta.ppf(alpha, k, n - k + 1))


def _safe_ppf(p: float, eps: float = 1e-12) -> float:
    """
    Guard norm.ppf against numerical drift that can yield 0 or 1 and give ±∞.
    
    Args:
        p: Probability value
        eps: Small epsilon to clamp probabilities away from 0 and 1
        
    Returns:
        Safely clamped probability for norm.ppf
    """
    return norm.ppf(min(max(p, eps), 1 - eps))


def angdiff_deg(a: float, b: float) -> float:
    """
    Compute smallest signed difference a-b in [-180, 180] degrees.
    
    Args:
        a: First angle in degrees
        b: Second angle in degrees
        
    Returns:
        Smallest signed difference
    """
    return (a - b + 180.0) % 360.0 - 180.0


def within_eps(pred: float, center: float, eps_y: float, circular: bool = False) -> bool:
    """
    Check if prediction is within eps_y of center.
    
    Args:
        pred: Predicted value
        center: Center value
        eps_y: Tolerance
        circular: If True, use circular distance (for angles in degrees)
        
    Returns:
        True if within tolerance
    """
    if circular:
        return abs(angdiff_deg(pred, center)) <= eps_y
    else:
        return abs(pred - center) <= eps_y


def probability_success_from_alpha(alpha_trim: float, n_sample: int, P: float) -> float:
    """
    Solve for q in BinomCDF(round(alpha*n), n, 1 - q) = P.
    Returns q \in (0,1).
    
    This is the Binomial inverse mapping from (alpha, n_sample, P) to q,
    as described in the alpha-smoothing paper.
    
    Args:
        alpha_trim: Trimming rate
        n_sample: Number of samples for g_alpha
        P: Target success probability in the radius formula
        
    Returns:
        Probability threshold q
    """
    # ✅ FIX 3: match DSAC* repo (uses round, not floor)
    k = int(np.round(alpha_trim * n_sample))
    
    def f(q):
        return binom.cdf(k, n_sample, 1 - q) - P
    
    return brentq(f, 1e-12, 1.0 - 1e-12)


def radius_from_probabilities(p_A: float, q: float, sigma: float) -> float:
    """
    Compute certified radius from smoothed probabilities.
    
    Based on the relationship:
        R = σ * (Φ^{-1}(p_A) - Φ^{-1}(q))
    
    where:
        p_A: Lower confidence bound on P(|f(x+ξ) - center| ≤ ε_y)
        q: Probability threshold from Binomial inverse
        σ: Noise standard deviation
    
    Args:
        p_A: Probability that raw noisy output stays within tolerance
        q: Threshold probability from (alpha, n_sample, P)
        sigma: Noise standard deviation
        
    Returns:
        Certified L2 radius
    """
    if p_A <= q:
        return 0.0
    
    # ✅ FIX 2: Guard ppf's against numerical drift
    return max(0.0, sigma * (_safe_ppf(p_A) - _safe_ppf(q)))


class AlphaTrimmingCertifier:
    """
    Certifier using α-trimming for robust regression certification.
    
    This implements the alpha-smoothing certification approach for high-dimensional
    regression tasks, following Rekavandi et al., NeurIPS 2024.
    
    Key steps:
    1. Estimate center (clean prediction or true label)
    2. Estimate p_A with n_tr trials (NO trimming) via Clopper-Pearson
    3. Compute q from (alpha, n_sample, P) via Binomial mapping
    4. Compute radius: R = σ * (Φ^{-1}(p_A^{LCB}) - Φ^{-1}(q))
    
    Args:
        sigma: Standard deviation of Gaussian noise
        eps_y: Output tolerance (maximum allowed deviation in output space)
        alpha: Trimming rate (fraction of extreme values to remove, e.g., 0.35)
        n_tr: Number of Monte Carlo samples for p_A estimation (Clopper-Pearson)
        n_sample: Number of samples for g_alpha / Binomial mapping to q
        confidence: Confidence level (1 - δ), where δ is failure probability
        P: Target success probability in the radius formula
        center: Center choice - 'pred' (clean prediction, for stability) or 'true' (true label, for accuracy)
        circular: If True, use circular distance for angles (MNIST rotation)
    """
    
    def __init__(
        self,
        sigma: float,
        eps_y: float,
        alpha: float = 0.35,
        n_tr: int = 1000,
        n_sample: int = 16,
        confidence: float = 0.95,
        P: float = 0.9,
        center: str = 'pred',
        circular: bool = False
    ):
        self.sigma = sigma
        self.eps_y = eps_y
        self.alpha = alpha
        self.n_tr = n_tr
        self.n_sample = n_sample
        self.delta = 1 - confidence
        self.P = P
        self.center = center
        self.circular = circular
        
        # ✅ FIX 4: Sanity check on n_sample (if ever computing g_α by trimming)
        kept = n_sample - 2 * int(np.round(alpha * n_sample))
        if kept < 1:
            # Calculate helpful suggestions
            n_trim_per_tail = int(np.round(alpha * n_sample))
            # Maximum alpha: ensure round(alpha * n_sample) <= floor((n_sample - 1) / 2)
            # Use a conservative bound: floor((n_sample - 1) / 2) / n_sample
            max_trim = (n_sample - 1) // 2
            max_alpha_for_n = max_trim / n_sample  # Guaranteed to keep at least 1 sample
            # Minimum n_sample: need n_sample - 2*round(alpha*n_sample) >= 1
            # So: 2*round(alpha*n_sample) <= n_sample - 1
            # For given alpha, need: n_sample >= 2*round(alpha*n_sample) + 1
            # Use a conservative estimate: n_sample >= 2*alpha*n_sample + 1
            # So: n_sample >= 1 / (1 - 2*alpha), but we need to account for rounding
            # More precise: find smallest n such that n - 2*round(alpha*n) >= 1
            min_n_for_alpha = n_sample
            while min_n_for_alpha - 2 * int(np.round(alpha * min_n_for_alpha)) < 1:
                min_n_for_alpha += 1
            raise ValueError(
                f"n_sample={n_sample} too small for alpha={alpha}: "
                f"would trim {n_trim_per_tail} samples from each tail, leaving {kept} samples.\n"
                f"  Suggestions:\n"
                f"    - Use alpha <= {max_alpha_for_n:.4f} with n_sample={n_sample} (keeps >= 1 sample), or\n"
                f"    - Use n_sample >= {min_n_for_alpha} with alpha={alpha} (keeps >= 1 sample)"
            )
        
        # NOTE: q is NOT Φ(ε/σ). We compute q later from (alpha, n_sample, P).
        
        print(f"AlphaTrimmingCertifier initialized:")
        print(f"  σ={sigma}, ε_y={eps_y}, α={alpha}")
        print(f"  n_tr={n_tr} (for p_A), n_sample={n_sample} (for g_alpha/q)")
        print(f"  confidence={confidence}, P={P}")
        print(f"  center={center}, circular={circular}")
    
    def apply_alpha_trimming(self, values: np.ndarray) -> np.ndarray:
        """
        Apply α-trimming to remove extreme values.
        
        Args:
            values: Array of values (predictions)
            
        Returns:
            Trimmed array with extreme values removed
        """
        n = len(values)
        n_trim = int(np.floor(self.alpha * n))
        
        if n_trim == 0:
            return values
        
        # Sort values
        sorted_values = np.sort(values)
        
        # Remove n_trim values from each tail
        trimmed_values = sorted_values[n_trim : n - n_trim]
        
        return trimmed_values
    
    def estimate_center(
        self,
        z: np.ndarray,
        model_fn: Callable,
        y_true: Optional[float] = None
    ) -> float:
        """
        Estimate the center value for certification.
        
        Args:
            z: Input point (flattened)
            model_fn: Function that maps input to output
            y_true: True label (required if center='true')
            
        Returns:
            Center value (scalar)
        """
        if self.center == 'true':
            # Use ground truth as center (for accuracy certification)
            if y_true is None:
                raise ValueError("center='true' requires y_true")
            return float(y_true)
        
        elif self.center == 'pred':
            # Use clean prediction as center (for stability certification)
            return float(model_fn(z))
        
        else:
            raise ValueError(f"Unknown center type: {self.center}")
    
    def certify_point(
        self,
        z: np.ndarray,
        model_fn: Callable,
        y_true: Optional[float] = None,
        seed: Optional[int] = None,
        return_both_radii: bool = False
    ) -> float:
        """
        Certify a single point using α-smoothing (correct implementation).
        
        Following Rekavandi et al., NeurIPS 2024:
        1. Estimate center (clean prediction or true label)
        2. Estimate p_A with n_tr trials (NO trimming)
        3. Compute q from (alpha, n_sample, P) via Binomial inverse
        4. Return radius: R = σ * (Φ^{-1}(p_A^{LCB}) - Φ^{-1}(q))
        
        Args:
            z: Input point (flattened array)
            model_fn: Function that maps input to scalar output
            y_true: True label (required if center='true')
            seed: Random seed for reproducibility
            return_both_radii: If True, return (r_g, r_base) tuple
            
        Returns:
            Certified L2 radius (or tuple if return_both_radii=True)
        """
        rng_tr = np.random.default_rng(42 if seed is None else seed)
        
        # 1) Center
        center_val = self.estimate_center(z, model_fn, y_true=y_true)
        
        # 2) Estimate p_A with n_tr trials (NO trimming)
        k = 0
        for _ in range(self.n_tr):
            noise = rng_tr.normal(0.0, self.sigma, size=z.shape)
            pred = float(model_fn(z + noise))
            k += int(within_eps(pred, center_val, self.eps_y, circular=self.circular))
        
        pA_lcb = clopper_pearson_lower(k, self.n_tr, self.delta)
        
        # 3) Compute q from (alpha, n_sample, P) via Binomial mapping
        q = probability_success_from_alpha(self.alpha, self.n_sample, self.P)
        
        # 4) ✅ FIX 5: Use the helper function (reuse radius_from_probabilities)
        r_g = radius_from_probabilities(pA_lcb, q, self.sigma)
        
        # ✅ FIX 6: Optionally return both base RS vs α-smoothed
        if return_both_radii:
            r_base = radius_from_probabilities(pA_lcb, self.P, self.sigma)
            return float(r_g), float(r_base)
        
        return float(r_g)
    
    def certify_batch(
        self,
        Z: np.ndarray,
        model_fn: Callable,
        y_true: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        return_both_radii: bool = False
    ) -> np.ndarray:
        """
        Certify multiple points.
        
        Args:
            Z: Input points (N x d array)
            model_fn: Function that maps input to scalar output
            y_true: True labels (N,) array, required if center='true'
            seed: Random seed for reproducibility
            verbose: Whether to print progress
            return_both_radii: If True, return (r_g, r_base) for each point
            
        Returns:
            Array of certified radii (N,) or tuple of arrays if return_both_radii=True
        """
        radii_g = []
        radii_base = []
        
        for i, z in enumerate(Z):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Certified {i+1}/{len(Z)} points...")
            
            y_i = y_true[i] if y_true is not None else None
            result = self.certify_point(
                z, model_fn, y_true=y_i, seed=seed + i if seed is not None else None,
                return_both_radii=return_both_radii
            )
            
            if return_both_radii:
                r_g, r_base = result
                radii_g.append(r_g)
                radii_base.append(r_base)
            else:
                radii_g.append(result)
        
        if return_both_radii:
            return np.array(radii_g), np.array(radii_base)
        return np.array(radii_g)

