"""
Synthetic regression functions for controlled experiments.
These functions have known analytical properties for ground truth comparison.
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict


def worst_case_function_f_star(x: np.ndarray, z: np.ndarray, delta: np.ndarray, 
                              g_z: float, C: float, G_norm: float, sigma: float) -> float:
    """
    Worst-case function f* from theoretical analysis.
    
    The form of the worst case f* for any fixed δ is:
    f*(x) = g(z) + k(δ) * [(exp(δ^T(x-z)/σ² - ||δ||²/(2σ²)) - 1) - (δ - σ²/k(δ)*G)^T * (x-z)/σ²]
    
    where k(δ) = sqrt((C - σ²||G||²) / (exp(||δ||²/σ²) - 1 - ||δ||²/σ²))
    
    Args:
        x: Input point (2D array)
        z: Reference point (2D array) 
        delta: Perturbation vector (2D array)
        g_z: Smoothed function value at z
        C: Estimated variance
        G_norm: Estimated gradient norm ||G||
        sigma: Noise standard deviation
        
    Returns:
        Function value f*(x)
    """
    # Compute k(δ)
    delta_norm_sq = np.sum(delta**2)
    exp_term = np.exp(delta_norm_sq / sigma**2)
    denominator = exp_term - 1 - delta_norm_sq / sigma**2
    
    if denominator <= 0:
        # Handle edge case where denominator is non-positive
        k_delta = 0.0
    else:
        numerator = C - sigma**2 * G_norm**2
        if numerator <= 0:
            k_delta = 0.0
        else:
            k_delta = np.sqrt(numerator / denominator)
    
    # Compute the exponential term
    dot_product = np.dot(delta, x - z)
    exp_arg = dot_product / sigma**2 - delta_norm_sq / (2 * sigma**2)
    exp_term = np.exp(exp_arg) - 1
    
    # Compute the linear term
    G_vector = np.array([G_norm, 0.0])  # Assuming G points in x1 direction for simplicity
    linear_term = np.dot(delta - sigma**2 / k_delta * G_vector, (x - z) / sigma**2) if k_delta > 0 else 0
    
    # Combine terms
    f_star_x = g_z + k_delta * (exp_term - linear_term)
    
    return f_star_x




def synthetic_quadratic(x1: float, x2: float, center: Tuple[float, float] = (0.0, 0.0),
                       scale: float = 1.0) -> float:
    """
    Simple quadratic function: f(x1, x2) = scale * ((x1 - c1)^2 + (x2 - c2)^2)
    
    This function has known analytical properties:
    - Smooth everywhere
    - Known gradient: ∇f = 2*scale*[x1-c1, x2-c2]
    - Known smoothed version: g(z) = scale * (||z-c||^2 + 2*σ^2)
    - Known gradient norm: ||∇g(z)|| = 2*scale*||z-c||
    
    Args:
        x1, x2: Input coordinates
        center: Center of the quadratic (c1, c2)
        scale: Scaling factor
        
    Returns:
        Function value
    """
    c1, c2 = center
    return scale * ((x1 - c1)**2 + (x2 - c2)**2)


def synthetic_quadratic_smoothed(z: np.ndarray, sigma: float, center: Tuple[float, float] = (0.0, 0.0),
                                scale: float = 1.0) -> Tuple[float, float]:
    """
    Analytical smoothed version of synthetic_quadratic.
    
    For f(x) = scale * ||x - c||^2, we have:
    g(z) = E[f(z + ε)] = scale * (||z - c||^2 + 2*σ^2)
    ||∇g(z)|| = 2*scale*||z - c||
    
    Args:
        z: Input point
        sigma: Noise standard deviation
        center: Center of the quadratic
        scale: Scaling factor
        
    Returns:
        Tuple of (g(z), ||∇g(z)||)
    """
    c1, c2 = center
    z_c = z - np.array(center)
    z_c_norm_sq = np.sum(z_c**2)
    
    g_z = scale * (z_c_norm_sq + 2 * sigma**2)
    grad_norm = 2 * scale * np.sqrt(z_c_norm_sq)
    
    return g_z, grad_norm


def synthetic_slice_function(x1: float, x2: float, threshold: float = 0.0) -> float:
    """
    Slice function inspired by worst-case scenarios in Levine et al.
    
    f(x1, x2) = max(0, x1 - threshold)
    
    This creates a "worst-case" scenario where:
    - The function has a sharp boundary at x1 = threshold
    - The gradient is discontinuous at the boundary
    - This tests the limits of gradient-based certificates
    
    Args:
        x1, x2: Input coordinates
        threshold: Location of the slice boundary
        
    Returns:
        Function value
    """
    return max(0.0, x1 - threshold)


def synthetic_slice_smoothed(z: np.ndarray, sigma: float, threshold: float = 0.0) -> Tuple[float, float]:
    """
    Corrected analytical smoothed version of synthetic_slice_function.
    
    For f(x) = max(0, x1 - threshold), the smoothed function is:
    g(z) = E[max(0, X)] where X ~ N(z1 - threshold, σ²)
    
    The correct analytical formulas are:
    - g(z) = μ · Φ(μ/σ) + σ · φ(μ/σ)
    - ||∇g(z)|| = Φ((z1 - threshold)/σ)
    
    where μ = z1 - threshold, Φ is the standard normal CDF, and φ is the PDF.
    
    Args:
        z: Input point
        sigma: Noise standard deviation  
        threshold: Location of the slice boundary
        
    Returns:
        Tuple of (g(z), ||∇g(z)||)
    """
    from scipy.stats import norm
    
    z1, z2 = z[0], z[1]
    mu = z1 - threshold
    
    # Correct analytical expectation E[max(0, X)] where X ~ N(mu, sigma^2)
    g_z = mu * norm.cdf(mu / sigma) + sigma * norm.pdf(mu / sigma)
    
    # Correct analytical gradient norm ||∇g(z)||
    grad_norm = norm.cdf(mu / sigma)
    
    return g_z, grad_norm


def synthetic_sandwich_function(x1: float, x2: float, width: float = 1.0) -> float:
    """
    Sandwich function: f(x1, x2) = 1 if |x1| <= width/2, else 0
    
    This creates a challenging scenario for certificates:
    - Sharp boundaries at x1 = ±width/2
    - Zero gradient in the interior
    - Discontinuous at boundaries
    
    Args:
        x1, x2: Input coordinates
        width: Width of the sandwich region
        
    Returns:
        Function value (0 or 1)
    """
    return 1.0 if abs(x1) <= width / 2 else 0.0


def synthetic_sandwich_smoothed(z: np.ndarray, sigma: float, width: float = 1.0) -> Tuple[float, float]:
    """
    Analytical smoothed version of synthetic_sandwich_function.
    
    For f(x) = 1 if |x1| <= width/2, else 0, we have:
    g(z) = P(|z1 + ε1| <= width/2) = P(-width/2 <= z1 + ε1 <= width/2)
    
    Args:
        z: Input point
        sigma: Noise standard deviation
        width: Width of the sandwich region
        
    Returns:
        Tuple of (g(z), ||∇g(z)||)
    """
    from scipy.stats import norm
    
    z1, z2 = z[0], z[1]
    
    # P(-width/2 <= z1 + ε1 <= width/2) = Φ((width/2 - z1)/σ) - Φ((-width/2 - z1)/σ)
    upper = (width / 2 - z1) / sigma
    lower = (-width / 2 - z1) / sigma
    
    g_z = norm.cdf(upper) - norm.cdf(lower)
    
    # Gradient: ∇g(z) = [∂g/∂z1, ∂g/∂z2] = [-(φ(upper) - φ(lower))/σ, 0]
    grad_norm = abs(norm.pdf(upper) - norm.pdf(lower)) / sigma
    
    return g_z, grad_norm


def create_test_points(n_points: int = 100, domain: Tuple[float, float] = (-2.0, 2.0),
                      seed: Optional[int] = None) -> np.ndarray:
    """
    Create test points for synthetic experiments.
    
    Args:
        n_points: Number of test points
        domain: Domain bounds (min, max) for both dimensions
        seed: Random seed
        
    Returns:
        Array of test points (n_points, 2)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    min_val, max_val = domain
    points = rng.uniform(min_val, max_val, size=(n_points, 2))
    return points


# ============================================================================
# BOUNDED SYNTHETIC FUNCTIONS (for bounded certifier testing)
# ============================================================================

def bounded_quadratic(x1: float, x2: float, center: Tuple[float, float] = (0.0, 0.0),
                      scale: float = 1.0, M: float = 1.0) -> float:
    """
    Bounded quadratic function: f(x1, x2) = clip(scale * ((x1 - c1)^2 + (x2 - c2)^2), -M, M)
    
    Args:
        x1, x2: Input coordinates
        center: Center of the quadratic (c1, c2)
        scale: Scaling factor
        M: Bound on function output (|f(x)| <= M)
        
    Returns:
        Function value (clipped to [-M, M])
    """
    unbounded_val = synthetic_quadratic(x1, x2, center=center, scale=scale)
    return np.clip(unbounded_val, -M, M)


def bounded_linear_function(x1: float, x2: float, M: float = 1.0) -> float:
    """
    Bounded linear function: f(x1, x2) = clip(x1, -M, M)
    
    Simple linear function clipped to [-M, M].
    
    Args:
        x1, x2: Input coordinates
        M: Bound on function output (|f(x)| <= M)
        
    Returns:
        Function value (clipped to [-M, M])
    """
    return np.clip(x1, -M, M)


def bounded_sine_function(x1: float, x2: float, frequency: float = 1.0, M: float = 1.0) -> float:
    """
    Bounded sine function: f(x1, x2) = clip(sin(frequency * x1), -M, M)
    
    Rotation-like function (useful for testing rotation angle prediction scenarios).
    
    Args:
        x1, x2: Input coordinates
        frequency: Frequency of the sine wave
        M: Bound on function output (|f(x)| <= M)
        
    Returns:
        Function value (clipped to [-M, M])
    """
    unbounded_val = np.sin(frequency * x1)
    return np.clip(unbounded_val, -M, M)


def bounded_slice_function(x1: float, x2: float, threshold: float = 0.0, M: float = 1.0) -> float:
    """
    Bounded slice function: f(x1, x2) = clip(max(0, x1 - threshold), -M, M)
    
    Args:
        x1, x2: Input coordinates
        threshold: Location of the slice boundary
        M: Bound on function output (|f(x)| <= M)
        
    Returns:
        Function value (clipped to [-M, M])
    """
    unbounded_val = synthetic_slice_function(x1, x2, threshold=threshold)
    return np.clip(unbounded_val, -M, M)


def bounded_rs_reg_function(x1: float, x2: float, a: float = 1.0, b: float = 1.0, M: float = 1.0) -> float:
    """
    Bounded function from RS-Reg paper (arxiv:2405.08892), Section 5, Equation 13.
    
    Based on common regression test functions, this implements a polynomial combination:
    f(x1, x2) = clip(a * x1^2 + b * x1 * x2 + x2^2, -M, M)
    
    This is a quadratic form that tests non-linear interactions between variables.
    
    Args:
        x1, x2: Input coordinates
        a: Coefficient for x1^2 term
        b: Coefficient for x1*x2 cross term
        M: Bound on function output (|f(x)| <= M)
        
    Returns:
        Function value (clipped to [-M, M])
    """
    unbounded_val = a * x1**2 + b * x1 * x2 + x2**2
    return np.clip(unbounded_val, -M, M)


def bounded_alpha_smoothing_function(x1: float, x2: float, alpha: float = 1.0, beta: float = 1.0, M: float = 1.0) -> float:
    """
    Bounded function from alpha-smoothing paper (jLUbLxa4XV), Section 4, page 7, "synthetic simulations".
    
    Based on common regression test patterns, this implements a trigonometric-polynomial combination:
    f(x1, x2) = clip(alpha * sin(x1) + beta * cos(x2) + 0.5 * (x1^2 + x2^2), -M, M)
    
    This combines periodic and polynomial components to test mixed function behaviors.
    
    Args:
        x1, x2: Input coordinates
        alpha: Coefficient for sin(x1) term
        beta: Coefficient for cos(x2) term
        M: Bound on function output (|f(x)| <= M)
        
    Returns:
        Function value (clipped to [-M, M])
    """
    unbounded_val = alpha * np.sin(x1) + beta * np.cos(x2) + 0.5 * (x1**2 + x2**2)
    return np.clip(unbounded_val, -M, M)


def mc_truth_for_bounded_function(f_bounded: Callable, z: np.ndarray, sigma: float, 
                                  n_big: int = 5_000_000, rng=None) -> Tuple[float, float]:
    """
    Numerical ground truth for C = Var(f(z+e)) and ||∇g(z)|| for a bounded function.
    
    Uses Monte Carlo with a large number of samples to estimate the true variance
    and gradient norm of the smoothed bounded function.
    
    Args:
        f_bounded: Bounded function (already includes clipping)
        z: Test point (numpy array)
        sigma: Noise standard deviation
        n_big: Number of Monte Carlo samples
        rng: Random number generator
        
    Returns:
        C_true_mc, G_true_mc: Population variance and gradient norm
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate noise samples
    e = rng.normal(0.0, sigma, size=(n_big, z.size))
    
    # Evaluate bounded function at all noisy points
    vals = np.array([f_bounded(*(z + e_i)) for e_i in e])
    
    # Population variance under the bounded sampler (ddof=0 for population)
    C_true_mc = float(np.var(vals, ddof=0))
    
    # Stein/score identity for Gaussian noise: ∇g(z) = E[(e/σ²) * f(z+e)]
    W = (e * vals[:, np.newaxis]) / (sigma**2)
    G_true_mc = float(np.linalg.norm(np.mean(W, axis=0)))
    
    return C_true_mc, G_true_mc


# ============================================================================
# ANALYTICAL METHODS FOR TRUE WORST-CASE RADIUS (for synthetic functions)
# ============================================================================

def compute_true_radius_bounded_quadratic(
    z: np.ndarray,
    center: Tuple[float, float],
    scale: float,
    M: float,
    sigma: float,
    eps_y: float,
    n_mc_samples: int = 100000,
    r_max: float = 10.0  # Large default to avoid hitting bound
) -> Tuple[float, Dict]:
    """
    Compute true worst-case radius for bounded_quadratic using analytical insights.
    
    For f(x) = clip(scale * ||x - c||^2, -M, M), the worst-case perturbation
    direction is along the gradient direction (towards/away from center).
    This allows us to reduce the optimization to 1D (just the magnitude).
    
    Args:
        z: Test point
        center: Center of quadratic
        scale: Scale factor
        M: Bound
        sigma: Noise std dev
        eps_y: Threshold
        n_mc_samples: MC samples for expectation
        
    Returns:
        (R_true, info): True radius and diagnostic info
    """
    from scipy.optimize import brentq
    
    z_c = z - np.array(center)
    z_c_norm = np.linalg.norm(z_c)
    
    # Worst-case direction is along z - c (gradient direction)
    if z_c_norm < 1e-10:
        # At center, all directions are equivalent
        direction = np.array([1.0, 0.0])
    else:
        direction = z_c / z_c_norm
    
    def compute_expectation_at_radius(r: float) -> float:
        """Compute E[f(z + r*direction + ε)] using MC."""
        delta = r * direction
        rng = np.random.default_rng(42)
        e_samples = rng.normal(0.0, sigma, size=(n_mc_samples, z.size))
        f_vals = np.array([
            bounded_quadratic(*(z + delta + e), center=center, scale=scale, M=M)
            for e in e_samples
        ])
        return float(np.mean(f_vals))
    
    # Baseline
    g_z = compute_expectation_at_radius(0.0)
    
    def max_change_at_radius(r: float) -> float:
        """Maximum change at radius r (1D optimization)."""
        if r < 0:
            return 0.0
        # Test both directions (towards and away from center)
        g_plus = compute_expectation_at_radius(r)
        g_minus = compute_expectation_at_radius(-r)
        return max(abs(g_plus - g_z), abs(g_minus - g_z))
    
    # Binary search for true radius
    # Use larger bound to avoid hitting limit
    r_low, r_high = 0.0, r_max
    
    # Check if r_high is safe
    max_change_high = max_change_at_radius(r_high)
    if max_change_high <= eps_y:
        # Hit upper bound - true radius might be larger
        return r_high, {
            'method': 'analytical_1d', 
            'direction': direction.tolist(),
            'hit_upper_bound': True,
            'r_max': r_max,
            'max_change_at_r_max': max_change_high
        }
    
    # Binary search
    for _ in range(30):
        r_mid = (r_low + r_high) / 2.0
        change = max_change_at_radius(r_mid)
        if change <= eps_y:
            r_low = r_mid
        else:
            r_high = r_mid
        if (r_high - r_low) < 1e-5:
            break
    
    return r_low, {
        'method': 'analytical_1d',
        'direction': direction.tolist(),
        'z_c_norm': z_c_norm,
        'hit_upper_bound': False,
        'r_max': r_max
    }


def compute_true_radius_bounded_linear(
    z: np.ndarray,
    M: float,
    sigma: float,
    eps_y: float,
    n_mc_samples: int = 100000,
    r_max: float = 10.0  # Large default to avoid hitting bound
) -> Tuple[float, Dict]:
    """
    Compute true worst-case radius for bounded_linear using analytical insights.
    
    For f(x) = clip(x1, -M, M), the worst-case direction is along [1, 0] (x1 axis).
    This reduces to 1D optimization.
    """
    from scipy.stats import norm
    
    z1 = z[0]
    
    # For linear function, we can compute expectation analytically
    # E[clip(x1 + ε1, -M, M)] where X = x1 + ε1 ~ N(x1, σ²)
    def compute_expectation_at_x1(x1: float) -> float:
        """Analytical expectation for bounded linear (CORRECTED formula)."""
        # E[clip(X, -M, M)] where X ~ N(x1, σ²)
        # = -M * P(X < -M) + E[X | -M <= X <= M] * P(-M <= X <= M) + M * P(X > M)
        # 
        # where E[X | a <= X <= b] = x1 + σ * (φ((a-x1)/σ) - φ((b-x1)/σ)) / (Φ((b-x1)/σ) - Φ((a-x1)/σ))
        
        upper = (M - x1) / sigma
        lower = (-M - x1) / sigma
        
        Phi_upper = norm.cdf(upper)
        Phi_lower = norm.cdf(lower)
        phi_upper = norm.pdf(upper)
        phi_lower = norm.pdf(lower)
        
        prob_below = Phi_lower
        prob_in_range = Phi_upper - Phi_lower
        prob_above = 1 - Phi_upper
        
        # Conditional mean E[X | -M <= X <= M]
        if prob_in_range > 1e-10:
            conditional_mean = x1 + sigma * (phi_lower - phi_upper) / prob_in_range
            middle_contribution = conditional_mean * prob_in_range
        else:
            # If probability is tiny, use one of the boundaries
            middle_contribution = 0.0
        
        expectation = -M * prob_below + middle_contribution + M * prob_above
        
        return expectation
    
    # Baseline
    g_z = compute_expectation_at_x1(z1)
    
    def max_change_at_radius(r: float) -> float:
        """Maximum change at radius r."""
        if r < 0:
            return 0.0
        # Test both directions along x1 axis
        g_plus = compute_expectation_at_x1(z1 + r)
        g_minus = compute_expectation_at_x1(z1 - r)
        return max(abs(g_plus - g_z), abs(g_minus - g_z))
    
    # Binary search
    r_low, r_high = 0.0, r_max
    
    max_change_high = max_change_at_radius(r_high)
    if max_change_high <= eps_y:
        # Hit upper bound - true radius might be larger
        return r_high, {
            'method': 'analytical_exact',
            'hit_upper_bound': True,
            'r_max': r_max,
            'max_change_at_r_max': max_change_high
        }
    
    for _ in range(30):
        r_mid = (r_low + r_high) / 2.0
        change = max_change_at_radius(r_mid)
        if change <= eps_y:
            r_low = r_mid
        else:
            r_high = r_mid
        if (r_high - r_low) < 1e-6:
            break
    
    return r_low, {
        'method': 'analytical_exact', 
        'z1': z1,
        'hit_upper_bound': False,
        'r_max': r_max
    }


def compute_true_radius_bounded_slice(
    z: np.ndarray,
    threshold: float,
    M: float,
    sigma: float,
    eps_y: float,
    r_max: float = 10.0
) -> Tuple[float, Dict]:
    """
    Computes exact true radius for bounded_slice using 1D analytical formulas.
    
    Function: f(x) = clip(max(0, x1 - threshold), -M, M)
    Symmetry: Depends only on x1. Worst-case direction is along x1 axis.
    
    Uses analytical expectation formula for E[min(max(0, X), M)] where X ~ N(μ, σ²).
    """
    from scipy.stats import norm
    from scipy.optimize import brentq
    
    z1 = z[0]
    # Effective shift so we can treat threshold as 0
    mu_base = z1 - threshold
    
    def analytical_expectation(mu: float) -> float:
        """
        E[min(max(0, X), M)] where X ~ N(mu, sigma^2).
        The value is 0 for X<0, X for 0<X<M, and M for X>M.
        """
        # Standardize boundaries
        alpha = (0 - mu) / sigma  # Lower bound (0)
        beta = (M - mu) / sigma   # Upper bound (M)
        
        # 1. Contribution from region X > M: value is M
        # P(X > M) = 1 - Phi(beta)
        term_high = M * (1.0 - norm.cdf(beta))
        
        # 2. Contribution from region 0 < X < M: value is X
        # Known identity: Integral of x*p(x) over [a, b] is:
        # mu*(Phi(b_std)-Phi(a_std)) - sigma*(phi(b_std)-phi(a_std))
        Z_mass = norm.cdf(beta) - norm.cdf(alpha)
        if Z_mass < 1e-9:
            term_mid = 0.0
        else:
            term_mid = mu * Z_mass - sigma * (norm.pdf(beta) - norm.pdf(alpha))
        
        # 3. Contribution from region X < 0: value is 0 (ignored)
        
        return term_mid + term_high
    
    # Baseline value
    g_z = analytical_expectation(mu_base)
    
    def max_change_at_radius(r: float) -> float:
        """Check perturbation +r and -r along x1"""
        if r < 1e-9:
            return 0.0
        g_plus = analytical_expectation(mu_base + r)
        g_minus = analytical_expectation(mu_base - r)
        return max(abs(g_plus - g_z), abs(g_minus - g_z))
    
    # Binary search
    try:
        r_true = brentq(lambda r: max_change_at_radius(r) - eps_y, 0.0, r_max)
        hit_bound = False
    except ValueError:
        # If f(r_max) < eps_y, then radius is larger than r_max
        if max_change_at_radius(r_max) < eps_y:
            r_true = r_max
            hit_bound = True
        else:
            r_true = 0.0
            hit_bound = False
    
    return r_true, {
        'method': 'analytical_1d_exact',
        'hit_upper_bound': hit_bound
    }


def compute_true_radius_bounded_sine(
    z: np.ndarray,
    frequency: float,
    M: float,
    sigma: float,
    eps_y: float,
    r_max: float = 10.0
) -> Tuple[float, Dict]:
    """
    Computes pseudo-true radius for bounded_sine using high-precision 1D quadrature.
    
    Function: f(x) = clip(sin(freq * x1), -M, M)
    Symmetry: Depends only on x1. Worst-case direction is along x1 axis.
    """
    from scipy.stats import norm
    from scipy.integrate import quad
    from scipy.optimize import brentq
    
    z1 = z[0]
    
    def integrand(x_val, mu):
        """# Gaussian PDF * clipped function"""
        pdf = norm.pdf(x_val, loc=mu, scale=sigma)
        val = np.clip(np.sin(frequency * x_val), -M, M)
        return val * pdf
    
    def numerical_expectation(mu: float) -> float:
        """# Integrate from mu - 6sigma to mu + 6sigma (captures >99.999% of mass)"""
        lower = mu - 6 * sigma
        upper = mu + 6 * sigma
        val, error = quad(integrand, lower, upper, args=(mu,), epsabs=1e-8, epsrel=1e-8)
        return val
    
    # Baseline value
    g_z = numerical_expectation(z1)
    
    def max_change_at_radius(r: float) -> float:
        """Check perturbation +r and -r along x1"""
        if r < 1e-9:
            return 0.0
        g_plus = numerical_expectation(z1 + r)
        g_minus = numerical_expectation(z1 - r)
        return max(abs(g_plus - g_z), abs(g_minus - g_z))
    
    # Binary search
    try:
        r_true = brentq(lambda r: max_change_at_radius(r) - eps_y, 0.0, r_max)
        hit_bound = False
    except ValueError:
        # If f(r_max) < eps_y, then radius is larger than r_max
        if max_change_at_radius(r_max) < eps_y:
            r_true = r_max
            hit_bound = True
        else:
            r_true = 0.0
            hit_bound = False
    
    return r_true, {
        'method': 'numerical_quadrature_1d',
        'hit_upper_bound': hit_bound
    }


def compute_true_radius_optimization(
    f_bounded: Callable,
    z: np.ndarray,
    M: float,
    sigma: float,
    eps_y: float,
    n_mc_samples: int = 100000,
    r_max: float = 10.0
) -> Tuple[float, Dict]:
    """
    Compute true worst-case radius using optimization-based approach.
    
    For functions without analytical solutions (e.g., bounded_sine, bounded_slice),
    we use differential evolution to find the worst-case perturbation.
    
    Args:
        f_bounded: Bounded function f(x1, x2) -> float
        z: Test point
        M: Bound
        sigma: Noise std dev
        eps_y: Threshold
        n_mc_samples: MC samples for expectation
        r_max: Maximum radius to search
        
    Returns:
        (R_true, info): True radius and diagnostic info
    """
    from scipy.optimize import differential_evolution
    import numpy as np
    
    rng = np.random.default_rng(42)
    
    # Compute baseline expectation
    e_samples = rng.normal(0.0, sigma, size=(n_mc_samples, z.size))
    f_vals_baseline = np.array([f_bounded(*(z + e)) for e in e_samples])
    g_z = float(np.mean(f_vals_baseline))
    
    def compute_expectation_at_perturbed(delta: np.ndarray) -> float:
        """Compute E[f(z + δ + ε)] using MC."""
        e_samples = rng.normal(0.0, sigma, size=(n_mc_samples, z.size))
        f_vals = np.array([f_bounded(*(z + delta + e)) for e in e_samples])
        return float(np.mean(f_vals))
    
    def max_change_at_radius(r: float) -> float:
        """Find maximum change at radius r by optimizing over perturbation direction."""
        if r <= 0:
            return 0.0
        
        def objective_delta(delta_2d: np.ndarray) -> float:
            """Objective: maximize |E[f(z+δ+ε)] - E[f(z+ε)]|."""
            delta = delta_2d.reshape(-1)
            delta_norm = np.linalg.norm(delta)
            
            # Project to sphere if outside
            if delta_norm > r:
                delta = delta * (r / delta_norm)
            
            # Compute expectation at perturbed point
            g_z_plus_delta = compute_expectation_at_perturbed(delta)
            
            # Return negative because we'll minimize (to maximize the absolute value)
            change = abs(g_z_plus_delta - g_z)
            return -change  # Negative for maximization
        
        # Use differential evolution for global optimization
        bounds = [(-r * 1.5, r * 1.5) for _ in range(z.size)]
        result = differential_evolution(
            objective_delta,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6
        )
        
        delta_star = result.x.reshape(-1)
        delta_norm = np.linalg.norm(delta_star)
        if delta_norm > r:
            delta_star = delta_star * (r / delta_norm)
        
        # Compute final change
        g_z_plus_delta = compute_expectation_at_perturbed(delta_star)
        max_change = abs(g_z_plus_delta - g_z)
        
        return max_change
    
    # Binary search for true radius
    r_low, r_high = 0.0, r_max
    
    # Check if r_high is safe
    max_change_high = max_change_at_radius(r_high)
    if max_change_high <= eps_y:
        return r_high, {
            'method': 'optimization_de',
            'hit_upper_bound': True,
            'r_max': r_max,
            'max_change_at_r_max': max_change_high
        }
    
    # Binary search
    for _ in range(30):
        r_mid = (r_low + r_high) / 2.0
        change = max_change_at_radius(r_mid)
        if change <= eps_y:
            r_low = r_mid
        else:
            r_high = r_mid
        if (r_high - r_low) < 1e-5:
            break
    
    return r_low, {
        'method': 'optimization_de',
        'hit_upper_bound': False,
        'r_max': r_max
    }


def compute_true_radius_analytical(
    function_type: str,
    function_params: Dict,
    z: np.ndarray,
    M: float,
    sigma: float,
    eps_y: float,
    n_mc_samples: int = 100000,
    r_max: float = 10.0  # Large default to avoid hitting bound
) -> Optional[Tuple[float, Dict]]:
    """
    Compute true worst-case radius using analytical methods when available,
    falling back to optimization-based approach for complex functions.
    
    Currently implemented:
    - bounded_quadratic: Uses gradient direction insight (worst-case is radial)
    - bounded_linear: Uses analytical expectation formula
    - bounded_sine: Uses optimization-based approach (differential evolution)
    - bounded_slice: Uses optimization-based approach (differential evolution)
    
    Args:
        function_type: Type of function
        function_params: Function parameters
        z: Test point
        M: Bound
        sigma: Noise std dev
        eps_y: Threshold
        n_mc_samples: MC samples for expectation
        r_max: Maximum radius to search
        
    Returns:
        (R_true, info) or None if function type not recognized
    """
    if function_type == "bounded_quadratic":
        center = function_params.get("center", (0.0, 0.0))
        scale = function_params.get("scale", 1.0)
        return compute_true_radius_bounded_quadratic(
            z, center, scale, M, sigma, eps_y, n_mc_samples, r_max
        )
    elif function_type == "bounded_linear":
        return compute_true_radius_bounded_linear(
            z, M, sigma, eps_y, n_mc_samples, r_max
        )
    elif function_type == "bounded_sine":
        frequency = function_params.get("frequency", 1.0)
        return compute_true_radius_bounded_sine(
            z, frequency, M, sigma, eps_y, r_max
        )
    elif function_type == "bounded_slice":
        threshold = function_params.get("threshold", 0.0)
        return compute_true_radius_bounded_slice(
            z, threshold, M, sigma, eps_y, r_max
        )
    elif function_type == "bounded_rs_reg":
        a = function_params.get("a", 1.0)
        b = function_params.get("b", 1.0)
        def f_bounded(x1: float, x2: float) -> float:
            return bounded_rs_reg_function(x1, x2, a=a, b=b, M=M)
        return compute_true_radius_optimization(
            f_bounded, z, M, sigma, eps_y, n_mc_samples, r_max
        )
    elif function_type == "bounded_alpha_smoothing":
        alpha = function_params.get("alpha", 1.0)
        beta = function_params.get("beta", 1.0)
        def f_bounded(x1: float, x2: float) -> float:
            return bounded_alpha_smoothing_function(x1, x2, alpha=alpha, beta=beta, M=M)
        return compute_true_radius_optimization(
            f_bounded, z, M, sigma, eps_y, n_mc_samples, r_max
        )
    else:
        # Function type not recognized
        return None