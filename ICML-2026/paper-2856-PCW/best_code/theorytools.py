#!/usr/bin/env python3
"""
Theoretical analysis tools for LLM watermarking.

Core functions for KL divergence, detection power, and parameter solving.
Used by solver notebook, experiment schedulers, and figure notebooks.
"""

import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.stats import norm

default_temp = 0.7


def KL_theory(gamma, delta, temp=default_temp):
    """KL divergence of watermarked vs original distribution."""
    gamma, delta = float(gamma), float(delta)
    gamma = min(max(gamma, 1e-12), 1 - 1e-12)
    em1 = np.expm1(delta)
    denom = 1.0 + gamma * em1
    return (em1 + 1.0) * gamma * delta / denom - np.log(denom)


def deltaP(gamma, delta, temp=default_temp):
    """Change in green token probability: P'(green) - gamma."""
    Palt = np.exp(delta) * gamma / (1 + np.expm1(delta) * gamma)
    return Palt - gamma


def power_stat(gamma, delta, alpha, n_samples, temp=default_temp):
    """Detection power z-statistic (before applying Phi)."""
    z_alpha = norm.ppf(1 - alpha)
    dp = deltaP(gamma, delta, temp)
    gamma_prime = gamma + dp
    num = np.sqrt(n_samples) * dp - z_alpha * np.sqrt(gamma * (1 - gamma))
    den = np.sqrt(gamma_prime * (1 - gamma_prime))
    return num / den


def power(gamma, delta, alpha, n_samples, temp=default_temp):
    """Detection power (probability of rejecting H0 when watermark is present)."""
    return norm.cdf(power_stat(gamma, delta, alpha, n_samples, temp))


def power_stat_gprime(n, gamma, gamma_prime, alpha):
    """Detection power z-statistic given gamma_prime directly."""
    z_alpha = norm.ppf(1 - alpha)
    num = np.sqrt(n) * (gamma_prime - gamma) - z_alpha * np.sqrt(gamma * (1 - gamma))
    den = np.sqrt(gamma_prime * (1 - gamma_prime))
    return num / den


def solve_delta(kl_target, gamma, temp=default_temp, delta_max=50000.0):
    """Solve KL_theory(gamma, delta) = kl_target for delta >= 0 using brentq."""
    kl_target, gamma = float(kl_target), float(gamma)
    if kl_target <= 0:
        return 0.0
    kl_cap = -np.log(max(gamma, 1e-12))
    if kl_target > kl_cap * (1 + 1e-12):
        return np.nan
    f = lambda d: KL_theory(gamma, d, temp) - kl_target
    lo, hi = 0.0, 1.0
    for _ in range(60):
        fhi = f(hi)
        if np.isfinite(fhi) and fhi > 0:
            break
        hi *= 2.0
        if hi > delta_max:
            raise ValueError(f'Cannot bracket root up to delta={delta_max}')
    return float(brentq(f, lo, hi, maxiter=200, xtol=1e-12, rtol=1e-10))


def solve_delta_power(power_s, gamma, alpha, n_samples, temp=default_temp):
    """Solve for delta given target power statistic and gamma."""
    f = lambda delta: power_stat(gamma, delta, alpha, n_samples, temp) - power_s
    delta_guess = 3
    return float(fsolve(f, delta_guess)[0])


# ============================================================================
# EXPERIMENT PARAMETER GENERATION
# ============================================================================

def compute_dp_params():
    """DP method: min-KL (gamma*, delta*) for each deltaP in 0.1..0.9.

    Returns list of (gamma, delta) tuples.
    """
    import math
    from scipy.optimize import minimize_scalar

    def logit(p):
        return math.log(p) - math.log(1.0 - p)

    params = []
    for dp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        eps = 1e-12
        lo = max(0.0, -dp) + eps
        hi = min(1.0, 1.0 - dp) - eps
        if lo >= hi:
            continue

        def objective(gamma, _dp=dp):
            if not (0.0 < gamma < 1.0):
                return np.inf
            p_alt = gamma + _dp
            if not (0.0 < p_alt < 1.0):
                return np.inf
            d = logit(p_alt) - logit(gamma)
            if not np.isfinite(d):
                return np.inf
            k = KL_theory(gamma, d)
            return k if np.isfinite(k) else np.inf

        res = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-12, "maxiter": 500})
        if res.success and np.isfinite(res.fun):
            g = float(res.x)
            params.append((g, float(logit(g + dp) - logit(g))))
    return params


def load_klfront_params(csv_path=None):
    """KLFRONT method: (gamma, delta) pairs from optimal_power_results CSV.

    Returns list of (gamma, delta) tuples.
    """
    import pandas as pd
    from pathlib import Path
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / 'optimal_power_results_combined.csv'
    df = pd.read_csv(csv_path).dropna()
    return list(zip(df['gamma'].values, df['delta'].values))
