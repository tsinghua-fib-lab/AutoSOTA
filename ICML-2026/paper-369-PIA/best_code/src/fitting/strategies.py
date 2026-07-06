"""
Fitting Strategy Configuration
Defines parameter constraint strategies for different scenario groups

Design principles:
1. Baseline = weakest assumptions
2. Each strategy touches at most ONE cognitive layer:
   - Learning: alpha_*
   - Utility / perception: rho, lambda_LA, R_perc
   - Decision: theta, beta, lambda
3. All 9 parameters are always available (alpha_pos, alpha_neg, rho, theta,
   lambda, phi, beta, R_perc, lambda_LA)
4. Strategy defines which parameters are 'free' vs 'fixed'
5. No strategy hard-codes conclusions
"""
from typing import List

# ---------------------------------------------------------------------
# Strategy mapping: strategy name → parameters to fit
# ---------------------------------------------------------------------
# Note: All strategies now work with the full 9-parameter model:
# alpha_pos, alpha_neg, rho, theta, lambda, phi, beta, R_perc, lambda_LA
# The strategy defines which parameters are 'free' to be optimized.
# ---------------------------------------------------------------------
STRATEGY_MAP = {
    # Weak baseline: decision noise / bias / memory
    'BASELINE': ['lambda', 'beta', 'phi', 'theta'],

    # Learning-focused manipulations
    'OPTIMISM': ['alpha_pos', 'alpha_neg', 'rho', 'R_perc', 'theta', 'lambda'],
    'PUNISHMENT': ['alpha_pos', 'alpha_neg', 'rho', 'R_perc', 'theta', 'lambda'],
    'REGRET': ['alpha_pos', 'alpha_neg', 'rho', 'R_perc', 'theta', 'lambda'],
    'MAGNITUDE': ['alpha_pos', 'alpha_neg', 'rho', 'R_perc', 'theta', 'lambda'],

    # Utility curvature (value sensitivity)
    'STIMULUS': ['alpha_pos', 'alpha_neg', 'rho', 'R_perc', 'theta', 'lambda'],

    # Decision bias (not learning)
    'THREAT': ['alpha_pos', 'alpha_neg', 'rho', 'R_perc', 'theta', 'lambda'],
    'AUTHORITY': ['alpha_pos', 'alpha_neg', 'rho', 'R_perc', 'theta', 'lambda'],
    # 'SYCOPHANCY': ['theta'],
}


# ---------------------------------------------------------------------
# Strategy selector
# ---------------------------------------------------------------------
def get_strategy_for_group(group_name: str) -> str:
    """
    Automatically select fitting strategy based on scenario group name

    Each strategy defines which parameters are 'free' to be optimized:
    - alpha_pos, alpha_neg: Learning rates
    - rho: Risk preference
    - theta: Decision bias
    - lambda: Behavioral inertia
    - phi: Memory decay
    - beta: Choice sensitivity
    - R_perc: Perceptual reward amplification
    - lambda_LA: Loss aversion
    """

    name = group_name.lower()

    if 'baseline' in name:
        return 'BASELINE'
    elif 'punishment' in name:
        return 'PUNISHMENT'
    elif 'optimism' in name:
        return 'OPTIMISM'
    elif 'stimulus' in name:
        return 'STIMULUS'
    elif 'magnitude' in name:
        return 'MAGNITUDE'
    elif 'threat' in name:
        return 'THREAT'
    elif 'authority' in name:
        return 'AUTHORITY'
    elif 'sycophancy' in name:
        return 'SYCOPHANCY'
    elif 'regret' in name:
        return 'REGRET'
    else:
        # Safe fallback: weakest model
        return 'BASELINE'
