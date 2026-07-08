"""
Fitting utilities for RCGP models.

This module provides various fitting strategies for GP models, including
standard MLL fitting and robust WLOO-CV fitting.
"""

from .wloo_mll import WeightedRobustLeaveOneOutMLL, RobustLeaveOneOutMLL
from .rcgp_wloo import (
    fit_rcgp_wloo,
    calculate_robust_heuristics,
    create_constant_center_fn,
    extract_parameters
)
from .scipy_optimizer import optimize_with_scipy_lbfgs

__all__ = [
    'WeightedRobustLeaveOneOutMLL',
    'RobustLeaveOneOutMLL',
    'fit_rcgp_wloo',
    'calculate_robust_heuristics', 
    'create_constant_center_fn',
    'extract_parameters',
    'optimize_with_scipy_lbfgs'
]