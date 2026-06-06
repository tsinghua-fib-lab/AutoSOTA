"""
Sampling Methods Module (patched for rpy2 >= 3.5, fixed cube arg order)
"""

import numpy as np
from typing import Tuple, Optional
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

# Import R BalancedSampling package
try:
    BalancedSampling = importr('BalancedSampling')
    cube = BalancedSampling.cube
except Exception as e:
    print(f"Warning: Could not import BalancedSampling package: {e}")
    cube = None


def uniform_poisson_sampling(
    N: int,
    budget: float,
    random_state=None
) -> Tuple[np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)
    inclusion_probs = np.full(N, budget)
    sampling_indicators = np.random.binomial(1, inclusion_probs)
    return sampling_indicators, inclusion_probs


def active_poisson_sampling(
    uncertainty: np.ndarray,
    budget: float,
    tau: float = 0.5,
    random_state=None
) -> Tuple[np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)
    N = len(uncertainty)
    mean_uncertainty = uncertainty.mean()
    inclusion_probs = (
        tau * (uncertainty / mean_uncertainty) * budget +
        (1.0 - tau) * budget
    )
    inclusion_probs = np.minimum(inclusion_probs, 1.0)
    sampling_indicators = np.random.binomial(1, inclusion_probs)
    return sampling_indicators, inclusion_probs


def cube_active_sampling(
    auxiliary_vars: np.ndarray,
    inclusion_probs: np.ndarray,
    random_state=None
) -> Tuple[np.ndarray, np.ndarray]:
    if cube is None:
        raise RuntimeError("BalancedSampling package not available.")
    if random_state is not None:
        np.random.seed(random_state)
    N = len(inclusion_probs)
    if auxiliary_vars.ndim == 1:
        auxiliary_vars = auxiliary_vars.reshape(-1, 1)
    # Use new rpy2 conversion context
    # Correct call: cube(prob, x) -- prob first, then balancing variables
    with (ro.default_converter + numpy2ri.converter).context():
        cube_result = cube(inclusion_probs, auxiliary_vars)
        selected_indices = np.array(cube_result).astype(int) - 1
    sampling_indicators = np.zeros(N, dtype=int)
    valid_idx = selected_indices[(selected_indices >= 0) & (selected_indices < N)]
    sampling_indicators[valid_idx] = 1
    return sampling_indicators, inclusion_probs


def classical_simple_random_sampling(
    N: int,
    budget: float,
    random_state=None
) -> Tuple[np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)
    inclusion_probs = np.full(N, budget)
    sampling_indicators = np.random.binomial(1, inclusion_probs)
    return sampling_indicators, inclusion_probs


def compute_sampling_probabilities(
    uncertainty: np.ndarray,
    budget: float,
    tau: float = 0.5
) -> np.ndarray:
    mean_uncertainty = uncertainty.mean()
    inclusion_probs = (
        tau * (uncertainty / mean_uncertainty) * budget +
        (1.0 - tau) * budget
    )
    inclusion_probs = np.minimum(inclusion_probs, 1.0)
    return inclusion_probs
