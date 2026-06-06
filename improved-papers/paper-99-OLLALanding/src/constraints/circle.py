# OLLA_NIPS/src/constraints/circle.py

import numpy as np
import torch

# --- Default Experiment Settings ---
# These settings control the runner script.
EXPERIMENT_SETTINGS = {
    'n_steps': 1000,
    'n_particles': 200,      # 2D problems run with multiple particles
    'is_single_chain': False,# Differentiates analysis style (no burn-in/thinning)
}

# --- Sampler Hyperparameters ---
# Hyperparameters for each sampler, tailored for this specific problem.
SAMPLER_SETTINGS = {
    'OLLA':      {'step_size': 5e-4, 'alpha': 200.0, 'epsilon': 1.0, 'proj_damp': 0.0},
    'OLLA-H':    {'step_size': 5e-4, 'alpha': 100.0, 'epsilon': 1.0, 'num_hutchinson_samples': 5, 'proj_damp': 0.0},
    'CLangevin': {'step_size': 5e-4, 'proj_iters': 3, 'proj_tol': 1e-4, 'proj_damp': 0.1},
    'CHMC':      {'step_size': 5e-4, 'gamma': 1.0, 'proj_iters': 3, 'proj_tol': 1e-4, 'proj_damp': 0.0},
    'CGHMC':     {'step_size': 5e-4, 'gamma': 1.0, 'proj_iters': 3, 'proj_tol': 1e-4, 'proj_damp': 0.0},
}

# The dimension of the sampling space.
dim = 2

# --- Problem Definition ---

def potential_fn(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x.pow(2).sum()

# Equality constraint function h(x) = 0.
# This forces samples onto the unit circle: x^2 + y^2 - 1 = 0.
h_fns = [
    lambda x: x.pow(2).sum() - 1.0
]

# No inequality constraints for this problem.
g_fns = []

# --- Initial Sample Generation ---

def generate_samples(num_samples: int, seed: int = 42) -> np.ndarray:
    """
    Generates initial samples uniformly on the unit circle.
    """
    rng = np.random.default_rng(seed)
    
    # Generate random angles and map them to points on the unit circle.
    theta = rng.uniform(0, 2 * np.pi, num_samples)
    x = np.cos(theta)
    y = np.sin(theta)
    
    return np.stack([x, y], axis=1)

