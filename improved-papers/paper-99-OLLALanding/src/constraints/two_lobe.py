# OLLA_NIPS/src/constraints/two_lobe.py

import numpy as np
import torch
import math


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
    'OLLA-H':    {'step_size': 5e-4, 'alpha': 200.0, 'epsilon': 1.0, 'num_hutchinson_samples': 5, 'proj_damp': 0.0},
    'CLangevin': {'step_size': 5e-4, 'proj_iters': 3, 'proj_tol': 1e-4, 'proj_damp': 1.0},
    'CHMC':      {'step_size': 5e-4, 'gamma': 1.0, 'proj_iters': 3, 'proj_tol': 1e-4, 'proj_damp': 0.0},
    'CGHMC':     {'step_size': 5e-4, 'gamma': 1.0, 'proj_iters': 3, 'proj_tol': 1e-4, 'proj_damp': 0.0},
}

# The dimension of the sampling space.
dim = 2

# --- Problem Definition ---

# The double-moon potential function.
def q_fn(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[0], x[1]
    num = torch.exp(-2.0 * (x1 - 3.0)**2) + torch.exp(-2.0 * (x1 + 3.0)**2)
    den = torch.exp(2.0 * (torch.norm(x) - 3.0)**2)
    return num / den

# The potential for the sampler is uniform over the manifold.
def potential_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(0.0, dtype=x.dtype, device=x.device)

# Inequality: -log(q(x)) <= 2, which defines the sampling domain.
g_fns = [
    lambda x: -torch.log(q_fn(x)) - 2.0
]
h_fns = []

# --- Initial Sample Generation ---

def generate_samples(
    num_samples: int,
    seed: int = 42,
    proposal_radius: float = 6.0
) -> np.ndarray:
    """
    Generates initial samples via rejection sampling.
    """
    rng = np.random.default_rng(seed)
    samples = []
    
    g_ineq = g_fns[0]

    while len(samples) < num_samples:
        # Propose a point from a uniform distribution.
        x_np = rng.uniform(-proposal_radius, proposal_radius, size=2)
        x = torch.tensor(x_np, dtype=torch.float64)
        
        # Accept only if it satisfies the inequality constraint.
        if g_ineq(x) <= 0:
            samples.append(x_np)
            
    return np.array(samples)
