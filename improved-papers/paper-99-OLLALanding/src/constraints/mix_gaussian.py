# OLLA_NIPS/src/constraints/mix_gaussian.py

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
    'CLangevin': {'step_size': 5e-4, 'proj_iters': 3, 'proj_tol': 1e-4, 'proj_damp': 0.1},
    'CHMC':      {'step_size': 5e-4, 'gamma': 1.0, 'proj_iters': 3, 'proj_tol': 1e-4, 'proj_damp': 0.0},
    'CGHMC':     {'step_size': 5e-4, 'gamma': 1.0, 'proj_iters': 3, 'proj_tol': 1e-4, 'proj_damp': 0.0},
}

# The dimension of the sampling space.
dim = 2

# --- Problem Definition ---

# Define the centers for the Gaussian mixture model.
_CENTERS = torch.tensor([
    [dx, dy]
    for dx in [-2.0, 0.0, 2.0]
    for dy in [-2.0, 0.0, 2.0]
], dtype=torch.float64)

# Potential: A mixture of 9 isotropic Gaussians.
def potential_fn(x: torch.Tensor) -> torch.Tensor:
    centers = _CENTERS.to(device=x.device, dtype=x.dtype)
    inv_var = 0.1
    
    # Calculate the log-likelihood for each component.
    diffs = x - centers
    exps = -0.5 * inv_var * diffs.pow(2).sum(dim=1)
    
    # The potential is the negative log of the summed likelihoods.
    return -torch.logsumexp(exps, dim=0)

# Equality: A 7-lobed star shape.
def h_star(x: torch.Tensor) -> torch.Tensor:
    k, a = 7.0, 1.0
    theta = torch.atan2(x[1], x[0])
    r_boundary = 3.0 + a * torch.cos(k * theta)
    return x.norm() - r_boundary

h_fns = [h_star]

# Inequality: A polynomial constraint.
g_fns = [
    lambda x: (x[0] - 2.0)**2 - 5*x[0]*(x[1]**3) + 0.5*x[1]**5 - 40.0
]


# --- Initial Sample Generation ---

def generate_samples(
    num_samples: int,
    seed: int = 0,
    batch_multiplier: int = 5
) -> np.ndarray:
    """
    Generates initial samples via rejection sampling on the star boundary.
    """
    rng = np.random.default_rng(seed)
    samples = []
    
    g_ineq = g_fns[0]
    
    # Oversample to account for rejections.
    while len(samples) < num_samples:
        needed = (num_samples - len(samples)) * batch_multiplier
        
        # 1. Propose points on the star-shaped equality boundary.
        theta = rng.uniform(0, 2 * math.pi, size=needed)
        k, a = 7.0, 1.0
        r = 3.0 + a * np.cos(k * theta)
        
        x_coords = r * np.cos(theta)
        y_coords = r * np.sin(theta)
        
        proposals = np.column_stack([x_coords, y_coords])
        
        # 2. Accept only those that satisfy the inequality.
        for p_np in proposals:
            p_torch = torch.tensor(p_np, dtype=torch.float64)
            if g_ineq(p_torch) <= 0:
                samples.append(p_np)
    
    return np.array(samples[:num_samples])
