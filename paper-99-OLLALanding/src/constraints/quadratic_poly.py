# OLLA_NIPS/src/constraints/x4y2_x3y3.py

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
    'OLLA-H':    {'step_size': 5e-4, 'alpha': 200.0, 'epsilon': 1.0, 'num_hutchinson_samples': 5, 'proj_damp': 0.0},
    'CLangevin': {'step_size': 5e-4, 'proj_iters': 3, 'proj_tol': 1e-4, 'proj_damp': 0.1},
    'CHMC':      {'step_size': 5e-4, 'gamma': 1.0, 'proj_iters': 3, 'proj_tol': 1e-4, 'proj_damp': 0.0},
    'CGHMC':     {'step_size': 5e-4, 'gamma': 1.0, 'proj_iters': 3, 'proj_tol': 1e-4, 'proj_damp': 0.0},
}
# The dimension of the sampling space.
dim = 2

# --- Problem Definition ---

# Potential: Negative log-density of a standard Gaussian.
def potential_fn(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x**2).sum()

# Equality: x^4*y^2 + x^2 + y - 1 = 0.
h_fns = [
    lambda x: x[0]**4 * x[1]**2 + x[0]**2 + x[1] - 1.0
]

# Inequality: x^3 - y^3 - 1 <= 0.
g_fns = [
    lambda x: x[0]**3 - x[1]**3 - 1.0
]

# --- Initial Sample Generation ---

def generate_samples(
    num_samples: int,
    seed: int = 42,
    x_min: float = -2.0,
    x_max: float = 2.0,
    batch_size: int = 10000
) -> np.ndarray:
    """
    Generates initial samples by solving the equality constraint as a
    quadratic equation in y for randomly sampled x values.
    """
    rng = np.random.default_rng(seed)
    samples = []

    while len(samples) < num_samples:
        # 1. Sample a batch of x coordinates.
        xs = rng.uniform(x_min, x_max, size=batch_size)

        # 2. Define quadratic coefficients a*y^2 + b*y + c = 0.
        a = xs**4
        b = 1.0
        c = xs**2 - 1.0

        # 3. Find real roots using the discriminant.
        discriminant = b**2 - 4 * a * c
        valid_mask = discriminant >= 0
        
        if not np.any(valid_mask):
            continue

        # Use only x values that yield real roots.
        xs_valid = xs[valid_mask]
        a_valid = a[valid_mask]
        disc_valid = discriminant[valid_mask]

        # 4. Calculate the two possible y roots.
        sqrt_disc = np.sqrt(disc_valid)
        y1 = (-b + sqrt_disc) / (2 * a_valid)
        y2 = (-b - sqrt_disc) / (2 * a_valid)

        # 5. Check which (x, y) pairs satisfy the inequality.
        for x_val, y_val in zip(np.tile(xs_valid, 2), np.concatenate([y1, y2])):
            if x_val**3 - y_val**3 - 1.0 <= 0:
                samples.append([x_val, y_val])
    
    return np.array(samples[:num_samples])
