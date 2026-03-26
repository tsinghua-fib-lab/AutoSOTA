# OLLA_NIPS/src/constraints/highdim_stress.py

import numpy as np
import torch

# --- Default Experiment Settings ---
EXPERIMENT_SETTINGS = {
    'n_steps': 1000,
    'n_particles': 1,
    'is_single_chain': True,
}

# --- Sampler Hyperparameters ---
SAMPLER_SETTINGS = {
    'OLLA':      {'step_size': 1e-2, 'alpha': 200.0, 'epsilon': 1.0, 'proj_damp': 0.0},
    'OLLA-H':    {'step_size': 1e-2, 'alpha': 200.0, 'epsilon': 1.0, 'num_hutchinson_samples': 5, 'proj_damp': 0.0},
    'CLangevin': {'step_size': 1e-2, 'proj_iters': 5, 'proj_tol': 1e-4, 'proj_damp': 0.1},
    'CHMC':      {'step_size': 1e-2, 'gamma': 1.0, 'proj_iters': 5, 'proj_tol': 1e-4, 'proj_damp': 0.0},
    'CGHMC':     {'step_size': 1e-2, 'gamma': 1.0, 'proj_iters': 5, 'proj_tol': 1e-4, 'proj_damp': 0.0},
}

# --- Globals for the Current Problem ---
# These are set by the build_constraints function.
dim = None
h_fns = []
g_fns = []
potential_fn = None

# Store constraint data for the sampler
A_mat, b_vec, c_mat, r_val = None, None, None, None

# --- Potential Function ---
# Uniform potential on the feasible set.
def uniform_potential_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(0.0, dtype=x.dtype, device=x.device)

# --- Dynamic Problem Builder ---
def build_constraints(
    n_dim: int = 10,
    n_eq: int = 10,  # Number of equalities, INCLUDING the sphere
    n_ineq: int = 10,
    sphere_radius: float = 5.0,
    obstacle_radius: float = 1.0,
    seed: int = 2025
):
    """Dynamically builds and sets the constraint and potential functions."""
    global dim, h_fns, g_fns, potential_fn, A_mat, b_vec, c_mat, r_val
    
    dim = n_dim
    rng = np.random.default_rng(seed)
    
    # --- Generate Constraint Data ---
    # Linear equalities (A @ x = b)
    A = torch.tensor(rng.standard_normal((n_eq - 1, n_dim)), dtype=torch.float64)
    b = torch.tensor(rng.standard_normal(n_eq - 1) * 0.1, dtype=torch.float64)
    
    # Spherical inequality obstacles (|x - c_j|^2 >= r^2)
    # Scale centers to be within the main sphere for a non-trivial problem
    c_scale = (sphere_radius / 2)**0.5
    c = torch.tensor(rng.standard_normal((n_ineq, n_dim)) * c_scale, dtype=torch.float64)
    r = obstacle_radius
    
    # Store for the sampler
    A_mat, b_vec, c_mat, r_val = A, b, c, r

    # --- Equality Constraints, h(x) = 0 ---
    h_list = [lambda x, i=i: A[i] @ x - b[i] for i in range(n_eq - 1)]
    h_list.append(lambda x: x @ x - sphere_radius**2)
    h_fns = h_list

    # --- Inequality Constraints, g(x) <= 0 ---
    g_fns = [lambda x, j=j: r**2 - (x - c[j]) @ (x - c[j]) for j in range(n_ineq)]
    
    potential_fn = uniform_potential_fn

# --- Initial Sample Generation ---
def generate_samples(num_samples: int, seed: int = 0) -> np.ndarray:
    """
    Generates initial samples via nullspace parameterization and rejection.
    Ensures samples satisfy linear and spherical equalities, then checks obstacles.
    """
    rng = np.random.default_rng(seed)
    
    # Use the globally configured constraint data
    A, b, c, r = A_mat, b_vec, c_mat, r_val
    sphere_radius = np.sqrt(h_fns[-1](torch.zeros(dim)).abs().item()) # Infer from h_fn

    # Find a particular solution to A @ x = b
    xp = torch.linalg.lstsq(A, b).solution
    
    # Find the nullspace of A
    _, S, Vh = torch.linalg.svd(A, full_matrices=True)
    rank = (S.abs() > 1e-8).sum().item()
    N = Vh[rank:].T  # Basis for the nullspace
    
    samples = []
    while len(samples) < num_samples:
        # 1. Sample in the affine subspace satisfying A @ x = b
        z = torch.tensor(rng.standard_normal(N.shape[1]), dtype=torch.float64)
        x = xp + N @ z
        
        # 2. Project exactly onto the sphere
        x = x * (sphere_radius / torch.norm(x))
        
        # 3. Reject if inside any obstacle
        is_valid = True
        for g_fn in g_fns:
            if g_fn(x) > 0:
                is_valid = False
                break
        
        if is_valid:
            samples.append(x.numpy())
            
    return np.stack(samples, axis=0)

# Build a default problem upon module import
build_constraints()

