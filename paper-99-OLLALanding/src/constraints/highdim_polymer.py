# OLLA_NIPS/src/constraints/highdim_polymer.py

import numpy as np
import torch

from torch.func import grad

# --- Default Experiment Settings ---
EXPERIMENT_SETTINGS = {
    'n_steps': 5000,
    'n_particles': 1,
    'is_single_chain': True, # Use burn-in and thinning for analysis
}

# --- Sampler Hyperparameters ---
SAMPLER_SETTINGS = {
    'OLLA':      {'step_size': 1e-5, 'alpha': 500.0, 'epsilon': 1.0, 'proj_damp': 0.0},
    'OLLA-H':    {'step_size': 1e-5, 'alpha': 500.0, 'epsilon': 1.0, 'num_hutchinson_samples': 0, 'proj_damp': 0.0},
    'CLangevin': {'step_size': 1e-5, 'proj_iters': 30, 'proj_tol': 1e-4, 'proj_damp': 0.5},
    'CHMC':      {'step_size': 1e-5, 'gamma': 1.0, 'proj_iters': 30, 'proj_tol': 1e-4, 'proj_damp': 0.0},
    'CGHMC':     {'step_size': 1e-5, 'gamma': 1.0, 'proj_iters': 30, 'proj_tol': 1e-4, 'proj_damp': 0.0},
}

# --- Globals for the Current Problem ---
# These are set by the build_constraints function.
dim = None
h_fns = []
g_fns = []
potential_fn = None
N_ATOMS = 10  # Default number of atoms

# --- Physics-Based Potential Function ---

class PolymerPotentialParams:
    """Configurable parameters for the polymer's potential energy."""
    def __init__(
        self,
        beta: float = 1.0,
        torsion_ms=(1, 3),
        torsion_k=(0.5, 0.2),
        torsion_delta=(0.0, 0.0),
        wca_eps: float = 1.0,
        wca_sigma = None, # Derived from steric_min if None
    ):
        self.beta = beta
        self.torsion_ms = tuple(torsion_ms)
        self.torsion_k = torch.tensor(torsion_k, dtype=torch.float64)
        self.torsion_delta = torch.tensor(torsion_delta, dtype=torch.float64)
        self.wca_eps = float(wca_eps)
        self.wca_sigma = wca_sigma        
        
def _dihedrals(P: torch.Tensor) -> torch.Tensor:
    """Calculates dihedral angles for a chain of atoms. P shape: (N, 3)"""
    b1 = P[1:-2, :] - P[0:-3, :]
    b2 = P[2:-1, :] - P[1:-2, :]
    b3 = P[3:  , :] - P[2:-1, :]
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    b2n = b2 / torch.clamp(b2.norm(dim=-1, keepdim=True), min=1e-12)
    x = (n1 * n2).sum(dim=-1)
    y = (torch.cross(n1, n2, dim=-1) * b2n).sum(dim=-1)
    return torch.atan2(y, x) # shape (N-3,)

def energy_fn(p: torch.Tensor, n_atoms: int, r_min: float, params: PolymerPotentialParams) -> torch.Tensor:
    """Calculates the total potential energy of a single polymer configuration."""
    P = p.view(n_atoms, 3)
    
    # Torsion energy
    phi = _dihedrals(P)
    U_tor = torch.tensor(0.0, device=p.device, dtype=p.dtype)
    if phi.numel() > 0 and len(params.torsion_ms) > 0:
        ms = torch.as_tensor(params.torsion_ms, dtype=p.dtype, device=p.device)
        km = params.torsion_k.to(p.device)
        dm = params.torsion_delta.to(p.device)
        # phi (N-3), ms (M) -> val (N-3, M)
        val = 1.0 + torch.cos(phi.unsqueeze(-1) * ms - dm)
        U_tor = (km * val).sum()

    # WCA non-bonded energy (|i-j| > 2)
    diff = P.unsqueeze(1) - P.unsqueeze(0)
    D = torch.sqrt(torch.clamp((diff * diff).sum(dim=-1), min=1e-12))
    
    mask = torch.ones((n_atoms, n_atoms), dtype=torch.bool, device=p.device)
    mask = torch.triu(mask, diagonal=3) # only pairs |i-j| > 2
    
    sigma = (r_min) / (2.0**(1.0/6.0)) if params.wca_sigma is None else params.wca_sigma
    rc = (2.0**(1.0/6.0)) * sigma
    
    Rij = D[mask]
    sr = (sigma / torch.clamp(Rij, min=1e-12))
    sr6, sr12 = sr**6, sr**12
    U_LJ = 4.0 * params.wca_eps * (sr12 - sr6) + params.wca_eps

    U_nb = torch.where(Rij < rc, U_LJ, torch.zeros_like(U_LJ)).sum()

    return U_tor + U_nb 

# --- Dynamic Problem Builder ---

def build_constraints(
    n_atoms: int = 10,
    bond_len: float = 1.0,
    angle_rad: float = 109.5 * np.pi/180 ,
    steric_min: float = 1.0,
    potential_params: PolymerPotentialParams = None
):
    """Dynamically builds and sets the constraint and potential functions."""
    global dim, h_fns, g_fns, potential_fn, N_ATOMS
    
    dim = 3 * n_atoms
    N_ATOMS = n_atoms

    # --- Equality Constraints ---
    h_list = []
    for k in range(n_atoms - 1):
        def h_bond_k(p: torch.Tensor, k=k):
            P = p.view(n_atoms, 3)
            return (P[k] - P[k+1]).pow(2).sum() - bond_len**2
        h_list.append(h_bond_k)

    for k in range(1, n_atoms - 1):
        def h_angle_norm_k(p: torch.Tensor, k=k):
            P = p.view(n_atoms, 3)
            v1 = P[k-1] - P[k]
            v2 = P[k+1] - P[k]
            n1 = torch.clamp(v1.norm(), min=1e-12)
            n2 = torch.clamp(v2.norm(), min=1e-12)
            cos_val = (v1 @ v2) / (n1 * n2)
            return cos_val - np.cos(angle_rad)
        h_list.append(h_angle_norm_k)
    h_fns = h_list

    # --- Inequality Constraints ---
    g_list = []
    for i in range(n_atoms):
        for j in range(i + 2, n_atoms):
            def g_steric_ij(p: torch.Tensor, i=i, j=j):
                P = p.view(n_atoms, 3)
                return steric_min**2 - (P[i] - P[j]).pow(2).sum()
            g_list.append(g_steric_ij)
    g_fns = g_list

    # --- Set the Potential Function ---
    if potential_params is None:
        potential_params = PolymerPotentialParams()
        
    def final_potential_fn(p: torch.Tensor) -> torch.Tensor:
        U = energy_fn(p, n_atoms, steric_min, potential_params)
        return potential_params.beta * U
    potential_fn = final_potential_fn

# --- Initial Sample Generation ---

def generate_samples(num_samples: int, seed: int = 0) -> np.ndarray:
    """Generates initial samples via a zig-zag start and SHAKE projection."""
    rng = np.random.default_rng(seed)
    n_atoms = N_ATOMS
    
    # Initial zig-zag guess
    base = np.zeros((n_atoms, 3))
    for k in range(1, n_atoms):
        base[k, 0] = base[k-1, 0] + 1.0
        base[k, 1] = ( -1 if k%2==0 else 1 ) * 0.2 
    
    samples = []
    while len(samples) < num_samples:
        p0 = base + 0.01 * rng.standard_normal((n_atoms, 3))
        p = torch.tensor(p0.flatten(), dtype=torch.float64)
        
        # SHAKE projection
        for _ in range(10): 
            vals = torch.stack([fn(p) for fn in h_fns])
            if vals.abs().max() < 1e-6: break
            grads = torch.stack([grad(fn)(p) for fn in h_fns])
            A = grads @ grads.T
            # Use lstsq for stability
            lam = torch.linalg.lstsq(A, vals).solution
            corr = grads.T @ lam
            p = p - corr
        samples.append(p.detach().numpy())
        
    return np.stack(samples)

