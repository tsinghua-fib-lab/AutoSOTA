from typing import List, Tuple, Optional
import itertools
import numpy as np
from dag.dag_ops import sample_weights
from dag.dag_ops import would_create_cycle

# ---------------------- Priors / q0 construction -------------------------

def make_prior_particles_from_truth(
    W_star: np.ndarray,
    S: int,
    flip_prob: float,
    add_remove_prob: float,
    weight_noise: float,
    rng: np.random.Generator,
) -> List[np.ndarray]:
    """Noisy neighborhood around truth (for demo). Replace with your q0(W|X) in real use."""
    D = W_star.shape[0]
    particles = []
    for _ in range(S):
        A = (np.abs(W_star) > 1e-12).astype(int).copy()
        # Orient flips (cycle-safe)
        for i, j in itertools.permutations(range(D), 2):
            if A[i, j] == 1 and rng.random() < flip_prob:
                A[i, j] = 0
                if not would_create_cycle(A, j, i):
                    A[j, i] = 1
                else:
                    A[i, j] = 1
        # Add/remove edges (cycle-safe)
        for i, j in itertools.permutations(range(D), 2):
            if i == j:
                continue
            if rng.random() < add_remove_prob:
                if A[i, j] == 1:
                    A[i, j] = 0
                else:
                    if not would_create_cycle(A, i, j):
                        A[i, j] = 1
        # Weights
        W = sample_weights(A, w_scale=1.0, rng=rng)
        # Noise where truth has edge
        mask_overlap = np.logical_and(A == 1, np.abs(W_star) > 1e-12)
        if int(mask_overlap.sum()) > 0:
            W[mask_overlap] = W_star[mask_overlap] + rng.normal(0.0, weight_noise, size=int(mask_overlap.sum()))
        # New edges not in truth
        mask_extra = np.logical_and(A == 1, np.abs(W_star) <= 1e-12)
        if int(mask_extra.sum()) > 0:
            W[mask_extra] = rng.normal(0.4, 0.2, size=int(mask_extra.sum()))
        particles.append(W)
    return particles

def load_prior_particles_npz(path: str) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """Load your prior from a .npz file with arrays: particles (S,D,D) and optional weights (S,)."""
    data = np.load(path, allow_pickle=False)
    particles = [data["particles"][s].copy() for s in range(data["particles"].shape[0])]
    weights = data["weights"] if "weights" in data.files else None
    return particles, weights

def sparse_prior_logprob(W, lam_sparsity=2.0):
    """Rough log prior preferring sparse graphs.
    TODO: Need to estimate actual log density of current posterior """
    return -lam_sparsity * np.sum(np.abs(W) > 1e-12)

