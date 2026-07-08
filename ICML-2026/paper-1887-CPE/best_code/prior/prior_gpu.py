import torch
from typing import Optional
from typing import Tuple
import numpy as np

_EPS = 1e-12


def make_prior_particles_from_truth(
    W_star: torch.Tensor,
    S: int,
    flip_prob: float,
    add_remove_prob: float,
    weight_noise: float,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    generator: Optional[torch.Generator] = None,
    extra_edge_prob: Optional[float] = None,
) -> torch.Tensor:
    """
    Super-efficient GPU prior sampler around truth.

    Design goals:
      - DAG guaranteed (by construction): all edges follow a per-particle random order
      - Very fast for S~1000, D~200
      - Memory disciplined: structure is sampled in upper-triangular edge-list form (S,E)

    Semantics:
      - Start from truth *skeleton* (undirected)
      - Randomly drop some truth skeleton edges (flip_prob used as "drop" knob)
      - Randomly remove some remaining edges (add_remove_prob)
      - Randomly add new undirected edges (p_add)
      - Orient all undirected edges according to the particle’s random order (DAG-safe)
      - Weights:
          * if the edge corresponds to a truth skeleton edge: use the truth weight magnitude (+ noise)
          * if it's newly added: draw N(0.4, 0.2)

    Returns:
      W: (S,D,D) dense float tensor on device.
    """
    if S <= 0:
        raise ValueError("S must be positive")
    if not (0.0 <= flip_prob <= 1.0):
        raise ValueError("flip_prob must be in [0,1]")
    if not (0.0 <= add_remove_prob <= 1.0):
        raise ValueError("add_remove_prob must be in [0,1]")

    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W_star = torch.as_tensor(W_star, device=device, dtype=dtype)
    D = int(W_star.shape[0])

    # Truth skeleton (undirected presence)
    A0_dir = (W_star.abs() > _EPS)
    eye = torch.eye(D, device=device, dtype=torch.bool)
    A0_dir = A0_dir & ~eye
    A0_skel = (A0_dir | A0_dir.transpose(0, 1)) & ~eye  # (D,D) bool

    # Upper-triangular edge list indices (u < v)
    triu = torch.triu_indices(D, D, offset=1, device=device)  # (2,E)
    u_idx, v_idx = triu[0], triu[1]
    E = int(u_idx.numel())

    # Per-particle random order via ranks -> permutation (fast on GPU)
    # perm[s, k] = node index in position k of topo order
    ranks = torch.rand((S, D), device=device, generator=generator, dtype=torch.float32)
    perm = torch.argsort(ranks, dim=1)  # (S,D), int64

    # For each particle, map each upper-tri position (u,v) to original node labels (pi,pj)
    pi = perm[:, u_idx]  # (S,E)
    pj = perm[:, v_idx]  # (S,E)

    # Truth skeleton membership for these unordered pairs in original labels
    skel0 = A0_skel[pi, pj]  # (S,E) bool

    # Start from truth skeleton edges (upper-tri representation)
    edges = skel0.clone()  # (S,E) bool

    # "flip_prob" used as a cheap structural perturbation knob: drop some truth edges
    if flip_prob > 0.0 and edges.any():
        drop = (torch.rand((S, E), device=device, generator=generator) < flip_prob) & edges
        edges = edges & ~drop

    # remove edges (toggle off)
    if add_remove_prob > 0.0 and edges.any():
        rem = (torch.rand((S, E), device=device, generator=generator) < add_remove_prob) & edges
        edges = edges & ~rem

    # add new edges (toggle on) among currently absent pairs
    if add_remove_prob > 0.0:
        p_add = extra_edge_prob
        if p_add is None:
            # conservative default to avoid exploding density for large D
            # (Expected added edges per node stays O(1))
            p_add = min(add_remove_prob, 2.0 / max(D, 1))

        can_add = ~edges
        add = (torch.rand((S, E), device=device, generator=generator) < float(p_add)) & can_add
        edges = edges | add

    # Allocate final dense weight tensor (this is the big unavoidable output)
    W = torch.zeros((S, D, D), device=device, dtype=dtype)

    # Classify which sampled edges correspond to truth skeleton vs new
    overlap = edges & skel0          # (S,E)
    extra = edges & ~skel0           # (S,E)

    # For overlap edges: use truth weight magnitude (regardless of original direction), plus optional noise
    if overlap.any():
        w_ij = W_star[pi, pj]
        w_ji = W_star[pj, pi]
        base = torch.where(w_ij.abs() > _EPS, w_ij, w_ji)  # (S,E)

        if weight_noise > 0.0:
            noise = torch.randn((S, E), device=device, dtype=dtype, generator=generator) * float(weight_noise)
            base = base.to(dtype) + noise
        else:
            base = base.to(dtype)

        se = overlap.nonzero(as_tuple=False)  # (N,2): columns [s,e]
        s_idx = se[:, 0]
        e_idx = se[:, 1]
        W[s_idx, pi[s_idx, e_idx], pj[s_idx, e_idx]] = base[s_idx, e_idx]

    # For extra edges: N(0.4, 0.2)
    if extra.any():
        extra_w = torch.randn((S, E), device=device, dtype=dtype, generator=generator) * 0.2 + 0.4
        se = extra.nonzero(as_tuple=False)
        s_idx = se[:, 0]
        e_idx = se[:, 1]
        W[s_idx, pi[s_idx, e_idx], pj[s_idx, e_idx]] = extra_w[s_idx, e_idx]

    return W



def _sample_weights_torch(
    A: torch.Tensor,
    w_scale: float,
    *,
    dtype: torch.dtype = torch.float64,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Torch analogue of `sample_weights`.

    Args:
        A: (D,D) adjacency with 0/1 (bool/int).
        w_scale: std-dev of Normal(0,w_scale).
        dtype: output dtype (default float64).
        generator: optional torch.Generator for determinism.

    Returns:
        W: (D,D) weight matrix with zeros where A==0.
    """
    if w_scale < 0:
        raise ValueError("w_scale must be non-negative")

    device = A.device
    mask = A.to(torch.bool)
    W = torch.zeros(A.shape, device=device, dtype=dtype)

    if w_scale == 0.0:
        sampled = torch.zeros_like(W)
    else:
        sampled = torch.randn(W.shape, device=device, dtype=dtype, generator=generator) * w_scale

    # Nudge away from zero so log(eps+|W|) is informative; sign(0)=0.
    sampled = sampled + torch.sign(sampled) * 0.5
    W[mask] = sampled[mask]
    return W

def load_prior_particles_npz(
    path: str,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Load a prior from a .npz with arrays: particles (S,D,D) and optional weights (S,).

    Returns:
        particles: (S,D,D) torch tensor
        weights: optional (S,) torch tensor
    """
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(path, allow_pickle=False)

    particles_np = data["particles"]
    particles = torch.as_tensor(particles_np, device=device, dtype=dtype).contiguous()

    weights = None
    if "weights" in data.files:
        weights = torch.as_tensor(data["weights"], device=device, dtype=dtype).contiguous()

    return particles, weights


def sparse_prior_logprob(W: torch.Tensor, lam_sparsity: float = 2.0) -> torch.Tensor:
    """Rough log prior preferring sparse graphs.

    Torch version of the baseline:
        -lam_sparsity * sum(|W| > 1e-12)

    Returns:
        scalar tensor on the same device as W.
    """
    return -float(lam_sparsity) * (torch.abs(W) > 1e-12).to(torch.float64).sum()
