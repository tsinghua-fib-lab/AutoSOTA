from __future__ import annotations
import numpy as np
import torch
from typing import Optional
from utils.utils_gpu import adjacency_from_weights

@torch.no_grad()
def adj_from_W(W):
    if isinstance(W, np.ndarray):
        return (np.abs(W) > 1e-12).astype(int)
    return (torch.abs(W) > 1e-12).to(dtype=torch.int64)

@torch.no_grad()
def skel_from_adj(A):
    if isinstance(A, np.ndarray):
        return ((A + A.T) > 0).astype(int)
    return ((A + A.transpose(-1, -2)) > 0).to(dtype=torch.int64)

@torch.no_grad()
def reachable(A: np.ndarray) -> np.ndarray:
    """Transitive closure (CPU numpy), kept for correctness.
    GPU batching for transitive closure is possible but is rarely the runtime bottleneck vs. EIG scoring.
    """
    D = A.shape[0]
    R = (A > 0).astype(int).copy()
    for k in range(D):
        for i in range(D):
            if R[i, k]:
                R[i, :] = np.maximum(R[i, :], R[k, :])
    return R

def would_create_cycle(A: np.ndarray, i: int, j: int) -> bool:
    if A[j, i] == 1:
        return True
    R = reachable(A)
    return bool(R[j, i])

def is_acyclic(W):
    # A = (np.abs(W) > 1e-12).astype(int)
    A = adjacency_from_weights(W)

    D = A.shape[0]
    visited = [0]*D
    recstack = [0]*D

    def visit(v):
        visited[v] = 1
        recstack[v] = 1
        for u in range(D):
            if A[v,u]:
                if not visited[u] and visit(u):
                    return True
                elif recstack[u]:
                    return True
        recstack[v] = 0
        return False

    for node in range(D):
        if not visited[node]:
            if visit(node):
                return False
    return True


def random_dag(
    D: int,
    edge_prob: float,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.int64,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Sample a DAG by random permutation and forward edges (GPU-friendly).

    Returns:
        A: (D, D) tensor with 0/1 entries (dtype int64 by default).
    """
    if not (0.0 <= edge_prob <= 1.0):
        raise ValueError("edge_prob must be in [0, 1].")

    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if generator is None:
        generator = torch.Generator(device="cpu")  # torch.Generator is CPU; randomness still works for CUDA tensors

    # Random topological order
    order = torch.randperm(D, generator=generator, device=device)

    # Upper-triangular mask (a<b) in the *order space*
    upper = torch.triu(torch.ones((D, D), device=device, dtype=torch.bool), diagonal=1)

    # Bernoulli edges in order-space, only above diagonal
    u = torch.rand((D, D), device=device, generator=generator)
    E_order = (u < edge_prob) & upper  # (D,D) bool

    # Map from order-space indices (a,b) -> node indices (i=order[a], j=order[b])
    A = torch.zeros((D, D), device=device, dtype=dtype)
    src = order.view(D, 1).expand(D, D)
    dst = order.view(1, D).expand(D, D)
    A[src[E_order], dst[E_order]] = 1
    return A


def sample_weights(
    A: torch.Tensor,
    w_scale: float,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Assign real weights to existing edges (GPU-friendly).

    Matches NumPy behavior:
      - W is zero where A==0
      - where A==1: W ~ Normal(0, w_scale)
      - then add sign(W)*0.5 (nudges away from zero; note sign(0)=0)

    Args:
        A: (D,D) 0/1 adjacency tensor (any integer/bool type accepted).
    Returns:
        W: (D,D) float tensor (float64 by default).
    """
    if w_scale < 0:
        raise ValueError("w_scale must be non-negative.")

    device = device if device is not None else (A.device if isinstance(A, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if generator is None:
        generator = torch.Generator(device="cpu")

    A = A.to(device=device)
    mask = A.to(torch.bool)

    W = torch.zeros_like(A, device=device, dtype=dtype)

    # Sample full matrix then mask (avoids variable-length sampling on GPU)
    if w_scale == 0.0:
        sampled = torch.zeros_like(W)
    else:
        sampled = torch.randn(W.shape, device=device, dtype=dtype, generator=generator) * w_scale

    sampled = sampled + torch.sign(sampled) * 0.5
    W[mask] = sampled[mask]
    return W
