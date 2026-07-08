# test_prior_ordered.py
# Run: pytest -q test_prior_ordered.py
import numpy as np
import torch

from prior.prior_gpu import  make_prior_particles_from_truth  # adjust import path if needed


def _is_dag_kahn(A: np.ndarray) -> bool:
    """Kahn's algorithm for DAG check. A is (D,D) with 0/1 entries."""
    D = A.shape[0]
    indeg = A.sum(axis=0).astype(int)
    stack = [i for i in range(D) if indeg[i] == 0]
    seen = 0
    while stack:
        v = stack.pop()
        seen += 1
        out = np.where(A[v] != 0)[0]
        for u in out:
            indeg[u] -= 1
            if indeg[u] == 0:
                stack.append(u)
    return seen == D


def test_make_prior_particles_from_truth_ordered_basic():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    D = 50
    S = 64

    # Make a small truth DAG-ish weight matrix (upper triangular) so it's acyclic.
    W_star = torch.zeros((D, D), dtype=dtype)
    W_star = W_star + torch.triu(torch.randn((D, D), dtype=dtype) * 0.5, diagonal=1)
    # sparsify
    mask = (torch.rand((D, D)) < 0.08)
    W_star = W_star * mask
    W_star.fill_diagonal_(0.0)

    W = make_prior_particles_from_truth(
        W_star,
        S=S,
        flip_prob=0.1,
        add_remove_prob=0.1,
        weight_noise=0.05,
        device=device,
        dtype=dtype,
        generator=torch.Generator().manual_seed(123),
        extra_edge_prob=0.01,  # keep extras sparse-ish for test stability
    )

    assert isinstance(W, torch.Tensor)
    assert W.shape == (S, D, D)
    assert W.dtype == torch.float64
    assert W.device.type == device.type

    # No self edges
    diag = torch.diagonal(W, dim1=1, dim2=2)
    assert torch.all(diag == 0)

    # Adjacency is where weights nonzero
    A = (W.abs() > 1e-12).to(torch.int64)
    assert torch.all(torch.diagonal(A, dim1=1, dim2=2) == 0)

    # Check DAG property on a subset (Kahn on CPU numpy)
    A_np = A[: min(S, 16)].cpu().numpy()
    for s in range(A_np.shape[0]):
        assert _is_dag_kahn(A_np[s])


def test_overlap_weights_are_close_to_truth_when_noise_zero():
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    D = 40
    S = 32

    # Truth weights on a sparse upper-triangular DAG
    W_star = torch.zeros((D, D), dtype=dtype)
    W_star = W_star + torch.triu(torch.randn((D, D), dtype=dtype), diagonal=1)
    W_star *= (torch.rand((D, D)) < 0.06)
    W_star.fill_diagonal_(0.0)

    W = make_prior_particles_from_truth(
        W_star,
        S=S,
        flip_prob=0.0,
        add_remove_prob=0.0,  # no structural changes: should keep truth skeleton and then orient by order
        weight_noise=0.0,
        device=device,
        dtype=dtype,
        generator=torch.Generator().manual_seed(999),
        extra_edge_prob=0.0,
    )

    # Wherever an edge exists in both W and truth (directed overlap), weights should equal truth exactly.
    overlap = (W.abs() > 1e-12) & (W_star.abs().to(device) > 1e-12).unsqueeze(0)
    if overlap.any():
        W_star_S = W_star.to(device).unsqueeze(0).expand(S, -1, -1)
        assert torch.max((W[overlap] - W_star_S[overlap]).abs()).item() == 0.0

