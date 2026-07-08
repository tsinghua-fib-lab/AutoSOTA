"""GPU-friendly Torch implementation of the BT-3way expert likelihood.

This is a faithful port of :file:`preference_likelihood_threeway.py` but supports
batched tensors, so you can evaluate all particles at once on GPU.

Baseline model (same semantics):
- direction_score(W,i,j) = log(eps + |W[i,j]|/tau) + lam * phi_ij
- a = max(s_ij, s_ji)              (evidence any edge)
- d = s_ij - s_ji                  (evidence direction)
- p_edge = sigmoid(beta_edge * a)
- p_ij_edge = sigmoid(beta_dir * d)
- p(i->j) = p_edge * p_ij_edge
- p(j->i) = p_edge * (1 - p_ij_edge)
- p(none) = 1 - p_edge

Outcome index convention matches baseline:
    y=0: i -> j
    y=1: j -> i
    y=2: none
"""

from __future__ import annotations

import torch


def direction_score(
    W: torch.Tensor,
    i: int,
    j: int,
    *,
    tau: float = 0.1,
    eps: float = 1e-6,
    lam: float = 0.0,
    phi_ij: float = 0.0,
) -> torch.Tensor:
    """Vectorized direction score.

    W may be (..., D, D). Returns shape (...) corresponding to W[..., i, j].
    """
    # Ensure float dtype (prefer float64 for numerical faithfulness)
    Wij = W[..., i, j]
    # Note: baseline uses abs(W[i,j]) / tau inside log(eps + ...)
    return torch.log(eps + torch.abs(Wij) / tau) + (lam * phi_ij)


def bt_threeway_hier(
    W: torch.Tensor,
    i: int,
    j: int,
    *,
    tau: float = 0.1,
    beta_edge: float = 5.0,
    beta_dir: float = 5.0,
    lam: float = 0.0,
    phi_ij: float = 0.0,
    phi_ji: float = 0.0,
) -> torch.Tensor:
    """Return p(y|W,i,j) for all particles in W.

    Args:
        W: torch tensor of shape (S,D,D) or (...,D,D)

    Returns:
        probs: tensor of shape (..., 3) with ordering [p_ij, p_ji, p_none].
    """
    s_ij = direction_score(W, i, j, tau=tau, eps=1e-6, lam=lam, phi_ij=phi_ij)
    s_ji = direction_score(W, j, i, tau=tau, eps=1e-6, lam=lam, phi_ij=phi_ji)

    a = torch.maximum(s_ij, s_ji)
    d = s_ij - s_ji

    p_edge = torch.sigmoid(torch.as_tensor(beta_edge, dtype=W.dtype, device=W.device) * a)
    p_ij_edge = torch.sigmoid(torch.as_tensor(beta_dir, dtype=W.dtype, device=W.device) * d)

    p_ij = p_edge * p_ij_edge
    p_ji = p_edge * (1.0 - p_ij_edge)
    p_none = 1.0 - p_edge

    return torch.stack([p_ij, p_ji, p_none], dim=-1)


def bt_threeway_hier_pairs(
    W: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    *,
    tau: float = 0.1,
    beta_edge: float = 5.0,
    beta_dir: float = 5.0,
    lam: float = 0.0,
    phi_ij: torch.Tensor | None = None,
    phi_ji: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return p(y|W, i_k, j_k) for many queried pairs.

    Args:
        W: tensor of shape (S,D,D) (or more generally (...,D,D) where the leading
           dims are treated as batch dims). This function is optimized for the
           common case (S,D,D).
        i_idx: int64 tensor of shape (K,)
        j_idx: int64 tensor of shape (K,)

    Returns:
        probs: tensor of shape (S,K,3) for W shaped (S,D,D). For more general
               batch shapes, returns (...,K,3).
    """
    if i_idx.shape != j_idx.shape:
        raise ValueError("i_idx and j_idx must have the same shape")
    if i_idx.dim() != 1:
        raise ValueError("i_idx and j_idx must be 1D tensors of shape (K,)")

    # Default phi terms to zeros.
    K = i_idx.numel()
    if phi_ij is None:
        phi_ij = torch.zeros((K,), dtype=W.dtype, device=W.device)
    if phi_ji is None:
        phi_ji = torch.zeros((K,), dtype=W.dtype, device=W.device)

    # Gather all queried weights in one go. For (S,D,D):
    #   W[:, i, j] -> (S,K)
    Wij = W[..., i_idx, j_idx]
    Wji = W[..., j_idx, i_idx]

    s_ij = torch.log(1e-6 + torch.abs(Wij) / tau) + (lam * phi_ij)
    s_ji = torch.log(1e-6 + torch.abs(Wji) / tau) + (lam * phi_ji)

    a = torch.maximum(s_ij, s_ji)
    d = s_ij - s_ji

    be = torch.as_tensor(beta_edge, dtype=W.dtype, device=W.device)
    bd = torch.as_tensor(beta_dir, dtype=W.dtype, device=W.device)

    p_edge = torch.sigmoid(be * a)
    p_ij_edge = torch.sigmoid(bd * d)

    p_ij = p_edge * p_ij_edge
    p_ji = p_edge * (1.0 - p_ij_edge)
    p_none = 1.0 - p_edge

    return torch.stack([p_ij, p_ji, p_none], dim=-1)


def _log_expert_likelihood_bt_threeway(
    y: int,
    W: torch.Tensor,
    i: int,
    j: int,
    *,
    tau: float = 0.1,
    beta_edge: float = 5.0,
    beta_dir: float = 5.0,
    lam: float = 0.0,
    phi_ij: float = 0.0,
    phi_ji: float = 0.0,
) -> torch.Tensor:
    """Vectorized log-likelihood log p(y | W,i,j).

    Returns:
        logp: tensor of shape (...) corresponding to W batch dims.
    """
    if y not in (0, 1, 2):
        raise ValueError("y must be 0 (i->j), 1 (j->i), or 2 (none)")

    p = bt_threeway_hier(
        W,
        i,
        j,
        tau=tau,
        beta_edge=beta_edge,
        beta_dir=beta_dir,
        lam=lam,
        phi_ij=phi_ij,
        phi_ji=phi_ji,
    )
    # baseline uses log(1e-12 + p[y])
    return torch.log(torch.clamp(p[..., y], min=1e-12))


# Convenience wrapper with baseline function name (but tensor-first).
# This helps drop-in usage where you swap the import.

def log_expert_likelihood_bt_threeway(
    y: int,
    W: torch.Tensor,
    i: int,
    j: int,
    tau: float = 0.1,
    beta_edge: float = 5.0,
    beta_dir: float = 5.0,
    lam: float = 0.0,
    phi_ij: float = 0.0,
    phi_ji: float = 0.0,
) -> torch.Tensor:
    return _log_expert_likelihood_bt_threeway(
        y,
        W,
        i,
        j,
        tau=tau,
        beta_edge=beta_edge,
        beta_dir=beta_dir,
        lam=lam,
        phi_ij=phi_ij,
        phi_ji=phi_ji,
    )
