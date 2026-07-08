import numpy as np
import torch

from likelihood.preference_likelihood_threeway_gpu import bt_threeway_hier_pairs


def _as_torch_int64(A_true, device: torch.device) -> torch.Tensor:
    if isinstance(A_true, torch.Tensor):
        return A_true.to(device=device, dtype=torch.int64)
    return torch.as_tensor(np.asarray(A_true, dtype=np.int64), device=device, dtype=torch.int64)


def _all_ordered_pairs(D: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ii, jj of shape (K,), K=D*(D-1), for all ordered pairs i!=j."""
    idx = torch.arange(D, device=device)
    ii = idx.repeat_interleave(D)
    jj = idx.repeat(D)
    mask = ii != jj
    return ii[mask], jj[mask]


def _posterior_predictive_for_pairs(
    posterior,
    ii: torch.Tensor,
    jj: torch.Tensor,
    beta_edge: float,
    beta_dir: float,
    lam: float,
) -> torch.Tensor:
    """
    Compute posterior predictive p(y | i,j) for all pairs in one batched pass.

    Returns:
        p: (K,3) torch float64 tensor on posterior device.
    """
    # Fast path: torch particle posterior
    if hasattr(posterior, "particles") and isinstance(posterior.particles, torch.Tensor) and hasattr(posterior, "weights"):
        W = posterior.particles
        device = W.device

        w = posterior.weights
        if not isinstance(w, torch.Tensor):
            w = torch.as_tensor(np.asarray(w, dtype=np.float64), device=device, dtype=torch.float64)
        else:
            w = w.to(device=device, dtype=torch.float64)

        w = w / w.sum().clamp_min(1e-300)

        tau = float(getattr(posterior, "tau", 0.1))

        # Batched likelihood for all particles and all queried pairs: (S,K,3)
        probs = bt_threeway_hier_pairs(
            W, ii.to(device), jj.to(device),
            tau=tau, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam,
            phi_ij=None, phi_ji=None,
        )

        # Posterior predictive mixture: (K,3)
        p = (w[:, None, None] * probs).sum(dim=0)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-300)
        return p.to(dtype=torch.float64)

    # Fallback: call posterior.predictive_answer_dist in a loop (CPU-ish)
    # (Still keeps signature compatibility for any posterior type.)
    D = int(max(ii.max().item(), jj.max().item()) + 1)
    p_list = []
    for a, b in zip(ii.detach().cpu().tolist(), jj.detach().cpu().tolist()):
        p_ab = posterior.predictive_answer_dist(a, b, beta_edge, beta_dir, lam)
        p_list.append(np.asarray(p_ab, dtype=np.float64))
    p_np = np.stack(p_list, axis=0)  # (K,3)
    return torch.as_tensor(p_np, device=torch.device("cpu"), dtype=torch.float64)


def expected_true_class_prob(posterior, beta_edge, beta_dir, lam, A_true):
    """
    Average probability the posterior assigns to the true label
    (i->j, j->i, or none).

    Signature matches metrics.py exactly.
    """
    # Choose device for computation
    if hasattr(posterior, "particles") and isinstance(posterior.particles, torch.Tensor):
        device = posterior.particles.device
    else:
        device = torch.device("cpu")

    A_true_t = _as_torch_int64(A_true, device=device)
    D = int(A_true_t.shape[0])

    ii, jj = _all_ordered_pairs(D, device=device)

    # True label y* in {0,1,2} for each ordered pair (i,j)
    a_ij = A_true_t[ii, jj]
    a_ji = A_true_t[jj, ii]
    ystar = torch.full((ii.shape[0],), 2, device=device, dtype=torch.long)
    ystar[(a_ij == 1) & (a_ji == 0)] = 0
    ystar[(a_ji == 1) & (a_ij == 0)] = 1

    p = _posterior_predictive_for_pairs(posterior, ii, jj, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam)

    # If fallback path returned CPU tensor but device is GPU, move ystar accordingly
    if p.device != ystar.device:
        ystar = ystar.to(p.device)

    acc = p[torch.arange(p.shape[0], device=p.device), ystar].mean()
    return float(acc.item())


def mean_brier_score(posterior, beta_edge, beta_dir, lam, A_true):
    """
    Multiclass Brier score averaged over node pairs.
    Lower is better (0 is perfect).

    Signature matches metrics.py exactly.
    """
    # Choose device for computation
    if hasattr(posterior, "particles") and isinstance(posterior.particles, torch.Tensor):
        device = posterior.particles.device
        dtype = posterior.particles.dtype
    else:
        device = torch.device("cpu")
        dtype = torch.float64

    A_true_t = _as_torch_int64(A_true, device=device)
    D = int(A_true_t.shape[0])

    ii, jj = _all_ordered_pairs(D, device=device)

    # True label y* in {0,1,2}
    a_ij = A_true_t[ii, jj]
    a_ji = A_true_t[jj, ii]
    ystar = torch.full((ii.shape[0],), 2, device=device, dtype=torch.long)
    ystar[(a_ij == 1) & (a_ji == 0)] = 0
    ystar[(a_ji == 1) & (a_ij == 0)] = 1

    p = _posterior_predictive_for_pairs(posterior, ii, jj, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam).to(dtype=torch.float64)

    if p.device != ystar.device:
        ystar = ystar.to(p.device)

    # one-hot: (K,3)
    e = torch.zeros((p.shape[0], 3), device=p.device, dtype=torch.float64)
    e[torch.arange(p.shape[0], device=p.device), ystar] = 1.0

    brier = ((p - e) ** 2).sum(dim=1).mean()
    return float(brier.item())
