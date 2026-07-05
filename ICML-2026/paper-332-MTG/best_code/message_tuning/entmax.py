
import torch
import torch.nn.functional as F


def entmax15(z, dim=-1, n_iter=30):
    """α-entmax with α=1.5 along dimension dim.

    For α=1.5: p_j = max(0, (τ - z_j) / 3)^2
    where τ is chosen so sum(p) = 1.
    Solved via bisection.
    """
    alpha = 1.5
    # Work with sorted values for efficiency
    tau_min = z.min(dim=dim, keepdim=True)[0] - 1.0
    tau_max = z.max(dim=dim, keepdim=True)[0] + 3.0  # τ is above max(z)

    for _ in range(n_iter):
        tau = (tau_min + tau_max) / 2
        p = torch.clamp((tau - z) / 3.0, min=0) ** 2
        sum_p = p.sum(dim=dim, keepdim=True)
        # Binary search: if sum > 1, decrease tau; else increase tau
        tau_max = torch.where(sum_p > 1.0, tau, tau_max)
        tau_min = torch.where(sum_p <= 1.0, tau, tau_min)

    tau = (tau_min + tau_max) / 2
    p = torch.clamp((tau - z) / 3.0, min=0) ** 2
    # Ensure exact unit sum
    p = p / p.sum(dim=dim, keepdim=True).clamp(min=1e-8)
    return p
