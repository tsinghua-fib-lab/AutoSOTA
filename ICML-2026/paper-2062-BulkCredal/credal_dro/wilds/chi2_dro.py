from __future__ import annotations

import math
from typing import Literal, Tuple


def chi2_rho_from_epsilon(
    epsilon: float,
    n: int,
    normalisation: Literal["max", "raw"] = "max",
) -> float:
    """Map a user-facing epsilon in [0,1] to a chi-square radius rho >= 0.

    Parameters
    ----------
    epsilon:
        User knob in [0,1]. epsilon=0 -> ERM (uniform weights).
    n:
        Number of points (typically minibatch size).
    normalisation:
        - "max": interpret epsilon as a *fraction of the maximum chi-square radius*
                 from uniform to a point mass for this n. Then rho_max = (n-1)/2.
        - "raw": interpret epsilon directly as rho (not recommended; batch-size sensitive).

    Notes
    -----
    For the Pearson chi-square divergence from uniform u_i=1/n:
        D_chi2(p||u) = 1/2 * sum_i (p_i-u_i)^2 / u_i
    The maximum over the simplex is attained at a point mass and equals (n-1)/2.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    eps = float(epsilon)
    if not (0.0 <= eps <= 1.0):
        raise ValueError(f"epsilon must be in [0,1], got {epsilon}.")

    if normalisation == "raw":
        return eps

    if normalisation == "max":
        rho_max = 0.5 * float(n - 1)
        return eps * rho_max

    raise ValueError(f"Unknown normalisation={normalisation!r}.")


def chi2_dro_weights(losses, rho: float):
    """Compute worst-case weights for chi-square DRO around uniform.

    Solves:
        maximise_p  sum_i p_i * l_i
        s.t. p in simplex,  D_chi2(p||u) <= rho,  u = uniform.

    Returns
    -------
    w : torch.Tensor shape (n,), nonnegative, sums to 1

    Implementation details
    ----------------------
    This uses an active-set KKT solver specialised to the uniform reference,
    exploiting the fact that the optimiser puts mass on the largest losses.
    """
    try:
        import torch
    except Exception as e:
        raise ImportError("PyTorch is required for chi2_dro_weights.") from e

    if rho <= 0.0:
        n = int(losses.numel())
        return torch.full_like(losses, 1.0 / max(1, n))

    l = losses.detach().reshape(-1)
    n = int(l.numel())
    if n == 0:
        raise ValueError("Empty losses tensor.")

    device = l.device
    dtype = l.dtype
    n_f = float(n)

    # Sort by decreasing loss: the active set will be a prefix of this ordering.
    l_sorted, idx = torch.sort(l, descending=True)

    # In the uniform-reference case, the chi-square ball is equivalent to an L2
    # ball around uniform in probability space:
    #   D_chi2(p||u) = (n/2) * ||p-u||^2  <= rho  =>  ||p-u||^2 <= 2*rho/n
    r2 = 2.0 * float(rho) / n_f

    # Minimum support size m such that the "drop others" baseline distribution
    # (uniform on m points, zero elsewhere) is inside the L2 ball.
    # This baseline has ||p-u||^2 = 1/m - 1/n.
    m_min = int(math.floor(1.0 / (r2 + 1.0 / n_f))) + 1
    m_min = max(1, min(n, m_min))

    m = n
    while True:
        if m < m_min:
            m = m_min

        l_m = l_sorted[:m]
        mean = l_m.mean()
        d = l_m - mean
        s2 = float(torch.sum(d * d).item())

        a = (1.0 / float(m)) - (1.0 / n_f)   # baseline ||p-u||^2 if uniform-on-m
        denom = r2 - a
        if denom <= 1e-12:
            # Too small a radius for this m; increase support.
            m += 1
            if m > n:
                return torch.full_like(l, 1.0 / n_f)
            continue

        if s2 <= 1e-12:
            w_m = torch.full((m,), 1.0 / float(m), device=device, dtype=dtype)
        else:
            lam = 0.5 * math.sqrt(s2 / denom)
            w_m = (1.0 / float(m)) + d / (2.0 * lam)

        if float(w_m.min().item()) >= -1e-10:
            w_m = torch.clamp(w_m, min=0.0)
            w_m = w_m / w_m.sum()

            w = torch.zeros((n,), device=device, dtype=dtype)
            w[idx[:m]] = w_m
            return w

        # If some weights are negative, drop all non-positive weights from the active set.
        pos = w_m > 0
        new_m = int(pos.sum().item())
        if new_m >= m:
            new_m = m - 1
        if new_m < m_min:
            new_m = m_min
        if new_m == m:
            new_m = m - 1
        m = new_m
        if m <= 0:
            return torch.full_like(l, 1.0 / n_f)


def chi2_dro_loss(
    losses,
    epsilon: float,
    normalisation: Literal["max", "raw"] = "max",
) -> Tuple["object", "object"]:
    """Return (robust_loss, weights) for chi-square DRO.

    Parameters
    ----------
    losses:
        Per-example losses, shape (n,). Must be unreduced.
    epsilon:
        User knob in [0,1]. For "max" normalisation we map it to a chi-square radius
        rho = epsilon * (n-1)/2.
    """
    try:
        import torch
    except Exception as e:
        raise ImportError("PyTorch is required for chi2_dro_loss.") from e

    l = losses.reshape(-1)
    n = int(l.numel())
    rho = chi2_rho_from_epsilon(float(epsilon), n=n, normalisation=normalisation)
    w = chi2_dro_weights(l, rho=rho).detach()  # envelope theorem => detach weights
    robust = torch.sum(w * l)
    return robust, w
