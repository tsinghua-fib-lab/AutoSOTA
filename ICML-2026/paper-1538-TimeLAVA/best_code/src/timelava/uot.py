"""Entropy-regularised unbalanced optimal transport (Algorithm 1, line 13).

Solved entirely in the log domain following Sejourne et al. (2019) -- the
solver the paper cites -- so the optimal dual potentials (Appendix B.4) are
returned directly and the method stays numerically stable across the full
range of the entropy parameter, including the very small values used in the
convergence study of Theorem B.2.
"""

from __future__ import annotations

import numpy as np

__all__ = ["unbalanced_sinkhorn_dual", "psi_kappa"]


def psi_kappa(u: np.ndarray, kappa: float) -> np.ndarray:
    """psi_kappa(u) = kappa * (1 - exp(-u / kappa))   (Theorem 4.5)."""
    return kappa * (1.0 - np.exp(-u / kappa))


def _logsumexp(M: np.ndarray, axis: int) -> np.ndarray:
    mx = np.max(M, axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    out = mx.squeeze(axis) + np.log(np.sum(np.exp(M - mx), axis=axis))
    return out


def unbalanced_sinkhorn_dual(
    a: np.ndarray,
    b: np.ndarray,
    D: np.ndarray,
    reg: float,
    kappa: float,
    numItermax: int = 2000,
    stopThr: float = 1e-9,
):
    """Entropy-regularised *unbalanced* OT with KL marginal penalties.

    Solves

        min_{T >= 0} <D, T> + kappa KL(T 1_m || a) + kappa KL(T^T 1_n || b)
                      - reg H(T)

    The log-domain scaling-iteration update for the KL-penalised problem is

        f_i <- -rho * reg * lse_j( (g_j - D_ij) / reg + log b_j )
        g_j <- -rho * reg * lse_i( (f_i - D_ij) / reg + log a_i )

    with contraction factor ``rho = kappa / (kappa + reg)``. Here (f, g) are
    exactly the optimal dual potentials of Appendix B.4, in the cost's own
    units, so no rescaling is required afterwards.

    Returns
    -------
    f : (n,) optimal dual potential for the evaluated side (= f* in the paper)
    g : (m,) optimal dual potential for the reference side
    T : (n, m) optimal transport plan
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    n, m = D.shape

    log_a = np.log(a)
    log_b = np.log(b)
    rho = kappa / (kappa + reg)

    f = np.zeros(n)
    g = np.zeros(m)

    # The entropic scaling iteration contracts at a rate that degrades as
    # reg -> 0, so a fixed small iteration cap silently returns an
    # *unconverged* potential at tiny reg. We therefore (i) check the change
    # of BOTH potentials every sweep against an absolute tolerance, and
    # (ii) let the iteration ceiling scale with 1/reg, which is the known
    # dependence of Sinkhorn's iteration complexity (O(1/reg)).
    max_iter = max(numItermax, int(50.0 / reg))

    for it in range(max_iter):
        f_prev = f
        g_prev = g
        M_g = (g[None, :] - D) / reg + log_b[None, :]
        f = -rho * reg * _logsumexp(M_g, axis=1)
        M_f = (f[:, None] - D) / reg + log_a[:, None]
        g = -rho * reg * _logsumexp(M_f, axis=0)

        # Absolute change of the full potential pair across one sweep.
        delta = max(
            np.max(np.abs(f - f_prev)),
            np.max(np.abs(g - g_prev)),
        )
        if delta < stopThr:
            break

    log_T = (
        (f[:, None] + g[None, :] - D) / reg
        + log_a[:, None]
        + log_b[None, :]
    )
    T = np.exp(log_T)
    return f, g, T
