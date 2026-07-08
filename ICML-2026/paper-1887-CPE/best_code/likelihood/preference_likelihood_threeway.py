import math
import numpy as np
from scipy.special import expit  # <-- tested, stable sigmoid


# ---------------------- Expert likelihood (BT-3way) ---------------------
def logistic(x):
    return expit(x)


def direction_score(W: np.ndarray, i: int, j: int,
                    tau: float=0.1, eps: float=1e-6,
                    lam: float = 0.0, phi_ij: float = 0.0) -> float:
    # g(|W_ij|) = log(eps + |W_ij|) + lambda * phi
    return math.log(eps + abs(W[i, j])/tau) + lam * phi_ij
    # return math.log1p(eps + abs(W[i, j])) + lam * phi_ij
    #

def bt_threeway_hier(
    W: np.ndarray,
    i: int,
    j: int,
    tau: float = 0.1,
    beta_edge: float = 5.0,
    beta_dir: float = 5.0,
    lam: float = 0.0,
    phi_ij: float = 0.0,
    phi_ji: float = 0.0,
):
    s_ij = direction_score(W, i, j, tau=tau, eps=1e-6, lam=lam, phi_ij=phi_ij)
    s_ji = direction_score(W, j, i, tau=tau, eps=1e-6, lam=lam, phi_ij=phi_ji)

    a = max(s_ij, s_ji)       # absolute evidence any edge
    d = s_ij - s_ji           # relative evidence

    p_edge = logistic(beta_edge * a)
    p_ij_edge = logistic(beta_dir * d)

    p_ij = p_edge * p_ij_edge
    p_ji = p_edge * (1.0 - p_ij_edge)
    p_none = 1.0 - p_edge

    return np.array([p_ij, p_ji, p_none], dtype=float)

def log_expert_likelihood_bt_threeway(
    y: int,
    W: np.ndarray,
    i: int,
    j: int,
    tau: float = 0.1,
    beta_edge: float = 5.0,
    beta_dir: float = 5.0,
    lam: float = 0.0,
    phi_ij: float = 0.0,
    phi_ji: float = 0.0,
):
    """
    Compute log p(y | W) for y ∈ {0,1,2} (0=i->j, 1=j->i, 2=none),
    using the hierarchical expert model but working directly with logits.
    """
    # TODO: It feels like working on logits then compute likelihoods is better

    if y not in (0, 1, 2):
        raise ValueError("y must be 0 (i->j), 1 (j->i), or 2 (none)")

    p = bt_threeway_hier(W, i, j, tau=tau, beta_edge=beta_edge, beta_dir=beta_dir,
                     lam=lam, phi_ij=phi_ij, phi_ji=phi_ji)

    return math.log(1e-12 + p[y])
