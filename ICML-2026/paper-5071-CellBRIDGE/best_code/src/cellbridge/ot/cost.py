import logging

import numpy as np
import ot

from cellbridge.ot.fgw_multi import fused_gromov_wasserstein_multi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _channel_dim(D: np.ndarray) -> int:
    return 1 if D.ndim == 2 else D.shape[-1]


def _as_channel_tensor(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D)
    if D.ndim == 2:
        return D[:, :, None]
    if D.ndim == 3:
        return D
    raise ValueError(f"Structure matrix must be 2D or 3D, got shape {D.shape}.")


def pairwise_sqeuclidean(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean costs."""
    # X is (n,d), Y is (m,d)
    Xn = (X**2).sum(1, keepdims=True)
    Yn = (Y**2).sum(1, keepdims=True).T
    return Xn + Yn - 2 * X @ Y.T


def fgw_var_term_mahalanobis_fast(D1, D2, T, M, assume_M_symmetric=False):
    """
    Computes: 2 * sum_{i,j,k,l} T[i,k] T[j,l] * (D1[i,j,:]^T M D2[k,l,:])
    """
    D1 = _as_channel_tensor(D1)
    D2 = _as_channel_tensor(D2)
    if D1.shape[-1] != D2.shape[-1]:
        raise ValueError(
            f"D1 and D2 must have the same number of channels, got "
            f"{D1.shape[-1]} and {D2.shape[-1]}."
        )
    if M is None:
        M = np.eye(D1.shape[-1])
    M = np.asarray(M, dtype=float)
    if not assume_M_symmetric:
        M = 0.5 * (M + M.T)  # harmless if already symmetric

    # Move channel to front for D1
    D1a = np.transpose(D1, (2, 0, 1))  # (p, n1, n1)

    # U[a] = T^T @ D1^{(a)} @ T   (batched over a)
    # akl = ik, aij, jl
    U = np.einsum("ik,aij,jl->akl", T, D1a, T, optimize=True)  # (p, n2, n2)

    # Apply M to D2 once along the channel (last) axis
    D2M = np.tensordot(D2, M, axes=([2], [0]))  # (n2, n2, p)

    # Final scalar: 2 * sum_{a,k,l} U[a,k,l] * D2M[k,l,a]
    return float(2.0 * np.einsum("akl,kla->", U, D2M, optimize=True))


def contributions_multi(M, D1, D2, T, Q):
    """Return linear and multi-channel GW cost terms."""
    return float(np.sum(M * T)), float(fgw_var_term_mahalanobis_fast(D1, D2, T, Q))


def lift_D1D2(f):
    """Turn (D1, D2)->(D1, D2) into (C, D1, D2)->(C, D1, D2)."""

    def wrapped(C, D1, D2):
        D1n, D2n = f(D1=D1, D2=D2)
        return C, D1n, D2n

    return wrapped


def scale_FGW_multi(
    C: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    Q: np.ndarray | None = None,
    numIterFGW=30000,
    numIterEMD=200000,
):
    """Scale linear and structural costs for multi-FGW."""
    n, m = C.shape
    a = np.ones(n, float) / n
    b = np.ones(m, float) / m
    if Q is None:
        Q = np.eye(_channel_dim(D1))

    # We get the costs via the endpoings (alpha = 0 and alpha = 1).
    logger.info("Endpoint with alpha = 1")
    logger.info("Using max iter = %d", numIterFGW)
    # Alpha = 1

    # Scale D1 and D2 by the same constant, which is the 99th percentile
    cons = max(
        float(np.quantile(D1[D1 > 0], 0.99)), float(np.quantile(D2[D2 > 0], 0.99))
    )

    D1_scaled = D1 / (cons + 1e-12)
    D2_scaled = D2 / (cons + 1e-12)

    T_gw = fused_gromov_wasserstein_multi(
        C,
        D1_scaled,
        D2_scaled,
        alpha=1.0,
        Q=Q,
        p=a,
        q=b,
        max_iter=numIterFGW,
        numItermaxEmd=numIterEMD,
    )

    logger.info("Endpoint with alpha = 0")

    # Alpha = 0
    T_ot = ot.emd(a, b, C, numItermax=numIterEMD)

    # Compute the costs (linear and GW) for each term

    logger.info("Computing contributions at both endpoints")

    lin_gw, gw_gw = contributions_multi(C, D1, D2, T_gw, Q)
    lin_ot, gw_ot = contributions_multi(C, D1, D2, T_ot, Q)

    # Dynamic ranges (avoid zeros)
    eps = np.finfo(float).eps
    Delta_lin = max(
        abs(lin_ot - lin_gw), eps
    )  # how much the linear term actually moves across relevant T's
    Delta_gw = max(
        abs(gw_gw - gw_ot), eps
    )  # how much the GW variable part moves across relevant T's
    logger.info(
        "Dynamic ranges: Delta_lin = %.6f, Delta_gw = %.6f", Delta_lin, Delta_gw
    )
    logger.info("Costs at alpha=0: linear = %.6f, GW = %.6f", lin_ot, gw_ot)
    logger.info("Costs at alpha=1: linear = %.6f, GW = %.6f", lin_gw, gw_gw)

    # Rescale the cost matrices
    M_f = C / Delta_lin
    D1_f = D1 / np.sqrt(Delta_gw)
    D2_f = D2 / np.sqrt(Delta_gw)

    return M_f, D1_f, D2_f


def shuffle_all(D1, D2):
    """Shuffle all structural entries independently."""
    logger.info("Shuffling all entries")
    # Shuffle all entries in each matrix (separately)
    D1_flat = D1.copy().flatten()
    np.random.shuffle(D1_flat)
    D1_shuffled = D1_flat.reshape(D1.shape)
    D2_flat = D2.copy().flatten()
    np.random.shuffle(D2_flat)
    D2_shuffled = D2_flat.reshape(D2.shape)
    return D1_shuffled, D2_shuffled


class CostPipeline:
    """Apply cost transforms in sequence."""

    def __init__(self, steps):
        """
        steps: list of callables, each (C, D1, D2) -> (C, D1, D2)
        """
        self.steps = steps

    def __call__(self, C, D1, D2):
        """Transform linear and structural costs."""
        for step in self.steps:
            C, D1, D2 = step(C=C, D1=D1, D2=D2)
        return C, D1, D2
