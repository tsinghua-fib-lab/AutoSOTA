import numpy as np


def prolongation_weighted(
    S: np.ndarray,
    weights: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> np.ndarray:
    """
    P = Diag(weights) S (S^T Diag(weights) S + ridge I)^{-1}
    which guarantees S^T P ≈ I_K (exact as ridge -> 0).
    """
    w = np.asarray(weights, dtype=float).ravel()
    WS = w[:, None] * S
    G = S.T @ WS
    K = G.shape[0]
    return WS @ np.linalg.solve(
        G + ridge * np.eye(K), np.eye(K)
    )


def lift_dense_two_sided(
    U: np.ndarray,  # (N x K)
    C: np.ndarray,  # (K x K)
    V: np.ndarray,  # (N x K)
) -> np.ndarray:
    """
    Build the full N x N lifted matrix:
        D_up = U @ C @ V.T
    """
    return (U @ C) @ V.T
