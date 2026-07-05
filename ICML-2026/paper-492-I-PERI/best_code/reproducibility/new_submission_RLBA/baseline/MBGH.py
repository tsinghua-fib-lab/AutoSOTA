from __future__ import annotations

import warnings
import numpy as np
from typing import Optional


def soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    """Elementwise soft-thresholding operator S_tau(X)."""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def estimate_delta_omega_admm(Sigma1: np.ndarray, Sigma2: np.ndarray, lam: float = 0.1, rho: float = 1.0, max_iter: int = 1000, tol: float = 1e-7,) -> np.ndarray:
    """
    :param Sigma1: sample covariance matrix of the first regime.
    :param Sigma2: sample covariance matrix of the second regime.
    :param lam: L1 regularisation parameter.
    :param rho: ADMM penalty parameter.
    :param max_iter: maximum number of ADMM iterations.
    :param tol: convergence tolerance.
    :return: estimated precision-matrix difference DeltaOmega.
    This function estimates the difference between the two precision matrices using the sparse ADMM algorithm of Jiang et al. (2018). 
    It solves an L1-regularized convex optimization problem whose solution is used to recover the Difference DAG.
    """

    m = Sigma1.shape[0]

    # --- Step 1: SVD / eigen decompositions ---
    d1, U1 = np.linalg.eigh(Sigma1)   # Σ1 = U1 diag(d1) U1^T
    d2, U2 = np.linalg.eigh(Sigma2)   # Σ2 = U2 diag(d2) U2^T

    # B_{jk} = 1 / (d1_j * d2_k + rho)
    B = 1.0 / (np.outer(d1, d2) + rho)

    # --- Initialize variables ---
    Omega = np.zeros((m, m))
    Psi   = np.zeros((m, m))
    Lambda = np.zeros((m, m))

    for _ in range(max_iter):
        Omega_prev = Omega.copy()
        Psi_prev = Psi.copy()

        # --- Step 2: Ω-update ---
        # A = (Σ1 - Σ2) - Λ + ρΨ
        A = (Sigma1 - Sigma2) - Lambda + rho * Psi

        # Ω = U1 [ B ∘ (U1^T A U2) ] U2^T
        A_tilde = U1.T @ A @ U2
        Omega = U1 @ (B * A_tilde) @ U2.T

        # --- Step 3: Ψ-update (soft-thresholding) ---
        Psi = soft_threshold(Omega + Lambda / rho, lam / rho)

        # --- Step 4: Λ-update ---
        Lambda = Lambda + rho * (Omega - Psi)

        # --- Convergence ---
        primal_res = np.linalg.norm(Omega - Psi, ord="fro")
        dual_change = np.linalg.norm(Psi - Psi_prev, ord="fro")
        omega_change = np.linalg.norm(Omega - Omega_prev, ord="fro")

        if max(primal_res, dual_change, omega_change) < tol:
            break

    DeltaOmega = 0.5 * (Psi + Psi.T)
    return DeltaOmega

def learn_ddag(
    X1: np.ndarray,
    X2: np.ndarray,
    estimator: str = "admm",
    lam: float = 0.1,
    diagonal_tol: Optional[float] = None,
    edge_tol: Optional[float] = None, 
    diagonal_tol_fraction: float = 0.05,
    edge_tol_fraction: float = 0.05,
    admm_kwargs: Optional[dict] = None,
) -> np.ndarray:
    """
    :param X1: samples from the first regime, with shape (n1, p).
    :param X2: samples from the second regime, with shape (n2, p).
    :param estimator: estimator used for the precision-matrix difference. Currently only admm is supported.
    :param lam: L1 regularisation parameter used by the ADMM estimator.
    :param diagonal_tol: absolute threshold for declaring a diagonal entry of DeltaOmega equal to zero.
    :param edge_tol: absolute threshold for declaring an off-diagonal entry of DeltaOmega equal to zero.
    :param diagonal_tol_fraction: fraction of the maximum absolute DeltaOmega entry used as adaptive diagonal threshold when diagonal_tol is None.
    :param edge_tol_fraction: fraction of the maximum absolute DeltaOmega entry used as adaptive edge threshold when edge_tol is None.
    :param admm_kwargs: optional keyword arguments passed to the ADMM estimator.
    :return: boolean adjacency matrix of the estimated Difference DAG.
    This function estimates the Difference DAG (DDAG) between two linear SEMs without estimating the two individual DAGs. 
    The returned matrix follows the convention ddag[i, j] = True for a directed edge j -> i, corresponding to a changed structural coefficient.
    """
    if admm_kwargs is None:
        admm_kwargs = {}

    _, p = X1.shape

    # Compute sample covariance matrices (unbiased, ddof=1)
    Sigma1_hat = np.cov(X1, rowvar=False)
    Sigma2_hat = np.cov(X2, rowvar=False)

    # Adjacency matrix of the output DDAG (original node indices)
    ddag = np.zeros((p, p), dtype=bool)

    # Active variable set (original indices); processed all-at-once per layer
    active = list(range(p))

    # ------------------------------------------------------------------ #
    # Main loop — Algorithm 1                                             #
    # ------------------------------------------------------------------ #
    while len(active) > 1:
        m = len(active)     

        # Extract sub-covariance matrices for active variables
        S1 = Sigma1_hat[np.ix_(active, active)]
        S2 = Sigma2_hat[np.ix_(active, active)]

        # Step 3 — estimate Delta_Omega over V
        if estimator == "admm":
            DeltaOmega = estimate_delta_omega_admm(S1, S2, lam=lam,**admm_kwargs)
        else:
            raise ValueError(f"Unknown estimator '{estimator}'. Choose 'admm'.")
        
        DeltaOmega = 0.5 * (DeltaOmega + DeltaOmega.T)
                               
        scale = np.max(np.abs(DeltaOmega))
        diag_tol = diagonal_tol if diagonal_tol is not None else (
            diagonal_tol_fraction * scale if scale > 0 else 1e-8
        )
        e_tol = edge_tol if edge_tol is not None else (
            edge_tol_fraction * scale if scale > 0 else 1e-8
        )

        diag_abs = np.abs(np.diag(DeltaOmega))
        terminal_local = [i for i in range(m) if diag_abs[i] < diag_tol]

        # Guard: if no terminals found the algorithm cannot progress.
        # This can happen when estimates are noisy; warn and stop.
        if len(terminal_local) == 0:
            warnings.warn(f"No terminal vertices identified in an active set of size {m}. ", RuntimeWarning, stacklevel=2,)
            break
        terminal_local_set = set(terminal_local)

        # Steps 5–12 — identify incoming edges for each terminal vertex
        for i_loc in terminal_local:                                                            
            i_glob = active[i_loc]                                                              
            for j_loc in range(m):                                                              
                if j_loc in terminal_local_set:
                    continue
                if abs(DeltaOmega[i_loc, j_loc]) > e_tol:                                         
                    j_glob = active[j_loc]
                    # Condition (Algorithm 1, line 8):
                    #   edge not already recorded  AND  j is not itself a terminal
                    # Edge:  i_glob ← j_glob
                    ddag[i_glob, j_glob] = True                                            

        # Step 13 — remove terminals from active set
        terminal_global = {active[i] for i in terminal_local}                                   
        active = [v for v in active if v not in terminal_global]                                
    return ddag