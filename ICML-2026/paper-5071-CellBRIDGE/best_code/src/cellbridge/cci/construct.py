import logging
from collections.abc import Callable
from typing import Literal

import anndata as ad  # AnnData container for single-cell matrices
import numpy as np  # numerical ops
from scipy import sparse  # sparse matrices

from .lifting import lift_dense_two_sided, prolongation_weighted
from .pair_extraction import (
    PairTable,
    pair_indices,
)

logger = logging.getLogger(__name__)


def soft_means_from_S(
    adata: ad.AnnData,  # AnnData with cells x genes counts (nonnegative)
    S: np.ndarray,  # soft memberships (N x K), rows should sum to 1
    *,
    layer: str | None = None,
    weights: np.ndarray | None = None,  # optional per-cell weights (N,)
    eps: float = 1e-12,  # numerical stabilizer
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """
    Compute metacell (soft cluster) means from a membership matrix S
    of size (N_cells, K), using the gene-expression matrix X of size (N_cells, G).

    Returns:
        means: (K x G) CSR  # average expression per metacell
        m:     (K,)         # metacell masses (effective sizes, sum of weights)
    """
    if not layer:
        X = adata.X
    else:
        logger.info(f"Using layer '{layer}' for gene expression matrix")
        X = adata.layers[layer]

    if X is None:
        raise ValueError("Gene expression matrix is required.")

    # Ensure array types we rely on
    S = np.asarray(S, dtype=float)
    N, K = S.shape

    # Validate shapes
    if X.shape[0] != N:
        raise ValueError(f"Rows of X ({X.shape[0]}) must match rows of S ({N}).")

    if transform is not None:
        X = transform(X)  # type: ignore

    # Cell weights
    if weights is None:
        w = np.ones(N, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != N:
            raise ValueError(f"weights must have length {N}, got {w.shape[0]}.")

    # Weighted memberships: (N x K)
    SW = w[:, None] * S

    # Metacell masses m_k = sum_i w_i * S_{ik}  -> (K,)
    m = SW.sum(axis=0, dtype=float)
    # Avoid divide-by-zero; keep behavior stable for near-empty metacells
    m_safe = np.where(m < eps, 1.0, m)

    # Weighted sums per metacell: (K x G)
    # Choose multiplication order based on X storage to avoid unnecessary conversions.
    if sparse.issparse(X):
        # For sparse X, multiply as (G x N) @ (N x K) -> (G x K), then transpose.
        # This uses sparse @ dense -> dense efficiently.
        sum_expr = (X.T @ SW).T  # result is dense (K x G) # type: ignore
    else:
        # For dense X, directly do (K x N) @ (N x G) -> (K x G)
        X_dense = np.asarray(X, dtype=float)
        sum_expr = SW.T @ X_dense  # dense (K x G)

    # Convert sums to means by row-wise division by masses
    means_dense = sum_expr / m_safe[:, None]

    # Return CSR to keep the original function's API
    means_csr = sparse.csr_matrix(means_dense)

    return means_csr, m


def cci_from_means(
    means: sparse.csr_matrix,  # (K x G) metacell means
    L_idx: np.ndarray,  # (P,) ligand column indices
    R_idx: np.ndarray,  # (P,) receptor column indices
    aggregation: Literal["sum", "concatenation"] = "sum",
) -> np.ndarray:
    """
    Compute a single-channel CCI: C = (A) @ (B).T where A=means[:,L] and B=means[:,R],
    optionally after an element-wise transform controlled by `score`.
    """
    assert len(L_idx) == len(R_idx), "L_idx and R_idx must have the same length."
    A = means[:, L_idx].toarray()
    B = means[:, R_idx].toarray()
    if aggregation == "sum":
        # sum over pairs: (K x P) @ (P x K)  → (K x K)
        C = A @ B.T
    elif aggregation == "concatenation":
        logger.info("Using concatenation aggregation for CCI matrix")
        # concatenate over pairs: stack (K x K) for each pair to get (K, K, P)
        C = A[:, None, :] * B[None, :, :]  # (K, K, P)
        logger.info("Finished concatenation")

    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    return np.asarray(C, dtype=float)


def cci_from_adata(
    adata: ad.AnnData,  # input AnnData
    assignment_key: str,  # obs cluster labels (soft memberships)  # type: ignore
    pairs: PairTable,  # interaction pairs
    var_col: str = "UP",  # var column with uppercase gene symbols
    weights: np.ndarray | None = None,
    ridge: float = 1e-8,
    layer: str | None = None,
    transform: Callable | None = None,
    aggregation: Literal["sum", "concatenation"] = "sum",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pipeline: Get metacells -> pair indices -> CCI matrix (single- or multi-channel).

    Returns
    -------
    D_up : np.ndarray
        If aggregation == "sum": shape (N, N).
        If aggregation == "concatenation": shape (N, N, P) where P is #pairs.
    clus : np.ndarray
        Metacell masses (effective sizes), length K.
    """
    logger.info(
        f"Generating CCI matrix from AnnData with assignment_key {assignment_key}"
    )

    # 1) Memberships, metacell means/masses
    S = np.asarray(adata.obsm[assignment_key], dtype=float)  # (N, K)

    means, clus = soft_means_from_S(
        adata, S, layer=layer, weights=weights, transform=transform
    )  # means: (K x G) CSR, clus (masses): (K,)

    # 2) Map LR pairs to var columns & build coarse CCI(s)
    L_idx, R_idx, _ = pair_indices(adata, pairs, var_col=var_col)
    logger.info("Beginning CCI matrix construction")
    C_KK = cci_from_means(means, L_idx, R_idx, aggregation=aggregation)
    logger.info("Finished cci from means")
    assert C_KK.ndim in (2, 3)
    assert (C_KK.ndim == 2 and aggregation == "sum") or (
        C_KK.ndim == 3 and aggregation == "concatenation"
    )

    # 3) Convert means-scale to sums-scale via left/right mass rescaling
    #    (equivalent to H = D @ C_KK @ D but faster and memory-friendly).
    m = np.asarray(clus, dtype=float)  # (K,)
    N = S.shape[0]
    ones = np.ones(N, dtype=float)  # uniform gates for both sides

    # Prolongations (same for all channels here)
    U = prolongation_weighted(S, ones, ridge=ridge)  # (N, K)
    V = prolongation_weighted(S, ones, ridge=ridge)  # (N, K)

    if C_KK.ndim == 2:
        # (K, K) -> rescale by masses on rows and cols
        # H = D @ C_KK @ D == (m[:,None] * C_KK) * m[None,:]
        H = (m[:, None] * C_KK) * m[None, :]
        # Lift to (N, N)
        D_up = lift_dense_two_sided(U, H, V)  # (N, N)

    else:
        logger.info("Lifting multi-channel CCI")
        U_s = U * m
        V_s = V * m

        # (K,K,P) · (N,K)^T over K (the 2nd axis of C_KK and 2nd of V_s)
        tmp = np.tensordot(C_KK, V_s, axes=([1], [1]))  # (K, P, N)
        D_up = np.tensordot(U_s, tmp, axes=([1], [0]))  # (N, P, N)
        D_up = np.moveaxis(D_up, 1, -1)  # (N, N, P)

    CCI = np.asarray(D_up, dtype=float, order="C")
    assert CCI.ndim in (2, 3)
    assert (CCI.ndim == 2 and aggregation == "sum") or (
        CCI.ndim == 3 and aggregation == "concatenation"
    )
    return CCI, clus
