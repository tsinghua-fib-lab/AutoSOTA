# solver.py
"""
Multi-channel (vector-edge) Gromov-Wasserstein and Fused Gromov-Wasserstein solvers
compatible with POT's optimizers, without modifying POT internals.

Supports:
- C1: (ns, ns) or (ns, ns, d)
- C2: (nt, nt) or (nt, nt, d)
- Optional Mahalanobis metric Q (d x d) on channels via one-time projection.

Retains POT's:
- Conditional Gradient (Frank-Wolfe) loop and Armijo option
- Closed-form quadratic line search (kept valid for squared Mahalanobis)
- Backends and dtype handling
"""

from __future__ import annotations

import warnings

import numpy as np
from ot.backend import NumpyBackend, get_backend
from ot.optim import cg, line_search_armijo, solve_1d_linesearch_quad

# POT imports (public API)
from ot.utils import list_to_array, unif

# ---------- helpers ----------


def _as_channel_tensor(C):
    """Represent scalar structures as one-channel tensors."""
    C = np.asarray(C)
    if C.ndim == 2:
        return C[:, :, None]
    if C.ndim == 3:
        return C
    raise ValueError(f"Structure matrix must be 2D or 3D, got shape {C.shape}.")


def _quad_value_true(C1, C2, T, p, q):
    """
    Canonical GW quadratic term:
      sum_{i,k,j,l} ||C1[i,k]-C2[j,l]||^2 * T[i,j] * T[k,l]
    Works for 2D (scalar edges) and 3D (channels already projected by Q^{1/2}).
    """
    if C1.ndim == 2:
        C1n2 = C1**2
        C2n2 = C2**2
    else:
        C1n2 = np.sum(C1**2, axis=2)
        C2n2 = np.sum(C2**2, axis=2)

    term1 = np.sum(C1n2 * np.outer(p, p))
    term2 = np.sum(C2n2 * np.outer(q, q))
    cross = np.sum(_bilinear(C1, T, C2) * T)
    return term1 + term2 - 2.0 * cross


def _transpose_structure(C):
    """Transpose structure matrix (2D) or swap axes (0,1) for 3D."""
    return C.T if C.ndim == 2 else np.transpose(C, (1, 0, 2))


def _is_sym_structure(C, atol=1e-10):
    """Symmetry test for 2D or 3D structure matrices."""
    if C.ndim == 2:
        return np.allclose(C, C.T, atol=atol)
    return np.allclose(C, np.transpose(C, (1, 0, 2)), atol=atol)


def _project_channels(D, Q: np.ndarray | None) -> np.ndarray:
    """
    Apply Mahalanobis sqrt(Q) to the last dim of D.
    If Q is None, returns D unchanged. D can be 2D (ignored) or 3D.
    """
    if Q is None or D.ndim == 2:
        return D
    # Cholesky is fastest; fall back to eigh for semi-definite
    try:
        L = np.linalg.cholesky(Q)
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(Q)
        evals = np.clip(evals, 0.0, None)
        L = evecs @ np.diag(np.sqrt(evals))
    # tensordot over channel dim
    return np.tensordot(D, L, axes=([2], [0]))  # (n, n, d)


def _bilinear(C1, X, C2):
    """
    Sum over channels of C1_r @ X @ C2_r^T.

    Returns:
        (ns, nt) matrix.
    """
    if C1.ndim == 2:
        # classic (single channel)
        return C1 @ X @ C2.T
    # 3D case: einsum is efficient and readable
    # (i,k,r),(k,j),(j,l,r) -> (i,l)
    # return np.einsum("ikr,kj,jlr->il", C1, X, C2, optimize=True)
    return np.einsum("ikr,kl,jlr->ij", C1, X, C2, optimize=True)


def _init_matrix_multich(C1, C2, p, q, Q):
    """
    Build constC and channel stacks hC1, hC2 for multi-channel squared loss.

    For 2D, falls back to POT's algebra (compatibility kept).
    For 3D, uses ||edge||^2 per pair summed over channels.
    """
    C1 = _as_channel_tensor(C1)
    C2 = _as_channel_tensor(C2)

    if C1.shape[-1] != C2.shape[-1]:
        raise ValueError(
            f"C1 and C2 must have the same number of channels, got "
            f"{C1.shape[-1]} and {C2.shape[-1]}."
        )
    if Q is None:
        Q = np.eye(C1.shape[-1])
    Q = np.asarray(Q, dtype=float)

    # Sum of squared channel norms.
    # shapes: (ns, ns) and (nt, nt)
    C1n2 = np.einsum("ijk, kl, ijl -> ij", C1, Q, C1)
    C2n2 = np.einsum("ijk, kl, ijl -> ij", C2, Q, C2)

    a = C1n2 @ p  # (ns,)
    b = C2n2 @ q  # (nt,)
    constC = a[:, None] + b[None, :]  # (ns, nt)

    hC1 = C1
    hC2 = 2 * np.einsum("ij, abj -> abi", Q, C2)  # absorb factor 2 in hC2

    return constC, hC1, hC2


def _gwloss(constC, hC1, hC2, G) -> float:
    """
    GW loss under squared (Mahalanobis) vector-edge mismatch.
    """
    cst = np.sum(constC * G)
    cross = np.sum(_bilinear(hC1, G, hC2) * G)
    return cst - 1.0 * cross


def _gwggrad(constC, hC1, hC2, G):
    """
    Gradient wrt G for the multi-channel squared loss.
    """
    return 2 * (constC - _bilinear(hC1, G, hC2))


def _solve_gromov_linesearch_multich(
    G,
    deltaG,
    cost_G,
    C1,
    C2,
    M,
    reg,
    alpha_min=None,
    alpha_max=None,
    nx=None,
    symmetric=False,
):
    """
    Closed-form quadratic line search for multi-channel squared loss.
    Mirrors POT's solve_gromov_linesearch but uses _bilinear.
    """
    if nx is None:
        if isinstance(M, int | float):  # type: ignore
            nx = get_backend(G, deltaG, C1, C2)
        else:
            nx = get_backend(G, deltaG, C1, C2, M)

    dot = _bilinear(C1, deltaG, C2)  # (ns, nt)
    a = -reg * nx.sum(dot * deltaG)

    if symmetric:
        b = nx.sum(M * deltaG) - 2 * reg * nx.sum(_bilinear(C1, G, C2) * deltaG)
    else:
        b = nx.sum(M * deltaG) - reg * (
            nx.sum(_bilinear(C1, G, C2) * deltaG)
            + nx.sum(_bilinear(C1, deltaG, C2) * G)
        )

    alpha = solve_1d_linesearch_quad(a, b)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)

    # update cost via the quadratic model
    cost_G = cost_G + a * (alpha**2) + b * alpha
    return alpha, 1, cost_G


def fused_gromov_wasserstein_multi(
    M,
    C1,
    C2,
    p=None,
    q=None,
    Q: np.ndarray | None = None,
    symmetric: bool | None = None,
    alpha: float = 0.5,
    armijo: bool = False,
    G0=None,
    log: bool = False,
    max_iter: int = int(1e4),
    tol_rel: float = 1e-9,
    tol_abs: float = 1e-9,
    **kwargs,
):
    r"""
    Multi-channel Fused Gromov-Wasserstein with squared Mahalanobis vector-edge loss.

    Args:
        M: (ns, nt) linear feature cost matrix (unchanged).
        C1: (ns, ns) or (ns, ns, d) structure tensor (source).
        C2: (nt, nt) or (nt, nt, d) structure tensor (target).
        p, q: optional weights, default uniform.
        Q: (d, d) PSD Mahalanobis matrix on channels. If None, identity.
        symmetric: if None, auto-detect symmetry.
        alpha: trade-off parameter in [0,1].
        armijo, G0, log, max_iter, tol_rel, tol_abs: same semantics as POT.
        **kwargs: forwarded to ot.optim.cg.

    Returns:
        T (ns, nt) and optional log dict (if log=True).
    """
    arr = [C1, C2, M]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=M)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(C2.shape[0], type_as=M)
    if G0 is not None:
        G0_ = G0
        arr.append(G0)

    nx = get_backend(*arr)
    p0, q0, C10, C20, M0, alpha0 = p, q, C1, C2, M, alpha

    p = nx.to_numpy(p0)
    q = nx.to_numpy(q0)
    C1_np = nx.to_numpy(C10)
    C2_np = nx.to_numpy(C20)
    M_np = nx.to_numpy(M0)
    alpha = float(alpha0)
    C1_np = _as_channel_tensor(C1_np)
    C2_np = _as_channel_tensor(C2_np)
    if C1_np.shape[-1] != C2_np.shape[-1]:
        raise ValueError(
            f"C1 and C2 must have the same number of channels, got "
            f"{C1_np.shape[-1]} and {C2_np.shape[-1]}."
        )
    if Q is None:
        Q = np.eye(C1_np.shape[-1])
    Q = np.asarray(Q, dtype=float)

    # Optional Mahalanobis projection
    C1_np = _project_channels(C1_np, Q)
    C2_np = _project_channels(C2_np, Q)

    if symmetric is None:
        symmetric = _is_sym_structure(C1_np) and _is_sym_structure(C2_np)

    if G0 is None:
        G0_np = p[:, None] * q[None, :]
    else:
        G0_np = nx.to_numpy(G0_)
        np.testing.assert_allclose(G0_np.sum(axis=1), p, atol=1e-6)
        np.testing.assert_allclose(G0_np.sum(axis=0), q, atol=1e-6)

    np_ = NumpyBackend()

    constC, hC1, hC2 = _init_matrix_multich(C1_np, C2_np, p, q, Q)

    def f(G):
        # POT’s cg expects the pure quadratic term in `f`; linear gets handled via M/reg
        return _gwloss(constC, hC1, hC2, G)

    if symmetric:

        def df(G):
            return _gwggrad(constC, hC1, hC2, G)
    else:
        constCt, hC1t, hC2t = _init_matrix_multich(
            _transpose_structure(C1_np), _transpose_structure(C2_np), p, q, Q
        )

        # Symmetrize the gradient contribution for nonsymmetric structures.
        def df(G):
            return 0.5 * (
                _gwggrad(constC, hC1, hC2, G) + _gwggrad(constCt, hC1t, hC2t, G)
            )

    if armijo:

        def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kw):  # type: ignore
            return line_search_armijo(cost, G, deltaG, Mi, cost_G, nx=np_, **kw)
    else:

        def line_search(cost, G, deltaG, Mi, cost_G, df_G, **kw):
            return _solve_gromov_linesearch_multich(
                G,
                deltaG,
                cost_G,
                hC1,
                hC2,
                M=(1 - alpha) * M_np,
                reg=alpha,
                nx=np_,
                symmetric=symmetric,
                **kw,
            )

    if not nx.is_floating_point(M0):
        warnings.warn(
            "Input feature matrix consists of integers. The transport plan will be "
            "cast accordingly, possibly losing precision. Provide float inputs.",
            stacklevel=2,
        )

    if log:
        res, logd = cg(
            p,
            q,
            (1 - alpha) * M_np,
            alpha,
            f,
            df,
            G0_np,
            line_search,
            log=True,
            numItermax=max_iter,
            stopThr=tol_rel,
            stopThr2=tol_abs,
            **kwargs,
        )
        # Recompute canonical terms from final T (matches POT’s definition)
        quad = _quad_value_true(C1_np, C2_np, res, p, q)
        lin = float(np.sum(M_np * res))
        fgw_true = alpha * quad + (1 - alpha) * lin

        logd["fgw_dist"] = nx.from_numpy(np.array(fgw_true), type_as=M0)
        logd["quad_loss"] = nx.from_numpy(np.array(alpha * quad), type_as=M0)
        logd["lin_loss"] = nx.from_numpy(np.array((1 - alpha) * lin), type_as=M0)
        logd["u"] = nx.from_numpy(logd["u"], type_as=M0)
        logd["v"] = nx.from_numpy(logd["v"], type_as=M0)
        return nx.from_numpy(res, type_as=M0), logd
    else:
        res = cg(
            p,
            q,
            (1 - alpha) * M_np,
            alpha,
            f,
            df,
            G0_np,
            line_search,
            log=False,
            numItermax=max_iter,
            stopThr=tol_rel,
            stopThr2=tol_abs,
            **kwargs,
        )
        return nx.from_numpy(res, type_as=M0)
