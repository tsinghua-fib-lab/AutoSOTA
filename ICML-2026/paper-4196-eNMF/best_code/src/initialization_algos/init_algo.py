# NMF clustering initialization algorithms

import numpy as np
from .pba_algo_utils import nmf_init_pba


def nmf_init_random(V, r, seed=0, scale=0.01, dist="uniform"):
    rng = np.random.default_rng(seed)
    m, n = V.shape
    if dist == "uniform":
        W = rng.random((m, r)) * scale
        H = rng.random((r, n)) * scale
    elif dist == "normal":
        W = np.maximum(rng.normal(loc=0.0, scale=scale, size=(m, r)), 0.0)
        H = np.maximum(rng.normal(loc=0.0, scale=scale, size=(r, n)), 0.0)
    else:
        raise ValueError("dist must be 'uniform' or 'normal'")
    return W, H


def _safe_norm(x):
    return np.sqrt(np.sum(x * x))


def nmf_init_nndsvd(V, r, variant="nndsvd", seed=0, eps=1e-12):
    """
    variant:
      - 'nndsvd'  : pure NNDSVD (zeros kept)
      - 'nndsvda' : fill zeros with mean(V)
      - 'nndsvdar': fill zeros with small random values * mean(V)
    """
    if np.any(V < 0):
        raise ValueError("V must be nonnegative for NMF init.")
    rng = np.random.default_rng(seed)
    m, n = V.shape

    # Economy SVD
    U, S, VT = np.linalg.svd(V, full_matrices=False)

    r_eff = min(r, U.shape[1], VT.shape[0])
    W = np.zeros((m, r), dtype=float)
    H = np.zeros((r, n), dtype=float)

    # First component (nonnegative by abs)
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(VT[0, :])

    # Other components
    for k in range(1, r_eff):
        u = U[:, k]
        v = VT[k, :]

        u_p = np.maximum(u, 0)
        u_n = np.maximum(-u, 0)
        v_p = np.maximum(v, 0)
        v_n = np.maximum(-v, 0)

        n_u_p, n_v_p = _safe_norm(u_p), _safe_norm(v_p)
        n_u_n, n_v_n = _safe_norm(u_n), _safe_norm(v_n)

        m_p = n_u_p * n_v_p
        m_n = n_u_n * n_v_n

        if m_p >= m_n:
            u_k = u_p / max(n_u_p, eps)
            v_k = v_p / max(n_v_p, eps)
            sigma = m_p
        else:
            u_k = u_n / max(n_u_n, eps)
            v_k = v_n / max(n_v_n, eps)
            sigma = m_n

        W[:, k] = np.sqrt(S[k] * sigma) * u_k
        H[k, :] = np.sqrt(S[k] * sigma) * v_k

    # If r > rank(SVD dims), fill remaining with small random
    if r_eff < r:
        meanV = float(np.mean(V))
        W[:, r_eff:] = rng.random((m, r - r_eff)) * (0.01 * meanV + eps)
        H[r_eff:, :] = rng.random((r - r_eff, n)) * (0.01 * meanV + eps)

    # Variants: fill zeros
    meanV = float(np.mean(V))
    if variant == "nndsvd":
        return W, H
    elif variant == "nndsvda":
        W[W == 0] = meanV
        H[H == 0] = meanV
        return W, H
    elif variant == "nndsvdar":
        small = 0.01 * meanV
        W[W == 0] = small * rng.random(np.sum(W == 0)) + eps
        H[H == 0] = small * rng.random(np.sum(H == 0)) + eps
        return W, H
    else:
        raise ValueError("variant must be 'nndsvd', 'nndsvda', or 'nndsvdar'")


def nmf_init_kmeans_columns(V, r, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    m, n = V.shape

    # init centers by picking random columns
    idx = rng.choice(n, size=r, replace=False if r <= n else True)
    C = V[:, idx].copy()  # (m, r)

    # assignments
    assign = np.zeros(n, dtype=int)

    for _ in range(iters):
        # compute distances to centers (squared L2)
        # dist(j,k) = ||V[:,j] - C[:,k]||^2
        V2 = np.sum(V * V, axis=0, keepdims=True)  # (1, n)
        C2 = np.sum(C * C, axis=0, keepdims=True).T  # (r, 1)
        dist = C2 + V2 - 2.0 * (C.T @ V)  # (r, n)
        new_assign = np.argmin(dist, axis=0)

        if np.all(new_assign == assign):
            break
        assign = new_assign

        # update centers
        for k in range(r):
            cols = V[:, assign == k]
            if cols.size == 0:
                # re-seed empty cluster
                C[:, k] = V[:, rng.integers(0, n)]
            else:
                C[:, k] = np.mean(cols, axis=1)

    W = np.maximum(C, 0.0)
    H = np.zeros((r, n), dtype=float)
    H[assign, np.arange(n)] = 1.0
    return W, H


# Double check NICA algorithm
def _sym_decorrelation(T, eps=1e-12):
    """
    Symmetric decorrelation:
        T <- (T T^T)^(-1/2) T
    Ensures T stays (approximately) orthonormal.
    """
    A = T @ T.T
    # eigen-decompose (A is symmetric PSD)
    d, E = np.linalg.eigh(A)
    d = np.maximum(d, eps)
    A_inv_sqrt = E @ np.diag(1.0 / np.sqrt(d)) @ E.T
    return A_inv_sqrt @ T


def _pca_project_no_center(M, k):
    """
    PCA projection without centering (as described in the NICA-for-NMF algorithm). :contentReference[oaicite:2]{index=2}
    Returns E_k (m x k) and projected X = E_k^T M (k x n).
    """
    m, n = M.shape
    # covariance-like matrix without centering
    C = (M @ M.T) / max(n, 1)
    d, E = np.linalg.eigh(C)
    idx = np.argsort(d)[::-1]
    E = E[:, idx]
    d = d[idx]
    E_k = E[:, :k]
    X = E_k.T @ M
    return E_k, X


def _whiten_no_center(X, eps=1e-12):
    """
    Whiten X without centering: Z = V X, with V = E D^{-1/2} E^T. :contentReference[oaicite:3]{index=3}
    """
    k, n = X.shape
    C = (X @ X.T) / max(n, 1)
    d, E = np.linalg.eigh(C)
    d = np.maximum(d, eps)
    V = E @ np.diag(1.0 / np.sqrt(d)) @ E.T
    Z = V @ X
    return V, Z


def nmf_init_nica(M, r, maxiter=500, tol=1e-7, gamma=0.05, seed=0, eps=1e-12):
    """
    NICA initialization for NMF.
    Input:
      M: (m,n) nonnegative data matrix
      r: target rank
    Output:
      W0: (m,r) nonnegative init
      H0: (r,n) nonnegative init

    Core loop follows:
      Y = T Z
      t_u <- t_u - 2*gamma*sum_n min(0, y_un) z_un
      T <- (T T^T)^(-1/2) T   (symmetric decorrelation)
    :contentReference[oaicite:4]{index=4}
    """
    if np.any(M < 0):
        raise ValueError("M must be nonnegative for NICA init.")
    rng = np.random.default_rng(seed)
    m, n = M.shape
    k = r

    # 1) PCA (no centering)
    E_k, X = _pca_project_no_center(M, k)

    # 2) Whitening (no centering)
    V, Z = _whiten_no_center(X, eps=eps)  # Z is (k,n)

    # 3) Initialize orthonormal rotation T
    A = rng.normal(size=(k, k))
    Q, _ = np.linalg.qr(A)
    T = Q

    # Helper: objective proxy = total negative mass in Y
    def neg_mass(Y):
        return np.sum(np.maximum(-Y, 0.0))

    prev = None
    for _ in range(maxiter):
        Y = T @ Z  # (k,n)

        # 4) Gradient step: for each row u, use min(0, y_un)
        neg = np.minimum(Y, 0.0)  # (k,n) negative part (<=0)
        # row-wise update: t_u <- t_u - 2*gamma * sum_n (neg[u,n] * Z[:,n])
        # Vectorized: T <- T - 2*gamma * (neg @ Z.T)
        T_new = T - 2.0 * gamma * (neg @ Z.T)

        # 5) Symmetric decorrelation
        T_new = _sym_decorrelation(T_new, eps=eps)

        cur = neg_mass(T_new @ Z)
        if prev is not None and abs(prev - cur) <= tol * (prev + eps):
            T = T_new
            break
        prev = cur
        T = T_new

    # 6) Build H, and map back to W:
    # Y = T V E_k^T M  =>  M ≈ E_k (T V)^(-1) Y
    H0 = T @ Z  # (r,n)
    TV = T @ V
    W0 = E_k @ np.linalg.pinv(TV)  # (m,r)

    # 7) Nonnegativize (simple projection; common in practice)
    W0 = np.maximum(W0, 0.0)
    H0 = np.maximum(H0, 0.0)

    # Optional: scale so that W0 H0 matches M roughly (helps MU)
    WH = W0 @ H0
    num = np.sum(M * WH)
    den = np.sum(WH * WH) + eps
    alpha = num / den
    W0 *= alpha

    return W0, H0


def get_init_factors(data_x, r, init_method="nndsvdar", seed=0, **kwargs):
    """
    Unified NMF initialization interface.

    Parameters
    ----------
    data_x : ndarray (m, n)
        Nonnegative data matrix V.
    r : int
        Target NMF rank / number of clusters.
    init_method : str
        One of:
          - "random"
          - "nndsvd", "nndsvda", "nndsvdar"
          - "kmeans"
          - "nica"
    seed : int
        Random seed.
    kwargs : dict
        Extra parameters forwarded to the selected initializer.

    Returns
    -------
    W0 : ndarray (m, r)
    H0 : ndarray (r, n)
    """

    V = np.asarray(data_x)
    if np.any(V < 0):
        raise ValueError("NMF initialization requires nonnegative data.")

    init_method = init_method.lower()

    # ---- Random ----
    if init_method == "random":
        scale = kwargs.get("scale", 0.01)
        dist = kwargs.get("dist", "uniform")
        return nmf_init_random(V, r, seed=seed, scale=scale, dist=dist)

    # ---- NNDSVD family ----
    elif init_method in ("nndsvd", "nndsvda", "nndsvdar"):
        return nmf_init_nndsvd(
            V,
            r,
            variant=init_method,
            seed=seed,
            eps=kwargs.get("eps", 1e-12),
        )

    # ---- K-means (columns) ----
    elif init_method in ("kmeans", "kmeans_columns"):
        iters = kwargs.get("iters", 20)
        return nmf_init_kmeans_columns(
            V,
            r,
            iters=iters,
            seed=seed,
        )

    # ---- NICA ----
    elif init_method == "nica":
        return nmf_init_nica(
            V,
            r,
            maxiter=kwargs.get("maxiter", 500),
            tol=kwargs.get("tol", 1e-7),
            gamma=kwargs.get("gamma", 0.05),
            seed=seed,
            eps=kwargs.get("eps", 1e-12),
        )

    elif init_method in ("pso", "de", "fss"):
        return nmf_init_pba(
            V,
            r,
            algo=init_method,
            pop=kwargs.get("pop", 40),
            eval_budget=kwargs.get("eval_budget", 2500),
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unknown init_method='{init_method}'. "
            "Choose from: random, nndsvd, nndsvda, nndsvdar, kmeans, nica, pso, de, fss."
        )
