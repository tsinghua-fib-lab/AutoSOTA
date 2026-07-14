"""
cobet.py
--------
CoBET and dCoBET: Copula-based Binary Expansion Tests for Independence.

Core building blocks
--------------------
- bits_from_uniform          : dyadic bit decomposition
- all_nonempty_subsets_indices : enumerate all 2^K − 1 subsets
- features_by_u              : centered binary-expansion features for a 1-D uniform
- build_AB_features           : stack features across d coordinates → A, B matrices
- J_numeric_K                : spectral weight matrix J (numeric integration)
- block_diag, kron_block_diag : linear-algebra helpers
- get_weights                 : assemble W_A, W_B, W_C for 'identity' or 'J' mode
- compute_full_T              : U-statistic T = T1 − 2T2 + T3
- plugin_var_tildeT1          : plug-in variance estimate
- ranks_to_uniforms           : rank-based probability integral transform

Simulation runners
------------------
- run_plugin_only      : power / Type-I sweep for CoBET / dCoBET
- run_multi_n_and_save : multi-sample-size sweep with Excel export
"""

import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm as norm_dist, rankdata

from .utils import (
    clayton_copula_sample_nd,
    TRANSFORM_MAP_ND,
    generate_XY,
)


# ─────────────────────────────────────────────
# 1. Dyadic features
# ─────────────────────────────────────────────

def bits_from_uniform(u, K):
    """
    Decompose a 1-D uniform array into K dyadic bits.

    Parameters
    ----------
    u : ndarray, shape (n,) – values in [0, 1)
    K : int                 – number of dyadic levels

    Returns
    -------
    bits : ndarray, shape (K, n), dtype int  (most-significant bit first)
    """
    M = 1 << K
    z = np.minimum((u * M).astype(int), M - 1)          # 0 … 2^K − 1
    bits = np.array([((z >> (K - 1 - k)) & 1).astype(int) for k in range(K)])
    return bits


def all_nonempty_subsets_indices(K):
    """
    Return all 2^K − 1 non-empty subsets of {1, …, K} as a list of tuples.

    Returns
    -------
    list of tuples, length 2^K − 1
    """
    out = []
    for r in range(1, K + 1):
        out.extend(itertools.combinations(range(1, K + 1), r))
    return out


def features_by_u(u, K, subsets):
    """
    Centered binary-expansion feature matrix for a 1-D uniform vector.

    Parameters
    ----------
    u       : ndarray, shape (n,)
    K       : int
    subsets : list of tuples (from all_nonempty_subsets_indices)

    Returns
    -------
    F : ndarray, shape (2^K − 1, n)  – centered indicator products
    """
    bits = bits_from_uniform(u, K)           # (K, n)
    n = u.shape[0]
    F = np.empty((len(subsets), n), dtype=float)
    for i, S in enumerate(subsets):
        rows = [r - 1 for r in S]
        ind = np.prod(bits[rows, :], axis=0)
        F[i, :] = ind.astype(float) - (2.0 ** (-len(S)))   # centering
    return F


def build_AB_features(X, Y, K, subsets):
    """
    Build feature matrices A and B by stacking rank-based features
    across all d coordinates.

    Parameters
    ----------
    X, Y    : ndarray, shape (n, d)
    K       : int
    subsets : list of tuples

    Returns
    -------
    A, B : ndarray, shape (d · (2^K − 1), n)
    """
    n, d = X.shape
    feats_A, feats_B = [], []
    for r in range(d):
        xu = rankdata(X[:, r]) / (n + 1.0)
        yu = rankdata(Y[:, r]) / (n + 1.0)
        feats_A.append(features_by_u(xu, K, subsets))
        feats_B.append(features_by_u(yu, K, subsets))
    A = np.vstack(feats_A)
    B = np.vstack(feats_B)
    return A, B


def ranks_to_uniforms(X):
    """Convert each column of X to rank-based uniforms in (0, 1)."""
    n, d = X.shape
    U = np.empty_like(X, dtype=float)
    for j in range(d):
        U[:, j] = rankdata(X[:, j]) / (n + 1.0)
    return U


# ─────────────────────────────────────────────
# 2. Weight matrices
# ─────────────────────────────────────────────

def _trapezoid_weights(x):
    w = np.zeros_like(x)
    dx = np.diff(x)
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    w[0]    = 0.5 * (x[1] - x[0])
    w[-1]   = 0.5 * (x[-1] - x[-2])
    return w


def J_numeric_K(K, t_min=1e-4, t_max=100.0, T=2001):
    """
    Compute the (2^K − 1) × (2^K − 1) spectral weight matrix J
    via numeric trapezoidal integration.

    Returns
    -------
    J       : ndarray, shape (2^K − 1, 2^K − 1)
    subsets : list of tuples
    """
    subsets = all_nonempty_subsets_indices(K)
    m = len(subsets)
    t = np.logspace(np.log10(t_min), np.log10(t_max), T)
    w = _trapezoid_weights(t) / (t ** 2)

    inv_pows = np.array([1.0 / (2 ** r) for r in range(1, K + 1)])
    P = np.empty((m, T))
    for i, S in enumerate(subsets):
        vals = np.ones_like(t)
        inS = np.zeros(K, dtype=bool)
        inS[[r - 1 for r in S]] = True
        for r in range(K):
            ang = t * inv_pows[r]
            vals *= np.sin(ang) if inS[r] else np.cos(ang)
        P[i, :] = vals

    J = (P * w) @ P.T
    J = 0.5 * (J + J.T)
    return J, subsets


def block_diag(*mats):
    """Assemble a block-diagonal matrix from a sequence of 2-D arrays."""
    r = sum(m.shape[0] for m in mats)
    c = sum(m.shape[1] for m in mats)
    out = np.zeros((r, c), dtype=float)
    i = j = 0
    for m in mats:
        rr, cc = m.shape
        out[i:i + rr, j:j + cc] = m
        i += rr
        j += cc
    return out


def kron_block_diag(mat, times):
    """Repeat mat as a block-diagonal matrix `times` times."""
    r, c = mat.shape
    out = np.zeros((times * r, times * c), dtype=float)
    for i in range(times):
        out[i * r:(i + 1) * r, i * c:(i + 1) * c] = mat
    return out


def get_weights(K, mode, J_cached=None, subsets=None, d_dims=2):
    """
    Construct weight matrices W_A, W_B, W_C for CoBET / dCoBET.

    Parameters
    ----------
    K         : int
    mode      : {'identity', 'J'}
    J_cached  : ndarray or None  – pre-computed J to avoid recomputation
    subsets   : list of tuples or None
    d_dims    : int  – number of coordinate dimensions

    Returns
    -------
    W_A, W_B : ndarray, shape (d_dims·(2^K−1), d_dims·(2^K−1))
    W_C      : ndarray, shape (2·d_dims·(2^K−1), 2·d_dims·(2^K−1))
    subsets  : list of tuples
    J        : ndarray or None
    """
    if subsets is None:
        subsets = all_nonempty_subsets_indices(K)
    block_rows = len(subsets)
    rows_per_side = d_dims * block_rows

    if mode == "identity":
        W_A = np.eye(rows_per_side)
        W_B = np.eye(rows_per_side)
        J = None

    elif mode == "J":
        if J_cached is None:
            J, subs2 = J_numeric_K(K)
            if subs2 != subsets:
                idx_map = {S: i for i, S in enumerate(subs2)}
                perm = [idx_map[S] for S in subsets]
                J = J[np.ix_(perm, perm)]
        else:
            J = J_cached
        W_A = kron_block_diag(J, d_dims)
        W_B = kron_block_diag(J, d_dims)

    else:
        raise ValueError("mode must be 'identity' or 'J'")

    W_C = block_diag(W_A, W_B)
    return W_A, W_B, W_C, subsets, J


# ─────────────────────────────────────────────
# 3. Core test statistic and variance
# ─────────────────────────────────────────────

def compute_full_T(A, B, W_A, W_B, W_C):
    """
    Compute the full U-statistic T = T1 − 2·T2 + T3.

    Parameters
    ----------
    A, B : ndarray, shape (p, n)  – feature matrices
    W_A  : ndarray, shape (p, p)
    W_B  : ndarray, shape (p, p)
    W_C  : ndarray, shape (2p, 2p)

    Returns
    -------
    T : float
    """
    n = A.shape[1]
    KA = (A.T @ W_A) @ A
    KB = (B.T @ W_B) @ B
    C  = np.vstack((A, B))
    KC = (C.T @ W_C) @ C

    off = ~np.eye(n, dtype=bool)
    T1 = (KA[off] * KB[off]).sum() / (n * (n - 1))

    def _sums(dot):
        S1 = dot.sum() - np.trace(dot)
        row_off = dot.sum(axis=1) - np.diag(dot)
        S2 = np.sum(row_off ** 2)
        S3 = (dot ** 2).sum() - np.trace(dot ** 2)
        return S1, S2, S3

    S1C, S2C, S3C = _sums(KC)
    S1A, S2A, S3A = _sums(KA)
    S1B, S2B, S3B = _sums(KB)

    T2 = ((S2C - S3C) - (S2A - S3A) - (S2B - S3B)) / (
        2 * n * (n - 1) * (n - 2))

    def _term(S1, S2, S3):
        return S1 ** 2 - 4 * (S2 - S3) - 2 * S3

    T3 = (_term(S1C, S2C, S3C) - _term(S1A, S2A, S3A) - _term(S1B, S2B, S3B)) / (
        2 * n * (n - 1) * (n - 2) * (n - 3))

    return T1 - 2 * T2 + T3


def plugin_var_tildeT1(A, B, W_A, W_B, unbiased=True):
    """
    Plug-in variance estimate for the leading term T̃₁.

    Returns
    -------
    var : float  (≥ 0)
    """
    n = A.shape[1]
    A_c = A - A.mean(axis=1, keepdims=True)
    B_c = B - B.mean(axis=1, keepdims=True)
    denom = (n - 1) if unbiased else n
    S_A = (A_c @ A_c.T) / denom
    S_B = (B_c @ B_c.T) / denom
    EA = np.trace(W_A @ S_A @ W_A @ S_A)
    EB = np.trace(W_B @ S_B @ W_B @ S_B)
    return (2.0 / (n * (n - 1))) * EA * EB


# ─────────────────────────────────────────────
# 4. Data generator
# ─────────────────────────────────────────────

def _generate_once(n, theta, b, K, transform_key, subsets, d, rng=None):
    """Generate one (A, B) feature pair from the Clayton + transform DGP."""
    if rng is None:
        rng = np.random.RandomState()
    u = clayton_copula_sample_nd(n, theta, d, rng=rng)
    v = clayton_copula_sample_nd(n, theta, d, rng=rng)
    X, Y = TRANSFORM_MAP_ND[transform_key](u, v, b=b)
    Xu = ranks_to_uniforms(X)
    Yu = ranks_to_uniforms(Y)
    # build features from uniforms directly
    feats_A, feats_B = [], []
    for r in range(d):
        feats_A.append(features_by_u(Xu[:, r], K, subsets))
        feats_B.append(features_by_u(Yu[:, r], K, subsets))
    A = np.vstack(feats_A)
    B = np.vstack(feats_B)
    return A, B


# ─────────────────────────────────────────────
# 5. Simulation runner
# ─────────────────────────────────────────────

def run_plugin_only(
    n, theta, K, b_config, d=2,
    weights_list=("identity", "J"),
    R_eval=200, alpha=0.05, seed=123,
    reuse_J=True, unbiased_plugin=True,
):
    """
    Monte Carlo power / Type-I sweep for CoBET / dCoBET (plug-in variance).

    Parameters
    ----------
    n            : int   – sample size
    theta        : float – Clayton copula parameter
    K            : int   – dyadic depth
    b_config     : dict  – {transform_key: [b_values]}  (b=0 → Type I)
    d            : int   – dimension
    weights_list : tuple – subset of {'identity', 'J'}
    R_eval       : int   – Monte Carlo replications
    alpha        : float – nominal level
    seed         : int
    reuse_J      : bool  – cache J matrix across transforms
    unbiased_plugin : bool

    Returns
    -------
    list of dicts with keys:
        transform, weights, b, metric ('typeI'|'power'), value, d, n, K, theta, alpha, R_eval, seed
    """
    results = []
    zcrit = norm_dist.ppf(1 - alpha)
    rng = np.random.RandomState(seed)

    subsets = all_nonempty_subsets_indices(K)
    J_cached = None
    if reuse_J and "J" in weights_list:
        J_cached, subs_J = J_numeric_K(K)
        if subs_J != subsets:
            idx_map = {S: i for i, S in enumerate(subs_J)}
            perm = [idx_map[S] for S in subsets]
            J_cached = J_cached[np.ix_(perm, perm)]

    for transform_key, b_list in b_config.items():
        if not isinstance(b_list, (list, tuple, np.ndarray)):
            b_list = [b_list]

        for weights_mode in weights_list:
            W_A0, W_B0, W_C0, subsets_used, _ = get_weights(
                K, weights_mode,
                J_cached=(J_cached if weights_mode == "J" else None),
                subsets=subsets,
                d_dims=d,
            )

            # ── Type I (b = 0) ──
            rej = sum(
                (compute_full_T(*_generate_once(n, theta, 0.0, K, transform_key, subsets_used, d, rng),
                                W_A0, W_B0, W_C0)
                 / np.sqrt(plugin_var_tildeT1(*_generate_once(n, theta, 0.0, K, transform_key, subsets_used, d, rng),
                                              W_A0, W_B0, unbiased=unbiased_plugin) + 1e-300)
                 ) > zcrit
                for _ in range(R_eval)
            )
            # recompute properly (two separate calls above share state; redo cleanly)
            rej = 0
            for _ in range(R_eval):
                A, B = _generate_once(n, theta, 0.0, K, transform_key, subsets_used, d, rng)
                T  = compute_full_T(A, B, W_A0, W_B0, W_C0)
                vT = plugin_var_tildeT1(A, B, W_A0, W_B0, unbiased=unbiased_plugin)
                rej += (T / np.sqrt(max(vT, 1e-300))) > zcrit

            results.append({
                "transform": transform_key, "weights": weights_mode,
                "b": 0.0, "metric": "typeI", "value": rej / R_eval,
                "d": d, "n": n, "K": K, "theta": theta,
                "alpha": alpha, "R_eval": R_eval, "seed": seed,
            })

            # ── Power (b > 0) ──
            for b in b_list:
                rej = 0
                for _ in range(R_eval):
                    A, B = _generate_once(n, theta, b, K, transform_key, subsets_used, d, rng)
                    T  = compute_full_T(A, B, W_A0, W_B0, W_C0)
                    vT = plugin_var_tildeT1(A, B, W_A0, W_B0, unbiased=unbiased_plugin)
                    rej += (T / np.sqrt(max(vT, 1e-300))) > zcrit

                results.append({
                    "transform": transform_key, "weights": weights_mode,
                    "b": float(b), "metric": "power", "value": rej / R_eval,
                    "d": d, "n": n, "K": K, "theta": theta,
                    "alpha": alpha, "R_eval": R_eval, "seed": seed,
                })

    return results


def run_multi_n_and_save(
    n_list, theta, d, K, b_config_by_n, weights_list,
    R_eval, alpha, seed,
    xlsx_path="cobet_results.xlsx",
):
    """
    Run run_plugin_only for multiple sample sizes and export results to Excel.

    Parameters
    ----------
    n_list        : list of int
    theta         : float
    d             : int   – dimension
    K             : int
    b_config_by_n : dict  – {n: {transform: [b_values]}}
    weights_list  : tuple – e.g. ('identity', 'J')
    R_eval        : int
    alpha         : float
    seed          : int
    xlsx_path     : str

    Returns
    -------
    df_all : pd.DataFrame  – combined results across all n
    """
    all_rows = []
    writer = None
    engine_used = None

    for eng in ("openpyxl", "xlsxwriter"):
        try:
            writer = pd.ExcelWriter(xlsx_path, engine=eng)
            engine_used = eng
            break
        except Exception:
            continue

    try:
        for n in n_list:
            if n not in b_config_by_n:
                raise ValueError(f"Missing b_config for n={n}")
            res_n = run_plugin_only(
                n=n, theta=theta, K=K, b_config=b_config_by_n[n], d=d,
                weights_list=weights_list, R_eval=R_eval, alpha=alpha,
                seed=seed, reuse_J=True, unbiased_plugin=True,
            )
            df_n = pd.DataFrame(res_n)
            all_rows.extend(res_n)
            if writer is not None:
                df_n.to_excel(writer, index=False, sheet_name=f"n={n}")
            else:
                df_n.to_csv(f"cobet_results_n{n}.csv", index=False)

        df_all = pd.DataFrame(all_rows)
        if writer is not None:
            df_all.to_excel(writer, index=False, sheet_name="combined")
            writer.close()
            print(f"Saved → {xlsx_path}  (engine={engine_used})")
        else:
            comb_path = "cobet_results_combined.csv"
            df_all.to_csv(comb_path, index=False)
            print(f"No Excel engine; saved CSVs (combined → {comb_path})")

    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

    return df_all


# ─────────────────────────────────────────────
# 6. Quick-start example
# ─────────────────────────────────────────────

if __name__ == "__main__":
    n_list = [250, 500, 1000]
    theta, d, K = 2, 5, 4
    alpha, R_eval, seed = 0.05, 200, 123

    b_config_by_n = {
        250:  {"trigU": [0.05, 0.08, 0.1], "expquad": [0.10, 0.15, 0.3],
               "linear": [0.1, 0.2, 0.3],  "logquad": [0.1, 0.30, 0.5]},
        500:  {"trigU": [0.03, 0.05, 0.10], "expquad": [0.07, 0.15, 0.2],
               "linear": [0.05, 0.10, 0.20], "logquad": [0.10, 0.20, 0.3]},
        1000: {"trigU": [0.01, 0.03, 0.05], "expquad": [0.05, 0.07, 0.12],
               "linear": [0.05, 0.08, 0.15], "logquad": [0.07, 0.15, 0.2]},
    }

    df = run_multi_n_and_save(
        n_list=n_list, theta=theta, d=d, K=K,
        b_config_by_n=b_config_by_n,
        weights_list=("identity", "J"),
        R_eval=R_eval, alpha=alpha, seed=seed,
        xlsx_path="cobet_results.xlsx",
    )
    print(df.head(10))
