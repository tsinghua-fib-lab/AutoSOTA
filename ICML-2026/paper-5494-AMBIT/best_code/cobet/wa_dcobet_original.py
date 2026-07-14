"""
wa_dcobet.py
------------
Weight-Adaptive dCoBET (wa-dCoBET).

For each coordinate pair (r, s), the weight matrix (identity vs. J) is
selected by 10-fold SNR comparison, then blended in proportion to the
vote counts and applied to the full dataset.

Public API
----------
aggregated_weights_power      – power / Type-I for the wa-dCoBET statistic
power_and_selection_one_setting – fixed-weight baselines + SNR selection stats
run_full_grid_and_export      – batch runner over n × transform × b, Excel export
pairwise_heatmap_new_stat     – pairwise Z-heatmap with BH-FDR stars
bh_fdr_mask                   – Benjamini-Hochberg FDR correction
"""

import itertools
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm as norm_dist, rankdata

from .utils import (
    clayton_copula_sample_nd,
    TRANSFORM_MAP_ND,
    generate_XY,
)
from .cobet import (
    all_nonempty_subsets_indices,
    features_by_u,
    build_AB_features,
    J_numeric_K,
    block_diag,
    compute_full_T,
    plugin_var_tildeT1,
)


# ─────────────────────────────────────────────
# 1. Weight helpers
# ─────────────────────────────────────────────

def _get_base_weights(K, subsets, d_coords, reuse_J=True):
    """
    Pre-compute identity and J base weight matrices (base_dim × base_dim)
    for a single coordinate pair.

    Returns
    -------
    W_id : ndarray, shape (base_dim, base_dim) – identity
    W_J  : ndarray, shape (base_dim, base_dim) – spectral J
    """
    base_dim = (1 << K) - 1
    W_id = np.eye(base_dim)

    J_cached = None
    if reuse_J:
        J_cached, subsJ = J_numeric_K(K)
        if subsJ != subsets:
            idx = {S: i for i, S in enumerate(subsJ)}
            perm = [idx[S] for S in subsets]
            J_cached = J_cached[np.ix_(perm, perm)]
    else:
        J_cached, _ = J_numeric_K(K)

    W_J = J_cached
    return W_id, W_J


def _precache_full_weights(K, subsets, d_coords, reuse_J=True):
    """
    Pre-compute full-size weight tuples (W_A, W_B, W_C, subsets)
    for both 'identity' and 'J' modes over d_coords coordinates.

    Returns
    -------
    dict: {'identity': (W_A, W_B, W_C, subsets), 'J': ...}
    """
    base_dim = (1 << K) - 1
    W_id_base, W_J_base = _get_base_weights(K, subsets, d_coords, reuse_J=reuse_J)

    result = {}
    for mode, W_base in (("identity", W_id_base), ("J", W_J_base)):
        W_A = block_diag(*([W_base] * d_coords))
        W_B = block_diag(*([W_base] * d_coords))
        W_C = block_diag(W_A, W_B)
        result[mode] = (W_A, W_B, W_C, subsets)
    return result


# ─────────────────────────────────────────────
# 2. Z-statistic helpers
# ─────────────────────────────────────────────

def _Z_stat(A, B, W_A, W_B, W_C, unbiased_plugin=True):
    """Z = T / sqrt(Var̂(T̃₁))."""
    T  = compute_full_T(A, B, W_A, W_B, W_C)
    vT = plugin_var_tildeT1(A, B, W_A, W_B, unbiased=unbiased_plugin)
    return T / np.sqrt(max(vT, 1e-16))


def Z_for_pair(A_pair, B_pair, W_base, unbiased_plugin=True):
    """
    Z-statistic for a single coordinate pair using a given base weight matrix.

    Parameters
    ----------
    A_pair, B_pair : ndarray, shape (base_dim, n)
    W_base         : ndarray, shape (base_dim, base_dim)

    Returns
    -------
    Z : float
    """
    W_C = block_diag(W_base, W_base)
    return _Z_stat(A_pair, B_pair, W_base, W_base, W_C, unbiased_plugin=unbiased_plugin)


# ─────────────────────────────────────────────
# 3. 10-fold SNR blending (pairwise)
# ─────────────────────────────────────────────

def _ten_folds_indices(n, rng):
    idx = rng.permutation(n)
    return np.array_split(idx, 10)


def blended_weight_from_10fold(A_pair, B_pair, W_id, W_J, rng, unbiased_plugin=True):
    """
    Select identity vs. J weighting on each of 10 folds by SNR,
    then return the blended base weight matrix W_blend = w_id·I + w_J·J.

    Parameters
    ----------
    A_pair, B_pair : ndarray, shape (base_dim, n)
    W_id, W_J      : ndarray, shape (base_dim, base_dim)
    rng            : np.random.RandomState

    Returns
    -------
    W_blend : ndarray, shape (base_dim, base_dim)
    w_id    : float – fraction of folds voting identity
    w_J     : float – fraction of folds voting J
    """
    folds = _ten_folds_indices(A_pair.shape[1], rng)
    picks = []
    for fidx in folds:
        A_f, B_f = A_pair[:, fidx], B_pair[:, fidx]
        Z_id = Z_for_pair(A_f, B_f, W_id, unbiased_plugin=unbiased_plugin)
        Z_J  = Z_for_pair(A_f, B_f, W_J,  unbiased_plugin=unbiased_plugin)
        picks.append("identity" if Z_id >= Z_J else "J")

    cnt_id = sum(p == "identity" for p in picks)
    w_id   = cnt_id / 10.0
    w_J    = 1.0 - w_id
    W_blend = w_id * W_id + w_J * W_J
    return W_blend, w_id, w_J


# ─────────────────────────────────────────────
# 4. 10-fold blending (full-dimension, for power sims)
# ─────────────────────────────────────────────

def _blend_full_weights(W_id_tuple, W_J_tuple, w_id, w_J):
    """Blend full-size (W_A, W_B, W_C, subsets) tuples."""
    W_A_id, W_B_id, _, _ = W_id_tuple
    W_A_J,  W_B_J,  _, _ = W_J_tuple
    W_A_new = w_id * W_A_id + w_J * W_A_J
    W_B_new = w_id * W_B_id + w_J * W_B_J
    W_C_new = block_diag(W_A_new, W_B_new)
    return W_A_new, W_B_new, W_C_new, None


def _generate_once_nd(n, theta, b, K, transform_key, subsets, d_coords, rng):
    u = clayton_copula_sample_nd(n, theta, d_coords, rng=rng)
    v = clayton_copula_sample_nd(n, theta, d_coords, rng=rng)
    X, Y = TRANSFORM_MAP_ND[transform_key](u, v, b=b)
    A, B = build_AB_features(X, Y, K, subsets)
    return A, B


# ─────────────────────────────────────────────
# 5. Main simulation functions
# ─────────────────────────────────────────────

def aggregated_weights_power(
    n, theta, K, transform_key, b,
    R_eval=1000, alpha=0.05, seed=123,
    d_coords=10, unbiased_plugin=True, reuse_J=True,
):
    """
    Monte Carlo power estimate for wa-dCoBET.

    Per replicate:
      1. Generate n × d_coords data.
      2. Split into 10 folds; on each fold pick identity vs. J by max Z (SNR).
      3. Blend: W_new = w_id·W_id + w_J·W_J.
      4. Compute Z on the full dataset with W_new.

    Parameters
    ----------
    n             : int
    theta         : float
    K             : int
    transform_key : str
    b             : float  (0 → Type I error)
    R_eval        : int    – replications
    alpha         : float
    seed          : int
    d_coords      : int    – dimension
    unbiased_plugin : bool
    reuse_J       : bool

    Returns
    -------
    dict with keys: transform, b, alpha, R_eval, d, n,
                    power_aggregated, avg_w_identity, avg_w_J,
                    Z_mean_full, Z_std_full
    """
    rng   = np.random.RandomState(seed)
    zcrit = norm_dist.ppf(1 - alpha)

    subsets = all_nonempty_subsets_indices(K)
    W_all   = _precache_full_weights(K, subsets, d_coords, reuse_J=reuse_J)
    W_id    = W_all["identity"]
    W_J     = W_all["J"]

    rejections    = 0
    sel_counts_id = 0
    sel_counts_J  = 0
    Z_full_list   = []

    for _ in range(R_eval):
        A, B = _generate_once_nd(n, theta, b, K, transform_key, subsets, d_coords, rng)

        # 10-fold weight selection
        folds = _ten_folds_indices(n, rng)
        fold_picks = []
        for fidx in folds:
            A_f, B_f = A[:, fidx], B[:, fidx]
            W_A_id, W_B_id, W_C_id, _ = W_id
            W_A_J,  W_B_J,  W_C_J,  _ = W_J
            Z_id = _Z_stat(A_f, B_f, W_A_id, W_B_id, W_C_id, unbiased_plugin)
            Z_J  = _Z_stat(A_f, B_f, W_A_J,  W_B_J,  W_C_J,  unbiased_plugin)
            fold_picks.append("identity" if Z_id >= Z_J else "J")

        cnt_id = sum(p == "identity" for p in fold_picks)
        cnt_J  = 10 - cnt_id
        w_id   = cnt_id / 10.0
        w_J    = cnt_J  / 10.0
        sel_counts_id += cnt_id
        sel_counts_J  += cnt_J

        W_A_new, W_B_new, W_C_new, _ = _blend_full_weights(W_id, W_J, w_id, w_J)
        T_new  = compute_full_T(A, B, W_A_new, W_B_new, W_C_new)
        vT_new = plugin_var_tildeT1(A, B, W_A_new, W_B_new, unbiased=unbiased_plugin)
        Z_new  = T_new / np.sqrt(max(vT_new, 1e-16))
        Z_full_list.append(Z_new)
        if Z_new > zcrit:
            rejections += 1

    return {
        "transform":      transform_key,
        "b":              float(b),
        "alpha":          alpha,
        "R_eval":         R_eval,
        "d":              d_coords,
        "n":              n,
        "power_aggregated": rejections / R_eval,
        "avg_w_identity": sel_counts_id / (10.0 * R_eval),
        "avg_w_J":        sel_counts_J  / (10.0 * R_eval),
        "Z_mean_full":    float(np.mean(Z_full_list)),
        "Z_std_full":     float(np.std(Z_full_list, ddof=1)),
    }


def power_and_selection_one_setting(
    n, theta, K, transform_key, b,
    R_eval=1000, alpha=0.05, seed=123,
    d_coords=10, reuse_J=True, unbiased_plugin=True,
):
    """
    Fixed-weight baseline comparison: identity vs. J, plus SNR-selection power.

    Returns
    -------
    dict with keys:
        transform, b, alpha, R_eval, d, n,
        power_identity, power_J, selected_power,
        pct_selected_identity, pct_selected_J,
        Z_mean_identity, Z_mean_J
    """
    rng   = np.random.RandomState(seed)
    zcrit = norm_dist.ppf(1 - alpha)

    subsets = all_nonempty_subsets_indices(K)
    W_all   = _precache_full_weights(K, subsets, d_coords, reuse_J=reuse_J)

    rejs_by_mode   = defaultdict(int)
    Z_sums_by_mode = defaultdict(float)
    select_counts  = defaultdict(int)
    selected_rejs  = 0

    for _ in range(R_eval):
        A, B = _generate_once_nd(n, theta, b, K, transform_key, subsets, d_coords, rng)
        Zs = {}
        for mode, (W_A, W_B, W_C, _) in W_all.items():
            z = _Z_stat(A, B, W_A, W_B, W_C, unbiased_plugin)
            Zs[mode] = z
            if z > zcrit:
                rejs_by_mode[mode] += 1
            Z_sums_by_mode[mode] += z

        picked = max(Zs, key=Zs.get)
        select_counts[picked] += 1
        if Zs[picked] > zcrit:
            selected_rejs += 1

    return {
        "transform":               transform_key,
        "b":                       float(b),
        "alpha":                   alpha,
        "R_eval":                  R_eval,
        "d":                       d_coords,
        "n":                       n,
        "power_identity":          rejs_by_mode["identity"] / R_eval,
        "power_J":                 rejs_by_mode["J"] / R_eval,
        "selected_power":          selected_rejs / R_eval,
        "pct_selected_identity":   100.0 * select_counts["identity"] / R_eval,
        "pct_selected_J":          100.0 * select_counts["J"] / R_eval,
        "Z_mean_identity":         Z_sums_by_mode["identity"] / R_eval,
        "Z_mean_J":                Z_sums_by_mode["J"] / R_eval,
    }


def run_full_grid_and_export(
    n_list, b_config_by_n,
    theta=2, K=4, d_coords=10,
    transforms=("trigU", "expquad", "linear", "logquad"),
    R_eval=500, alpha=0.05, seed=123,
    unbiased_plugin=True, reuse_J=True,
    out_path=None, also_baseline=True,
):
    """
    Batch runner: wa-dCoBET power over n_list × transforms × b grids,
    with optional fixed-weight baselines. Saves to Excel.

    Parameters
    ----------
    n_list        : list of int
    b_config_by_n : dict  – {n: {transform: [b_values]}}
    theta         : float
    K             : int
    d_coords      : int
    transforms    : tuple of str
    R_eval        : int
    alpha         : float
    seed          : int
    out_path      : str or None  (auto-named if None)
    also_baseline : bool  – also run power_and_selection_one_setting

    Returns
    -------
    df_agg  : pd.DataFrame  – wa-dCoBET results
    df_base : pd.DataFrame  – baseline results (empty if also_baseline=False)
    """
    aggregated_rows = []
    baseline_rows   = []

    for n in n_list:
        config = b_config_by_n.get(n, {})
        for tkey in transforms:
            b_list = config.get(tkey, [])
            for b in b_list:
                agg = aggregated_weights_power(
                    n=n, theta=theta, K=K, transform_key=tkey, b=b,
                    R_eval=R_eval, alpha=alpha, seed=seed,
                    d_coords=d_coords, unbiased_plugin=unbiased_plugin,
                    reuse_J=reuse_J,
                )
                aggregated_rows.append(agg)

                if also_baseline:
                    base = power_and_selection_one_setting(
                        n=n, theta=theta, K=K, transform_key=tkey, b=b,
                        R_eval=R_eval, alpha=alpha, seed=seed,
                        d_coords=d_coords, reuse_J=reuse_J,
                        unbiased_plugin=unbiased_plugin,
                    )
                    baseline_rows.append(base)

    df_agg  = pd.DataFrame(aggregated_rows)
    df_base = pd.DataFrame(baseline_rows) if also_baseline else pd.DataFrame()

    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"wa_dcobet_results_{ts}.xlsx"

    for eng in ("xlsxwriter", "openpyxl"):
        try:
            with pd.ExcelWriter(out_path, engine=eng) as writer:
                df_agg.to_excel(writer, index=False, sheet_name="aggregated")
                if also_baseline and not df_base.empty:
                    df_base.to_excel(writer, index=False, sheet_name="baseline")
            print(f"Saved → {out_path}  (engine={eng})")
            break
        except Exception as exc:
            last_exc = exc
    else:
        # CSV fallback
        base_name = out_path.replace(".xlsx", "")
        df_agg.to_csv(f"{base_name}_aggregated.csv", index=False)
        if also_baseline and not df_base.empty:
            df_base.to_csv(f"{base_name}_baseline.csv", index=False)
        print(f"Excel unavailable ({last_exc}); saved CSVs.")

    return df_agg, df_base


# ─────────────────────────────────────────────
# 6. BH-FDR
# ─────────────────────────────────────────────

def bh_fdr_mask(pvals_2d, q=0.05):
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals_2d : ndarray, shape (d, d)
    q        : float  – FDR level

    Returns
    -------
    reject : bool ndarray, shape (d, d)
    """
    p = pvals_2d.flatten()
    m = p.size
    order    = np.argsort(p)
    p_sorted = p[order]
    thresh   = q * (np.arange(1, m + 1) / m)
    below    = p_sorted <= thresh

    reject = np.zeros(m, dtype=bool)
    if np.any(below):
        kmax = np.max(np.where(below))
        reject[order[:kmax + 1]] = True
    return reject.reshape(pvals_2d.shape)


# ─────────────────────────────────────────────
# 7. Pairwise heatmap
# ─────────────────────────────────────────────

def pairwise_heatmap_new_stat(
    n=500, d_coords=10, theta=2, K=4,
    b=0.1, transform_key="logquad",
    q_fdr=0.05, seed_data=1234, seed_folds=999,
    unbiased_plugin=True, J_reuse=True,
):
    """
    Compute and plot the pairwise Z-statistic heatmap for wa-dCoBET.

    For each coordinate pair (r, s):
      - 10-fold SNR → blended W
      - Final Z on full data
    BH-FDR stars mark significant pairs.

    Parameters
    ----------
    n             : int
    d_coords      : int  – number of coordinate dimensions
    theta         : float
    K             : int
    b             : float  – signal strength
    transform_key : str
    q_fdr         : float  – FDR level
    seed_data     : int
    seed_folds    : int
    unbiased_plugin : bool
    J_reuse       : bool

    Returns
    -------
    dict with keys: Z, T, Var, p, sig_bh, w_id, w_J
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    subsets  = all_nonempty_subsets_indices(K)
    base_dim = (1 << K) - 1

    # ── Generate one dataset ──
    rng_data = np.random.RandomState(seed_data)
    A, B = _generate_once_nd(n, theta, b, K, transform_key, subsets, d_coords, rng_data)

    # ── Base weight matrices (base_dim × base_dim) ──
    W_id_base, W_J_base = _get_base_weights(K, subsets, d_coords, reuse_J=J_reuse)

    # ── Output matrices ──
    Zmat = np.zeros((d_coords, d_coords))
    Tmat = np.zeros((d_coords, d_coords))
    Vmat = np.zeros((d_coords, d_coords))
    wid  = np.zeros((d_coords, d_coords))
    wj   = np.zeros((d_coords, d_coords))

    rng_folds = np.random.RandomState(seed_folds)

    for r in range(d_coords):
        A_r = A[r * base_dim:(r + 1) * base_dim, :]
        for s in range(d_coords):
            B_s = B[s * base_dim:(s + 1) * base_dim, :]

            W_blend, w_id_val, w_J_val = blended_weight_from_10fold(
                A_r, B_s, W_id_base, W_J_base,
                rng=rng_folds, unbiased_plugin=unbiased_plugin,
            )
            wid[r, s] = w_id_val
            wj[r, s]  = w_J_val

            W_C = block_diag(W_blend, W_blend)
            T_rs = compute_full_T(A_r, B_s, W_blend, W_blend, W_C)
            v_rs = plugin_var_tildeT1(A_r, B_s, W_blend, W_blend, unbiased=unbiased_plugin)
            Z_rs = T_rs / np.sqrt(max(v_rs, 1e-16))

            Tmat[r, s] = T_rs
            Vmat[r, s] = v_rs
            Zmat[r, s] = Z_rs

    Pmat   = 1.0 - norm_dist.cdf(Zmat)
    sig_bh = bh_fdr_mask(Pmat, q=q_fdr)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 7))
    norm = mcolors.Normalize(vmin=float(np.min(Zmat)), vmax=float(np.max(Zmat)))
    im = ax.imshow(Zmat, aspect="equal", cmap="viridis_r", norm=norm)

    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Z statistic", fontsize=11)

    ax.set_title(f"Pairwise Z heatmap | ★ = BH-FDR discoveries (q={q_fdr})")
    ax.set_xlabel("V coordinates")
    ax.set_ylabel("U coordinates")
    ax.set_xticks(range(d_coords))
    ax.set_yticks(range(d_coords))
    ax.set_xticklabels([f"v{i + 1}" for i in range(d_coords)])
    ax.set_yticklabels([f"u{i + 1}" for i in range(d_coords)])

    for r in range(d_coords):
        for s in range(d_coords):
            if sig_bh[r, s]:
                ax.text(s, r, "★", ha="center", va="center",
                        fontsize=13, fontweight="bold", color="white")

    plt.tight_layout()
    plt.show()

    print(f"[BH-FDR] q={q_fdr}, discoveries: {int(sig_bh.sum())} / {d_coords * d_coords}")
    print(f"[weights] mean w_id={wid.mean():.3f},  mean w_J={wj.mean():.3f}")

    return {
        "Z": Zmat, "T": Tmat, "Var": Vmat,
        "p": Pmat, "sig_bh": sig_bh,
        "w_id": wid, "w_J": wj,
    }


# ─────────────────────────────────────────────
# 8. Quick-start example
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Pairwise heatmap
    out = pairwise_heatmap_new_stat(
        n=500, d_coords=10, theta=2, K=4,
        b=0.4, transform_key="logquad",
        q_fdr=0.05, seed_data=123,
        unbiased_plugin=True, J_reuse=True,
    )

    # Power grid
    n_list = [250, 500, 1000]
    b_config_by_n = {
        250:  {"trigU": [0, 0.05, 0.08, 0.1],  "logquad": [0, 0.1, 0.3, 0.5]},
        500:  {"trigU": [0, 0.03, 0.05, 0.10], "logquad": [0, 0.1, 0.2, 0.3]},
        1000: {"trigU": [0, 0.01, 0.03, 0.05], "logquad": [0, 0.07, 0.15, 0.2]},
    }
    df_agg, df_base = run_full_grid_and_export(
        n_list=n_list, b_config_by_n=b_config_by_n,
        theta=2, K=4, d_coords=10,
        R_eval=200, alpha=0.05, seed=123,
        out_path="wa_dcobet_results.xlsx",
    )
    print(df_agg.head())
