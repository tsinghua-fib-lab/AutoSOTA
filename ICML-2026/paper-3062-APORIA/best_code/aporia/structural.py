"""
Structural analysis of response geometry.

For each (model, prompt) pair this module produces:

* intra-/inter-class distance distributions in the original embedding space
  and in the 1-D Fisher space (``D_GG``, ``D_HH``, ``D_GH`` and the ``_z``
  counterparts);
* the Wasserstein distance ``W(D_GG, D_HH)`` together with a label-permutation
  null distribution;
* a scalar summary row (``n_G``, ``n_H``, ``W_GG_HH``, ``W_GG_HH_z`` …).

The :func:`run_structural_analysis` driver walks the full dataset and
optionally persists everything to disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from tqdm.auto import tqdm

from .caching import (
    ensure_dir,
    load_structural_cache,
    save_structural_cache,
    structural_cache_exists,
)
from .config import Config
from .data import extract_prompt_data, split_by_label
from .projections import FisherProjection


# ============================================================
# ===================== DISTANCE METRICS =====================
# ============================================================

def compute_distance_distributions(
    X_G: np.ndarray,
    X_H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(D_GG, D_HH, D_GH)`` Euclidean distance distributions."""
    D_GG = pdist(X_G) if len(X_G) > 1 else np.array([])
    D_HH = pdist(X_H) if len(X_H) > 1 else np.array([])
    D_GH = (
        cdist(X_G, X_H).ravel()
        if len(X_G) > 0 and len(X_H) > 0
        else np.array([])
    )
    return D_GG, D_HH, D_GH


def wasserstein_GG_HH(D_GG: np.ndarray, D_HH: np.ndarray) -> float:
    if len(D_GG) == 0 or len(D_HH) == 0:
        return np.nan
    return wasserstein_distance(D_GG, D_HH)


def wasserstein_null_model(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 100,
    random_state: int | None = None,
) -> np.ndarray:
    """Null distribution of ``W(D_GG, D_HH)`` under random class relabelling."""
    rng = np.random.default_rng(random_state)
    W_null: list[float] = []

    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        X_Gp, X_Hp = split_by_label(X, y_perm)
        if len(X_Gp) < 2 or len(X_Hp) < 2:
            continue
        D_GG_p, D_HH_p, _ = compute_distance_distributions(X_Gp, X_Hp)
        W_null.append(wasserstein_GG_HH(D_GG_p, D_HH_p))

    return np.asarray(W_null)


# ============================================================
# ================== PER-PROMPT ANALYSIS =====================
# ============================================================

def analyse_prompt(
    df: pd.DataFrame,
    model_id: int,
    prompt_id: int,
    cfg: Config,
    lambda_reg: float | None = None,
    n_permutations: int | None = None,
    random_state: int | None = None,
) -> dict:
    """Full geometric analysis of one (model, prompt) pair.

    ``n_permutations=0`` (or any falsy value other than ``None``) skips
    the null-model loop; ``None`` falls back to ``cfg.experiment``.
    """
    lambda_reg     = cfg.experiment.best_lambda  if lambda_reg is None     else lambda_reg
    n_permutations = cfg.experiment.n_permutations if n_permutations is None else n_permutations
    random_state   = cfg.experiment.random_state if random_state is None   else random_state

    X, y = extract_prompt_data(df, model_id, prompt_id, cfg)
    X_G, X_H = split_by_label(X, y)
    n_G, n_H = len(X_G), len(X_H)

    res: dict = {
        "model_id":  model_id,
        "prompt_id": prompt_id,
        "n_G": n_G,
        "n_H": n_H,
    }

    # ---- original space distances ----
    D_GG, D_HH, D_GH = compute_distance_distributions(X_G, X_H)
    res["D_GG"], res["D_HH"], res["D_GH"] = D_GG, D_HH, D_GH
    res["W_GG_HH"] = wasserstein_GG_HH(D_GG, D_HH)

    # ---- Fisher 1D space ----
    if n_G >= 2 and n_H >= 2:
        proj = FisherProjection(lambda_reg=lambda_reg).fit(X, y)
        Z = proj.transform(X)
        Z_G, Z_H = Z[~y.astype(bool)], Z[y.astype(bool)]

        D_GG_z, D_HH_z, D_GH_z = compute_distance_distributions(Z_G, Z_H)

        res["v_fisher"] = proj.v
        res["D_GG_z"], res["D_HH_z"], res["D_GH_z"] = D_GG_z, D_HH_z, D_GH_z
        res["W_GG_HH_z"] = wasserstein_GG_HH(D_GG_z, D_HH_z)

    # ---- null model ----
    if n_permutations and n_G >= 2 and n_H >= 2:
        W_null = wasserstein_null_model(
            X, y,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        res["W_null_samples"] = W_null
        res["W_null_mean"] = float(W_null.mean()) if len(W_null) else None
        res["W_null_std"]  = float(W_null.std())  if len(W_null) else None
    else:
        res["W_null_samples"] = None
        res["W_null_mean"] = None
        res["W_null_std"] = None

    return res


def collect_prompt_result(res: dict, min_per_class_plot: int = 5) -> tuple[dict, dict]:
    """Split an ``analyse_prompt`` output into a flat scalar row + geometry payload."""
    m, p = res["model_id"], res["prompt_id"]
    n_G, n_H = res["n_G"], res["n_H"]
    n_total = n_G + n_H

    frac_G = n_G / n_total if n_total > 0 else np.nan
    frac_H = n_H / n_total if n_total > 0 else np.nan

    row = {
        "model_id":   m,
        "prompt_id":  p,
        "n_total":    n_total,
        "n_G":        n_G,
        "n_H":        n_H,
        "frac_G":     frac_G,
        "frac_H":     frac_H,
        "W_GG_HH":    res.get("W_GG_HH",   np.nan),
        "W_GG_HH_z":  res.get("W_GG_HH_z", np.nan),
        "valid_geom": (n_G >= 2) and (n_H >= 2),
        "valid_plot": (n_G >= min_per_class_plot) and (n_H >= min_per_class_plot),
        "W_null_mean": res["W_null_mean"],
        "W_null_std":  res["W_null_std"],
    }

    if row["W_null_mean"] is not None:
        row["delta_W"]   = row["W_GG_HH"]   - row["W_null_mean"]
        if "W_GG_HH_z" in res:
            row["delta_W_z"] = row["W_GG_HH_z"] - row["W_null_mean"]
    else:
        row["delta_W"] = None

    geometry = {
        "D_GG":   res.get("D_GG"),
        "D_HH":   res.get("D_HH"),
        "D_GH":   res.get("D_GH"),
        "D_GG_z": res.get("D_GG_z"),
        "D_HH_z": res.get("D_HH_z"),
        "D_GH_z": res.get("D_GH_z"),
        "v_fisher": res.get("v_fisher"),
    }
    return row, geometry


# ============================================================
# ======================= STUDY RUNNER =======================
# ============================================================

def run_structural_analysis(
    df: pd.DataFrame,
    cfg: Config,
    lambda_reg: float | None = None,
    n_permutations: int | None = None,
    random_state: int | None = None,
    min_per_class_plot: int | None = None,
    use_cache: bool = False,
    cache_dir: str | None = None,
    overwrite_cache: bool = False,
) -> tuple[pd.DataFrame, dict, dict]:
    """Run :func:`analyse_prompt` over every (model, prompt) pair.

    Parameters
    ----------
    df, cfg
        Data and configuration.
    lambda_reg, random_state, min_per_class_plot
        Defaults come from ``cfg.experiment`` when left as ``None``.
    n_permutations
        ``None`` → use the cfg default (typically 100).
        ``0`` (or any falsy value) → skip the null-model computation;
        useful when only the geometry / counts / Fisher Wasserstein
        are needed and the permutation loop would be wasted time.
    use_cache, cache_dir, overwrite_cache
        Standard parquet/pickle/JSON caching.  ``cache_dir`` defaults to
        ``f"{cfg.cache.root}/S-data"``.

    Returns
    -------
    results_df : pd.DataFrame
        One row per pair with the scalar summaries.
    geometry_store : dict[(int, int), dict]
        Per-pair geometry payload (distance arrays + Fisher direction).
    null_store : dict[(int, int), np.ndarray]
        Per-pair null-permutation Wasserstein samples (empty when
        ``n_permutations`` is 0).
    """
    lambda_reg         = cfg.experiment.best_lambda    if lambda_reg     is None else lambda_reg
    n_permutations     = cfg.experiment.n_permutations if n_permutations is None else n_permutations
    random_state       = cfg.experiment.random_state   if random_state   is None else random_state
    min_per_class_plot = cfg.experiment.min_per_class  if min_per_class_plot is None else min_per_class_plot

    if cache_dir is None:
        cache_dir = f"{cfg.cache.root}/S-data"

    # ---- try cache ----
    if use_cache and not overwrite_cache and structural_cache_exists(cache_dir, lambda_reg):
        print("Cache correctly loaded.")
        return load_structural_cache(cache_dir, lambda_reg)

    # ---- compute ----
    rows: list[dict] = []
    geometry_store: dict = {}
    null_store: dict = {}

    grouped = df.groupby([cfg.dataset.model_column, cfg.dataset.prompt_column])

    for (m, p), _ in tqdm(grouped, desc="Structural"):
        res = analyse_prompt(
            df, model_id=m, prompt_id=p, cfg=cfg,
            lambda_reg=lambda_reg,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        row, gs = collect_prompt_result(res, min_per_class_plot=min_per_class_plot)
        geometry_store[(m, p)] = gs
        null_store[(m, p)] = res.get("W_null_samples")
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # ---- persist ----
    if use_cache:
        ensure_dir(cache_dir)
        meta = {
            "lambda_reg": lambda_reg,
            "n_permutations": n_permutations,
            "random_state": random_state,
            "min_per_class_plot": min_per_class_plot,
            "n_rows": len(results_df),
            "dataset_name": cfg.dataset.name,
        }
        save_structural_cache(
            cache_dir, lambda_reg,
            results_df, geometry_store, null_store, meta,
        )
        print("Cache correctly dumped.")

    return results_df, geometry_store, null_store



# ============================================================
# ============== SAMPLE-SIZE STRUCTURAL ABLATION =============
# ============================================================

from scipy.stats import mannwhitneyu


def run_structural_ablation_for_prompt(
    df,
    model_id: int,
    prompt_id: int,
    cfg,
    N_values,
    lambda_reg: float        = 1.2,
    n_iter: int              = 20,
    n_permutations: int      = 200,
    random_state: int        = 42,
    alpha: float             = 0.05,
):
    """Subsample-based structural ablation for a single (model, prompt) pair.

    For each ``N`` in ``N_values``, draws ``n_iter`` random subsamples of
    size ``N`` (preserving the natural genuine/hallucinated class ratio).
    A subsample is accepted only if it contains at least two of each
    class.  For each accepted subsample, computes:

      * Wasserstein distance ``W(D_GG, D_HH)`` and its permutation p-value;
      * Mann-Whitney U p-value comparing ``D_GG`` vs ``D_HH``;
      * separability ratio (inter / intra class) in the original space;
      * the same ratio after Fisher projection.

    Returns
    -------
    pd.DataFrame | None
        One row per accepted subsample, or ``None`` if no subsample met
        the minimum-class-count constraint.
    """
    from .data import extract_prompt_data, split_by_label
    from .projections import FisherProjection

    X, y = extract_prompt_data(df, model_id, prompt_id, cfg)
    n_total = len(y)

    rng  = np.random.default_rng(random_state)
    rows: list[dict] = []

    for N in N_values:
        if N > n_total:
            continue

        accepted     = 0
        attempts     = 0
        max_attempts = n_iter * 10

        while accepted < n_iter and attempts < max_attempts:
            attempts += 1

            idx   = rng.choice(n_total, size=N, replace=False)
            X_sub = X[idx]
            y_sub = y[idx]

            X_G, X_H = split_by_label(X_sub, y_sub)
            if len(X_G) < 2 or len(X_H) < 2:
                continue
            accepted += 1

            D_GG, D_HH, D_GH = compute_distance_distributions(X_G, X_H)

            W_obs = wasserstein_GG_HH(D_GG, D_HH)
            if np.isnan(W_obs):
                continue
            W_null = wasserstein_null_model(
                X_sub, y_sub,
                n_permutations=n_permutations,
                random_state=int(rng.integers(1_000_000)),
            )
            p_wass = float(np.mean(W_null >= W_obs))

            D_GG_fin = D_GG[np.isfinite(D_GG)]
            D_HH_fin = D_HH[np.isfinite(D_HH)]
            if len(D_GG_fin) >= 2 and len(D_HH_fin) >= 2:
                _, p_mw = mannwhitneyu(D_GG_fin, D_HH_fin, alternative="two-sided")
            else:
                p_mw = np.nan

            mu_GG = float(np.mean(D_GG_fin)) if len(D_GG_fin) else np.nan
            mu_HH = float(np.mean(D_HH_fin)) if len(D_HH_fin) else np.nan
            mu_GH = float(np.mean(D_GH[np.isfinite(D_GH)])) if len(D_GH) else np.nan
            intra = 0.5 * (mu_GG + mu_HH)
            sep   = mu_GH / intra if intra > 0 else np.nan

            sep_z = np.nan
            try:
                fisher = FisherProjection(
                    lambda_reg=lambda_reg,
                    normalise=True,
                    normalise_by_trace=True,
                )
                fisher.fit(X_sub, y_sub)
                Z_sub = fisher.transform(X_sub).ravel()
                Z_G = Z_sub[~y_sub].reshape(-1, 1)
                Z_H = Z_sub[ y_sub].reshape(-1, 1)
                D_GG_z, D_HH_z, D_GH_z = compute_distance_distributions(Z_G, Z_H)
                mu_GG_z = float(np.mean(D_GG_z)) if len(D_GG_z) else np.nan
                mu_HH_z = float(np.mean(D_HH_z)) if len(D_HH_z) else np.nan
                mu_GH_z = float(np.mean(D_GH_z)) if len(D_GH_z) else np.nan
                intra_z = 0.5 * (mu_GG_z + mu_HH_z)
                sep_z   = mu_GH_z / intra_z if intra_z > 0 else np.nan
            except Exception:
                pass

            rows.append({
                "model_id":    model_id,
                "prompt_id":   prompt_id,
                "N":           N,
                "n_genuine":   int(len(X_G)),
                "n_hall":      int(len(X_H)),
                "iter_id":     accepted - 1,
                "W_obs":       W_obs,
                "W_null_mean": float(W_null.mean()),
                "W_null_std":  float(W_null.std()),
                "p_wass":      p_wass,
                "sig_wass":    bool(p_wass < alpha),
                "p_mw":        float(p_mw) if not np.isnan(p_mw) else np.nan,
                "sig_mw":      bool(p_mw < alpha) if not np.isnan(p_mw) else False,
                "sep":         sep,
                "sep_z":       sep_z,
            })

    return pd.DataFrame(rows) if rows else None


# ============================================================
# ============== SEPARABILITY HELPERS ========================
# ============================================================

def compute_separability(gs: dict, agg: str = "mean") -> dict:
    """Separability ratio (inter/intra class) from a geometry-store entry.

    Returns ``{"sep": <original space>, "sep_z": <Fisher space>}``.
    Both are computed from the per-pair distance arrays stored in ``gs``;
    ``agg`` is either ``"mean"`` (default) or ``"median"``.
    """
    def _agg(x):
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return np.nan
        return float(np.mean(x)) if agg == "mean" else float(np.median(x))

    mu_GG = _agg(gs["D_GG"])
    mu_HH = _agg(gs["D_HH"])
    mu_GH = _agg(gs["D_GH"])
    mu_GG_z = _agg(gs["D_GG_z"])
    mu_HH_z = _agg(gs["D_HH_z"])
    mu_GH_z = _agg(gs["D_GH_z"])

    intra   = 0.5 * (mu_GG + mu_HH)
    intra_z = 0.5 * (mu_GG_z + mu_HH_z)
    return {
        "sep":   mu_GH   / intra   if intra   > 0 else np.nan,
        "sep_z": mu_GH_z / intra_z if intra_z > 0 else np.nan,
    }


def build_separability_df(
    results_df: pd.DataFrame,
    geometry_store: dict,
) -> pd.DataFrame:
    """One row per valid (model, prompt) pair with ``sep`` and ``sep_z`` columns."""
    rows = []
    for (m, p), gs in geometry_store.items():
        sel = results_df[(results_df["model_id"] == m) & (results_df["prompt_id"] == p)]
        if not len(sel) or not bool(sel["valid_geom"].values[0]):
            continue
        s = compute_separability(gs)
        rows.append({
            "model_id":  m,
            "prompt_id": p,
            "sep":       s["sep"],
            "sep_z":     s["sep_z"],
        })
    return pd.DataFrame(rows)


def prepare_separability_violin_df(df_ans: pd.DataFrame) -> pd.DataFrame:
    """Long-form ``(model_id, prompt_id, space, separability)`` for violin plots."""
    rows = []
    for _, r in df_ans.iterrows():
        rows.append({
            "model_id":     r["model_id"],
            "prompt_id":    r["prompt_id"],
            "space":        "original",
            "separability": r["sep"],
        })
        rows.append({
            "model_id":     r["model_id"],
            "prompt_id":    r["prompt_id"],
            "space":        "fisher",
            "separability": r["sep_z"],
        })
    return pd.DataFrame(rows)
