import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import itertools
from copy import deepcopy
from .BRCD.utils import gm_to_nx_Digraph
import cliquepicking as cp
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from scipy.special import logsumexp
import time 
import pyAgrum as gum
from multiprocessing import Pool
from functools import partial
from scipy.stats import chi2_contingency, chi2
from scipy.stats import entropy
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Dict, Optional, List,Literal,  Tuple, Dict, Any, Callable
from .BRCD.boss import boss
from causallearn.graph.Endpoint import Endpoint  
from tqdm import tqdm  # optional, install with `pip install tqdm`
from collections import Counter, defaultdict
from graphical_models.classes.dags.pdag import PDAG
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import traceback


from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    preprocess,
    select_useful_cols,
)



def df_to_prefix_graph(df, G):
    """
    Create directed graph H from df columns, matching G's prefix edges.
    
    Args:
    - df: DataFrame with columns like 'frontend_cpu_1', 'catalogue_mem_abc'
    - G: nx.DiGraph with prefix nodes (e.g. 'front-end' -> 'catalogue')
    
    Returns: nx.DiGraph H with exact df column names as nodes.
    """
    cols = df.columns.tolist()
    
    # Map G prefix -> matching df columns (exact prefix match)
    prefix_to_nodes = {}
    for prefix in G.nodes:
        prefix_to_nodes[prefix] = [col for col in cols if col.startswith(prefix)]
    
    H = nx.DiGraph()
    
    # Add all df columns as nodes
    H.add_nodes_from(cols)
    
    # Add edges: for every (u_prefix, v_prefix) in G, connect *all* u_nodes -> *all* v_nodes
    for u_prefix, v_prefix in G.edges:
        if u_prefix in prefix_to_nodes and v_prefix in prefix_to_nodes:
            u_nodes = prefix_to_nodes[u_prefix]
            v_nodes = prefix_to_nodes[v_prefix]
            for u in u_nodes:
                for v in v_nodes:
                    H.add_edge(u, v)
    
    return H

#### New Esimtator ##################

# Worker wrapper must be top-level so it's picklable by multiprocessing
def _worker_brcd(args):
    """
    Unpack args tuple and call brcd_update.
    Returns (index, success, posterior_array_or_none, elapsed_or_none, error_str_or_none)
    """
    idx, joint_df, cpdag, cols, combos, isdiscrete, node_transform, transform_parents, prior, brcd_update = args
    try:
        posterior, elapsed = brcd_update(
            joint_df,
            cpdag,
            cols,
            combos,
            isdiscrete,
            node_transform,
            transform_parents,
            prior
        )
        posterior_arr = np.asarray(posterior)
        return (idx, True, posterior_arr, float(elapsed), None)
    except Exception as e:
        tb = traceback.format_exc()
        return (idx, False, None, None, f"{e}\n{tb}")

def parallel_weighted_posterior(
    topk_list: List[Dict[str, Any]],
    joint_df,
    cols,
    combos,
    isdiscrete,
    node_transform,
    transform_parents,
    prior,
    brcd_update: Callable,
    n_workers: Optional[int] = None,
    renormalize: bool = True,
    log_space: bool = False,
    use_threads: bool = False,
    show_progress: bool = True,
):
    """
    Run brcd_update in parallel over topk_list (items: {'cpdag':..., 'topk_ratio':...}),
    then compute final weighted posterior in vectorized manner.

    Returns:
      {
        "final_posterior": ndarray (or log-space ndarray if log_space=True),
        "normalized_final_posterior": ndarray or None (only computed for linear space),
        "per_cpdag": list of dicts with keys:
            {'idx', 'cpdag', 'ratio', 'posterior' (ndarray or None),
             'elapsed', 'success', 'error'},
        "failed_idxs": list of failed indices
      }
    Notes:
      - If log_space=True: brcd_update must return log-posterior in 'posterior' (scalar/array).
        Final output 'final_posterior' will be log-sum-exp aggregated.
      - If use_threads=True: uses ThreadPoolExecutor (useful if brcd_update not picklable).
    """
    m = len(topk_list)
    if m == 0:
        raise ValueError("topk_list is empty")

    # Extract ratios and optionally renormalize
    ratios = np.array([float(item["topk_ratio"]) for item in topk_list], dtype=float)
    if renormalize:
        s = ratios.sum()
        if s == 0:
            raise ValueError("Sum of topk_ratio is zero; cannot renormalize")
        ratios = ratios / s

    # Prepare args for workers
    worker_args = []
    for idx, item in enumerate(topk_list):
        cpdag = item["cpdag"]
        worker_args.append((idx, joint_df, cpdag, cols, combos, isdiscrete, node_transform, transform_parents, prior, brcd_update))

    # Choose executor
    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    per_cpdag = [None] * m
    failed_idxs = []

    # Launch parallel jobs
    with Executor(max_workers=n_workers) as exe:
        futures = {exe.submit(_worker_brcd, arg): arg[0] for arg in worker_args}
        if show_progress:
            pbar = tqdm(total=len(futures), desc="brcd_update (parallel)")
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                (i, success, posterior_arr, elapsed, error) = fut.result()
            except Exception as e:
                # Should not happen because worker catches exceptions, but safe fallback
                success = False
                posterior_arr = None
                elapsed = None
                error = f"Executor-side exception: {e}\n{traceback.format_exc()}"
            if success:
                per_cpdag[i] = {
                    "idx": i,
                    "cpdag": topk_list[i]["cpdag"],
                    "ratio": ratios[i],
                    "posterior": posterior_arr,
                    "elapsed": elapsed,
                    "success": True,
                    "error": None
                }
            else:
                per_cpdag[i] = {
                    "idx": i,
                    "cpdag": topk_list[i]["cpdag"],
                    "ratio": ratios[i],
                    "posterior": None,
                    "elapsed": None,
                    "success": False,
                    "error": error
                }
                failed_idxs.append(i)
            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()

    # Collect successful posteriors and check shapes
    success_items = [x for x in per_cpdag if x["success"]]
    if len(success_items) == 0:
        raise RuntimeError("All brcd_update calls failed")

    shapes = [x["posterior"].shape for x in success_items]
    # ensure shapes are identical
    first_shape = shapes[0]
    for s in shapes:
        if s != first_shape:
            raise ValueError(f"Posterior shape mismatch among CPDAGs: {first_shape} vs {s}")

    # Stack posteriors: shape (m_success, *posterior_shape)
    stacked = np.stack([x["posterior"] for x in success_items], axis=0)
    ratios_success = np.array([x["ratio"] for x in success_items], dtype=float)

    if log_space:
        # stacked contains log-posteriors; compute log-sum-exp weighted by ratios.
        # We want log(sum_i ratios_i * exp(logpost_i)) = logsumexp(logpost_i + log(ratio_i))
        from scipy.special import logsumexp
        # add log ratio per sample (broadcast)
        log_ratios = np.log(ratios_success + 1e-300)  # avoid -inf
        # stacked shape: (S, *shape). We'll reshape to (S, K) where K=prod(shape)
        S = stacked.shape[0]
        K = int(np.prod(first_shape))
        stacked2 = stacked.reshape(S, K)
        log_ratios2 = log_ratios[:, None]  # (S,1)
        weighted_logs = stacked2 + log_ratios2  # (S,K)
        # compute log-sum-exp over axis 0 -> (K,)
        agg_log = logsumexp(weighted_logs, axis=0)
        final_log_posterior = agg_log.reshape(first_shape)
        # normalized_final_posterior is not well-defined in log-space without exponentiating.
        return {
            "final_posterior": final_log_posterior,
            "normalized_final_posterior": None,
            "per_cpdag": per_cpdag,
            "failed_idxs": failed_idxs
        }
    else:
        # linear space: stacked contains linear posteriors
        # vectorized weighted sum: final = sum_i ratio_i * posterior_i
        # ratios_success shape (S,), stacked shape (S, *first_shape)
        # multiply and sum over axis=0
        weighted = stacked * ratios_success.reshape((ratios_success.shape[0],) + (1,)*len(first_shape))
        final_posterior = weighted.sum(axis=0)
        # also produce normalized_final_posterior (sum to 1)
        total = final_posterior.sum()
        if np.isfinite(total) and total > 0:
            normalized_final = final_posterior / total
        else:
            normalized_final = None
        return {
            "final_posterior": final_posterior,
            "normalized_final_posterior": normalized_final,
            "per_cpdag": per_cpdag,
            "failed_idxs": failed_idxs
        }


def causal_learn_graph_to_nx_digraph(G_cl, column_names):
    G_nx = nx.DiGraph()
    id_to_col = {i: name for i, name in enumerate(column_names)}
    for node in G_cl.get_nodes():
        node_id = G_cl.node_map[node]
        G_nx.add_node(id_to_col[node_id])
    arcs = []
    edges = []
    for edge in G_cl.get_graph_edges():
        n1 = id_to_col[G_cl.node_map[edge.node1]]
        n2 = id_to_col[G_cl.node_map[edge.node2]]
        ep1 = edge.endpoint1
        ep2 = edge.endpoint2
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            arcs.append((n1, n2))
        elif ep2 ==  Endpoint.TAIL and ep1 == Endpoint.ARROW:
            arcs.append((n2, n1))
        elif ep2 ==  Endpoint.TAIL and ep1 == Endpoint.TAIL:
            edges.append((n2, n1))
    return arcs, edges

# ---------- helpers ----------
def _normal_logpdf_1d(x: np.ndarray, mu: np.ndarray, var: float) -> np.ndarray:
    var = float(max(var, 1e-12))
    return -0.5 * (np.log(2*np.pi*var) + (x - mu)**2 / var)

def _logsumexp2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))

def _transform_and_jacobian(x: np.ndarray, kind: Literal["none","log","log1p","yeojohnson"]="none"):
    x = np.asarray(x, dtype=float)
    if kind == "none":
        return x, np.zeros_like(x)
    if kind == "log":
        if np.any(x <= 0): raise ValueError("log requires x>0.")
        z = np.log(x)
        return z, -np.log(x)                    # log |dz/dx| = -log x
    if kind == "log1p":
        if np.any(x < -1): raise ValueError("log1p requires x>=-1.")
        z = np.log1p(x)
        return z, -np.log1p(x)                  # log |dz/dx| = -log(1+x)
    if kind == "yeojohnson":
        try:
            from sklearn.preprocessing import PowerTransformer
        except Exception as e:
            raise ImportError("scikit-learn required for yeojohnson.") from e
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        z = pt.fit_transform(x.reshape(-1,1)).ravel()
        lam = float(pt.lambdas_[0])
        log_jac = np.empty_like(x)
        pos = x >= 0
        if abs(lam) < 1e-12:
            log_jac[pos] = -np.log1p(x[pos])
        else:
            log_jac[pos] = (lam - 1) * np.log1p(x[pos])
        if abs(lam - 2) < 1e-12:
            log_jac[~pos] = -np.log1p(-x[~pos])
        else:
            log_jac[~pos] = (1 - lam) * np.log1p(-x[~pos])
        return z, log_jac
    raise ValueError(f"Unknown transform kind: {kind}")

def _fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    # X includes intercept column; returns (beta, sigma2)
    n, p = X.shape
    XtX = X.T @ X + ridge * np.eye(p)
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    resid = y - X @ beta
    dof = max(n - p, 1)
    sigma2 = max(float(resid @ resid) / dof, 1e-12)
    return beta, sigma2

# ---------- main ----------
def gaussian_conditional_postpred_rowwise(
    df: pd.DataFrame,
    node: str,
    parents: List[str],
    F: Optional[str] = None,                      # name of binary indicator, optional
    node_transform: Literal["none","log","log1p","yeojohnson"] = "none",
    transform_parents: bool = False,              # apply same transform to continuous parents (no Jacobian)
    ridge: float = 1e-4,
    gating: Literal["auto","empirical"] = "auto", # when F not in parents: mixture gating p(F=1|X)
) -> np.ndarray:
    """
    Return per-row densities for p(node | parents).

    Automatic transform adaptation:
      - If node_transform == "log" but the node or (if transform_parents) any parent has zeros
        on the F=0 subset (if F exists) or on all rows (otherwise), we switch to "log1p".
      - If values < -1 are detected there (so log1p is invalid), we switch to "yeojohnson".

    Cases:
      - If F is in parents: condition on F (separate linear-Gaussian per F group).
      - If F is not in parents (or F is None): integrate F via mixture-of-experts:
            p(y|X) = (1-π(X)) N(z; μ0(X), σ0^2) + π(X) N(z; μ1(X), σ1^2), then add Jacobian.
        π(X)=P(F=1|X) learned via logistic regression (fallback to empirical prior).
    """
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)

    if node not in df.columns:
        raise ValueError(f"'{node}' not in dataframe.")
    for p in parents:
        if p not in df.columns:
            raise ValueError(f"Parent '{p}' not in dataframe.")

    # ===== choose effective transform (must be valid for ALL rows) =====
    eff_transform = node_transform
    if node_transform in {"log", "log1p"}:
        # Check node and (optionally) parents across *all rows*
        cols_to_check = [node]
        if transform_parents:
            cols_to_check.extend([p for p in parents if p != F])

        sub = df[cols_to_check].apply(pd.to_numeric, errors="coerce")
        has_lt_m1  = (sub < -1).any().any()
        has_le_zero = (sub <= 0).any().any()   # catches zeros & negatives

        if node_transform == "log":
            if has_lt_m1:
                eff_transform = "yeojohnson"
            elif has_le_zero:
                eff_transform = "log1p"
            else:
                eff_transform = "log"
        elif node_transform == "log1p":
            if has_lt_m1:
                eff_transform = "yeojohnson"
            else:
                eff_transform = "log1p"

    # Prepare y and parent matrix
    y_orig = pd.to_numeric(df[node], errors="coerce").to_numpy()
    if np.any(~np.isfinite(y_orig)):
        raise ValueError(f"Non-finite values in node '{node}'.")

    X_all = df[parents].apply(pd.to_numeric, errors="coerce").to_numpy() if parents else None
    if parents and np.isnan(X_all).any():
        # We'll mask non-finite rows during fitting; evaluation has fallbacks below.
        pass

    # ---------- Branch 1: F is an explicit parent → condition on F ----------
    if F is not None and F in parents:
        f_col_idx = parents.index(F)
        cont_idx = [i for i in range(len(parents)) if i != f_col_idx]
        probs = np.empty(len(df), dtype=float)

        for f_val in (0, 1):
            idxs = np.where(np.asarray(df[F]) == f_val)[0]
            if idxs.size == 0:
                continue

            y = y_orig[idxs].astype(float)
            z_y, log_jac = _transform_and_jacobian(y, eff_transform)

            if cont_idx:
                Xg = X_all[idxs][:, cont_idx]

                # optional transform of parents (NO Jacobian)
                if transform_parents and eff_transform != "none":
                    if eff_transform == "log":
                        if np.any(Xg <= 0):  # should be rare given the check above
                            raise ValueError("log parents require all values > 0.")
                        Xg = np.log(Xg)
                    elif eff_transform == "log1p":
                        if np.any(Xg < -1):
                            raise ValueError("log1p parents require all values >= -1.")
                        Xg = np.log1p(Xg)
                    elif eff_transform == "yeojohnson":
                        from sklearn.preprocessing import PowerTransformer
                        Xg = PowerTransformer(method="yeo-johnson", standardize=False).fit_transform(Xg)

                m = np.isfinite(z_y) & np.isfinite(Xg).all(axis=1)
                if not np.any(m):
                    mu = float(np.mean(z_y)) if z_y.size else 0.0
                    var = float(np.var(z_y, ddof=1)) if z_y.size > 1 else 1.0
                    probs[idxs] = np.exp(_normal_logpdf_1d(z_y, mu, var) + log_jac)
                    continue

                X_fit = np.c_[np.ones(np.sum(m)), Xg[m]]
                y_fit = z_y[m]
                n_g, p_g = X_fit.shape
                if n_g <= p_g:
                    mu = float(np.mean(y_fit))
                    var = float(np.var(y_fit, ddof=1)) if y_fit.size > 1 else 1.0
                    mu_pred = np.full_like(z_y, mu, dtype=float)
                    probs[idxs] = np.exp(_normal_logpdf_1d(z_y, mu_pred, var) + log_jac)
                    continue

                beta, sigma2 = _fit_ridge(X_fit, y_fit, ridge)
                Xg_all = np.c_[np.ones(len(z_y)), Xg]
                mu_pred = Xg_all @ beta
                probs[idxs] = np.exp(_normal_logpdf_1d(z_y, mu_pred, sigma2) + log_jac)
            else:
                # only F in parents → unconditional Gaussian within each F
                mu = float(np.mean(z_y)) if z_y.size else 0.0
                var = float(np.var(z_y, ddof=1)) if z_y.size > 1 else 1.0
                probs[idxs] = np.exp(_normal_logpdf_1d(z_y, mu, var) + log_jac)

        return probs

    # ---------- Branch 2: F not in parents (or F is None) ----------
    z_all, log_jac_all = _transform_and_jacobian(y_orig, eff_transform)

    if F is None or F not in df.columns:
        # Single expert ignoring F
        if parents:
            X = X_all
            if transform_parents and eff_transform != "none":
                if eff_transform == "log":
                    if np.any(X <= 0): raise ValueError("log parents require > 0.")
                    X = np.log(X)
                elif eff_transform == "log1p":
                    if np.any(X < -1): raise ValueError("log1p parents require >= -1.")
                    X = np.log1p(X)
                elif eff_transform == "yeojohnson":
                    from sklearn.preprocessing import PowerTransformer
                    X = PowerTransformer(method="yeo-johnson", standardize=False).fit_transform(X)
            m = np.isfinite(z_all) & np.isfinite(X).all(axis=1)
            if np.any(m):
                beta, sigma2 = _fit_ridge(np.c_[np.ones(np.sum(m)), X[m]], z_all[m], ridge)
                mu_pred = (np.c_[np.ones(len(z_all)), X] @ beta)
            else:
                mu_pred = np.full_like(z_all, np.mean(z_all), dtype=float)
                sigma2 = float(np.var(z_all, ddof=1)) if len(z_all) > 1 else 1.0
        else:
            mu_pred = np.full_like(z_all, np.mean(z_all), dtype=float)
            sigma2 = float(np.var(z_all, ddof=1)) if len(z_all) > 1 else 1.0

        return np.exp(_normal_logpdf_1d(z_all, mu_pred, sigma2) + log_jac_all)

    # Mixture of experts over F (F present in df but not conditioned on)
    F_vals = np.asarray(df[F]).astype(int)
    idx0 = np.where(F_vals == 0)[0]
    idx1 = np.where(F_vals == 1)[0]

    X = X_all if parents else None
    if parents and transform_parents and eff_transform != "none":
        if eff_transform == "log":
            if np.any(X <= 0): raise ValueError("log parents require > 0.")
            X = np.log(X)
        elif eff_transform == "log1p":
            if np.any(X < -1): raise ValueError("log1p parents require >= -1.")
            X = np.log1p(X)
        elif eff_transform == "yeojohnson":
            from sklearn.preprocessing import PowerTransformer
            X = PowerTransformer(method="yeo-johnson", standardize=False).fit_transform(X)

    def _fit_expert(idxs):
        if len(idxs) == 0:
            return np.array([np.mean(z_all)]), float(np.var(z_all, ddof=1) if len(z_all)>1 else 1.0)
        y = z_all[idxs]
        if parents:
            Xg = X[idxs]
            m = np.isfinite(y) & np.isfinite(Xg).all(axis=1)
            if not np.any(m):
                mu = float(np.mean(y)) if y.size else 0.0
                var = float(np.var(y, ddof=1)) if y.size > 1 else 1.0
                return np.array([mu]), var
            X_fit = np.c_[np.ones(np.sum(m)), Xg[m]]
            beta, var = _fit_ridge(X_fit, y[m], ridge)
            return beta, var
        else:
            mu = float(np.mean(y)) if y.size else 0.0
            var = float(np.var(y, ddof=1)) if y.size > 1 else 1.0
            return np.array([mu]), var

    beta0, var0 = _fit_expert(idx0)
    beta1, var1 = _fit_expert(idx1)

    if parents:
        X_design = np.c_[np.ones(len(z_all)), X]
        mu0 = X_design @ beta0
        mu1 = X_design @ beta1
    else:
        mu0 = np.full_like(z_all, beta0[0], dtype=float)
        mu1 = np.full_like(z_all, beta1[0], dtype=float)

    # Gating π(X)
    if gating == "empirical":
        pi1 = np.full(len(z_all), fill_value=len(idx1) / max(len(z_all), 1), dtype=float)
    else:
        try:
            from sklearn.linear_model import LogisticRegression
            if parents:
                X_gate = X
            else:
                X_gate = np.ones((len(z_all), 1))
            m = np.isfinite(F_vals) & np.isfinite(X_gate).all(axis=1)
            if not np.any(m):
                pi1 = np.full(len(z_all), fill_value=len(idx1)/max(len(z_all),1))
            else:
                clf = LogisticRegression(solver="lbfgs", penalty="l2", max_iter=1000)
                clf.fit(X_gate[m], F_vals[m])
                pi1 = clf.predict_proba(X_gate)[:, 1]
        except Exception:
            pi1 = np.full(len(z_all), fill_value=len(idx1)/max(len(z_all),1), dtype=float)

    logN0 = _normal_logpdf_1d(z_all, mu0, var0)
    logN1 = _normal_logpdf_1d(z_all, mu1, var1)
    log_mix = np.logaddexp(np.log1p(-pi1) + logN0, np.log(pi1) + logN1)
    return np.exp(log_mix + log_jac_all)

def continuous_likelihood_fn_gaussian(
    df: pd.DataFrame,
    node: str,
    parents: List[str],
    F: Optional[str] = None,            # optional: may be absent or not in parents
    node_transform: str = "none",       # "none" | "log" | "log1p" | "yeojohnson"
    transform_parents: bool = False,    # apply same transform to continuous parents (no Jacobian)
    ridge: float = 1e-4,                # small ridge for stability
    gating: str = "auto",               # "auto" (logistic) or "empirical" when F not in parents
) -> np.ndarray:
    """
    Per-row likelihoods p(node | parents) under a linear-Gaussian CPD.

    Behavior:
      - If F is provided and F ∈ parents: condition on F (separate expert per F).
      - If F is provided and F ∉ parents: integrate F via mixture-of-experts with gating.
      - If F is None: single expert ignoring F.
    """
    if F is not None and F not in df.columns:
        raise ValueError(f"F column '{F}' not found in dataframe.")

    return gaussian_conditional_postpred_rowwise(
        df=df,
        node=node,
        parents=parents,
        F=F,
        node_transform=node_transform,
        transform_parents=transform_parents,
        ridge=ridge,
        gating=gating,
    )

##########################################
def _infer_cardinality(col: pd.Series) -> int:
    """Infer #states; assumes discretized ints 0..K-1 if possible, else nunique()."""
    vals = col.to_numpy()
    if np.issubdtype(vals.dtype, np.integer):
        vmin, vmax = int(vals.min()), int(vals.max())
        if vmin >= 0:
            return vmax + 1
    return int(col.nunique())

def dirichlet_postpred_rowwise(
    df: pd.DataFrame,
    node: str,
    parents: list[str],
    alpha_star: float = 5.0,
    cardinalities: Optional[Dict[str, int]] = None,
    # order: 'as_is' uses df's row order; if you want, you can pass a fixed permutation for reproducibility
    order: str = 'as_is',
) -> np.ndarray:
    """
    Return per-row posterior-predictive probabilities under a symmetric Dirichlet(alpha_star/K)
    for the family p(node | parents). The product over rows equals the integrated marginal likelihood.
    """
    if cardinalities is None:
        cardinalities = {}

    # ensure RangeIndex so we can index by position efficiently
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)

    K = cardinalities.get(node, _infer_cardinality(df[node]))
    alpha0 = alpha_star / K

    probs = np.empty(len(df), dtype=float)

    if len(parents) == 0:
        # single stream over all rows
        counts = np.zeros(K, dtype=float)
        total = 0.0
        x_vals = df[node].astype(int).to_numpy()

        if order != 'as_is':
            # if you want to permute for prequential scoring, do it here
            idxs = np.arange(len(df))
        else:
            idxs = np.arange(len(df))

        for i in idxs:
            x = x_vals[i]
            # posterior predictive for this row given prior+past rows
            probs[i] = (counts[x] + alpha0) / (total + alpha_star)
            counts[x] += 1.0
            total += 1.0
        return probs

    # group by parent configurations; handle each stream independently
    groups = df.groupby(parents, sort=False).groups  # dict: key -> index labels
    node_vals = df[node].astype(int).to_numpy()

    for _, idx_labels in groups.items():
        # convert to positional indices (RangeIndex → same ints)
        idxs = np.fromiter(idx_labels, dtype=int)
        # (optional) you could permute idxs reproducibly here if desired
        counts = np.zeros(K, dtype=float)
        total = 0.0
        for i in idxs:
            x = node_vals[i]
            probs[i] = (counts[x] + alpha0) / (total + alpha_star)
            counts[x] += 1.0
            total += 1.0

    return probs

def discrete_likelihood_fn_dirichlet(
    df,
    node,
    parents,
    alpha_star: float = 5.0,
    cardinalities: Optional[Dict[str, int]] = None,
):
    return dirichlet_postpred_rowwise(
        df=df,
        node=node,
        parents=parents,
        alpha_star=alpha_star,
        cardinalities=cardinalities,
        order='as_is',
    )
# # ---- Precompute global moments once ----
# class GaussianMoments:
#     def __init__(self, cols, mean, cov):
#         self.cols = cols
#         self.col_to_idx = {c:i for i,c in enumerate(cols)}
#         self.mean = mean  # shape (d,)
#         self.cov = cov    # shape (d,d)

# def precompute_gaussian_moments(joint_df: pd.DataFrame, cols=None) -> GaussianMoments:
#     if cols is None:
#         cols = list(joint_df.columns)
#     X = joint_df[cols].to_numpy(dtype=float)
#     mu = X.mean(axis=0)
#     # ddof=0 → MLE covariance
#     Sigma = np.cov(X, rowvar=False, ddof=0)
#     return GaussianMoments(cols, mu, Sigma)

# # ---- Fast conditional Gaussian logpdf per row ----
# def gaussian_conditional_logpdf_rows(joint_df: pd.DataFrame, node: str, parents: list, gm: GaussianMoments,
#                                      ridge=1e-8) -> pd.Series:
#     idx_x = gm.col_to_idx[node]
#     idx_z = [gm.col_to_idx[p] for p in parents]

#     Xvals = joint_df[[node]].to_numpy(dtype=float)  # (n,1)

#     mu_x = gm.mean[idx_x]
#     var_xx = gm.cov[idx_x, idx_x]

#     if len(idx_z) == 0:
#         # p(x) ~ N(mu_x, var_xx)
#         resid = Xvals[:, 0] - mu_x
#         # clamp variance to be positive
#         var = max(var_xx, ridge)
#         log_norm_const = -0.5 * (np.log(2*np.pi*var))
#         ll = log_norm_const - 0.5 * (resid**2) / var
#         return pd.Series(ll, index=joint_df.index, name=f'logpdf_{node}')

#     Zvals = joint_df[parents].to_numpy(dtype=float)         # (n,k)
#     mu_z = gm.mean[idx_z]                                   # (k,)
#     Sigma_xz = gm.cov[idx_x, idx_z].reshape(1, -1)          # (1,k)
#     Sigma_zx = Sigma_xz.T                                   # (k,1)
#     Sigma_zz = gm.cov[np.ix_(idx_z, idx_z)].copy()          # (k,k)

#     # Numerical stabilization
#     trace = np.trace(Sigma_zz)
#     lam = ridge if trace == 0 else ridge * trace / len(idx_z)
#     Sigma_zz.flat[::len(idx_z)+1] += lam  # add lam to diag

#     # Solve Sigma_zz^{-1} via Cholesky
#     try:
#         L = np.linalg.cholesky(Sigma_zz)
#         def chol_solve(b):
#             # solve Sigma_zz * x = b for x
#             y = np.linalg.solve(L, b)
#             return np.linalg.solve(L.T, y)
#         Sigma_zz_inv_Sigma_zx = chol_solve(Sigma_zx)                # (k,1)
#         A = Sigma_xz @ chol_solve(np.eye(len(idx_z)))               # (1,k) @ (k,k) = (1,k)
#     except np.linalg.LinAlgError:
#         # fallback
#         Sigma_zz_inv = np.linalg.pinv(Sigma_zz)
#         Sigma_zz_inv_Sigma_zx = Sigma_zz_inv @ Sigma_zx
#         A = Sigma_xz @ Sigma_zz_inv

#     # Conditional mean/variance
#     mu_cond = mu_x + (A @ (Zvals - mu_z).T).ravel()                 # (n,)
#     var_cond = var_xx - (Sigma_xz @ Sigma_zz_inv_Sigma_zx)[0, 0]
#     var_cond = float(max(var_cond, ridge))

#     # Log-density per row
#     resid = Xvals[:, 0] - mu_cond
#     log_norm_const = -0.5 * (np.log(2*np.pi*var_cond))
#     ll = log_norm_const - 0.5 * (resid**2) / var_cond
#     return pd.Series(ll, index=joint_df.index, name=f'logpdf_{node}|{",".join(parents)}')

# # ---- Convenience wrapper with caching across families ----
# _moments_cache = {}
# def rowwise_loglik_gaussian(joint_df, node, parents, cols=None, cache_key="__all__"):
#     gm = _moments_cache.get(cache_key)
#     if gm is None:
#         gm = precompute_gaussian_moments(joint_df if cols is None else joint_df[cols])
#         _moments_cache[cache_key] = gm
#     return gaussian_conditional_logpdf_rows(joint_df, node, parents, gm)

# @dataclass
# class ConditionalKDE:
#     x_name: str
#     z_names: list
#     scaler_joint: StandardScaler | None
#     scaler_z: StandardScaler | None
#     kde_joint: KernelDensity        # KDE on [x, z] if z exists, else KDE on x
#     kde_z: KernelDensity | None     # KDE on z (None if no parents)

#     def logpdf(self, x_val, z_val=None):
#         """
#         Evaluate log p(x | z) at a single point.
#         If no parents, returns log p(x).
#         """
#         if len(self.z_names) == 0:
#             x_val = np.asarray([x_val], dtype=float).reshape(1, 1)
#             X_std = self.scaler_joint.transform(x_val)
#             return self.kde_joint.score_samples(X_std)[0]

#         z_val = np.asarray(z_val, dtype=float).reshape(1, -1)
#         x_val = np.asarray([x_val], dtype=float).reshape(1, 1)
#         X_joint = np.hstack([x_val, z_val])

#         X_joint_std = self.scaler_joint.transform(X_joint)
#         Z_std = self.scaler_z.transform(z_val)

#         log_p_joint = self.kde_joint.score_samples(X_joint_std)[0]
#         log_p_z = self.kde_z.score_samples(Z_std)[0]
#         return log_p_joint - log_p_z

#     def pdf(self, x_val, z_val=None):
#         return np.exp(self.logpdf(x_val, z_val))

#     def logpdf_rows(self, df: pd.DataFrame):
#         """
#         Vectorized per-row log p(x | z). If no parents, returns log p(x).
#         """
#         if len(self.z_names) == 0:
#             X = df[[self.x_name]].to_numpy(dtype=float)
#             X_std = self.scaler_joint.transform(X)
#             return pd.Series(self.kde_joint.score_samples(X_std), index=df.index,
#                              name=f'logpdf_{self.x_name}')

#         X = df[[self.x_name] + self.z_names].to_numpy(dtype=float)
#         Z = df[self.z_names].to_numpy(dtype=float)

#         X_joint_std = self.scaler_joint.transform(X)
#         Z_std = self.scaler_z.transform(Z)

#         log_p_joint = self.kde_joint.score_samples(X_joint_std)
#         log_p_z = self.kde_z.score_samples(Z_std)
#         return pd.Series(log_p_joint - log_p_z, index=df.index,
#                          name=f'logpdf_{self.x_name}|{",".join(self.z_names)}')

#     def pdf_rows(self, df: pd.DataFrame):
#         return self.logpdf_rows(df).apply(np.exp)

# def _fit_kde(X_std, bandwidth_grid=(0.1, 0.2, 0.5, 1.0, 2.0)):
#     grid = {'bandwidth': bandwidth_grid}
#     search = GridSearchCV(KernelDensity(kernel='gaussian'), grid, cv=5, n_jobs=-1)
#     search.fit(X_std)
#     return search.best_estimator_

# def estimate_conditional_pdf(joint_df: pd.DataFrame, x: str, ls_of_parents: list,
#                              bandwidth_grid_joint=(0.1, 0.2, 0.5, 1.0, 2.0),
#                              bandwidth_grid_z=(0.1, 0.2, 0.5, 1.0, 2.0)) -> ConditionalKDE:
#     """
#     Build a conditional KDE estimator for continuous data.
#     Handles the no-parents case by fitting a marginal KDE p(x).
#     """
#     # safety checks
#     if x not in joint_df.columns:
#         raise ValueError(f"Column '{x}' not found in joint_df.")
#     for c in ls_of_parents:
#         if c not in joint_df.columns:
#             raise ValueError(f"Column '{c}' not found in joint_df.")

#     cols_joint = [x] + ls_of_parents
#     data_joint = joint_df[cols_joint].dropna().to_numpy(dtype=float)

#     if len(ls_of_parents) == 0:
#         # Fit marginal p(x)
#         X = data_joint[:, [0]]  # shape (n,1)
#         scaler_x = StandardScaler().fit(X)
#         X_std = scaler_x.transform(X)
#         kde_x = _fit_kde(X_std, bandwidth_grid_joint)
#         return ConditionalKDE(
#             x_name=x,
#             z_names=[],
#             scaler_joint=scaler_x,  # scaler over x only
#             scaler_z=None,
#             kde_joint=kde_x,        # stores marginal KDE
#             kde_z=None
#         )

#     # parents present: fit p(x,z) and p(z)
#     Z = data_joint[:, 1:]  # parents only
#     scaler_joint = StandardScaler().fit(data_joint)
#     scaler_z = StandardScaler().fit(Z)

#     data_joint_std = scaler_joint.transform(data_joint)
#     data_z_std = scaler_z.transform(Z)

#     kde_joint = _fit_kde(data_joint_std, bandwidth_grid_joint)
#     kde_z = _fit_kde(data_z_std, bandwidth_grid_z)

#     return ConditionalKDE(
#         x_name=x,
#         z_names=ls_of_parents,
#         scaler_joint=scaler_joint,
#         scaler_z=scaler_z,
#         kde_joint=kde_joint,
#         kde_z=kde_z
#     )

# def mutual_info(df, x_col, y_col):
#     """
#     Compute mutual information between two columns in a dataframe.
    
#     Args:
#         df (pd.DataFrame): DataFrame containing the columns
#         x_col (str): Name of first column
#         y_col (str): Name of second column
        
#     Returns:
#         float: Mutual information between x_col and y_col
#     """
#     # Create contingency table
#     contingency = pd.crosstab(df[x_col], df[y_col])
    
#     # Convert to probabilities
#     p_xy = contingency / contingency.sum().sum()
    
#     # Compute marginal probabilities
#     p_x = p_xy.sum(axis=1)
#     p_y = p_xy.sum(axis=0)
    
#     # Compute mutual information
#     mi = 0
#     for i in range(len(p_x)):
#         for j in range(len(p_y)):
#             if p_xy.iloc[i,j] > 0:  # Avoid log(0)
#                 mi += p_xy.iloc[i,j] * np.log(p_xy.iloc[i,j] / (p_x[i] * p_y[j]))
    
#     return mi


################ FORST-KDE ##################
# ===== Forest-KDE utilities =====
from dataclasses import dataclass
from typing import Optional, List, Literal, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.neighbors import KernelDensity

def _kde_pdf_1d(z_samples: np.ndarray, z_query: np.ndarray, h: float) -> np.ndarray:
    kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(z_samples.reshape(-1, 1))
    return np.exp(kde.score_samples(z_query.reshape(-1, 1)))

def _gauss_pdf(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u*u) / np.sqrt(2*np.pi)

def _silverman_bandwidth(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = max(len(y), 1)
    if n <= 1:
        return 1.0
    sd = np.std(y, ddof=1)
    iqr = np.subtract(*np.percentile(y, [75, 25]))
    s = min(sd, iqr/1.349) if (sd > 0 and iqr > 0) else max(sd, iqr/1.349, 1e-3)
    h = 0.9 * s * n ** (-1/5)
    return float(max(h, 1e-4))

class _NodeTransformer:
    """Fit-once transform for the node; keeps params for consistent Jacobian."""
    def __init__(self, kind: Literal["none","log","log1p","yeojohnson"]="none"):
        self.kind = kind
        self.pt: Optional[PowerTransformer] = None
        self.lam: Optional[float] = None

    def fit(self, y: np.ndarray):
        y = np.asarray(y, dtype=float)
        if self.kind == "yeojohnson":
            self.pt = PowerTransformer(method="yeo-johnson", standardize=False).fit(y.reshape(-1,1))
            self.lam = float(self.pt.lambdas_[0])
        return self

    def transform_with_log_jac(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float)
        if self.kind == "none":
            return y, np.zeros_like(y)
        if self.kind == "log":
            if np.any(y <= 0): raise ValueError("log requires y>0.")
            z = np.log(y)
            return z, -np.log(y)                      # log|dz/dy|
        if self.kind == "log1p":
            if np.any(y < -1): raise ValueError("log1p requires y>=-1.")
            z = np.log1p(y)
            return z, -np.log1p(y)
        if self.kind == "yeojohnson":
            if self.pt is None: raise RuntimeError("YJ transformer not fit.")
            z = self.pt.transform(y.reshape(-1,1)).ravel()
            lam = self.lam
            log_jac = np.empty_like(y)
            pos = y >= 0
            # d z / d y for YJ; see sklearn docs
            if abs(lam) < 1e-12:
                log_jac[pos] = -np.log1p(y[pos])
            else:
                log_jac[pos] = (lam - 1) * np.log1p(y[pos])
            if abs(lam - 2) < 1e-12:
                log_jac[~pos] = -np.log1p(-y[~pos])
            else:
                log_jac[~pos] = (1 - lam) * np.log1p(-y[~pos])
            return z, log_jac
        raise ValueError(f"Unknown transform: {self.kind}")

def _choose_effective_transform(df: pd.DataFrame, node: str, parents: List[str],
                                requested: str, F: Optional[str], transform_parents: bool) -> str:
    """Make sure transform is valid for ALL rows we score."""
    eff = requested
    if requested in {"log", "log1p"}:
        cols = [node] + ([p for p in parents if p != F] if transform_parents else [])
        sub = df[cols].apply(pd.to_numeric, errors="coerce")
        has_lt_m1 = (sub < -1).any().any()
        has_le_0 = (sub <= 0).any().any()
        if requested == "log":
            eff = "yeojohnson" if has_lt_m1 else ("log1p" if has_le_0 else "log")
        else:  # log1p
            eff = "yeojohnson" if has_lt_m1 else "log1p"
    return eff



@dataclass
class _ForestKDE1D:
    rf: "ExtraTreesRegressor"
    leaf_inv: List[Dict[int, np.ndarray]]
    y_train_z: np.ndarray
    h: float
    parents: List[str]
    kde_uncond: Optional[KernelDensity] = None  # NEW

    def pdf_rows_z(self, Xq: np.ndarray, zq: np.ndarray) -> np.ndarray:
        out = np.zeros(len(zq), dtype=float)
        if not self.leaf_inv:  # unconditional model fast-path
            return np.exp(self.kde_uncond.score_samples(zq.reshape(-1, 1)))

        h = self.h
        leaves_all = self.rf.apply(Xq)  # shape (m, n_estimators)
        T = leaves_all.shape[1]

        for t in range(T):
            leaves_q = leaves_all[:, t]
            inv = self.leaf_inv[t]
            uniq, rpos = np.unique(leaves_q, return_inverse=True)

            for li, leaf_id in enumerate(uniq):
                r_idx = np.where(rpos == li)[0]
                tr_idx = inv.get(int(leaf_id))
                if tr_idx is None or tr_idx.size == 0 or r_idx.size == 0:
                    continue
                zi = self.y_train_z[tr_idx]
                Z = zq[r_idx][:, None]
                contrib = np.sum(np.exp(-0.5*((Z - zi[None, :])/h)**2)/np.sqrt(2*np.pi), axis=1) / (h * len(tr_idx) * T)
                out[r_idx] += contrib
        return out

        
def _fit_forest_kde(df, node, parents, idxs, z_all, rf_params: Optional[Dict] = None) -> _ForestKDE1D:
    params = dict(
        n_estimators=64,       # leaner defaults
        max_depth=6,
        min_samples_leaf=32,
        max_features="sqrt",
        bootstrap=False,
        n_jobs=-1,
        random_state=0,
    )
    if rf_params:
        params.update(rf_params)

    P = list(parents)
    ztr = z_all[idxs]
    ztr = ztr[np.isfinite(ztr)]
    if len(P) == 0:
        h = _silverman_bandwidth(ztr)
        kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(ztr.reshape(-1, 1))
        dummy = ExtraTreesRegressor(random_state=params["random_state"]).fit(np.zeros((1, 1)), [0])
        return _ForestKDE1D(dummy, [], ztr, h, P, kde_uncond=kde)

    Xtr = df.iloc[idxs][P].apply(pd.to_numeric, errors="coerce").to_numpy()
    m = np.isfinite(Xtr).all(axis=1)
    idxs = idxs[m]; Xtr = Xtr[m]; ztr = z_all[idxs]
    if len(idxs) == 0:
        h = _silverman_bandwidth(ztr)
        kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(ztr.reshape(-1, 1))
        dummy = ExtraTreesRegressor(random_state=params["random_state"]).fit(np.zeros((1, 1)), [0])
        return _ForestKDE1D(dummy, [], ztr, h, P, kde_uncond=kde)

    rf = ExtraTreesRegressor(**params).fit(Xtr, ztr)

    leaf_inv = []
    for est in rf.estimators_:
        leaves = est.apply(Xtr)
        inv = {int(leaf_id): np.where(leaves == leaf_id)[0] for leaf_id in np.unique(leaves)}
        leaf_inv.append(inv)

    h = _silverman_bandwidth(ztr)
    kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(ztr.reshape(-1, 1))
    return _ForestKDE1D(rf=rf, leaf_inv=leaf_inv, y_train_z=ztr, h=h, parents=P, kde_uncond=kde)

# Global cache to avoid refitting for the same family across sampled DAGs
_fkde_cache: Dict[Tuple, _ForestKDE1D] = {}

def continuous_likelihood_fn_forestkde(
    df: pd.DataFrame,
    node: str,
    parents: List[str],
    F: Optional[str] = None,
    node_transform: Literal["none","log","log1p","yeojohnson"] = "none",
    transform_parents: bool = False,   # trees don't need it; kept for API symmetry
    gating: Literal["auto","empirical"] = "auto",
    rf_params: Optional[Dict] = None,
) -> np.ndarray:
    """
    Per-row densities p(node | parents) using forest-weighted 1D KDE on a transformed node.
    Returns densities on the ORIGINAL scale (includes Jacobian).

    Expects these helpers to exist in scope:
      - _choose_effective_transform, _NodeTransformer
      - _fit_forest_kde (returns _ForestKDE1D with .y_train_z, .h, .pdf_rows_z)
      - _kde_pdf_1d (scikit-learn KernelDensity wrapper)
      - _gauss_pdf, _silverman_bandwidth
      - cache: _fkde_cache: Dict[Tuple, _ForestKDE1D]
    """
    # 0) Basic checks
    if node not in df.columns:
        raise ValueError(f"'{node}' not in df.")
    for p in parents:
        if p not in df.columns:
            raise ValueError(f"Parent '{p}' missing.")
    y_orig = pd.to_numeric(df[node], errors="coerce").to_numpy()
    if np.any(~np.isfinite(y_orig)):
        raise ValueError(f"Non-finite in '{node}'.")

    # 1) Transform (fit once) + Jacobian
    eff_tf = _choose_effective_transform(df, node, parents, node_transform, F, transform_parents)
    tf = _NodeTransformer(eff_tf).fit(y_orig)
    z_all, log_jac_all = tf.transform_with_log_jac(y_orig)

    # Parents matrix (trees only need raw parents)
    X_all = df[parents].apply(pd.to_numeric, errors="coerce").to_numpy() if parents else None
    n = len(df)

    # === Case A: F is an explicit parent (separate experts; pick matching expert per row) ===
    if F is not None and F in parents:
        P_woF = [p for p in parents if p != F]
        probs = np.zeros(n, dtype=float)
        F_vals = df[F].to_numpy().astype(int)

        for f_val in (0, 1):
            rows = np.where(F_vals == f_val)[0]
            if rows.size == 0:
                continue

            key = ("fkde", node, tuple(P_woF), f_val, eff_tf,
                   tuple(sorted((rf_params or {}).items())))
            model = _fkde_cache.get(key)
            if model is None:
                model = _fit_forest_kde(df, node, P_woF, rows, z_all, rf_params or {})
                _fkde_cache[key] = model

            if len(P_woF) == 0:
                # Unconditional KDE (within regime) → use KernelDensity
                ztr, h = model.y_train_z, model.h
                zq = z_all[rows]
                probs[rows] = np.exp(model.kde_uncond.score_samples(zq.reshape(-1,1)))

            else:
                # Forest-weighted KDE with NaN-safe fallback
                Xq = df.iloc[rows][P_woF].apply(pd.to_numeric, errors="coerce").to_numpy()
                zq = z_all[rows]
                mq = np.isfinite(Xq).all(axis=1)

                if np.any(mq):
                    probs[rows[mq]] = model.pdf_rows_z(Xq[mq], zq[mq])

                if np.any(~mq):
                    ztr, h = model.y_train_z, model.h
                    zb = zq[~mq]
                    probs[rows[~mq]] = np.exp(model.kde_uncond.score_samples(zb.reshape(-1,1)))


        return np.maximum(probs, 0.0) * np.exp(log_jac_all)

    # === Case B: F not in parents (and maybe not in df at all) ===
    if F is None or (F not in df.columns):
        if not parents:
            # Global unconditional KDE → use KernelDensity
            h = _silverman_bandwidth(z_all)
            probs = _kde_pdf_1d(z_all, z_all, h)
            return np.maximum(probs, 0.0) * np.exp(log_jac_all)

        key = ("fkde", node, tuple(parents), "all", eff_tf,
               tuple(sorted((rf_params or {}).items())))
        model = _fkde_cache.get(key)
        if model is None:
            idxs = np.arange(n)
            model = _fit_forest_kde(df, node, parents, idxs, z_all, rf_params or {})
            _fkde_cache[key] = model

        # NaN-safe evaluation
        m = np.isfinite(X_all).all(axis=1)
        probs_z = np.empty(n, dtype=float)
        if np.any(m):
            probs_z[m] = model.pdf_rows_z(X_all[m], z_all[m])
        if np.any(~m):
            ztr, h = model.y_train_z, model.h
            zb = z_all[~m]
            probs_z[~m] = _kde_pdf_1d(ztr, zb, h)

        return np.maximum(probs_z, 0.0) * np.exp(log_jac_all)

    # === Case C: F present in df but NOT in parents → mixture of two experts with gate ===
    F_vals = df[F].to_numpy().astype(int)
    idx0 = np.where(F_vals == 0)[0]
    idx1 = np.where(F_vals == 1)[0]
    force_empirical = (len(idx0) == 0 or len(idx1) == 0)

    # Experts
    if not parents:
        # Two unconditional experts → KernelDensity for each
        if len(idx0) > 0:
            z0 = z_all[idx0]; h0 = _silverman_bandwidth(z0)
            p0_z = _kde_pdf_1d(z0, z_all, h0)
        else:
            p0_z = np.zeros(n, dtype=float)
        if len(idx1) > 0:
            z1 = z_all[idx1]; h1 = _silverman_bandwidth(z1)
            p1_z = _kde_pdf_1d(z1, z_all, h1)
        else:
            p1_z = np.zeros(n, dtype=float)
    else:
        key0 = ("fkde", node, tuple(parents), 0, eff_tf,
                tuple(sorted((rf_params or {}).items())))
        key1 = ("fkde", node, tuple(parents), 1, eff_tf,
                tuple(sorted((rf_params or {}).items())))
        model0 = _fkde_cache.get(key0)
        model1 = _fkde_cache.get(key1)
        if model0 is None and len(idx0) > 0:
            model0 = _fit_forest_kde(df, node, parents, idx0, z_all, rf_params or {})
            _fkde_cache[key0] = model0
        if model1 is None and len(idx1) > 0:
            model1 = _fit_forest_kde(df, node, parents, idx1, z_all, rf_params or {})
            _fkde_cache[key1] = model1

        m = np.isfinite(X_all).all(axis=1)
        if len(idx0) > 0:
            p0_z = np.empty(n, dtype=float)
            if np.any(m):
                p0_z[m] = model0.pdf_rows_z(X_all[m], z_all[m])
            if np.any(~m):
                z0tr, h0 = model0.y_train_z, model0.h
                zb = z_all[~m]
                p0_z[~m] = np.exp(model0.kde_uncond.score_samples(zb.reshape(-1,1)))

        else:
            p0_z = np.zeros(n, dtype=float)

        if len(idx1) > 0:
            p1_z = np.empty(n, dtype=float)
            if np.any(m):
                p1_z[m] = model1.pdf_rows_z(X_all[m], z_all[m])
            if np.any(~m):
                z1tr, h1 = model1.y_train_z, model1.h
                zb = z_all[~m]
                p1_z[~m] = np.exp(model1.kde_uncond.score_samples(zb.reshape(-1,1)))
        else:
            p1_z = np.zeros(n, dtype=float)

    # Gate π(F=1|X)
    if gating == "empirical" or force_empirical:
        pi1 = np.full(n, fill_value=len(idx1) / max(n, 1), dtype=float)
    else:
        try:
            from sklearn.linear_model import LogisticRegression
            X_gate = X_all if parents else np.ones((n, 1))
            m = np.isfinite(X_gate).all(axis=1)
            uniq = np.unique(F_vals[m])
            if uniq.size < 2:
                pi1 = np.full(n, fill_value=len(idx1) / max(n, 1), dtype=float)
            else:
                clf = LogisticRegression(solver="lbfgs", penalty="l2", max_iter=1000)
                clf.fit(X_gate[m], F_vals[m])
                pi1 = clf.predict_proba(X_gate)[:, 1]
        except Exception:
            pi1 = np.full(n, fill_value=len(idx1) / max(n, 1), dtype=float)

    # Mixture on z-scale → back to original scale with Jacobian
    probs_z = (1.0 - pi1) * p0_z + pi1 * p1_z
    return np.maximum(probs_z, 0.0) * np.exp(log_jac_all)



################################################



def conditional_chi2_test(df, x_col, y_col, z_cols=[]):
    """
    Conducts a conditional chi-square test for X and Y given multiple Zs.

    If z_cols is empty, does a regular chi-square test.

    Args:
        df (pd.DataFrame): DataFrame containing X, Y, and (optionally) Zs.
        x_col (str): Name of the X column.
        y_col (str): Name of the Y column.
        z_cols (list[str]): List of names of Z columns (can be empty).

    Returns:
        chi2_stat_total, dof_total, p_value_total
    """
    chi2_stat_total = 0
    dof_total = 0

    if len(z_cols) == 0:
        # No conditioning: regular chi-square test
        contingency = pd.crosstab(df[x_col], df[y_col])

        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return None, None, None

        chi2_stat, p, dof, expected = chi2_contingency(contingency, correction=False)
        return chi2_stat, dof, p
    else:
        # Conditioning on Z
        grouped = df.groupby(z_cols)

        for _, group in grouped:
            contingency = pd.crosstab(group[x_col], group[y_col])

            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue

            chi2_stat, p, dof, expected = chi2_contingency(contingency, correction=False)
            chi2_stat_total += chi2_stat
            dof_total += dof

        if dof_total == 0:
            return None, None, None

        from scipy.stats import chi2
        p_value_total = 1 - chi2.cdf(chi2_stat_total, dof_total)

        return chi2_stat_total, dof_total, p_value_total
    
def build_cpt(df, node, parents, alpha=1e-6):
    """
    Returns a dict mapping
       (parents_vals_tuple, node_val) -> P(node=node_val | parents=parents_vals_tuple)
    computed by one groupby+merge.
    """
    if not parents:
        # just a prior P(node)
        probs = df[node].value_counts(normalize=True).to_dict()
        return { ((), val): p for val,p in probs.items() }

    joint = (
        df
        .groupby(parents + [node])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = (
        df
        .groupby(parents)
        .size()
        .rename("total")
        .reset_index()
    )
    merged = joint.merge(totals, on=parents)
    return {
        (tuple(row[p] for p in parents), row[node]): row["count"]/row["total"]
        for _, row in merged.iterrows()
    }

def _is_adjacent_pdag(pdg, a, b) -> bool:
    """
    True if a and b are adjacent in the PDAG in ANY way:
    - undirected edge a—b, or
    - directed arc a->b or b->a.
    Some PDAG libs differentiate pdg.has_edge (undirected) vs pdg.has_arc (directed).
    We guard for both directions just in case.
    """
    return (
        pdg.has_edge(a, b) or pdg.has_edge(b, a) or
        pdg.has_arc(a, b)  or pdg.has_arc(b, a)
    )

def has_new_unshielded_collider_at(pdg_completed, node, original_parents_at_node):
    """
    Return True if, at `node`, we created an unshielded collider a->node<-b
    that wasn't already present in the baseline completion *at node*.
    """
    current_parents = list(pdg_completed.parents_of(node))
    # any pair of parents that are NOT adjacent create a collider at node
    for i in range(len(current_parents)):
        for j in range(i+1, len(current_parents)):
            a, b = current_parents[i], current_parents[j]
            if not _is_adjacent_pdag(pdg_completed, a, b):
                # It’s a collider at node. Decide whether it is “new”.
                if not (a in original_parents_at_node and b in original_parents_at_node):
                    return True
    return False


# --- new multi-node helpers ---
def _normalize_nodes(node_or_nodes):
    """Accept a single node or an iterable of nodes; return a tuple of unique nodes."""
    if isinstance(node_or_nodes, (tuple, list, set)):
        return tuple(dict.fromkeys(node_or_nodes))  # preserve order, dedup
    return (node_or_nodes,)

def _collect_original_parents_map(cpdag, nodes):
    """Map each target node -> set of its current parents in the baseline CPDAG."""
    return {u: set(cpdag.parents_of(u)) for u in nodes}

def _collect_incident_undirected_edges(cpdag, nodes):
    """
    Collect unique undirected edges incident on any node in `nodes`.
    Return as a list of 2-tuples (u, v). Each undirected edge appears once.
    """
    seen = set()
    edges = []
    for u in nodes:
        # cpdag._undirected_neighbors[u] is assumed available as in your code
        for v in cpdag._undirected_neighbors[u]:
            # Use frozenset to dedup undirected edge {u,v}
            key = frozenset((u, v))
            if key not in seen:
                seen.add(key)
                # Store in a deterministic tuple order (sorted by string repr)
                # (You can change this ordering policy if you prefer.)
                a, b = sorted((u, v), key=lambda x: str(x))
                edges.append((a, b))
    return edges

def _has_new_unshielded_collider_any(pdg_completed, nodes, original_parents_map):
    """
    True if any node in `nodes` gained a NEW unshielded collider at that node.
    """
    for u in nodes:
        if has_new_unshielded_collider_at(pdg_completed, u, original_parents_map[u]):
            return True
    return False


def _is_valid_configuration_multi(pdg, original_parents_map, nodes) -> bool:
    """
    Validate a candidate:
      1) complete PDAG with Meek,
      2) ensure arcs are acyclic,
      3) forbid NEW unshielded colliders *at any of the target nodes* (local check).
    """
    pdg_completed = pdg.copy()
    pdg_completed.to_complete_pdag()

    G = nx.DiGraph()
    G.add_edges_from(pdg_completed.arcs)
    if not nx.is_directed_acyclic_graph(G):
        return False

    if _has_new_unshielded_collider_any(pdg_completed, nodes, original_parents_map):
        return False

    return True
# ------------------------------------------

# --- generalized getConfigurations ---
def getConfigurations_multi(cpdag, node_or_nodes):
    """
    Generalization of getConfigurations that accepts:
      - node_or_nodes: a single node (str) or a tuple/list of nodes (str, ...)

    It collects *all* undirected edges incident on *any* of the target nodes,
    and enumerates all 2^E orientations of those edges (E = number of such edges).
    Each orientation is validated by the same criteria you used before:
    Meek completion, acyclicity, and NO new unshielded colliders at *any* target node.
    """
    nodes = _normalize_nodes(node_or_nodes)
    configurations = []

    # (1) Baseline parent sets at each target node (to detect "new" colliders locally)
    original_parents_map = _collect_original_parents_map(cpdag, nodes)

    # (2) Collect the undirected edges incident on any of the target nodes (unique)
    undirected_edges = _collect_incident_undirected_edges(cpdag, nodes)
    E = len(undirected_edges)

    # (3) Handle special cases 0 or 1 (fast paths)
    if E == 0:
        cpdag_copy = cpdag.copy()
        configurations.append(cpdag_copy)
        return configurations

    if E == 1:
        (u, v) = undirected_edges[0]

        # Orientation u -> v
        cpdag_copy1 = cpdag.copy()
        cpdag_copy1.replace_edge_with_arc((u, v))
        if _is_valid_configuration_multi(cpdag_copy1, original_parents_map, nodes):
            configurations.append(cpdag_copy1)

        # Orientation v -> u
        cpdag_copy2 = cpdag.copy()
        cpdag_copy2.replace_edge_with_arc((v, u))
        if _is_valid_configuration_multi(cpdag_copy2, original_parents_map, nodes):
            configurations.append(cpdag_copy2)

        return configurations

    # (4) General case: enumerate all directions for all incident undirected edges
    # Each undirected edge (a,b) becomes either a->b or b->a.
    for bits in itertools.product((0, 1), repeat=E):
        cpdag_copy = cpdag.copy()

        # Apply orientation choices
        for (choice, (a, b)) in zip(bits, undirected_edges):
            if choice == 0:
                cpdag_copy.replace_edge_with_arc((a, b))
            else:
                cpdag_copy.replace_edge_with_arc((b, a))

        # Validate this configuration
        if _is_valid_configuration_multi(cpdag_copy, original_parents_map, nodes):
            configurations.append(cpdag_copy)

    return configurations

def convert_graphical_model_object_to_nx_Digraph(cpdag, name_to_id_dict):
    nx_graph = nx.DiGraph()
    # add nodes to the graph
    cpdag_nodeids = [name_to_id_dict[nodename] for nodename in cpdag.nodes]
    nx_graph.add_nodes_from(cpdag_nodeids)

    for u, v in cpdag.edges:
        uid = name_to_id_dict[u]
        vid = name_to_id_dict[v]
        nx_graph.add_edge(uid, vid)
        nx_graph.add_edge(vid, uid)

    for u, v in cpdag.arcs:
        uid = name_to_id_dict[u]
        vid = name_to_id_dict[v]
        nx_graph.add_edge(uid, vid)

    return nx_graph

def sampleAugmentedGraphs(cpdag, nodenames, potential_root_causes):
    # given a cpdag, get a set of I-Markov in-equavialent augmented DAGs
    arugmented_dags_dict = defaultdict(list)
    mec_sizes = defaultdict(list)
    name_to_id = {nodename:i for i, nodename in enumerate(nodenames)}
    name_to_id['FNODE'] =  max(name_to_id.values()) + 1
    #print('name_to_id:{}'.format(name_to_id))
    id_to_name = {id : name for name, id in name_to_id.items()}
    #print('id_to_names:{}'.format(id_to_name))

    # for each node x , this follows numeric ascending order for the names
    # parallelize configuration‐generation across roots
    ########## NEW  ############
    partial_get = partial(getConfigurations_multi, cpdag)

    num_cores_to_use = min(4, os.cpu_count())
    with Pool(num_cores_to_use) as pool:
        # results is a list of lists of cpdag copies
        configs_per_root = pool.map(partial_get, potential_root_causes)

    # now proceed as before, but using the parallel results
    for root, all_configs_of_root in zip(potential_root_causes, configs_per_root):
        rootids = [name_to_id[rt] for rt in root]
        for config_of_root in all_configs_of_root:
            # apply meek rules
            config_of_root.to_complete_pdag()
            ########## NEW  ############
            # convert indices from strings to integer
            nx_graph = convert_graphical_model_object_to_nx_Digraph(config_of_root, name_to_id)
            #print('nx_graph edges:{}'.format(nx_graph.edges))
            nx_graph.add_node(name_to_id['FNODE'])
            for rootid in rootids:
                nx_graph.add_edge(name_to_id['FNODE'], rootid)
            size = cp.mec_size(list(nx_graph.edges))
            sampler = cp.MecSampler(list(nx_graph.edges))
            sampled_augmented_dag = sampler.sample_dag()
            new_sample_dag_edges_ls = [(id_to_name[e1], id_to_name[e2]) for e1, e2 in sampled_augmented_dag]
            graph_to_be_added = nx.DiGraph(new_sample_dag_edges_ls)
            # add back any isolated nodes
            ls_of_nodes = list(nx_graph.nodes)
            nodes_to_add = [id_to_name[nod] for nod in ls_of_nodes]
            graph_to_be_added.add_nodes_from(nodes_to_add) # this will automatically take care of duplicate nodes
            # add the graph
            arugmented_dags_dict[root].append(graph_to_be_added)
            mec_sizes[root].append(size)
    return arugmented_dags_dict, mec_sizes


def compute_conditional_prob(row, ie, target_var, conditioning_vars):
    # Clear any previous evidence
    ie.setEvidence({})  # reset inference engine
    # If conditioning variables exist, build and set evidence
    if conditioning_vars:
        evidence = {var: int(row[var]) for var in conditioning_vars}
        ie.setEvidence(evidence)

    # Compute posterior P(target_var | evidence)
    posterior = ie.posterior(target_var)

    # Return the probability of the value observed in the row
    return posterior[int(row[target_var])]




def compute_conditional_probs_cached(df, ie, target_var,
                                     conditioning_vars=None,
                                     alpha=1e-6):
    """
    Computes P(target_var = y | conditioning_vars = x) for each row in df,
    by caching unique evidence combinations in a dict, then doing one
    pandas merge to broadcast back to all rows.

    Parameters:
    - df: pandas DataFrame containing all relevant variables
    - ie: pyAgrum inference engine (e.g., gum.LazyPropagation(bn))
    - target_var: name of the target variable (str)
    - conditioning_vars: list of variable names to condition on (can be empty or None)
    - alpha: fallback probability for unseen combinations (default: 1e-6)

    Returns:
    - A NumPy array of conditional probabilities, aligned with df rows
    """
    if conditioning_vars is None:
        conditioning_vars = []

    lookup = {}

    # Case 1: No conditioning variables → compute once (prior)
    if not conditioning_vars:
        ie.setEvidence({})
        posterior = ie.posterior(target_var)
        # map each row's value to its prior
        return df[target_var].astype(int).map(lambda val: posterior[val]).values

    # Case 2: With conditioning → build lookup for all unique combinations
    unique_rows = df[conditioning_vars + [target_var]].drop_duplicates()
    for _, row in unique_rows.iterrows():
        evidence = {var: int(row[var]) for var in conditioning_vars}
        ie.setEvidence(evidence)
        posterior = ie.posterior(target_var)
        key = tuple(int(row[var]) for var in conditioning_vars) \
              + (int(row[target_var]),)
        lookup[key] = posterior[int(row[target_var])]

    # ──────────────── vectorized broadcast ────────────────

    if not lookup:
        # no combos? fallback to alpha
        return np.full(len(df), alpha)

    # Build a tiny lookup‐table DataFrame
    # each key is (c1, c2, ..., target), value is prob
    rows = []
    cols = conditioning_vars + [target_var]
    for key, prob in lookup.items():
        entry = dict(zip(cols, key))
        entry["prob"] = prob
        rows.append(entry)
    lookup_df = pd.DataFrame(rows)

    # Merge it onto the full df at once
    merged = df.merge(lookup_df, on=cols, how="left")

    # Fill any missing combos with alpha
    return merged["prob"].fillna(alpha).values



def discrete_likelihood_fn(df, node, parents, obs_ie, int_ie, obs_ground_truth, int_ground_truth, alpha=1e-6):
    """
    Compute the likelihood for a discrete node with string entries.
    
    Parameters:
        df (pd.DataFrame): The dataset.
        node (str): The name of the current node.
        parents (list): A list of parent node names.
        alpha (float): A small number to assign to unseen parent-child combinations.
        
    Returns:
        np.ndarray: A vector of likelihoods for the node over all rows in df.
    """
    if not parents:
        # Compute the empirical probability of each value in the column
        if obs_ground_truth:
            if node == 'FNODE':
                # directly compute the distribution for F-node from data 
                probs = df[node].value_counts(normalize=True).to_dict()
                return df[node].map(probs).values
            else:
                # REPLACE WITH PRE-COMPUTE
                return compute_conditional_probs_cached(df, obs_ie, node, conditioning_vars=None, alpha=1e-6)
        else:
            # REPLACE WITH PRE-COMPUTE
            probs = df[node].value_counts(normalize=True).to_dict()
            # Map each observation to its probability
            return df[node].map(probs).values
    else:
        if obs_ground_truth:
            if 'FNODE' in parents:
                n_df = df[df['FNODE'] == 0]
                a_df = df[df['FNODE'] == 1]
                pa_without_F = [pa for pa in parents if pa != 'FNODE']
                if not pa_without_F:
                    # if there is no other parent besides F-NODE
                    probs = a_df[node].value_counts(normalize=True).to_dict()
                    # Map each observation to its probability
                    int_dist = a_df[node].map(probs).values
                else:
                    if not int_ground_truth:
                        # get the int distribution from data
                        joint_counts = a_df.groupby(pa_without_F + [node]).size().reset_index(name='count')
                        # Then, compute the totals for each parent's configuration
                        parent_totals = a_df.groupby(pa_without_F).size().reset_index(name='total')
                        # Merge to get the conditional probability for each joint combination
                        merged = pd.merge(joint_counts, parent_totals, on=pa_without_F)
                        merged['prob'] = merged['count'] / merged['total']
                        # Merge the computed probabilities back to the original dataframe
                        df_merged = pd.merge(a_df, merged[pa_without_F + [node, 'prob']], on=pa_without_F + [node], how='left')
                        # For any unseen parent-child combination, fill in a small probability
                        df_merged['prob'].fillna(alpha, inplace=True)
                        int_dist = df_merged['prob'].values
                    else:
                        int_dist = compute_conditional_probs_cached(a_df, int_ie, node, conditioning_vars=pa_without_F, alpha=1e-6)
                obs_dist = compute_conditional_probs_cached(n_df, obs_ie, node, conditioning_vars=pa_without_F, alpha=1e-6)
                return np.append(obs_dist, int_dist)
            else:
                # REPLACE WITH PRE-COMPUTE
                return compute_conditional_probs_cached(df, obs_ie, node, conditioning_vars=parents, alpha=1e-6)
            
        else:
            if 'FNODE' in parents:
                pa_without_F = [pa for pa in parents if pa != 'FNODE']
                if int_ground_truth:
                    n_df = df[df['FNODE'] == 0]
                    a_df = df[df['FNODE'] == 1]
                    int_dist = compute_conditional_probs_cached(a_df, int_ie, node, conditioning_vars=pa_without_F, alpha=1e-6)
                    obs_dist = compute_conditional_probs_cached(n_df, int_ie, node, conditioning_vars=pa_without_F, alpha=1e-6)
                    return np.append(obs_dist, int_dist)
                else:
                    joint_counts = df.groupby(parents + [node]).size().reset_index(name='count')
                    # Then, compute the totals for each parent's configuration
                    parent_totals = df.groupby(parents).size().reset_index(name='total')
                    # Merge to get the conditional probability for each joint combination
                    merged = pd.merge(joint_counts, parent_totals, on=parents)
                    merged['prob'] = merged['count'] / merged['total']
                    # Merge the computed probabilities back to the original dataframe
                    df_merged = pd.merge(df, merged[parents + [node, 'prob']], on=parents + [node], how='left')
                    # For any unseen parent-child combination, fill in a small probability
                    df_merged['prob'].fillna(alpha, inplace=True)
                    return df_merged['prob'].values
            else:
                # TO DO: If F is in the parents, then we compute
                # IF NOT, we then just get the pre-computed likelihood

                # Compute conditional probabilities for each combination of parent's values and node
                # First, count the joint occurrences
                joint_counts = df.groupby(parents + [node]).size().reset_index(name='count')
                # Then, compute the totals for each parent's configuration
                parent_totals = df.groupby(parents).size().reset_index(name='total')
                # Merge to get the conditional probability for each joint combination
                merged = pd.merge(joint_counts, parent_totals, on=parents)
                merged['prob'] = merged['count'] / merged['total']
                # Merge the computed probabilities back to the original dataframe
                df_merged = pd.merge(df, merged[parents + [node, 'prob']], on=parents + [node], how='left')
                # For any unseen parent-child combination, fill in a small probability
                df_merged['prob'].fillna(alpha, inplace=True)
                return df_merged['prob'].values



_local_factor_cache = {}
def compute_local_likelihoods(dag, df, ie,  obs_ground_truth, int_ground_truth):
    """
    Compute local likelihoods for each node in the DAG over the dataframe using a common discrete likelihood function.
    Works in log-space to prevent numerical underflow.
    
    Parameters:
        dag (networkx.DiGraph): The DAG with nodes representing variables.
        df (pd.DataFrame): The discrete dataset where each column is a variable.
        
    Returns:
        np.ndarray: A 1D array of shape (n_samples, 1) where each row corresponds to joint distribution based on a sample.
    """
    factors = []
    
    # Process nodes in topological order to respect dependency order
    for node in nx.topological_sort(dag):
        # Use the discrete likelihood function for every node
        parents = list(dag.predecessors(node))
        parents = [str(pa) for pa in parents]

        cache_key = (node, tuple(parents), obs_ground_truth)
        if cache_key not in _local_factor_cache:
            _local_factor_cache[cache_key] = discrete_likelihood_fn(
                df,
                str(node),
                parents,
                ie,
                obs_ground_truth,
                int_ground_truth
            )
        
        local_factor = _local_factor_cache[cache_key]
        
        if len(local_factor) != len(df):
            raise ValueError(f"Likelihood for node '{node}' returned an array of incorrect length.")
        
        # Convert to log space to prevent underflow
        log_local_factor = np.log(local_factor + 1e-300)  # Add small constant to avoid log(0)
        
        # p(v|pa(v)) over all rows in the dataframe
        factors.append(log_local_factor)
    
    # Stack the log factors
    log_joint = np.column_stack(factors)
    
    # Sum the log factors (equivalent to multiplying the original factors)
    log_joint = np.sum(log_joint, axis=1)
    
    # Convert back to probability space if needed
    # joint = np.exp(log_joint)
    
    # Return the log joint probability
    return log_joint



def cpdag_to_key(cpdag: Any) -> str:
    """
    Produce a canonical string key for a PDAG/CPDAG so we can deduplicate.
    Expected cpdag has attributes or methods to list directed arcs and undirected edges.
    Adjust attribute access if your PDAG has different API.
    """
    # Try common attribute names; otherwise try methods .arcs / .edges
    try:
        arcs = list(cpdag.arcs)
        edges = list(cpdag.edges)
    except Exception:
        # fallback assuming cpdag has edges_list() like earlier examples
        try:
            el = cpdag.edges_list()
            return ";".join(sorted(el))
        except Exception:
            # try to introspect
            arcs = []
            edges = []
    # normalize tuples to sorted strings
    arcs_s = sorted([f"{u}->{v}" for (u, v) in arcs])
    edges_s = sorted([f"{min(u,v)}--{max(u,v)}" for (u, v) in edges])
    return "|".join(arcs_s + edges_s)

def bootstrap_cpdag_list(
    df_obs: pd.DataFrame,
    bootstrap_samples: int,
    random_state: int = None,
    show_progress: bool = True
) -> Tuple[List[Any], Dict[str, Dict[str, Any]]]:
    """
    Run bootstrap_samples times:
      - resample df_obs with replacement -> b_df
      - run boss(b_df.to_numpy())
      - convert to arcs, edges
      - construct PDAG(nodes=colnames, arcs=arcs, edges=edges)
    Returns:
      - list_of_cpdags: raw list of PDAG objects in sampling order
      - summary: dict keyed by canonical cpdag key with:
           { 'cpdag': PDAG_obj, 'count': int, 'freq': float, 'examples': [indices_of_occurrences] }
    """
    rng = np.random.default_rng(random_state)
    n = len(df_obs)
    colnames = list(df_obs.columns)
    list_of_cpdags: List[Any] = []
    keys_order: List[str] = []
    # store indices of bootstrap runs that produced each unique CPDAG
    key_to_info: Dict[str, Dict[str, Any]] = {}

    it = range(bootstrap_samples)
    if show_progress:
        it = tqdm(it, desc="bootstrapping CPDAGs")

    for b in it:
        # reproducible resampling per iteration (optionally deterministic given seed)
        # draw indices with replacement
        idxs = rng.integers(low=0, high=n, size=n)
        b_df = df_obs.iloc[idxs].reset_index(drop=True)

        try:
            # run your algorithm: boss expects numpy array
            G_cl = boss(b_df.to_numpy())

            # convert to networkx-style arcs/edges using your helper
            arcs, edges = causal_learn_graph_to_nx_digraph(G_cl, colnames)

            # build PDAG object
            cpdag = PDAG(nodes=colnames, arcs=arcs, edges=edges)

            # record
            list_of_cpdags.append(cpdag)

            key = cpdag_to_key(cpdag)
            keys_order.append(key)
            if key not in key_to_info:
                key_to_info[key] = {"cpdag": cpdag, "count": 0, "examples": [], 'df': b_df.copy()}
            key_to_info[key]["count"] += 1
            key_to_info[key]["examples"].append(b)  # store bootstrap run index

        except Exception as e:
            # handle failures gracefully (log or print). Continue looping.
            # You can also collect failed indices if needed.
            print(f"bootstrap iter {b} failed: {e}")
            continue

    # compute frequencies
    total = sum(info["count"] for info in key_to_info.values())
    for key, info in key_to_info.items():
        info["freq"] = info["count"] / total if total > 0 else 0.0

    return list_of_cpdags, key_to_info

# GLOBAL_DATA = None        # numpy array shape (N, d)
# GLOBAL_COLNAMES = None
# GLOBAL_N = None
# GLOBAL_BOSS = None
# GLOBAL_C2NX = None

# def _init_worker(data_np, colnames, boss_callable, c2nx_callable):
#     """Initializer for worker processes. Stores common objects in module globals to avoid pickling each task."""
#     global GLOBAL_DATA, GLOBAL_COLNAMES, GLOBAL_N, GLOBAL_BOSS, GLOBAL_C2NX
#     GLOBAL_DATA = data_np
#     GLOBAL_COLNAMES = colnames
#     GLOBAL_N = GLOBAL_DATA.shape[0]
#     GLOBAL_BOSS = boss_callable
#     GLOBAL_C2NX = c2nx_callable

# def _worker_task(args):
#     """
#     Worker receives (b_index, seed_base).
#     Returns (b_index, arcs, edges, error_str_or_None)
#     """
#     b_index, seed_base = args
#     try:
#         rng = np.random.default_rng(seed_base + int(b_index))
#         # draw bootstrap indices
#         idxs = rng.integers(low=0, high=GLOBAL_N, size=GLOBAL_N)
#         b_np = GLOBAL_DATA[idxs]  # views/copies managed by numpy

#         # call boss on the resampled numpy array
#         G_cl = GLOBAL_BOSS(b_np)

#         # convert to arcs/edges using helper
#         arcs, edges = GLOBAL_C2NX(G_cl, GLOBAL_COLNAMES)

#         return (b_index, arcs, edges, None)
#     except Exception as e:
#         # return error string to handle gracefully
#         import traceback
#         return (b_index, None, None, f"{str(e)}\n{traceback.format_exc()}")

# def parallel_bootstrap_cpdag_list(
#     df_obs,
#     bootstrap_samples: int,
#     random_state: int = None,
#     n_workers: int = None,
#     chunksize: int = 1,
#     show_progress: bool = True
# ) -> Tuple[List[Any], Dict[str, Dict[str, Any]]]:
#     """
#     Parallel bootstrap version returning (list_of_cpdags, summary).
#     - Returns PDAG objects reconstructed in main process.
#     - summary keyed by canonical key -> {'cpdag': PDAG_obj, 'count': int, 'freq': float, 'examples': [b_indices]}
#     """
#     # Convert dataframe once
#     data_np = df_obs.to_numpy()
#     colnames = list(df_obs.columns)
#     N = data_np.shape[0]

#     # seed base for reproducibility
#     seed_base = 0 if random_state is None else int(random_state)

#     # prepare worker arguments (b indices)
#     args = [(b, seed_base) for b in range(bootstrap_samples)]

#     list_of_cpdags: List[Any] = []
#     key_to_info: Dict[str, Dict[str, Any]] = {}

#     # Use multiprocessing Pool with initializer to publish GLOBAL_DATA etc.
#     with Pool(processes=n_workers, initializer=_init_worker,
#               initargs=(data_np, colnames, boss, causal_learn_graph_to_nx_digraph)) as pool:

#         if show_progress:
#             it = pool.imap_unordered(_worker_task, args, chunksize=chunksize)
#             pbar = tqdm(total=bootstrap_samples, desc="bootstrap (parallel)")
#         else:
#             it = pool.imap_unordered(_worker_task, args, chunksize=chunksize)
#             pbar = None

#         for res in it:
#             if pbar:
#                 pbar.update(1)
#             b_index, arcs, edges, error = res
#             if error is not None:
#                 # log & continue
#                 print(f"bootstrap iter {b_index} failed: {error}")
#                 continue
#             # reconstruct PDAG in main process (safer than pickling PDAG objects across processes)
#             cpdag = PDAG(nodes=colnames, arcs=arcs, edges=edges)
#             list_of_cpdags.append(cpdag)

#             key = cpdag_to_key(cpdag)   # you must have a canonical key function available in main process
#             if key not in key_to_info:
#                 key_to_info[key] = {"cpdag": cpdag, "count": 0, "examples": []}
#             key_to_info[key]["count"] += 1
#             key_to_info[key]["examples"].append(b_index)

#         if pbar:
#             pbar.close()

#     total = sum(info["count"] for info in key_to_info.values())
#     for key, info in key_to_info.items():
#         info["freq"] = info["count"] / total if total > 0 else 0.0

#     return list_of_cpdags, key_to_info


def get_top_k_cpdags_with_ratio(summary, isdiscrete, node_transform, transform_parents, k=10):
    """
    From summary, return a list of dicts:
    [
        {'cpdag': CPDAG_object, 'topk_ratio': float},
        ...
    ]
    where topk_ratio is normalized only among the top-k CPDAGs.
    """
    # Sort by count descending
    sorted_items = sorted(summary.items(), key=lambda kv: -kv[1]["count"])
    
    # Take the top-k entries
    top_k = sorted_items[:k]
    
    # Sum of counts within top-k
    total_top_k_counts = sum(info["count"] for _, info in top_k)
    
    # Build output: ONLY cpdag and normalized ratio
    result = []
    for _, info in top_k:
        obs_df = info["df"]
        nodenames = list(info["cpdag"].nodes)
        name_to_id = {nodename:i for i, nodename in enumerate(nodenames)}
        id_to_name = {id : name for name, id in name_to_id.items()}

        nx_graph = convert_graphical_model_object_to_nx_Digraph(info["cpdag"], name_to_id)
        sampler = cp.MecSampler(list(nx_graph.edges))
        sampled_dag = sampler.sample_dag()
        new_sample_dag_edges_ls = [(id_to_name[e1], id_to_name[e2]) for e1, e2 in sampled_dag]
        dag = nx.DiGraph(new_sample_dag_edges_ls)
        ls_of_nodes = list(nx_graph.nodes)
        nodes_to_add = [id_to_name[nod] for nod in ls_of_nodes]
        dag.add_nodes_from(nodes_to_add)
        unique_families = {}
        for node in dag.nodes():
            parents = tuple(sorted(dag.predecessors(node)))
            unique_families.setdefault((node, parents), None)
            if isdiscrete:
                probs = discrete_likelihood_fn_dirichlet(
                        obs_df,
                        str(node),
                        [str(p) for p in parents],
                        alpha_star=5.0,                  # tune: 1..10 common; larger = stronger smoothing
                        cardinalities=None               # or pass a precomputed {var: K} dict
                )
            else:
                probs = continuous_likelihood_fn_gaussian(
                        obs_df,
                        str(node),
                        [str(p) for p in parents],
                        node_transform = node_transform,       # "none" | "log" | "log1p" | "yeojohnson"
                        transform_parents = transform_parents,           
                        gating = "empirical",              
                    )
            unique_families[(node, parents)] = np.log(probs + 1e-300)
        log_joint = sum(
                unique_families[(node, tuple(sorted(dag.predecessors(node))))]
                for node in dag.nodes()
        )
        log_joint = np.sum(log_joint)
        
        
        ratio_among_topk = info["count"] / total_top_k_counts
        log_p_c = np.log(1/k + 1e-300)
        log_ratio_among_topk = np.log(ratio_among_topk + 1e-300)
        posterior_of_this_cpdag = log_joint + log_p_c - log_ratio_among_topk
        result.append({
            "cpdag": info["cpdag"],
            'topk_ratio': posterior_of_this_cpdag
            # "topk_ratio": ratio_among_topk
        })
    denom = np.array([])
    for info in result:
        denom = np.append(denom, info['topk_ratio'])
    log_Z = logsumexp(denom)
    for info in result:
        info['topk_ratio'] = np.exp(info['topk_ratio'] - log_Z)
        print(info['topk_ratio'])
    return result


def brcd_update(joint_df,
                cpdag, 
                cols, 
                combos, 
                isdiscrete, 
                node_transform, 
                transform_parents, 
                prior):
     # ───────────────────────────────────────────────────────────────
    # 2) Sample augmented DAGs

    sampled_augmented, mec_sizes = sampleAugmentedGraphs(cpdag, cols,
                                                         combos)


    # ───────────────────────────────────────────────────────────────
    # 3) Build a cache of every unique (node, parents) family *once*
    unique_families = {}
    for dags in sampled_augmented.values():
        for dag in dags:
            for node in dag.nodes():
                parents = tuple(sorted(dag.predecessors(node)))
                unique_families.setdefault((node, parents), None)
    start_time = time.time()
    # Compute & store their log‑likelihood vectors


    for (node, parents) in unique_families:
        if isdiscrete:
            probs = discrete_likelihood_fn_dirichlet(
                    joint_df,
                    str(node),
                    [str(p) for p in parents],
                    alpha_star=5.0,                  # tune: 1..10 common; larger = stronger smoothing
                    cardinalities=None               # or pass a precomputed {var: K} dict
            )
        else:
            pa_set = [str(p) for p in parents]
            if 'FNODE' in pa_set:
                probs = continuous_likelihood_fn_gaussian(
                        joint_df,
                        str(node),
                        [str(p) for p in parents],
                        F='FNODE',
                        node_transform = node_transform,       # "none" | "log" | "log1p" | "yeojohnson"
                        transform_parents = transform_parents,           
                        gating = "auto",              
                )
            else:
                probs = continuous_likelihood_fn_gaussian(
                        joint_df,
                        str(node),
                        [str(p) for p in parents],
                        node_transform = node_transform,       # "none" | "log" | "log1p" | "yeojohnson"
                        transform_parents = transform_parents,           
                        gating = "empirical",              
                    )
            # --- Forest-KDE instead of Gaussian ---
            # common = dict(
            #     df=joint_df,
            #     node=str(node),
            #     parents=pa_set,
            #     node_transform=node_transform,
            #     transform_parents=False,
            #     rf_params=dict(n_estimators=64, max_depth=6, min_samples_leaf=32, max_features="sqrt", bootstrap=False),
            # )
            # if 'FNODE' in pa_set:
            #     probs = continuous_likelihood_fn_forestkde(
            #     **common, F='FNODE', gating='auto'  # gating ignored in this branch
            # )
            # else:
            #     probs = continuous_likelihood_fn_forestkde(
            #         **common, F=None  # or just omit F
            #     )
        
        unique_families[(node, parents)] = np.log(probs + 1e-300)


   
    # 4) For each root r, assemble per‑DAG joint logs, add log‑prior, then log‑sum‑exp
    log_p_data_given_R = []
    root_causes = []
    all_config_size = 0
    for r in sampled_augmented.keys():
        all_config_size += len(sampled_augmented[r])

    for r, dags in sampled_augmented.items():
        sizes = np.array(mec_sizes[r], dtype=float)
        log_p_G = np.log(sizes / sizes.sum() + 1e-300)

        # build an (n_samples × num_dags) array where each column = log P(data|G) + log P(G)
        cols = []
        for i, dag in enumerate(dags):
            # sum cached log‑factors over that DAG's families
            log_joint = sum(
                unique_families[(node, tuple(sorted(dag.predecessors(node))))]
                for node in dag.nodes()
            )
            cols.append(log_joint + log_p_G[i])
            #cols.append(log_joint + log_p_G)

        matrix = np.column_stack(cols)
        # log P(data | root=r) = logsumexp over DAGs
        log_p_data_given_R.append(logsumexp(matrix, axis=1))
        root_causes.append(r)

    # stack over roots → shape = (num_roots, num_samples)
    log_p_data_given_R = np.stack(log_p_data_given_R, axis=0)



    # ───────────────────────────────────────────────────────────────
    # 5) Sum over samples to get log-likelihood per root, add uniform prior, normalize
    log_likelihood = log_p_data_given_R.sum(axis=1)            # shape=(num_roots,)
    log_posterior = log_likelihood + np.log(prior)
    end_time = time.time()
    elasped = end_time - start_time

    return log_posterior, elasped


def brcd_helper(normal_df,
         anomalous_df,
         cpdag=None,
         isdiscrete=False,
         node_transform = "none",       # "none" | "log" | "log1p" | "yeojohnson"
         transform_parents= True,
         num_root_causes_candidates = 1,
         bootstrap_samples = 10):
    # ───────────────────────────────────────────────────────────────
    # 1) Prepare data + inference engine
    _local_factor_cache.clear()
    _fkde_cache.clear()
    df_obs = normal_df.copy()
    df_int = anomalous_df.copy()
    joint_df = pd.concat([df_obs, df_int], ignore_index=True)

    #discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
    #discretizer.fit(joint_df.to_numpy())
    #disc_d = discretizer.transform(joint_df.to_numpy())
    #joint_df = pd.DataFrame(disc_d, columns=joint_df.columns.values.tolist()).astype(int)
    joint_df['FNODE'] = np.r_[np.zeros(len(df_obs), dtype=int), np.ones(len(df_int), dtype=int)]
    potential_root_causes = list(normal_df.columns)

    cols = list(normal_df.columns)
    combos = list(itertools.combinations(cols, num_root_causes_candidates))
    prior = np.ones(len(combos)) / len(combos)


    # boostrapping the data
    if cpdag is None:
        list_of_cpdags, summary = bootstrap_cpdag_list(
                                    df_obs=df_obs,
                                    bootstrap_samples=bootstrap_samples,
                                    random_state=22
                                )
        # list_of_cpdags, summary = parallel_bootstrap_cpdag_list(
        #     df_obs,
        #     bootstrap_samples,
        #     random_state=22,
        #     n_workers=8,
        #     chunksize=10,
        #     show_progress=True
        # )

        print('Number of unique CPDAGs: {}'.format(len(summary)))
        #top_k_cpdags = get_top_k_cpdags_with_ratio(summary, k=len(summary))
        top_k_cpdags = get_top_k_cpdags_with_ratio(summary, isdiscrete, node_transform, transform_parents, k=len(summary))
        start_time = time.time()
        out = parallel_weighted_posterior(
                                    topk_list=top_k_cpdags,
                                    joint_df=joint_df,
                                    cols=cols,
                                    combos=combos,
                                    isdiscrete=isdiscrete,
                                    node_transform=node_transform,
                                    transform_parents=transform_parents,
                                    prior=prior,
                                    brcd_update=brcd_update,
                                    n_workers=8,            # tune to CPU cores
                                    renormalize=False,
                                    log_space=True,         # brcd_update returns log-posteriors
                                    use_threads=False,      # set True to use ThreadPoolExecutor
                                    show_progress=True
                                )

        # for idx, item in enumerate(top_k_cpdags):
        #     cpdag = item["cpdag"]
        #     ratio = float(item["topk_ratio"])
        elapsed = time.time() - start_time
        log_posterior = out['final_posterior']


    else:
         log_posterior, elapsed = brcd_update(joint_df,
                cpdag, 
                cols, 
                combos, 
                isdiscrete, 
                node_transform, 
                transform_parents, 
                prior)


   
    # Rank by p(R|D) ∝ p(D|R)p(R), i.e. by log_posterior (avoids exp underflow when one candidate dominates).
    sorted_indices = np.argsort(-log_posterior)
    sorted_root_causes = [combos[i] for i in sorted_indices]
    sorted_posterior = [log_posterior[i] for i in sorted_indices]
    if num_root_causes_candidates == 1:
        sorted_root_causes = [t[0] for t in sorted_root_causes]
    return {
        "ranks": sorted_root_causes
    }



def brcd(data, inject_time=None, dataset=None, graph=None, **kwargs):
    normal_df = data[data["time"] < inject_time]
    anomal_df = data[data["time"] >= inject_time]
    normal_df = normal_df.drop(columns=["time"])
    anomal_df = anomal_df.drop(columns=["time"])

    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]
              
    # remove nodes from graph if
    if graph is not None:
        granular_graph = df_to_prefix_graph(normal_df, graph)
        cpdag = PDAG(nodes=list(normal_df.columns), arcs=list(granular_graph.edges), edges=list())

    # G_cl = boss(normal_df.to_numpy())
    # arcs, edges = causal_learn_graph_to_nx_digraph(G_cl, list(normal_df.columns))
    # cpdag = PDAG(nodes=list(normal_df.columns), arcs=arcs, edges=edges)

    return brcd_helper(normal_df,
         anomal_df ,
         cpdag=cpdag,
         isdiscrete=False,
         node_transform = "none",       # "none" | "log" | "log1p" | "yeojohnson"
         transform_parents= True,
         num_root_causes_candidates = 1)
    
   
   
