import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import itertools
from copy import deepcopy
from utils import gm_to_nx_Digraph
import cliquepicking as cp
import networkx as nx
from collections import defaultdict
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
from typing import Dict, Optional, List,Literal

#### New Esimtator ##################

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



def brcd(normal_df,
         anomalous_df,
         cpdag,
         isdiscrete=False,
         node_transform = "none",       # "none" | "log" | "log1p" | "yeojohnson"
         transform_parents = False, 
         version='brcd_u',
         num_root_causes_candidates = 1):
    # ───────────────────────────────────────────────────────────────
    # 1) Prepare data + inference engine
    _local_factor_cache.clear()
    _fkde_cache.clear()
    df_obs = normal_df.copy()
    df_int = anomalous_df.copy()
    joint_df = pd.concat([df_obs, df_int], ignore_index=True)

    
    joint_df['FNODE'] = np.r_[np.zeros(len(df_obs), dtype=int), np.ones(len(df_int), dtype=int)]
    potential_root_causes = list(normal_df.columns)

    cols = list(normal_df.columns)
    combos = list(itertools.combinations(cols, num_root_causes_candidates))
    prior = np.ones(len(combos)) / len(combos)



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
            # if 'FNODE' in pa_set:
            #     probs = continuous_likelihood_fn_gaussian(
            #             joint_df,
            #             str(node),
            #             [str(p) for p in parents],
            #             F='FNODE',
            #             node_transform = node_transform,       # "none" | "log" | "log1p" | "yeojohnson"
            #             transform_parents = transform_parents,           
            #             gating = "auto",              
            #     )
            # else:
            #     probs = continuous_likelihood_fn_gaussian(
            #             joint_df,
            #             str(node),
            #             [str(p) for p in parents],
            #             node_transform = node_transform,       # "none" | "log" | "log1p" | "yeojohnson"
            #             transform_parents = transform_parents,           
            #             gating = "empirical",              
            #         )
            # --- Forest-KDE instead of Gaussian ---
            common = dict(
                df=joint_df,
                node=str(node),
                parents=pa_set,
                node_transform=node_transform,
                transform_parents=False,
                rf_params=dict(n_estimators=64, max_depth=6, min_samples_leaf=32, max_features="sqrt", bootstrap=False),
            )
            if 'FNODE' in pa_set:
                probs = continuous_likelihood_fn_forestkde(
                **common, F='FNODE', gating='auto'  # gating ignored in this branch
            )
            else:
                probs = continuous_likelihood_fn_forestkde(
                    **common, F=None  # or just omit F
                )
        
        unique_families[(node, parents)] = np.log(probs + 1e-300)


   
    # 4) For each root r, assemble per‑DAG joint logs, add log‑prior, then log‑sum‑exp
    log_p_data_given_R = []
    root_causes = []
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
    # Rank by p(R|D) ∝ p(D|R)p(R), i.e. by log_posterior (avoids exp underflow when one candidate dominates).
    sorted_indices = np.argsort(-log_posterior)
    sorted_root_causes = [root_causes[i] for i in sorted_indices]
    sorted_posterior = [log_posterior[i] for i in sorted_indices]
    return sorted_root_causes, sorted_posterior, elasped


    
   
   
   