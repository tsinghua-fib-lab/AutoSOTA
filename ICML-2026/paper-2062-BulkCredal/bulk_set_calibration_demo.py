# -*- coding: utf-8 -*-
"""bulk_set_calibration_demo.ipynb
# Imports
"""

import numpy as np
import numpy.linalg as la
from scipy.stats import beta, chi2, multivariate_normal
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.stats import f as fdist
from scipy.optimize import brentq
from matplotlib.patches import Ellipse, Rectangle

np.set_printoptions(precision=4, suppress=True)

from pathlib import Path

import seaborn as sns
sns.set_theme(
  context="talk",           # larger base sizes
  style="whitegrid"        # clean background
)
sns.color_palette("husl", 9)

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,              # embed TrueType in vector PDFs
    "ps.fonttype": 42,

    # Global size baseline (increase overall scale)
    "font.size": 14,                 # body/labels baseline

    # Fine-grained sizes
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
})


from matplotlib import patheffects as pe
from matplotlib.patches import Polygon

"""## Utility functions"""

def safe_cov(X, ridge=1e-8):
    """
    Sample covariance with a tiny ridge to ensure positive definiteness.
    X: (n,d)
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / max(1, X.shape[0] - 1)
    return S + ridge * np.eye(S.shape[0])

def split_indices(n, fit_prop=0.5, rng=None):
    """
    Random split into:
      - fit   (a.k.a. 'train'): used to fit the centre / score function
      - select: used to select the DKW threshold
    Returns (ids_train, ids_select).
    """
    rng = np.random.default_rng() if rng is None else rng
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(fit_prop * n)
    ids_train = idx[:n_train]
    ids_select = idx[n_train:]
    return ids_train, ids_select

def run_bulk_certificate(
    X, dgp, gamma=0.05, delta=0.05, score_type="ellipsoid",
    split=0.5, use_pc=True, J=8, rng=None
):
    """
    DKW-only bulk-set construction.

    Splits X into:
      - train (fit):   used to fit the centre / score function
      - select:        used to select the DKW threshold

    Returns a dict with ids, fitted centre, score metadata, select scores,
    and the DKW selection result.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = X.shape[0]

    # --- parse split (allow float or tuple/list)
    if isinstance(split, (tuple, list, np.ndarray)):
        if len(split) != 2:
            raise ValueError("split must be a float or a tuple/list of length 2 (fit, select).")
        fit_prop = float(split[0])
    else:
        fit_prop = float(split)

    ids_train, ids_select = split_indices(n, fit_prop=fit_prop, rng=rng)
    X_train, X_select = X[ids_train], X[ids_select]

    # --- fit centre on train if requested
    center = fit_pc_gaussian(X_train) if use_pc else None

    # --- build score function
    s, meta = make_score_function(
        score_type=score_type,
        center=center if use_pc else None,
        X_ref=X_train if not use_pc else None,
        J=J,
    )

    # --- DKW selection uses select scores directly
    s_select = np.asarray(s(X_select), dtype=float).ravel()
    dkw = dkw_select_threshold(s_select, gamma=gamma, delta=delta)

    return {
        "ids": {"train": ids_train, "select": ids_select},
        "center": center,
        "score_meta": meta,
        "s_select": s_select,
        "dkw": dkw,
    }

@dataclass
class TrueDGP:
    name: str
    d: int
    params: dict
    rng: np.random.Generator

    def sample(self, n):
        raise NotImplementedError

    def logpdf(self, X):
        raise NotImplementedError

    def pdf(self, X):
        return np.exp(self.logpdf(X))

class GaussianDGP(TrueDGP):
    def __init__(self, mu, Sigma, rng=None):
        super().__init__(name="gaussian", d=len(mu), params={"mu": mu, "Sigma": Sigma},
                         rng=np.random.default_rng() if rng is None else rng)
        self.mu = np.asarray(mu)
        self.Sigma = np.asarray(Sigma)
        self._mv = multivariate_normal(mean=self.mu, cov=self.Sigma, allow_singular=False)

    def sample(self, n):
        return self._mv.rvs(size=n, random_state=self.rng)

    def logpdf(self, X):
        return self._mv.logpdf(X)

class MvtDGP(TrueDGP):
    """Multivariate Student t with df=nu, location mu, scatter Sigma (scale matrix)."""
    def __init__(self, mu, Sigma, nu=5.0, rng=None):
        super().__init__(name="mvt", d=len(mu), params={"mu": mu, "Sigma": Sigma, "nu": nu},
                         rng=np.random.default_rng() if rng is None else rng)
        self.mu = np.asarray(mu)
        self.Sigma = np.asarray(Sigma)
        self.nu = float(nu)
        self.d = self.mu.size
        self.Sig_inv = la.inv(self.Sigma)
        self._log_norm = (
            np.log(la.det(self.Sigma)) * (-0.5)
            - (self.d/2)*np.log(self.nu*np.pi)
            + np.sum(np.log(np.arange(1, int(self.nu+self.d)//2)) - np.log(np.arange(1, int(self.nu)//2)), initial=0.0)
        )  # constant; we will compute using scipy.special.gammaln instead (more stable)

        # Recompute with gammaln for stability
        from scipy.special import gammaln
        self._log_const = gammaln((self.nu + self.d)/2) - gammaln(self.nu/2) \
                          - (self.d/2)*np.log(self.nu*np.pi) - 0.5*np.log(la.det(self.Sigma))

    def sample(self, n):
        g = self.rng.chisquare(self.nu, size=n) / self.nu  # shape (n,)
        Z = self.rng.multivariate_normal(np.zeros(self.d), self.Sigma, size=n)
        return self.mu + Z / np.sqrt(g)[:, None]

    def logpdf(self, X):
        X = np.asarray(X)
        D = X - self.mu
        Q = np.einsum('...i,ij,...j->...', D, self.Sig_inv, D)  # Mahalanobis^2
        return self._log_const - 0.5*(self.nu + self.d)*np.log1p(Q/self.nu)

class ContaminatedGaussianDGP(TrueDGP):
    """(1-eps) N(mu, Sigma) + eps N(mu, c^2 Sigma) contamination."""
    def __init__(self, mu, Sigma, eps=0.05, scale_mult=2.0, rng=None):
        super().__init__(name="contaminated-gaussian", d=len(mu),
                         params={"mu": mu, "Sigma": Sigma, "eps": eps, "scale_mult": scale_mult},
                         rng=np.random.default_rng() if rng is None else rng)
        self.mu = np.asarray(mu)
        self.Sigma = np.asarray(Sigma)
        self.eps = float(eps)
        self.c = float(scale_mult)
        self._mv1 = multivariate_normal(mean=self.mu, cov=self.Sigma, allow_singular=False)
        self._mv2 = multivariate_normal(mean=self.mu, cov=(self.c**2)*self.Sigma, allow_singular=False)

    def sample(self, n):
        z = self.rng.uniform(size=n)
        n2 = (z < self.eps).sum()
        n1 = n - n2
        X = np.vstack([
            self._mv1.rvs(size=n1, random_state=self.rng),
            self._mv2.rvs(size=n2, random_state=self.rng)
        ])
        self.rng.shuffle(X)
        return X

    def logpdf(self, X):
        f1 = np.exp(self._mv1.logpdf(X))
        f2 = np.exp(self._mv2.logpdf(X))
        return np.log((1-self.eps)*f1 + self.eps*f2)

def create_synthetic_data(n=5000, d=2, kind="gaussian", rng=None):
    rng = np.random.default_rng() if rng is None else rng
    mu = np.zeros(d)
    # Make covariance moderately anisotropic for nicer plots
    A = rng.normal(size=(d, d))
    Sigma = A @ A.T / d
    # Inflate one axis a bit for visibility
    eigvals, eigvecs = la.eigh(Sigma)
    eigvals = np.maximum(eigvals, 0.2)  # avoid tiny
    Sigma = (eigvecs * eigvals) @ eigvecs.T

    if kind == "gaussian":
        dgp = GaussianDGP(mu=mu, Sigma=Sigma, rng=rng)
    elif kind == "t":
        dgp = MvtDGP(mu=mu, Sigma=Sigma, nu=5.0, rng=rng)
    elif kind == "contaminated":
        dgp = ContaminatedGaussianDGP(mu=mu, Sigma=Sigma, eps=0.05, scale_mult=3.0, rng=rng)
    else:
        raise ValueError("Unknown kind")

    X = dgp.sample(n)
    return X, dgp

@dataclass
class GaussianCenter:
    mu: np.ndarray
    Sigma: np.ndarray

    @property
    def Sig_inv(self):
        return la.inv(self.Sigma)

    def score_ellipsoid(self, X):
        D = X - self.mu
        return np.sqrt(np.einsum('...i,ij,...j->...', D, self.Sig_inv, D))

    def score_box(self, X, w):
        # w: feature scales (positive)
        D = (X - self.mu) / w
        return np.max(np.abs(D), axis=1)

    def score_directional(self, X, U, b):
        # U: (J,d) unit directions, b_j>0 scales
        D = X - self.mu
        proj = (U @ D.T).T  # (n,J)
        s = np.max(np.abs(proj) / b, axis=1)
        return s

def fit_pc_gaussian(X, ridge=1e-6):
    mu = X.mean(axis=0)
    Sigma = safe_cov(X, ridge=ridge)
    return GaussianCenter(mu=mu, Sigma=Sigma)

"""## Defining nonconformity scores"""

def make_score_function(score_type="ellipsoid",
                        center: GaussianCenter=None,
                        X_ref=None,
                        J=8):
    """
    Returns a callable s(X) producing the nonconformity score.
    - ellipsoid: s(x) = ||Sigma_c^{-1/2} (x - mu_c)||_2
    - box:       s(x) = max_i |(x_i - mu_i)/w_i|, with w_i = std_i from center or X_ref
    - directional polytope: s(x) = max_j |u_j^T (x - mu)| / b_j, with b_j^2 = u_j^T Sigma u_j
    """
    if center is None and X_ref is None:
        raise ValueError("Provide either a centre or reference data to define the score.")

    if score_type == "ellipsoid":
        if center is None:
            center = fit_pc_gaussian(X_ref)
        def s(X):
            return center.score_ellipsoid(X)
        meta = {"type": "ellipsoid", "mu": center.mu, "Sigma": center.Sigma}
        return s, meta

    elif score_type == "box":
        if center is None:
            mu = X_ref.mean(axis=0)
            w = X_ref.std(axis=0, ddof=1)
            w[w == 0.0] = 1.0
            center = GaussianCenter(mu=mu, Sigma=np.diag(w**2))
        else:
            # use marginal scales from Sigma
            w = np.sqrt(np.diag(center.Sigma))
            w[w == 0.0] = 1.0
        def s(X):
            return center.score_box(X, w=w)
        meta = {"type": "box", "mu": center.mu, "w": w}
        return s, meta

    elif score_type == "directional":
        if center is None:
            center = fit_pc_gaussian(X_ref)
        d = center.mu.size
        rng = np.random.default_rng(123)
        U = rng.normal(size=(J, d))
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        # scale b_j from center covariance
        b = np.sqrt(np.sum(U @ center.Sigma * U, axis=1))
        b[b == 0.0] = 1.0
        def s(X):
            return center.score_directional(X, U=U, b=b)
        meta = {"type": "directional", "mu": center.mu, "U": U, "b": b}
        return s, meta

    else:
        raise ValueError("Unknown score_type.")

def make_grid_from_baseline(t0, K=7, eta=0.05, mode="multiplicative"):
    """
    Pre-registered inflation grid on the score scale.
    - multiplicative: t_k = t0 * (1 + eta)^(k-1)
    - additive:       t_k = t0 + (k-1)*eta*t0
    """
    if t0 <= 0:
        raise ValueError("t0 must be positive.")
    if mode == "multiplicative":
        return t0 * (1.0 + eta) ** np.arange(K)
    elif mode == "additive":
        return t0 + (eta * t0) * np.arange(K)
    else:
        raise ValueError("Unknown mode.")

import numpy as np
import math
import matplotlib.pyplot as plt

def dkw_select_threshold(select_scores, gamma, delta):
    try:
        gamma = float(gamma); delta = float(delta)
    except Exception:
        raise ValueError("gamma and delta must be floats in (0,1).")
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must lie in (0,1).")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0,1).")

    scores = np.asarray(select_scores, dtype=float).ravel()
    m = scores.size
    if m < 1:
        raise ValueError("select_scores must be a non-empty 1-D array.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("select_scores contains NaN/Inf; please clean or filter before calling.")

    # ---- DKW radius
    r = float(np.sqrt(np.log(2.0 / delta) / (2.0 * m)))
    certifiable_max_coverage = 1.0 - r

    # Existence condition: we need 1 - gamma <= 1 - r  <=>  gamma >= r
    exists = gamma >= r
    if not exists:
        return {
            "t_hat": np.nan,
            "j_star": None,
            "r": r,
            "certifiable_max_coverage": certifiable_max_coverage,
            "exists": False,
            "Fm_at_t_hat": np.nan,
            "L_at_t_hat": np.nan,
        }

    # j* = ceil( m * (1 - gamma + r) ), clipped to [1, m] for numerical robustness
    j_star = int(np.clip(np.ceil(m * (1.0 - gamma + r)), 1, m))
    sorted_scores = np.sort(scores)
    t_hat = float(sorted_scores[j_star - 1])  # 1-based index -> 0-based array

    # ECDF at t_hat uses <= (right-continuous)
    count_le = int(np.searchsorted(sorted_scores, t_hat, side="right"))
    Fm_at_t_hat = count_le / m
    L_at_t_hat = max(Fm_at_t_hat - r, 0.0)

    return {
        "t_hat": t_hat,
        "j_star": j_star,
        "r": r,
        "certifiable_max_coverage": certifiable_max_coverage,
        "exists": True,
        "Fm_at_t_hat": Fm_at_t_hat,
        "L_at_t_hat": L_at_t_hat,
    }


def dkw_certificate(select_scores, gamma, delta):
    """
    One-step wrapper that prints a concise certificate and returns the result dict.
    """
    res = dkw_select_threshold(select_scores, gamma, delta)
    m = np.asarray(select_scores).size
    print("=== DKW bulk-mass certificate ===")
    print(f"m = {m}, gamma = {gamma:.6f}, delta = {delta:.6f}")
    print(f"r = sqrt(log(2/delta)/(2m)) = {res['r']:.6f}")
    print(f"certifiable_max_coverage = 1 - r = {res['certifiable_max_coverage']:.6f}")

    if not res["exists"]:
        shortfall = res["r"] - gamma
        print(f"\nNo solution exists because gamma < r by {shortfall:.6f}.")
        print("Action: increase m (more select data), use a larger delta, or relax gamma.")
        return res

    print("\nSolution exists (gamma >= r).")
    print(f"j_star = ceil( m * (1 - gamma + r) ) = {res['j_star']}")
    print(f"t_hat  = order-statistic value = {res['t_hat']:.6f}")
    print(f"F_m(t_hat) = {res['Fm_at_t_hat']:.6f}")
    print(f"L^{chr(123)}DKW{chr(125)}(t_hat) = max(F_m - r, 0) = {res['L_at_t_hat']:.6f}")
    print(f"Certificate: P^*(Xi_0(t_hat)) >= {1.0 - gamma:.6f} with prob >= {1.0 - delta:.6f}.")
    return res

def dkw_get_testing_scores(result):
    """
    Return 1-D array of scores for DKW selection = the select split scores.
    """
    if not isinstance(result, dict):
        raise TypeError("result must be a dict as returned by run_bulk_certificate(...).")
    if "s_select" not in result:
        raise KeyError("result must contain 's_select'.")
    scores = np.asarray(result["s_select"], dtype=float).ravel()
    if scores.size == 0:
        raise ValueError("Empty select scores (s_select is empty).")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite values present in s_select.")
    return scores

"""## Helper functions for plotting"""

def ellipse_points(mu, Sigma, t, n_pts=400):
    """
    Boundary of { (x-mu)^T Sigma^{-1} (x-mu) <= t^2 } in 2-D.
    Returns array (n_pts, 2).
    """
    mu = np.asarray(mu)
    vals, vecs = la.eigh(Sigma)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    angles = np.linspace(0, 2*np.pi, n_pts)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (n_pts, 2)
    # scale: t * sqrt(eigvals), rotate by vecs, shift by mu
    pts = (circle * (t * np.sqrt(vals)) ) @ vecs.T + mu
    return pts

def hdr_threshold_from_true(dgp, gamma, M=200_000, rng=None):
    """
    Estimate the HDR density threshold c such that P(f(X) >= c) = 1 - gamma,
    by Monte Carlo from the true DGP.
    """
    rng = np.random.default_rng() if rng is None else rng
    Xs = dgp.sample(M)
    dens = dgp.pdf(Xs)  # vector (M,)
    # c is the gamma-quantile of f(X): P(f(X) >= c) = 1 - gamma
    c = np.quantile(dens, gamma)
    return c

def plot_experiment_2d(
    X, dgp, result, gamma, delta,
    audit_names, t_hats, shape,
    title_suffix="",
    J=None, rasterize_scatter=True, add_kde=True
):
    assert isinstance(audit_names, (list, tuple)) and isinstance(t_hats, (list, tuple)), \
        "`audit_names` and `t_hats` must be lists/tuples."
    assert len(audit_names) == len(t_hats) and len(audit_names) >= 1, \
        "Lengths of `audit_names` and `t_hats` must match and be >= 1."

    shape = str(shape).lower()
    meta = result["score_meta"]
    center = result["center"]
    ids = result["ids"]

    assert "train" in ids and "select" in ids, "result['ids'] must contain 'train' and 'select'."
    assert meta["type"] == shape, f"Shape mismatch: meta['type']={meta['type']} vs requested shape={shape}"

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    pe_glow = [pe.Stroke(linewidth=2.5, foreground="white", alpha=0.9), pe.Normal()]
    pe_dash = [pe.Stroke(linewidth=2.2, foreground="white", alpha=0.9), pe.Normal()]

    hdr_color = "#222222"   # charcoal for true HDR
    dkw_color = "#C76E00"   # purple for DKW set
    pe_dkw    = [pe.Stroke(linewidth=2.2, foreground="white", alpha=0.9), pe.Normal()]

    def _style_for(name: str):
        key = str(name).strip().lower()
        if key == "dkw":
            return dict(color=dkw_color, linestyle=(0,(2,2)), linewidth=1.8,
                        label=r"$\widehat{\Xi}_0^{\mathrm{DKW}}$", path_effects=pe_dkw)
        # Fallback (keeps aesthetics close to your dashed style)
        return dict(color="#4C72B0", linestyle=(0,(6,4)), linewidth=1.6,
                    label=rf"$\widehat{{\Xi}}_0^{{\mathrm{{{name}}}}}$", path_effects=pe_dash)

    scat_kw = dict(s=12, alpha=0.25, linewidths=0, rasterized=rasterize_scatter)
    ax.scatter(X[ids["train"], 0], X[ids["train"], 1], c = "blue", label="train", **scat_kw)
    ax.scatter(X[ids["select"], 0], X[ids["select"], 1], c = "green", label="select", **scat_kw)

    # --- true HDR boundary
    if dgp.d == 2 and dgp.name == "gaussian":
        mu_true = dgp.params["mu"]
        Sigma_true = dgp.params["Sigma"]
        r_true = np.sqrt(chi2.ppf(1 - gamma, df=2))
        hdr_pts = ellipse_points(mu_true, Sigma_true, r_true)
        ax.plot(hdr_pts[:, 0], hdr_pts[:, 1], color=hdr_color, linewidth=1.6,
                label="true HDR", path_effects=pe_glow)

    elif dgp.d == 2 and dgp.name == "mvt":
        mu_true = dgp.params["mu"]
        Sigma_true = dgp.params["Sigma"]
        nu = float(dgp.params["nu"])
        r_true = np.sqrt(dgp.d * fdist.ppf(1 - gamma, dgp.d, nu))
        hdr_pts = ellipse_points(mu_true, Sigma_true, r_true)
        ax.plot(hdr_pts[:, 0], hdr_pts[:, 1], color=hdr_color, linewidth=1.6,
                label="true HDR", path_effects=pe_glow)

    elif dgp.d == 2 and dgp.name == "contaminated-gaussian":
        mu_true = dgp.params["mu"]
        Sigma_true = dgp.params["Sigma"]
        eps = float(dgp.params["eps"])
        c = float(dgp.params["scale_mult"])

        def mix_cdf(q):
            return (1 - eps) * chi2.cdf(q, dgp.d) + eps * chi2.cdf(q / (c ** 2), dgp.d)

        q_lo = 0.0
        q_hi = chi2.ppf(1 - gamma, dgp.d) * max(1.0, c ** 2) * 10.0
        while mix_cdf(q_hi) < (1 - gamma):
            q_hi *= 2.0
        q_star = brentq(lambda q: mix_cdf(q) - (1 - gamma), q_lo, q_hi)

        r_true = np.sqrt(q_star)
        hdr_pts = ellipse_points(mu_true, Sigma_true, r_true)
        ax.plot(hdr_pts[:, 0], hdr_pts[:, 1], linewidth=1.6, color=hdr_color,
                label="true HDR", path_effects=pe_glow)

    else:
        c = hdr_threshold_from_true(dgp, gamma, M=200_000)
        mins = np.percentile(X, 1, axis=0)
        maxs = np.percentile(X, 99, axis=0)
        pad = 0.2 * (maxs - mins)
        xs = np.linspace(mins[0] - pad[0], maxs[0] + pad[0], 300)
        ys = np.linspace(mins[1] - pad[1], maxs[1] + pad[1], 300)
        XX, YY = np.meshgrid(xs, ys)
        grid = np.c_[XX.ravel(), YY.ravel()]
        ZZ = dgp.pdf(grid).reshape(XX.shape)
        ax.contour(XX, YY, ZZ, levels=[c], linewidths=1.6, colors=[hdr_color])
        ax.plot([], [], color=hdr_color, linewidth=1.6,
                label="true HDR", path_effects=pe_glow)

    if shape == "ellipsoid":
        assert center is not None, "Center is required for ellipsoids."
        mu_c, Sigma_c = center.mu, center.Sigma
        for name, t_hat in zip(audit_names, t_hats):
            if not np.isfinite(t_hat):  # skip missing/invalid
                continue
            st = _style_for(name)
            pts = ellipse_points(mu_c, Sigma_c, float(t_hat))
            ax.plot(pts[:,0], pts[:,1], **st)

    elif shape == "box":
        mu = np.asarray(meta["mu"]).reshape(-1)
        w  = np.asarray(meta["w"]).reshape(-1)
        for name, t_hat in zip(audit_names, t_hats):
            if not np.isfinite(t_hat):
                continue
            st = _style_for(name)
            half = w * float(t_hat)
            rect = np.array([
                mu + [-half[0], -half[1]],
                mu + [ half[0], -half[1]],
                mu + [ half[0],  half[1]],
                mu + [-half[0],  half[1]],
                mu + [-half[0], -half[1]],
            ])
            ax.plot(rect[:,0], rect[:,1], **st)

    elif shape == "directional":
        mu = np.asarray(meta["mu"]).reshape(-1)
        U  = np.asarray(meta["U"])
        b  = np.asarray(meta["b"]).reshape(-1)
        assert U.shape[1] == 2, "Directional polytope plotting requires d=2."
        for name, t_hat in zip(audit_names, t_hats):
            if not np.isfinite(t_hat):
                continue
            st = _style_for(name)
            t_hat = float(t_hat)

            A_list, c_list = [], []
            for uj, bj in zip(U, b):
                A_list.append( uj); c_list.append(-(uj @ mu) - t_hat * bj)
                A_list.append(-uj); c_list.append( (uj @ mu) - t_hat * bj)

            A = np.vstack(A_list); c = np.asarray(c_list).reshape(-1, 1)
            halfspaces = np.hstack([A, c])
            x0 = mu.copy()
            try:
                hs = HalfspaceIntersection(halfspaces, x0)
                V = hs.intersections
                if V.shape[0] >= 3:
                    hull = ConvexHull(V)
                    poly = V[hull.vertices]
                    poly = np.vstack([poly, poly[0]])
                    ax.plot(poly[:, 0], poly[:, 1], **st)
                else:
                    ax.plot([], [], **st)
            except Exception:
                ax.plot([], [], **st)
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    ax.set_title(title_suffix)
    leg = ax.legend(loc="upper left", frameon=True, fancybox=True, borderpad=0.7)
    for lh in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
        try: lh.set_alpha(0.9)
        except Exception: pass

    ax.set_aspect('equal')
    plt.grid(True)

    fig.tight_layout()
    plt.show()

def dkw_plot(select_scores, gamma, delta, result=None, title=None):
    """
    Diagnostic plot: ECDF F_m(t), the lower envelope L^{DKW}(t), the line 1 - gamma,
    and a vertical line at t_hat if it exists.

    Renders inline as a PDF (no filesystem writes).
    """
    scores = np.asarray(select_scores, dtype=float).ravel()
    if scores.size == 0:
        raise ValueError("empty select_scores.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("select_scores contains NaN/Inf.")

    m = scores.size
    r = float(np.sqrt(np.log(2.0 / delta) / (2.0 * m)))
    x = np.sort(scores)
    y = np.arange(1, m + 1, dtype=float) / m
    y_lower = np.clip(y - r, 0.0, None)

    if result is None:
        result = dkw_select_threshold(scores, gamma, delta)

    fig = plt.figure()
    plt.step(x, y, where="post", label=r"$F_m(t)$")
    plt.step(x, y_lower, where="post", label=r"$L^{\mathrm{DKW}}(t)$")
    plt.axhline(1.0 - gamma, linestyle="--", label=r"$1-\gamma$")
    if result.get("exists", False):
        plt.axvline(result["t_hat"], linestyle=":", label=r"$\hat{t}_{\mathrm{DKW}}$")
    plt.xlabel("score threshold $t$")
    plt.ylabel("CDF value")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    fig.tight_layout()
    plt.show()


def print_t_hat_summary(audit_names, t_hats, precision: int = 6):
    import numpy as np

    if len(audit_names) != len(t_hats):
        raise ValueError("audit_names and t_hats must have the same length.")

    name_width = max(len(str(n)) for n in audit_names) if audit_names else 0
    print("\n=== Bulk-set thresholds (t_hat) ===")
    for name, t in zip(audit_names, t_hats):
        if t is None:
            val = "n/a"
        else:
            try:
                t_float = float(t)
                val = f"{t_float:.{precision}f}" if np.isfinite(t_float) else "n/a"
            except Exception:
                val = "n/a"
        print(f"{str(name):>{name_width}} : t_hat = {val}")

"""# Synthetic Experiments

## Gaussian DGP
"""

rng = np.random.default_rng(2025)
kind = "gaussian" # kind can be "gaussian" or "t" or "contaminated"
n=6000
X, dgp = create_synthetic_data(n=n, d=2, kind = kind, rng=rng)

gamma = 0.05      # target bulk mass 1 - gamma
delta = 0.05      # confidence level for the certificate
score_type = "ellipsoid"  # This can be "ellipsoid" or "box" or "directional"
use_pc = True

result = run_bulk_certificate(
    X, dgp, gamma=gamma, delta=delta, score_type=score_type,
    split=0.5, use_pc=use_pc, rng=rng
)

testing_scores = dkw_get_testing_scores(result)
dkw_result = dkw_certificate(testing_scores, gamma=gamma, delta=delta)

audit_names = ["DKW"]
t_hats = [float(dkw_result['t_hat'])]
print_t_hat_summary(audit_names, t_hats)

dkw_plot(testing_scores, gamma, delta, result=dkw_result, title = "Gaussian")

# Shape comes from your geometry
shape = result['score_meta']['type']  # 'ellipsoid' | 'box' | 'directional'

# Plot (visuals unchanged)
plot_experiment_2d(
    X, dgp, result, gamma, delta,
    audit_names=audit_names, t_hats=t_hats, shape=shape,
    title_suffix="Gaussian"
)

rng = np.random.default_rng(2025)
kind = "gaussian" # kind can be "gaussian" or "t" or "contaminated"
n=6000
X, dgp = create_synthetic_data(n=n, d=2, kind = kind, rng=rng)

gamma = 0.05      # target bulk mass 1 - gamma
delta = 0.05      # confidence level for the certificate
score_type = "box"  # This can be "ellipsoid" or "box" or "directional"
use_pc = True

result = run_bulk_certificate(
    X, dgp, gamma=gamma, delta=delta, score_type=score_type,
    split=0.5, use_pc=use_pc, rng=rng
)

testing_scores = dkw_get_testing_scores(result)
dkw_result = dkw_certificate(testing_scores, gamma=gamma, delta=delta)

audit_names = ["DKW"]
t_hats = [float(dkw_result['t_hat'])]
print_t_hat_summary(audit_names, t_hats)

# Shape comes from your geometry
shape = result['score_meta']['type']  # 'ellipsoid' | 'box' | 'directional'

# Plot (visuals unchanged)
plot_experiment_2d(
    X, dgp, result, gamma, delta,
    audit_names=audit_names, t_hats=t_hats, shape=shape,
    title_suffix=""
)

"""## Contaminated Gaussian DGP"""

rng = np.random.default_rng(2025)
kind = "contaminated" # kind can be "gaussian" or "t" or "contaminated"
n=6000
X, dgp = create_synthetic_data(n=n, d=2, kind = kind, rng=rng)

gamma = 0.05      # target bulk mass 1 - gamma
delta = 0.05      # confidence level for the certificate
score_type = "ellipsoid"  # This can be "ellipsoid" or "box" or "directional"
use_pc = True

result = run_bulk_certificate(
    X, dgp, gamma=gamma, delta=delta, score_type=score_type,
    split=0.5, use_pc=use_pc, rng=rng
)

testing_scores = dkw_get_testing_scores(result)
dkw_result = dkw_certificate(testing_scores, gamma=gamma, delta=delta)

audit_names = ["DKW"]
t_hats = [float(dkw_result['t_hat'])]
print_t_hat_summary(audit_names, t_hats)

dkw_plot(testing_scores, gamma, delta, result=dkw_result, title = "Contaminated Gaussian")

# Shape comes from your geometry
shape = result['score_meta']['type']  # 'ellipsoid' | 'box' | 'directional'

# Plot (visuals unchanged)
plot_experiment_2d(
    X, dgp, result, gamma, delta,
    audit_names=audit_names, t_hats=t_hats, shape=shape,
    title_suffix="Contaminated Gaussian"
)

rng = np.random.default_rng(2025)
kind = "contaminated" # kind can be "gaussian" or "t" or "contaminated"
n=6000
X, dgp = create_synthetic_data(n=n, d=2, kind = kind, rng=rng)

gamma = 0.05      # target bulk mass 1 - gamma
delta = 0.05      # confidence level for the certificate
score_type = "box"  # This can be "ellipsoid" or "box" or "directional"
use_pc = True

result = run_bulk_certificate(
    X, dgp, gamma=gamma, delta=delta, score_type=score_type,
    split=0.5, use_pc=use_pc, rng=rng
)

testing_scores = dkw_get_testing_scores(result)
dkw_result = dkw_certificate(testing_scores, gamma=gamma, delta=delta)

audit_names = ["DKW"]
t_hats = [float(dkw_result['t_hat'])]
print_t_hat_summary(audit_names, t_hats)

# Shape comes from your geometry
shape = result['score_meta']['type']  # 'ellipsoid' | 'box' | 'directional'

# Plot (visuals unchanged)
plot_experiment_2d(
    X, dgp, result, gamma, delta,
    audit_names=audit_names, t_hats=t_hats, shape=shape,
    title_suffix=""
)

"""## Multivariate-t DGP"""

rng = np.random.default_rng(2025)
kind = "t" # kind can be "gaussian" or "t" or "contaminated"
n=6000
X, dgp = create_synthetic_data(n=n, d=2, kind = kind, rng=rng)

gamma = 0.05      # target bulk mass 1 - gamma
delta = 0.05      # confidence level for the certificate
score_type = "ellipsoid"  # This can be "ellipsoid" or "box" or "directional"
use_pc = True

result = run_bulk_certificate(
    X, dgp, gamma=gamma, delta=delta, score_type=score_type,
    split=0.5, use_pc=use_pc, rng=rng
)

testing_scores = dkw_get_testing_scores(result)
dkw_result = dkw_certificate(testing_scores, gamma=gamma, delta=delta)

audit_names = ["DKW"]
t_hats = [float(dkw_result['t_hat'])]
print_t_hat_summary(audit_names, t_hats)

dkw_plot(testing_scores, gamma, delta, result=dkw_result, title = "Student-t")

# Shape comes from your geometry
shape = result['score_meta']['type']  # 'ellipsoid' | 'box' | 'directional'

# Plot (visuals unchanged)
plot_experiment_2d(
    X, dgp, result, gamma, delta,
    audit_names=audit_names, t_hats=t_hats, shape=shape,
    title_suffix="Student-t"
)

rng = np.random.default_rng(2025)
kind = "t" # kind can be "gaussian" or "t" or "contaminated"
n=6000
X, dgp = create_synthetic_data(n=n, d=2, kind = kind, rng=rng)

gamma = 0.05      # target bulk mass 1 - gamma
delta = 0.05      # confidence level for the certificate
score_type = "box"  # This can be "ellipsoid" or "box" or "directional"
use_pc = True

result = run_bulk_certificate(
    X, dgp, gamma=gamma, delta=delta, score_type=score_type,
    split=0.5, use_pc=use_pc, rng=rng
)

testing_scores = dkw_get_testing_scores(result)
dkw_result = dkw_certificate(testing_scores, gamma=gamma, delta=delta)

audit_names = ["DKW"]
t_hats = [float(dkw_result['t_hat'])]
print_t_hat_summary(audit_names, t_hats)

# Shape comes from your geometry
shape = result['score_meta']['type']  # 'ellipsoid' | 'box' | 'directional'

# Plot (visuals unchanged)
plot_experiment_2d(
    X, dgp, result, gamma, delta,
    audit_names=audit_names, t_hats=t_hats, shape=shape,
    title_suffix=""
)

