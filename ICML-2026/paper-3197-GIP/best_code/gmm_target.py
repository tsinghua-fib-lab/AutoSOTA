
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from numpy.linalg import slogdet

import numpy as np
from numpy.linalg import slogdet

def log_gaussian(x, m, S, eps=1e-12, quad_cap=1e300):
    """
    Robust log N(x | m, S).
    - Symmetrize S.
    - Use eig basis for quadratic (no explicit inverse).
    - Use slogdet when SPD; fall back to eig logdet otherwise.
    """
    d = len(x)
    Ssym = 0.5 * (S + S.T)

    # Log-det: try SPD path; else from clipped eigs
    sign, logdet = slogdet(Ssym)
    w, U = np.linalg.eigh(Ssym)
    w_clip = np.clip(w, eps, np.inf)
    if sign <= 0:
        logdet = np.log(w_clip).sum()

    # Quadratic form: ||U^T (x-m)||^2 weighted by 1/w
    dx = x - m
    y = U.T @ dx
    quad = float(np.minimum(np.sum((y * y) / w_clip), quad_cap))

    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


@dataclass
class GMM2D:
    w1: float = 0.5
    m1: tuple = (-2.0, 0.0)
    S1: tuple = (1.0, 0.0, 0.0, 0.5)  # as (a,b,b,c)
    m2: tuple = (2.0, 0.0)
    S2: tuple = (1.0, 0.0, 0.0, 0.5)

    def _S(self, t):
        a,b,c,d = t  # allow (a,b,b,c) or full tuple; interpret symmetric 2x2
        return np.array([[a, b],[b, d]], dtype=float)

    @property
    def components(self):
        m1 = np.array(self.m1, dtype=float)
        m2 = np.array(self.m2, dtype=float)
        S1 = self._S(self.S1)
        S2 = self._S(self.S2)
        w = float(self.w1)
        return w, m1, S1, (1.0-w), m2, S2

    def logp(self, x: np.ndarray) -> float:
        w, m1, S1, w2, m2, S2 = self.components
        lg1 = np.log(w)  + log_gaussian(x, m1, S1)
        lg2 = np.log(w2) + log_gaussian(x, m2, S2)
        m = max(lg1, lg2)
        return m + np.log(np.exp(lg1 - m) + np.exp(lg2 - m))

    def V(self, x: np.ndarray) -> float:
        return -self.logp(x)

    def gradV(self, x: np.ndarray) -> np.ndarray:
        w, m1, S1, w2, m2, S2 = self.components
        # responsibilities
        n1 = w  * np.exp(log_gaussian(x, m1, S1))
        n2 = w2 * np.exp(log_gaussian(x, m2, S2))
        Z = n1 + n2 + 1e-300
        r1 = n1 / Z
        r2 = n2 / Z
        S1i = np.linalg.inv(0.5*(S1+S1.T))
        S2i = np.linalg.inv(0.5*(S2+S2.T))
        g1 = S1i @ (m1 - x)
        g2 = S2i @ (m2 - x)
        grad_logp = r1*g1 + r2*g2
        return -grad_logp

    def hessV(self, x: np.ndarray) -> np.ndarray:
        w, m1, S1, w2, m2, S2 = self.components
        S1i = np.linalg.inv(0.5*(S1+S1.T))
        S2i = np.linalg.inv(0.5*(S2+S2.T))
        n1 = w  * np.exp(log_gaussian(x, m1, S1))
        n2 = w2 * np.exp(log_gaussian(x, m2, S2))
        Z = n1 + n2 + 1e-300
        r1 = n1 / Z
        r2 = n2 / Z
        g1 = S1i @ (m1 - x)
        g2 = S2i @ (m2 - x)
        gbar = r1*g1 + r2*g2
        # Hessian of -log p = - Hessian(log p)
        H_logp = r1*(-S1i) + r2*(-S2i) + r1*np.outer(g1 - gbar, g1) + r2*np.outer(g2 - gbar, g2)
        return -H_logp

    def KL_q_to_p(self, m: np.ndarray, S: np.ndarray, expect_scalar):
        # KL(q||p) = E_q[log q - log p]
        d = len(m)
        Ssym = 0.5*(S+S.T)
        sign, logdet = np.linalg.slogdet(Ssym)
        if sign <= 0:
            w, U = np.linalg.eigh(Ssym)
            w = np.clip(w, 1e-12, None)
            logdet = np.log(w).sum()
        logZq = -0.5*(d*np.log(2*np.pi) + logdet)  # E_q[ -0.5 log det term ] same as const
        # E_q[ (x-m)^T S^{-1} (x-m) ] = d
        Eq_logq = logZq - 0.5*d
        Eq_logp = expect_scalar(self.logp, m, S)
        return float(Eq_logq - Eq_logp)


@dataclass
class GMM4:
    # Three explicit weights; the 4th is 1 - (w1+w2+w3).
    # We'll renormalize to be safe if the sum drifts slightly.
    w1: float = 0.25
    w2: float = 0.25
    w3: float = 0.25

    # Component means (2D) and covariances (given as (a,b,b,d))
    m1: tuple = (-2.0, -2.0)
    S1: tuple = (1.0, 0.0, 0.0, 1.0)

    m2: tuple = ( 2.0, -2.0)
    S2: tuple = (1.0, 0.0, 0.0, 1.0)

    m3: tuple = (-2.0,  2.0)
    S3: tuple = (1.0, 0.0, 0.0, 1.0)

    m4: tuple = ( 2.0,  2.0)
    S4: tuple = (1.0, 0.0, 0.0, 1.0)

    def _S(self, t):
        a, b, c, d = t  # allow (a,b,b,d) or full tuple; interpret symmetric 2x2
        return np.array([[a, b], [b, d]], dtype=float)

    def _weights(self):
        w1 = float(self.w1)
        w2 = float(self.w2)
        w3 = float(self.w3)
        w4 = 1.0 - (w1 + w2 + w3)
        w = np.array([w1, w2, w3, w4], dtype=float)
        # Robust normalization & clipping to avoid negatives / zeros
        w = np.clip(w, 1e-12, None)
        w = w / w.sum()
        return w

    @property
    def components(self):
        ms = [np.array(self.m1, dtype=float),
              np.array(self.m2, dtype=float),
              np.array(self.m3, dtype=float),
              np.array(self.m4, dtype=float)]
        Ss = [self._S(self.S1),
              self._S(self.S2),
              self._S(self.S3),
              self._S(self.S4)]
        ws = self._weights()
        return ws, ms, Ss

    def logp(self, x: np.ndarray) -> float:
        ws, ms, Ss = self.components
        # log-sum-exp over 4 components
        lgs = np.array([np.log(w) + log_gaussian(x, m, S) for w, m, S in zip(ws, ms, Ss)])
        m = lgs.max()
        return float(m + np.log(np.exp(lgs - m).sum()))

    def V(self, x: np.ndarray) -> float:
        return -self.logp(x)

    def gradV(self, x: np.ndarray) -> np.ndarray:
        ws, ms, Ss = self.components
        # responsibilities
        ns = np.array([w * np.exp(log_gaussian(x, m, S)) for w, m, S in zip(ws, ms, Ss)])
        Z = ns.sum() + 1e-300
        rs = ns / Z

        S_invs = [np.linalg.inv(0.5 * (S + S.T)) for S in Ss]
        gs = [S_inv @ (m - x) for S_inv, m in zip(S_invs, ms)]  # g_i = S_i^{-1}(m_i - x)

        grad_logp = sum(r * g for r, g in zip(rs, gs))
        return -grad_logp

    def hessV(self, x: np.ndarray) -> np.ndarray:
        ws, ms, Ss = self.components
        ns = np.array([w * np.exp(log_gaussian(x, m, S)) for w, m, S in zip(ws, ms, Ss)])
        Z = ns.sum() + 1e-300
        rs = ns / Z

        S_invs = [np.linalg.inv(0.5 * (S + S.T)) for S in Ss]
        gs = [S_inv @ (m - x) for S_inv, m in zip(S_invs, ms)]
        gbar = sum(r * g for r, g in zip(rs, gs))

        # Hessian of -log p = - Hessian(log p)
        # Follow your 2-component pattern generalized to K=4:
        H_logp = sum(r * (-S_inv) for r, S_inv in zip(rs, S_invs))
        H_logp += sum(r * np.outer(g - gbar, g) for r, g in zip(rs, gs))
        return -H_logp

    def KL_q_to_p(self, m: np.ndarray, S: np.ndarray, expect_scalar):
        # KL(q||p) = E_q[log q - log p]
        d = len(m)
        Ssym = 0.5 * (S + S.T)
        sign, logdet = np.linalg.slogdet(Ssym)
        if sign <= 0:
            w, U = np.linalg.eigh(Ssym)
            w = np.clip(w, 1e-12, None)
            logdet = np.log(w).sum()
        logZq = -0.5 * (d * np.log(2 * np.pi) + logdet)
        Eq_logq = logZq - 0.5 * d
        Eq_logp = expect_scalar(self.logp, m, S)
        return float(Eq_logq - Eq_logp)
    


# ---------------------------
# Robust SPD helpers
# ---------------------------
def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def _spd_inv_and_logdet(S: np.ndarray, eps: float = 1e-12):
    """Return (S^{-1}, logdet S) for symmetric S, via eig with floor for stability."""
    S = _sym(S)
    w, U = np.linalg.eigh(S)
    w = np.clip(w, eps, None)
    Sinv = U @ np.diag(1.0 / w) @ U.T
    logdet = np.log(w).sum()
    return Sinv, logdet

def _rand_spd(d: int, rng: np.random.Generator, lam_min: float = 0.3, lam_max: float = 2.0):
    """Random SPD with spectrum in [lam_min, lam_max]."""
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    # sample eigenvalues log-uniform within [lam_min, lam_max] for better conditioning
    r = rng.uniform(0.0, 1.0, size=d)
    evals = lam_min * (lam_max/lam_min) ** r
    return Q @ np.diag(evals) @ Q.T

# ---------------------------
# N-D GMM
# ---------------------------
@dataclass
class GMMNd:
    """
    General d-dimensional, K-component Gaussian mixture.
    - w : shape (K,), mixture weights (sum to 1)
    - m : shape (K, d), component means
    - S : shape (K, d, d), SPD covariance matrices
    """
    w: np.ndarray
    m: np.ndarray
    S: np.ndarray

    @property
    def K(self): return int(self.w.shape[0])
    @property
    def d(self): return int(self.m.shape[1])

    # ---- log p(x) via log-sum-exp over components
    def logp(self, x: np.ndarray) -> float:
        x = np.asarray(x, float)
        vals = []
        for k in range(self.K):
            wk = float(self.w[k])
            mk = self.m[k]
            Sk = self.S[k]
            Sinv, logdet = _spd_inv_and_logdet(Sk)
            dx = x - mk
            lg = np.log(wk) - 0.5 * (self.d * np.log(2*np.pi) + logdet + dx @ Sinv @ dx)
            vals.append(lg)
        vals = np.array(vals)
        m = vals.max()
        return float(m + np.log(np.exp(vals - m).sum()))

    def V(self, x: np.ndarray) -> float:
        return -self.logp(x)

    # ---- grad of -log p(x): sum r_k S_k^{-1}(x - m_k); but we return -grad log p
    def gradV(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        # responsibilities r_k(x)
        lgs = []
        Sinvs = []
        for k in range(self.K):
            Sinv, logdet = _spd_inv_and_logdet(self.S[k])
            Sinvs.append(Sinv)
            dx = x - self.m[k]
            lg = np.log(self.w[k]) - 0.5 * (self.d*np.log(2*np.pi) + logdet + dx @ Sinv @ dx)
            lgs.append(lg)
        lgs = np.array(lgs)
        m = lgs.max()
        r = np.exp(lgs - (m + np.log(np.exp(lgs - m).sum())))  # softmax
        # grad log p(x) = sum_k r_k * S_k^{-1}(m_k - x)
        g = np.zeros(self.d)
        for k in range(self.K):
            g += r[k] * (Sinvs[k] @ (self.m[k] - x))
        return -g  # grad of -log p

    # ---- Hessian of -log p(x) (optional, generalized your 2D formula)
    def hessV(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        lgs = []
        Sinvs = []
        gs = []
        for k in range(self.K):
            Sinv, logdet = _spd_inv_and_logdet(self.S[k])
            Sinvs.append(Sinv)
            dx = x - self.m[k]
            lgs.append(np.log(self.w[k]) - 0.5 * (self.d*np.log(2*np.pi) + logdet + dx @ Sinv @ dx))
            gs.append(Sinv @ (self.m[k] - x))  # g_k = S_k^{-1}(m_k - x)
        lgs = np.array(lgs)
        m = lgs.max()
        r = np.exp(lgs - (m + np.log(np.exp(lgs - m).sum())))
        gbar = sum(r[k] * gs[k] for k in range(self.K))

        # Hessian log p = sum r_k*(-S_k^{-1}) + sum r_k (g_k - gbar)(g_k - gbar)^T
        H_logp = np.zeros((self.d, self.d))
        for k in range(self.K):
            H_logp += r[k] * (-Sinvs[k])
        for k in range(self.K):
            diff = (gs[k] - gbar).reshape(-1, 1)
            H_logp += r[k] * (diff @ diff.T)
        return -H_logp  # Hessian of -log p

    # ---- KL(q || p) with q = N(m,S) via E_q[log q - log p]
    def KL_q_to_p(self, m: np.ndarray, S: np.ndarray, expect_scalar):
        d = len(m)
        Ssym = _sym(S)
        # log |S|
        _, logdet = _spd_inv_and_logdet(Ssym)
        logZq = -0.5 * (d * np.log(2*np.pi) + logdet)
        Eq_logq = logZq - 0.5 * d
        Eq_logp = expect_scalar(self.logp, m, S)
        return float(Eq_logq - Eq_logp)

# ---------------------------
# Random generator
# ---------------------------
def random_gmm_nd(
    d: int,
    K: int,
    rng: np.random.Generator | None = None,
    mean_radius: float = 3.0,
    lam_min: float = 0.3,
    lam_max: float = 2.0,
) -> GMMNd:
    """
    Draw a random K-component, d-dimensional GMM that scales well with d.
    - Means lie roughly on a sphere of radius ~ mean_radius * sqrt(d).
    - Cov eigenvalues in [lam_min, lam_max] for reasonable conditioning.
    """
    rng = np.random.default_rng() if rng is None else rng

    # weights from Dirichlet
    w = rng.dirichlet(alpha=np.ones(K))
    # means ~ N(0, I), rescaled to radius ~ mean_radius * sqrt(d)
    m = rng.normal(size=(K, d))
    m_norms = np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    m = (m / m_norms) * (mean_radius * np.sqrt(d))

    # covariances
    S = np.stack([_rand_spd(d, rng, lam_min=lam_min, lam_max=lam_max) for _ in range(K)], axis=0)

    return GMMNd(w=w.astype(float), m=m.astype(float), S=S.astype(float))

