"""
iamp_sk_solver.py
=================
Full implementation of the Incremental Approximate Message Passing (IAMP)
algorithm from:
  "Optimization of the Sherrington-Kirkpatrick
  Hamiltonian." arXiv:1812.10897v2 (paper suggested by the reviewer iKh6)

Implements Algorithms 1 & 2 (Appendix B) with the Parisi PDE solved via
a stable Crank-Nicolson / implicit-diffusion scheme.

SK - Sherrington-Kirkpatrick model
"""

import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.filterwarnings("ignore")


# ======================================================================
# 1. Edwards-Anderson order parameter  q*(beta)
# ======================================================================

def estimate_q_star(beta, n_mc=200_000, seed=0):
    """
    Solve the Parisi/TAP self-consistency equation for the SK model:
        q* = E[tanh^2(beta * sqrt(q*) * Z)],   Z ~ N(0,1)

    This is the Edwards-Anderson order parameter. For beta <= 1 the
    paramagnetic solution q*=0 holds; for beta>1 there is a unique
    spin-glass fixed point in (0,1).
    """
    if beta <= 1.0:
        return 1e-3

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_mc)

    def resid(q):
        return float(np.mean(np.tanh(beta * np.sqrt(max(q, 1e-9)) * Z)**2)) - q

    lo, hi = 1e-6, 1.0 - 1e-6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ======================================================================
# 2. Parisi PDE  (Crank-Nicolson + explicit nonlinear term)
# ======================================================================

class ParisiPDE:
    """
    Solve the Parisi PDE backward in time:

        d_t Phi + (beta^2/2) d_xx Phi + (beta^2/2) mu(t) (d_x Phi)^2 = 0
        Phi(1, x) = log 2 cosh(x)

    using Crank-Nicolson for linear diffusion (unconditionally stable)
    + explicit treatment of the nonlinear (d_x Phi)^2 term.

    Parisi measure mu(t) = min(t/q*, 1)  (uniform-RSB / continuous-RSB ansatz).
    """

    def __init__(self, beta, q_star, nx=151, nt=6001, x_max=6.0):
        self.beta   = beta
        self.q_star = q_star
        self.nx     = nx
        self.nt     = nt
        self.x_max  = x_max
        self._solve()

    def _mu(self, t):
        if self.q_star <= 0:
            return 1.0
        return min(float(t) / self.q_star, 1.0)

    def _solve(self):
        beta   = self.beta
        nx, nt = self.nx, self.nt
        xs = np.linspace(-self.x_max, self.x_max, nx)
        ts = np.linspace(0.0, 1.0, nt)
        dx = xs[1] - xs[0]
        dt = ts[1] - ts[0]

        # Crank-Nicolson parameter r = (beta^2/2) * (dt/2) / dx^2
        r = (beta**2 / 2.0) * (dt / 2.0) / dx**2

        # Build banded LHS matrix  (I + r D^2)  with Neumann BCs
        # using scipy solve_banded format: ab[0]=superdiag, ab[1]=diag, ab[2]=subdiag
        ab_lhs = np.zeros((3, nx))
        ab_lhs[1, :] = 1.0 + 2.0 * r
        ab_lhs[1, 0]  -= r      # Neumann: d_x Phi = 0 at boundaries
        ab_lhs[1, -1] -= r
        ab_lhs[0, 1:]  = -r     # superdiagonal
        ab_lhs[2, :-1] = -r     # subdiagonal

        # Build dense RHS operator  (I - r D^2)  with Neumann BCs
        diag_rhs = np.full(nx, 1.0 - 2.0 * r)
        diag_rhs[0]  += r
        diag_rhs[-1] += r
        off_rhs = np.full(nx - 1, r)

        def apply_rhs(P):
            out = diag_rhs * P
            out[:-1] += off_rhs * P[1:]
            out[1:]  += off_rhs * P[:-1]
            return out

        def deriv1(P):
            d = np.empty_like(P)
            d[1:-1] = (P[2:] - P[:-2]) / (2.0 * dx)
            d[0]    = (P[1]  - P[0])   / dx
            d[-1]   = (P[-1] - P[-2])  / dx
            return d

        # Terminal condition
        Phi = np.log(2.0 * np.cosh(xs))

        Phi_all = np.empty((nt, nx))
        Phi_all[-1] = Phi

        # Backward integration t: 1 -> 0
        for k in range(nt - 2, -1, -1):
            t_cur = ts[k + 1]
            mu_t  = self._mu(t_cur)

            # Nonlinear source at current time level (explicit)
            Phi_x   = deriv1(Phi)
            nonlin  = (beta**2 / 2.0) * mu_t * Phi_x**2

            # RHS vector for CN step
            rhs = apply_rhs(Phi) - dt * nonlin

            # Solve (I + r D^2) Phi_new = rhs
            Phi = solve_banded((1, 1), ab_lhs, rhs)
            Phi_all[k] = Phi

        self.ts = ts
        self.xs = xs
        self.Phi_all = Phi_all

        # Precompute spatial derivative grids
        dPhi   = np.empty_like(Phi_all)
        d2Phi  = np.empty_like(Phi_all)

        dPhi[:, 1:-1]  = (Phi_all[:, 2:] - Phi_all[:, :-2]) / (2.0 * dx)
        dPhi[:, 0]     = (Phi_all[:, 1]  - Phi_all[:, 0])   / dx
        dPhi[:, -1]    = (Phi_all[:, -1] - Phi_all[:, -2])  / dx

        d2Phi[:, 1:-1] = (Phi_all[:, 2:] - 2*Phi_all[:, 1:-1] + Phi_all[:, :-2]) / dx**2
        d2Phi[:, 0]    = d2Phi[:, 1]
        d2Phi[:, -1]   = d2Phi[:, -2]

        kw = dict(method='linear', bounds_error=False, fill_value=None)
        self._itp_Phi   = RegularGridInterpolator((ts, xs), Phi_all, **kw)
        self._itp_dPhi  = RegularGridInterpolator((ts, xs), dPhi,    **kw)
        self._itp_d2Phi = RegularGridInterpolator((ts, xs), d2Phi,   **kw)

    def _query(self, itp, t, x):
        x  = np.asarray(x, dtype=float)
        ts = np.full(x.shape, float(t))
        return itp(np.stack([ts.ravel(), x.ravel()], axis=1)).reshape(x.shape)

    def Phi(self, t, x):    return self._query(self._itp_Phi,   t, x)
    def dPhi(self, t, x):   return self._query(self._itp_dPhi,  t, x)
    def d2Phi(self, t, x):  return self._query(self._itp_d2Phi, t, x)
    def mu(self, t):        return self._mu(t)


# ======================================================================
# 3. IAMP core  (Algorithm 1 / Appendix B)
# ======================================================================

def iamp_trajectory(A, pde, beta, delta, M_clip=5.0, seed=0):
    """
    Run one IAMP trajectory; return continuous z in R^n.

    Algorithm (Appendix B, Montanari 2019):
      u^{k+1} = A (g^{k-1} o u^k)  -  b_k * g^{k-2} o u^{k-1}
      x^k = x^{k-1} + beta^2 mu(k*delta) d_x Phi(k*delta, x^{k-1}) * delta
                     + beta * sqrt(delta) * u^k
      g^k = sqrt(n) * d_xx Phi(k*delta, x^k) / ||d_xx Phi||
      b_{k+1} = mean(g^k)
      z += sqrt(delta) * g^{k-1} o clip(u^k, -M, M)
    """
    n      = A.shape[0]
    rng    = np.random.default_rng(seed)
    q_star = pde.q_star
    K      = max(1, int(np.floor(q_star / delta)))
    sq_d   = np.sqrt(delta)

    # Initialise
    u_km1 = np.zeros(n)
    u_k   = rng.standard_normal(n)
    g_km2 = np.zeros(n)
    g_km1 = np.ones(n)
    b_k   = 0.0
    x_k   = np.zeros(n)
    z     = np.zeros(n)

    for k in range(K):
        t_k   = k * delta
        mu_k  = pde.mu(t_k)

        # AMP update
        u_kp1 = A @ (g_km1 * u_k) - b_k * (g_km2 * u_km1)

        # State (SDE) update
        dphi  = pde.dPhi(t_k, x_k)
        x_kp1 = x_k + beta**2 * mu_k * dphi * delta + beta * sq_d * u_k

        # Normalised second-derivative vector
        d2phi = pde.d2Phi(t_k, x_kp1)
        nm    = np.linalg.norm(d2phi)
        g_k   = np.sqrt(n) * d2phi / nm if nm > 1e-14 else np.ones(n) / np.sqrt(n)

        # Onsager correction
        b_kp1 = float(np.mean(g_k))

        # Accumulate z
        z += sq_d * g_km1 * np.clip(u_k, -M_clip, M_clip)

        # Advance
        u_km1, u_k   = u_k,   u_kp1
        g_km2, g_km1 = g_km1, g_k
        b_k          = b_kp1
        x_k          = x_kp1

    return z


# ======================================================================
# 4. Sequential rounding  (Algorithm 2, Appendix B)
# ======================================================================

def round_to_pm1(A, z):
    """
    Two-pass coordinate rounding (Lemma 3.5 / Algorithm 2):
      Pass 1: project z_i onto [-1,+1]
      Pass 2: for each l, set sigma_l = sign(sum_{j!=l} A_{lj} sigma_j)

    Guarantees the off-diagonal Hamiltonian does not decrease.
    """
    sigma = np.clip(z, -1.0, 1.0).copy()
    n     = len(sigma)
    for ell in range(n):
        h = float(A[ell] @ sigma) - A[ell, ell] * sigma[ell]
        sigma[ell] = 1.0 if h >= 0.0 else -1.0
    return sigma.astype(int)


# ======================================================================
# 5. High-level solver
# ======================================================================

def iamp_solve(A, beta=6.0, delta=0.02, M_clip=5.0, n_restarts=3,
               q_star=None, pde=None, seed=42):
    """
    Full IAMP pipeline:
      1. Estimate q*(beta)
      2. Solve Parisi PDE
      3. Run n_restarts IAMP trajectories
      4. Round each -> sigma in {+-1}^n
      5. Return best sigma by energy <sigma, A sigma>

    Returns dict(sigma, energy, q_star, pde).
    """
    if q_star is None:
        q_star = estimate_q_star(beta)
    if pde is None:
        pde = ParisiPDE(beta, q_star)

    best_e     = -np.inf
    best_sigma = None

    for r in range(n_restarts):
        z     = iamp_trajectory(A, pde, beta, delta, M_clip=M_clip,
                                seed=seed + r * 9999)
        sigma = round_to_pm1(A, z)
        e     = float(sigma @ A @ sigma)
        if e > best_e:
            best_e     = e
            best_sigma = sigma.copy()

    return dict(sigma=best_sigma, energy=best_e, q_star=q_star, pde=pde)


# ======================================================================
# 6. Metrics
# ======================================================================

def ising_energy(sigma, J):
    return -0.5 * float(sigma @ J @ sigma)

def sync_ratio(sigma, J):
    J_mat = J - np.diag(np.diag(J))  # zero out diagonal for sync ratio
    lf = J_mat @ sigma
    lf[lf == 0] = sigma[lf == 0]   # tie-break
    return float(np.mean(sigma == np.sign(lf)))
