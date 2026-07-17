"""
Exact numerical solutions for dynamic OT with Gaussian marginal potentials.

Author(s): Raghav Kansal
"""

import hashlib
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from experiments.gaussian import plotting

# =============================================================================
# Distance functions between isotropic Gaussians
# =============================================================================


def sq_norm(x: ArrayLike) -> float:
    """Squared Euclidean norm of a vector."""
    x = np.asarray(x)
    return float(np.sum(x**2))


def W2(
    m_p: float | ArrayLike,
    sigma_p: float,
    m_q: float | ArrayLike,
    sigma_q: float,
    d: int,
) -> float:
    """W2 distance between N(m_p, sigma_p^2 I) and N(m_q, sigma_q^2 I)."""
    delta2 = sq_norm(np.asarray(m_p) - np.asarray(m_q))
    return np.sqrt(delta2 + d * (sigma_p - sigma_q) ** 2)


def KL(
    m_p: float | ArrayLike,
    sigma_p: float,
    m_q: float | ArrayLike,
    sigma_q: float,
    d: int,
) -> float:
    """KL divergence KL(N(m_p, sigma_p^2 I) || N(m_q, sigma_q^2 I))."""
    var_p = sigma_p**2
    var_q = sigma_q**2
    delta2 = sq_norm(np.asarray(m_p) - np.asarray(m_q))
    var_ratio = var_p / var_q
    return 0.5 * (d * (var_ratio - 1.0 - np.log(var_ratio)) + delta2 / var_q)


def MMD2_RBF(
    m_p: float | ArrayLike,
    sigma_p: float,
    m_q: float | ArrayLike,
    sigma_q: float,
    gamma: float,
    d: int,
) -> float:
    """Exact MMD^2 with RBF kernel k(x,y) = exp(-||x - y||^2 / (2*gamma^2))."""
    var_p = sigma_p**2
    var_q = sigma_q**2
    tau_p_sq = gamma**2 + 2.0 * var_p
    tau_q_sq = gamma**2 + 2.0 * var_q
    tau_plus_sq = gamma**2 + var_p + var_q
    A_pp = (gamma**2 / tau_p_sq) ** (d / 2.0)
    A_qq = (gamma**2 / tau_q_sq) ** (d / 2.0)
    delta2 = sq_norm(np.asarray(m_p) - np.asarray(m_q))
    B_pq = (gamma**2 / tau_plus_sq) ** (d / 2.0) * np.exp(-delta2 / (2.0 * tau_plus_sq))
    return A_pp + A_qq - 2.0 * B_pq


def MMD2_Poly(
    m_p: float | ArrayLike,
    sigma_p: float,
    m_q: float | ArrayLike,
    sigma_q: float,
    coef0: float,
    d: int,
) -> float:
    """Exact MMD^2 with polynomial kernel k(x,y) = (x^T y + c)^2."""
    m_p = np.asarray(m_p)
    m_q = np.asarray(m_q)
    c = coef0
    var_p = sigma_p**2
    var_q = sigma_q**2
    norm_p_sq = sq_norm(m_p)
    norm_q_sq = sq_norm(m_q)
    dot_pq = float(np.dot(m_p.ravel(), m_q.ravel()))
    A_pp = norm_p_sq**2 + d * var_p**2 + 2 * var_p * norm_p_sq + 2 * c * norm_p_sq + c**2
    A_qq = norm_q_sq**2 + d * var_q**2 + 2 * var_q * norm_q_sq + 2 * c * norm_q_sq + c**2
    B_pq = (
        dot_pq**2
        + d * var_p * var_q
        + var_p * norm_q_sq
        + var_q * norm_p_sq
        + 2 * c * dot_pq
        + c**2
    )
    return A_pp + A_qq - 2.0 * B_pq


# =============================================================================
# Gradient functions
# =============================================================================


def grad_W2(m, s, mk, sigma_k, d):
    """Gradient of W2 distance w.r.t. mean m and std s."""
    g_m = 2.0 * (np.asarray(m) - np.asarray(mk))
    g_sigma = 2.0 * d * (s - sigma_k)
    return g_m, g_sigma


def grad_KL(m, s, mk, sigma_k, d, debug_dict=None):
    """Gradient of KL divergence w.r.t. mean m and std s."""
    s = np.maximum(s, 1e-3)
    g_m = (np.asarray(m) - np.asarray(mk)) / (sigma_k**2)
    g_sigma = d * (s / (sigma_k**2) - 1.0 / s)
    g_sigma = np.clip(g_sigma, -100, 100)
    return g_m, g_sigma


def grad_MMD2_RBF(m, s, mk, sigma_k, d, gamma):
    """Gradient of MMD^2 (RBF) w.r.t. mean m and std s."""
    m = np.asarray(m)
    mk = np.asarray(mk)
    var_p = s**2
    var_q = sigma_k**2
    tau_rho_sq = gamma**2 + 2.0 * var_p
    tau_plus_sq = gamma**2 + var_p + var_q
    A_rho = (gamma**2 / tau_rho_sq) ** (d / 2.0)
    delta_m = m - mk
    delta_m_sq = sq_norm(delta_m)
    B = (gamma**2 / tau_plus_sq) ** (d / 2.0) * np.exp(-delta_m_sq / (2.0 * tau_plus_sq))
    g_m = 2.0 * B * delta_m / tau_plus_sq
    dA_rho_ds = -2.0 * d * s / tau_rho_sq * A_rho
    dB_ds = B * s * (-d / tau_plus_sq + delta_m_sq / tau_plus_sq**2)
    g_sigma = dA_rho_ds - 2.0 * dB_ds
    return g_m, g_sigma


def grad_MMD2_Poly(m, s, mk, sigma_k, d, coef0, debug_dict=None, clip_grad=10.0):
    """Gradient of MMD^2 (polynomial) w.r.t. mean m and std s."""
    m = np.asarray(m)
    mk = np.asarray(mk)
    c = coef0
    var_p = s**2
    var_q = sigma_k**2
    norm_m_sq = sq_norm(m)
    norm_mk_sq = sq_norm(mk)
    dot_m_mk = float(np.dot(m.ravel(), mk.ravel()))
    coeff_m = norm_m_sq + var_p - var_q + c
    coeff_mk = dot_m_mk + c
    g_m = 4.0 * (coeff_m * m - coeff_mk * mk)
    g_sigma = 4.0 * s * (d * (var_p - var_q) + (norm_m_sq - norm_mk_sq))
    if clip_grad is not None:
        g_m = np.clip(g_m, -clip_grad, clip_grad)
        g_sigma = np.clip(g_sigma, -clip_grad, clip_grad)
    return g_m, float(g_sigma)


# =============================================================================
# GaussianMarginalSolver
# =============================================================================


class GaussianMarginalSolver:
    """
    Solves the OT problem with middle-marginal potentials for all-isotropic-Gaussian
    marginals under the Gaussian ansatz.

    Uses shooting method to numerically find the initial mean and sigma velocities
    that solve the final boundary conditions. Supports K >= 1 intermediate Gaussians.

    The potential is a sum over all intermediates:
        V(m, s, t) = Σ_k w_k * λ_k(t) * r(D(m, s, mk, sigma_k))

    Args:
        d: Dimension of the problem
        means: Array of means [m0, mk_1, ..., mk_K, m1] (shape (K+2,) or (K+2, d))
        sigmas: Array of standard deviations [sigma0, sigma_k_1, ..., sigma_k_K, sigma1]
        w: Strength(s) of intermediate potential(s). Scalar or shape (K,).
        t_k: Time(s) of intermediate marginal(s). Scalar or shape (K,).
        lambda_width: Width(s) of the potential(s) in time. Scalar or shape (K,).
        distD: Statistical distance ("W2", "KL", "MMD_RBF", "MMD_Poly")
        gamma: RBF kernel bandwidth (required for MMD_RBF)
        coef0: Polynomial kernel coefficient (default 1.0)
        r_D: Dependence of potential on distance ("-D", "-D^2", "1/D")
        lambda_type: Time localization ("gaussian", "triangle", "box")
        ode_args: Arguments for scipy.integrate.solve_ivp
        shooting_args: Arguments for shooting solver
        cache: Optional directory for caching solutions
    """

    def __init__(
        self,
        d: int,
        means: ArrayLike,
        sigmas: ArrayLike,
        w: float | ArrayLike,
        t_k: float | ArrayLike,
        lambda_width: float | ArrayLike,
        distD: str,
        gamma: float = None,
        coef0: float = 1.0,
        r_D: str = "-D",
        lambda_type: str = "gaussian",
        ode_args: dict = None,
        shooting_args: dict = None,
        cache: str | Path | None = None,
    ):
        self.d = d

        # Parse means array
        means = np.asarray(means)
        if means.ndim == 1:
            assert len(means) >= 3
            self.m0 = means[0]
            self._mk = means[1:-1]
            self.m1 = means[-1]
        else:
            assert means.shape[0] >= 3
            self.m0 = means[0]
            self._mk = means[1:-1]
            self.m1 = means[-1]

        self.K = len(self._mk)

        # Parse sigmas array
        sigmas = np.asarray(sigmas)
        assert len(sigmas) == self.K + 2
        self.sigma0 = float(sigmas[0])
        self._sigma_k = sigmas[1:-1].astype(float)
        self.sigma1 = float(sigmas[-1])

        # Parse w, t_k, lambda_width
        self._w = np.atleast_1d(np.asarray(w, dtype=float))
        self._t_k = np.atleast_1d(np.asarray(t_k, dtype=float))
        self._lambda_width = np.atleast_1d(np.asarray(lambda_width, dtype=float))

        if len(self._w) == 1:
            self._w = np.full(self.K, self._w[0])
        if len(self._t_k) == 1:
            self._t_k = np.full(self.K, self._t_k[0])
        if len(self._lambda_width) == 1:
            self._lambda_width = np.full(self.K, self._lambda_width[0])

        assert len(self._w) == self.K
        assert len(self._t_k) == self.K
        assert len(self._lambda_width) == self.K

        self.gamma = gamma
        self.coef0 = coef0
        self.distD = distD
        self.r_D = r_D
        self.lambda_type = lambda_type
        self.v0_opt, self.u0_opt = (None, None)
        self.ode_successful = False
        self.ode_error = ""

        # ODE solver arguments
        self.ode_args = dict(
            method="Radau",
            rtol=1e-5,
            atol=1e-9,
            fun=self.ode_system,
            t_span=(0.0, 1.0),
            max_step=self._lambda_width.min() / 10.0,
        )
        if ode_args is not None:
            self.ode_args.update(ode_args)

        # Shooting solver arguments
        default_shooting_args = dict(
            shooting_method="root",
            root_method="broyden1",
            ls_method="trf",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=1000,
        )
        self.shooting_args = default_shooting_args.copy()
        if shooting_args is not None:
            self.shooting_args.update(shooting_args)

        self.shooting_method = self.shooting_args["shooting_method"]
        self.root_method = self.shooting_args["root_method"]
        self.ls_method = self.shooting_args["ls_method"]
        self.ftol = self.shooting_args["ftol"]
        self.xtol = self.shooting_args["xtol"]
        self.gtol = self.shooting_args["gtol"]
        self.max_nfev = self.shooting_args["max_nfev"]

        self.debug_dict = {"shooting": [], "grad_KL": [], "grad_MMD": []}

        if self.distD == "MMD_RBF" and self.gamma is None:
            raise ValueError("gamma must be provided for MMD_RBF")

        self.cache_dir = Path(cache) if cache is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def means(self) -> np.ndarray:
        return np.concatenate([[self.m0], self._mk, [self.m1]])

    @property
    def sigmas(self) -> np.ndarray:
        return np.concatenate([[self.sigma0], self._sigma_k, [self.sigma1]])

    @property
    def mk(self):
        return self._mk[0] if self.K == 1 else self._mk

    @property
    def sigma_k(self):
        return float(self._sigma_k[0]) if self.K == 1 else self._sigma_k

    @property
    def w(self):
        return float(self._w[0]) if self.K == 1 else self._w

    @property
    def t_k(self):
        return float(self._t_k[0]) if self.K == 1 else self._t_k

    @property
    def lambda_width(self):
        return float(self._lambda_width[0]) if self.K == 1 else self._lambda_width

    def _get_cache_key(self, extra_params: dict = None):
        """Generate cache key from parameters."""

        def serialize_value(value):
            if isinstance(value, np.ndarray):
                return (value.shape, value.dtype.name, value.tobytes())
            elif isinstance(value, int | float | str | bool | type(None)):
                return value
            elif isinstance(value, list | tuple):
                return tuple(serialize_value(v) for v in value)
            else:
                return str(value)

        ode_args_for_cache = {k: v for k, v in self.ode_args.items() if k != "fun"}
        params = {
            "d": self.d,
            "K": self.K,
            "m0": serialize_value(self.m0),
            "m1": serialize_value(self.m1),
            "sigma0": serialize_value(self.sigma0),
            "sigma1": serialize_value(self.sigma1),
            "mk": serialize_value(self._mk),
            "sigma_k": serialize_value(self._sigma_k),
            "w": serialize_value(self._w),
            "t_k": serialize_value(self._t_k),
            "lambda_width": serialize_value(self._lambda_width),
            "gamma": self.gamma,
            "coef0": self.coef0,
            "distD": self.distD,
            "r_D": self.r_D,
            "lambda_type": self.lambda_type,
            "ode_args": tuple(
                sorted((k, serialize_value(v)) for k, v in ode_args_for_cache.items())
            ),
            "shooting_args": tuple(
                sorted((k, serialize_value(v)) for k, v in self.shooting_args.items())
            ),
        }
        if extra_params is not None:
            for key, value in extra_params.items():
                params[key] = serialize_value(value)
        param_items = sorted(params.items())
        param_bytes = pickle.dumps(param_items)
        return hashlib.md5(param_bytes).hexdigest()

    def _load_cache(self, cache_key: str, cache_type: str):
        if self.cache_dir is None:
            return None
        cache_file = self.cache_dir / f"{cache_type}_{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                warnings.warn(f"Failed to load cache: {e}", UserWarning, stacklevel=2)
        return None

    def _save_cache(self, cache_key: str, cache_type: str, data: dict):
        if self.cache_dir is None:
            return
        cache_file = self.cache_dir / f"{cache_type}_{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            warnings.warn(f"Failed to save cache: {e}", UserWarning, stacklevel=2)

    def _get_D_single(self, m, s, mk, sigma_k):
        match self.distD:
            case "W2":
                return W2(m, s, mk, sigma_k, self.d)
            case "KL":
                return KL(m, s, mk, sigma_k, self.d)
            case "MMD_RBF":
                return MMD2_RBF(m, s, mk, sigma_k, self.gamma, self.d)
            case "MMD_Poly":
                return MMD2_Poly(m, s, mk, sigma_k, self.coef0, self.d)
            case _:
                raise ValueError(f"Unknown D: {self.distD}")

    def _grad_D_single(self, m, s, mk, sigma_k):
        match self.distD:
            case "W2":
                return grad_W2(m, s, mk, sigma_k, self.d)
            case "KL":
                return grad_KL(m, s, mk, sigma_k, self.d, self.debug_dict)
            case "MMD_RBF":
                return grad_MMD2_RBF(m, s, mk, sigma_k, self.d, self.gamma)
            case "MMD_Poly":
                return grad_MMD2_Poly(m, s, mk, sigma_k, self.d, self.coef0, self.debug_dict)
            case _:
                raise ValueError(f"Unknown D: {self.distD}")

    def get_D(self, m, s, k=None):
        if k is not None:
            return self._get_D_single(m, s, self._mk[k], self._sigma_k[k])
        return np.array(
            [self._get_D_single(m, s, self._mk[i], self._sigma_k[i]) for i in range(self.K)]
        )

    def grad_D(self, m, s, k=None):
        if k is not None:
            return self._grad_D_single(m, s, self._mk[k], self._sigma_k[k])
        return [self._grad_D_single(m, s, self._mk[i], self._sigma_k[i]) for i in range(self.K)]

    def d_r_D(self, D_value):
        match self.r_D:
            case "-D":
                return -1.0
            case "-D^2":
                return np.clip(-D_value * 2.0, -100, 100)
            case "1/D":
                return np.clip(-1.0 / (D_value**2), -100, 100)
            case _:
                raise ValueError(f"Unknown r_D: {self.r_D}")

    def _lambda_t_single(self, t, t_k, lambda_width):
        match self.lambda_type:
            case "gaussian":
                return np.exp(-0.5 * ((t - t_k) / lambda_width) ** 2) / (
                    np.sqrt(2 * np.pi) * lambda_width
                )
            case "triangle":
                x = (t - t_k) / lambda_width
                return max(0.0, (1.0 - abs(x)) / lambda_width)
            case "box":
                inside = abs(t - t_k) <= lambda_width
                norm = 1.0 / (2 * lambda_width)
                return norm * inside
            case _:
                raise ValueError(f"Unknown lambda type: {self.lambda_type}")

    def lambda_t(self, t, k=None):
        if k is not None:
            return self._lambda_t_single(t, self._t_k[k], self._lambda_width[k])
        return np.array(
            [self._lambda_t_single(t, self._t_k[i], self._lambda_width[i]) for i in range(self.K)]
        )

    def _unpack_y(self, y):
        m = y[: self.d]
        s = y[self.d]
        v = y[self.d + 1 : -1]
        u = y[-1]
        if self.d == 1:
            m = m[0]
            v = v[0]
        return m, s, v, u

    def _unpack_vu(self, velocities):
        velocities = np.asarray(velocities)
        v = velocities[: self.d]
        u = velocities[self.d]
        if self.d == 1:
            v = v[0]
        return v, u

    def _pack_y(self, m, s, v, u):
        if np.asarray(m).ndim == 0:
            return np.array([m, s, v, u])
        else:
            return np.concatenate([m, [s], v, [u]])

    def _pack_vu(self, v, u):
        if np.asarray(v).ndim == 0:
            return np.array([v, u])
        else:
            return np.concatenate([v, [u]])

    def ode_system(self, t, y):
        m, s, v, u = self._unpack_y(y)
        dv = np.zeros_like(np.atleast_1d(m), dtype=float)
        du = 0.0
        for k in range(self.K):
            g_m_k, g_sigma_k = self.grad_D(m, s, k=k)
            lam_k = self.lambda_t(t, k=k)
            if self.r_D != "-D":
                D_k = self.get_D(m, s, k=k)
                dr_dD_k = self.d_r_D(D_k)
            else:
                dr_dD_k = -1.0
            coeff = self._w[k] * lam_k * dr_dD_k
            dv = dv - coeff * np.atleast_1d(g_m_k)
            du = du - coeff * g_sigma_k / self.d
        if self.d == 1:
            dv = float(dv[0])
        return self._pack_y(v, u, dv, du)

    def terminal_residual(self, init_velocities):
        from scipy.integrate import solve_ivp

        v0, u0 = self._unpack_vu(init_velocities)
        y0 = self._pack_y(self.m0, self.sigma0, v0, u0)
        ode_args = self.ode_args.copy()
        ode_args.update(dict(y0=y0, t_eval=[1.0]))
        sol = solve_ivp(**ode_args)
        m_end, s_end, _, _ = self._unpack_y(sol.y[:, -1])
        return self._pack_vu(m_end - self.m1, s_end - self.sigma1)

    def solve_bvp_root(self, init_guess):
        from scipy.optimize import root

        root_sol = root(self.terminal_residual, init_guess, method=self.root_method)
        if root_sol.success:
            self.ode_successful = True
        else:
            self.ode_successful = False
            self.ode_error = "Solution did not converge! Error: " + root_sol.message
        self.v0_opt, self.u0_opt = self._unpack_vu(root_sol.x)

    def solve_bvp_LS(self, init_guess):
        from scipy.optimize import least_squares

        res = least_squares(
            self.terminal_residual,
            x0=init_guess,
            method=self.ls_method,
            ftol=self.ftol,
            xtol=self.xtol,
            gtol=self.gtol,
            max_nfev=self.max_nfev,
        )
        self.v0_opt, self.u0_opt = self._unpack_vu(res.x)

    def solve_bvp(self):
        """Find initial velocities that solve terminal condition."""
        cache_key = self._get_cache_key()
        cached_data = self._load_cache(cache_key, "bvp")
        if cached_data is not None:
            self.v0_opt = cached_data["v0_opt"]
            self.u0_opt = cached_data["u0_opt"]
            self.ode_successful = cached_data["ode_successful"]
            self.ode_error = cached_data["ode_error"]
            return

        guess_v0 = self.m1 - self.m0
        guess_u0 = self.sigma1 - self.sigma0
        init_guess = self._pack_vu(guess_v0, guess_u0)

        if self.distD == "KL" and np.any(self._w > 10):
            self.ode_successful = False
            self.ode_error = r"KL solution prohibitive for large $w$ - did not attempt!"
            self.v0_opt, self.u0_opt = self._unpack_vu(init_guess)
            self._save_cache(
                cache_key,
                "bvp",
                {
                    "v0_opt": self.v0_opt,
                    "u0_opt": self.u0_opt,
                    "ode_successful": self.ode_successful,
                    "ode_error": self.ode_error,
                },
            )
            return

        match self.shooting_method:
            case "root":
                self.solve_bvp_root(init_guess)
            case "ls":
                self.solve_bvp_LS(init_guess)
            case _:
                raise ValueError(f"Unknown shooting method: {self.shooting_method}")

        self._save_cache(
            cache_key,
            "bvp",
            {
                "v0_opt": self.v0_opt,
                "u0_opt": self.u0_opt,
                "ode_successful": self.ode_successful,
                "ode_error": self.ode_error,
            },
        )

    def integrate_ode(self, t_eval):
        """Integrate ODE using solved initial velocities."""
        t_eval_array = np.asarray(t_eval)
        cache_key = self._get_cache_key({"t_eval": t_eval_array})
        cached_data = self._load_cache(cache_key, "ode")
        if cached_data is not None:
            self.m_path = cached_data["m_path"]
            self.sigma_path = cached_data["sigma_path"]
            self.v_path = cached_data["v_path"]
            self.u_path = cached_data["u_path"]
            self.t_eval = cached_data["t_eval"]
            return

        from scipy.integrate import solve_ivp

        y0 = self._pack_y(self.m0, self.sigma0, self.v0_opt, self.u0_opt)
        ode_args = self.ode_args.copy()
        ode_args.update(dict(y0=y0, t_eval=t_eval))
        sol_full = solve_ivp(**ode_args)
        if not sol_full.success:
            raise RuntimeError(f"ODE solver failed: {sol_full.message}")
        self.m_path, self.sigma_path, self.v_path, self.u_path = self._unpack_y(sol_full.y)
        self.t_eval = t_eval

        self._save_cache(
            cache_key,
            "ode",
            {
                "m_path": self.m_path,
                "sigma_path": self.sigma_path,
                "v_path": self.v_path,
                "u_path": self.u_path,
                "t_eval": self.t_eval,
            },
        )

    def _check_bvp_and_ode_solved(self, t_eval=None):
        if self.v0_opt is None or self.u0_opt is None:
            self.solve_bvp()
        if t_eval is None:
            if hasattr(self, "t_eval") and self.t_eval is not None:
                t_eval = self.t_eval
            else:
                t_eval = np.linspace(0, 1, 100)
        t_eval_array = np.asarray(t_eval)
        needs_integration = (
            not hasattr(self, "m_path")
            or self.m_path is None
            or not hasattr(self, "t_eval")
            or self.t_eval is None
        )
        if not needs_integration:
            existing = np.asarray(self.t_eval)
            if existing.shape != t_eval_array.shape or not np.allclose(existing, t_eval_array):
                needs_integration = True
        if needs_integration:
            self.integrate_ode(t_eval_array)
        return t_eval_array

    def get_trajectories(self, x0s: np.ndarray, t_eval: np.ndarray = None) -> tuple:
        """Get individual sample trajectories."""
        t_eval_array = self._check_bvp_and_ode_solved(t_eval)
        ms = np.tile(self.m_path, (len(x0s), 1, 1))
        ss = np.tile(self.sigma_path, (len(x0s), self.d, 1))
        xs = ms + x0s.reshape(-1, self.d, 1) * ss
        return t_eval_array, xs

    def plot_results(self, plot_velocities=False, t_eval=None):
        """Plot mean, sigma, and velocity trajectories."""
        if self.d != 1:
            raise ValueError("Plot results not implemented for d > 1")
        t_eval_array = self._check_bvp_and_ode_solved(t_eval)

        if plot_velocities:
            fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        else:
            fig, axs = plt.subplots(1, 2, figsize=(16, 5))
            axs = np.expand_dims(axs, axis=0)

        # Compute total lambda
        lambda_total = np.zeros_like(self.t_eval)
        for k in range(self.K):
            lambda_total += np.array([self.lambda_t(t, k=k) for t in self.t_eval])
        lambda_vals = lambda_total / lambda_total.max()

        titleD = self.distD if self.distD != "MMD" else r"MMD$^2$"
        m_size = 80
        linecolor = "#5386E4"
        lambda_color = "#ECECEC"
        linewidth = 2

        if self.K == 1:
            lambda_label = rf"$\lambda(t)$, width={self._lambda_width[0]:.2f}"
            title_params = f"($D=${titleD}, $w={self._w[0]:.2f}$)"
        else:
            lambda_label = rf"$\Sigma_k \lambda_k(t)$ (K={self.K})"
            title_params = f"($D=${titleD}, K={self.K})"

        rD_label = plotting.rD_labels.get(self.r_D, self.r_D)
        title_params = f"{title_params}, {rD_label}"

        if not self.ode_successful:
            fig.suptitle("Warning: " + self.ode_error, color="red")

        for j in range(2):
            if j == 0:
                path, vpath, endpoints, middle_points = (
                    self.m_path,
                    self.v_path,
                    [self.m0, self.m1],
                    self._mk,
                )
                axlabel, titlelabel = "m(t)", "Mean"
            else:
                path, vpath, endpoints, middle_points = (
                    self.sigma_path,
                    self.u_path,
                    [self.sigma0, self.sigma1],
                    self._sigma_k,
                )
                axlabel, titlelabel = r"$\sigma(t)$", r"$\sigma$"

            ax = axs[0, j]
            ax.plot(self.t_eval, path, color=linecolor, linewidth=linewidth)
            ax.plot(
                self.t_eval,
                np.max(self.m_path) * lambda_vals,
                color=lambda_color,
                linewidth=linewidth,
                label=lambda_label,
            )
            ax.scatter([0, 1], endpoints, color="k", zorder=3, label="Endpoints", s=m_size)
            ax.scatter(
                self._t_k,
                middle_points,
                color="#DC851F",
                zorder=3,
                label=f"Intermediate(s) (K={self.K})",
                s=m_size,
            )
            ax.set_ylabel(axlabel)
            ax.set_title(f"{titlelabel} trajectory {title_params}")

            if plot_velocities:
                ax = axs[1, j]
                ax.plot(self.t_eval, vpath, color=linecolor, linewidth=linewidth)
                ax.plot(
                    self.t_eval,
                    np.max(vpath) * lambda_vals,
                    color=lambda_color,
                    linewidth=linewidth,
                    label=lambda_label,
                )
                ax.set_ylabel(r"$v(t)$")
                ax.set_title(f"{titlelabel} velocity {title_params}")

        for ax in np.ravel(axs):
            ax.set_xlabel(r"$t$")
            ax.set_xlim(0, 1)
            ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_trajectories(self, x0s: np.ndarray, t_eval: np.ndarray = None):
        """Plot individual sample trajectories."""
        t_eval_array, xs = self.get_trajectories(x0s, t_eval)

        if self.d == 1:
            func = plotting.plot_trajectories_middle_marginal_1d
        elif self.d == 2:
            func = plotting.plot_trajectories_middle_marginal_2d
        else:
            raise ValueError(f"Unsupported dimension: {self.d}")

        means_list = [self.m0] + list(self._mk) + [self.m1]
        sigmas_list = [self.sigma0] + list(self._sigma_k) + [self.sigma1]
        title = f"{plotting.distD_labels.get(self.distD, self.distD)}    {plotting.rD_labels.get(self.r_D, self.r_D)}"

        if self.d == 1:
            func(
                means_list,
                sigmas_list,
                x0s,
                self._t_k,
                xs,
                t_eval_array,
                wks=self._w,
                title=title,
                lambda_width=self._lambda_width,
                lambda_type=self.lambda_type,
                plot_dir=self.cache_dir,
                show=True,
            )
        else:
            func(
                means_list,
                sigmas_list,
                x0s,
                xs,
                self._t_k,
                t_eval_array,
                plot_dir=self.cache_dir,
                show=True,
            )
