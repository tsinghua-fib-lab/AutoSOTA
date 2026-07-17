"""
Per-sample localized OTP-FM solver (Sec. 5) for the $W_2^2$ potential,
primarily for visualizing the conditional trajectories (e.g. for Fig. 1 of the paper).

For each conditional sample (x_0, x_1, x_{mu_k} for k=1..K), the localized
trajectory is

    X_t = x_0 + (x_1 - x_0) t + sum_k w_k * grad_g_k * [I2_k(t) - I2_k(1) * t],

where I2_k(t) = double time-integral of lambda_k from 0 to t. With the
sample-level W_2 force grad_g_k = X_{t_k} - x_{mu_k} (linear in X_{t_k}), the
self-consistency equations at t_j become a K x K linear system per sample:

    (I - A) X_tk = X_base - A x_mu,

where A[j, k] = w_k * (I2_k(t_j) - I2_k(1) * t_j) and X_base[j] = x_0 + (x_1 - x_0) * t_j.

Author(s): Raghav Kansal
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

# ---------------------------------------------------------------------------
# Closed-form double integrals of lambda_k(t)
# ---------------------------------------------------------------------------


def _Phi_vec(z):
    from scipy.special import erf

    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


def _phi_vec(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def double_integrated_lambda(
    t: ArrayLike, t_k: float, width: float, lambda_type: str
) -> np.ndarray:
    """
    I2(t) = integral_0^t Lambda(s) ds, with Lambda(s) = integral_0^s lambda(s') ds'.

    Closed forms for "gaussian", "triangle", and "box" lambdas. For all kernels
    this satisfies I2(0) = 0 and I2(t) is monotone non-decreasing.
    """
    t = np.asarray(t, dtype=float)

    if lambda_type == "gaussian":
        zt = (t - t_k) / width
        z0 = -t_k / width
        Lambda_t = _Phi_vec(zt) - _Phi(z0)
        return (t - t_k) * Lambda_t + width * (_phi_vec(zt) - _phi(z0))

    if lambda_type == "triangle":
        h = width
        a = t_k - h
        b = t_k + h

        out = np.zeros_like(t)

        # Region 2: a < t <= t_k -> Lambda(t) = (t - a)^2 / (2 h^2)
        #                          I2(t)     = (t - a)^3 / (6 h^2)
        m2 = (t > a) & (t <= t_k)
        if np.any(m2):
            u = t[m2] - a
            out[m2] = (u**3) / (6.0 * h * h)

        # Region 3: t_k < t < b -> Lambda(t) = 1/2 + (t - t_k)/h - (t - t_k)^2 / (2 h^2)
        #     I2(t_k) = h / 6
        #     For 0 < u = t - t_k < h:
        #       integral_{t_k}^{t_k+u} Lambda(s) ds
        #         = (1/2) u + u^2 / (2 h) - u^3 / (6 h^2)
        m3 = (t > t_k) & (t < b)
        if np.any(m3):
            u = t[m3] - t_k
            out[m3] = h / 6.0 + 0.5 * u + (u * u) / (2.0 * h) - (u**3) / (6.0 * h * h)

        # Region 4: t >= b -> Lambda(t) = 1, I2(t) = I2(b) + (t - b)
        m4 = t >= b
        if np.any(m4):
            out[m4] = h + (t[m4] - b)

        return out

    if lambda_type == "box":
        # lambda(s) = 1 / (2 h) on [t_k - h, t_k + h], 0 otherwise.
        h = width
        a = t_k - h
        b = t_k + h
        out = np.zeros_like(t)

        # Lambda(t) = clip((t - a) / (2 h), 0, 1)
        m2 = (t > a) & (t < b)
        if np.any(m2):
            u = t[m2] - a
            out[m2] = (u * u) / (4.0 * h)
        m3 = t >= b
        if np.any(m3):
            out[m3] = h + (t[m3] - b)
        return out

    raise ValueError(f"Unknown lambda type: {lambda_type}")


# ---------------------------------------------------------------------------
# Localized conditional solver
# ---------------------------------------------------------------------------


def localized_conditional_trajectories(
    x0: ArrayLike,
    x1: ArrayLike,
    x_mks: ArrayLike,
    w: float | ArrayLike,
    t_k: float | ArrayLike,
    lambda_width: float | ArrayLike,
    t_eval: ArrayLike,
    lambda_type: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the OTP-FM localized conditional trajectory (Eq. eq:conditional_Xt)
    in closed form per sample, using the sample-level W_2 force
    grad_g_k(X_{t_k}) = X_{t_k} - x_{mu_k}.

    Args:
        x0: (N,) or (N, d) source samples.
        x1: (N,) or (N, d) terminal samples.
        x_mks: (K, N) or (K, N, d) intermediate target samples.
        w: scalar or (K,) potential strengths.
        t_k: scalar or (K,) intermediate times in (0, 1).
        lambda_width: scalar or (K,) widths of the time kernels.
        t_eval: (T,) times to evaluate.
        lambda_type: one of {"gaussian", "triangle", "box"}.

    Returns:
        X_t: (N, T) or (N, T, d) trajectories.
        X_tks: (K, N) or (K, N, d) fixed-point positions at the intermediate times.
    """
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x_mks = np.asarray(x_mks, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    scalar_d = x0.ndim == 1
    if scalar_d:
        x0_ = x0[:, None]  # (N, 1)
        x1_ = x1[:, None]
        x_mks_ = x_mks[..., None]  # (K, N, 1)
    else:
        x0_ = x0
        x1_ = x1
        x_mks_ = x_mks

    N, d = x0_.shape
    K = x_mks_.shape[0]
    assert x_mks_.shape[1:] == (N, d), f"x_mks shape {x_mks_.shape} != ({K}, {N}, {d})"

    w_arr = np.broadcast_to(np.atleast_1d(np.asarray(w, dtype=float)), (K,)).astype(float)
    tk_arr = np.broadcast_to(np.atleast_1d(np.asarray(t_k, dtype=float)), (K,)).astype(float)
    lw_arr = np.broadcast_to(np.atleast_1d(np.asarray(lambda_width, dtype=float)), (K,)).astype(
        float
    )

    # I2_k(t) for t in t_eval, t_k_i, and t=1.
    I2_eval = np.stack(
        [double_integrated_lambda(t_eval, tk_arr[k], lw_arr[k], lambda_type) for k in range(K)],
        axis=0,
    )  # (K, T)
    I2_at_tks = np.stack(
        [double_integrated_lambda(tk_arr, tk_arr[k], lw_arr[k], lambda_type) for k in range(K)],
        axis=0,
    )  # (K, K), entry [k, j] = I2_k(t_j)
    I2_at_1 = np.array(
        [
            double_integrated_lambda(np.array([1.0]), tk_arr[k], lw_arr[k], lambda_type)[0]
            for k in range(K)
        ]
    )  # (K,)

    # A[j, k] = w_k * (I2_k(t_j) - I2_k(1) * t_j)
    # Note: I2_at_tks[k, j] = I2_k(t_j), so we transpose.
    I2_k_of_tj = I2_at_tks.T  # (K_j, K_k) = (j, k)
    A = (I2_k_of_tj - np.outer(tk_arr, I2_at_1)) * w_arr[None, :]  # (K, K)

    # X_base[j] = x0 + (x1 - x0) * t_j
    X_base = x0_[None, :, :] + (x1_ - x0_)[None, :, :] * tk_arr[:, None, None]  # (K, N, d)

    # rhs = X_base - sum_k A[j, k] * x_mu_k
    A_xmu = np.einsum("jk,knd->jnd", A, x_mks_)  # (K, N, d)
    rhs = X_base - A_xmu  # (K, N, d)

    # Solve (I - A) X_tk = rhs.
    I_minus_A = np.eye(K) - A
    I_minus_A_inv = np.linalg.inv(I_minus_A)
    X_tks = np.einsum("jk,knd->jnd", I_minus_A_inv, rhs)  # (K, N, d)

    # grad_g_k = X_tk - x_mu_k (force per sample)
    g_k = X_tks - x_mks_  # (K, N, d)

    # X_t = x0 + (x1 - x0) * t + sum_k w_k * g_k * (I2_k(t) - I2_k(1) * t)
    bracket = I2_eval - np.outer(I2_at_1, t_eval)  # (K, T)
    base_t = x0_[:, None, :] + (x1_ - x0_)[:, None, :] * t_eval[None, :, None]  # (N, T, d)
    corr_t = np.einsum("kt,k,knd->ntd", bracket, w_arr, g_k)  # (N, T, d)
    X_t = base_t + corr_t

    if scalar_d:
        return X_t[..., 0], X_tks[..., 0]
    return X_t, X_tks
