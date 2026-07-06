from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from softjax.utils import _canonicalize_axis, _validate_softness


@jax.custom_jvp
def _proj_unit_simplex_pnorm_q2(values: jax.Array) -> jax.Array:
    """Projection onto the unit simplex.

    Taken from https://github.com/google/jaxopt/blob/cf28b4563f5ad9354b76433622dbb9ee32af5f09/jaxopt/_src/projection.py#L96
    """
    s = 1.0
    n_features = values.shape[0]
    u = jnp.sort(values)[::-1]
    cumsum_u = jnp.cumsum(u)
    ind = jnp.arange(n_features) + 1
    cond = s / ind + (u - cumsum_u / ind) > 0
    idx = jnp.count_nonzero(cond)
    return jax.nn.relu(s / idx + (values - cumsum_u[idx - 1] / idx))


@_proj_unit_simplex_pnorm_q2.defjvp
def _proj_unit_simplex_pnorm_q2_jvp(
    primals: list[jax.Array], tangents: list[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    (values,) = primals
    (values_dot,) = tangents
    primal_out = _proj_unit_simplex_pnorm_q2(values)
    supp = primal_out > 0
    card = jnp.count_nonzero(supp)
    tangent_out = supp * values_dot - (jnp.dot(supp, values_dot) / card) * supp
    return primal_out, tangent_out


def _proj_unit_simplex_pnorm_q3_impl(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Closed-form simplex projection for p=3/2 (alpha=2) via quadratic formula.

    Returns (primal_out, theta) so the JVP can reuse theta.
    """
    n = values.shape[0]
    u = jnp.sort(values)[::-1]
    u0 = u[0]
    u_shift = u - u0  # shift so max is 0; improves float32 stability
    S = jnp.cumsum(u_shift)
    M2 = jnp.cumsum(u_shift**2)
    k_arr = jnp.arange(1, n + 1, dtype=values.dtype)

    disc = S**2 - k_arr * (M2 - 1.0)
    theta_k = (S - jnp.sqrt(jnp.maximum(disc, 0.0))) / k_arr

    cond = u_shift > theta_k
    idx = jnp.count_nonzero(cond)
    theta = theta_k[idx - 1] + u0  # convert back to original coordinates
    y = jnp.maximum(values - theta, 0.0) ** 2
    return y / jnp.sum(y), theta


@jax.custom_jvp
def _proj_unit_simplex_pnorm_q3(values: jax.Array) -> jax.Array:
    return _proj_unit_simplex_pnorm_q3_impl(values)[0]


@_proj_unit_simplex_pnorm_q3.defjvp
def _proj_unit_simplex_pnorm_q3_jvp(
    primals: list[jax.Array], tangents: list[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    (values,) = primals
    (values_dot,) = tangents
    primal_out, theta = _proj_unit_simplex_pnorm_q3_impl(values)

    supp = (primal_out > 0).astype(values.dtype)
    t = jnp.maximum(values - theta, 0.0)
    w = t * supp  # alpha=2: weight propto 2*t
    w_sum = jnp.sum(w)
    w_sum = jnp.where(w_sum > 0, w_sum, 1.0)

    raw_tangent = 2.0 * t * (values_dot - jnp.dot(w, values_dot) / w_sum) * supp
    # y = t^2 / sum(t^2), quotient rule
    sum_t2 = jnp.sum(t**2)
    sum_t2 = jnp.where(sum_t2 > 0, sum_t2, 1.0)
    tangent_out = raw_tangent / sum_t2 - primal_out * jnp.sum(raw_tangent) / sum_t2
    return primal_out, tangent_out


def _proj_unit_simplex_pnorm_q4_impl(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Closed-form simplex projection for p=4/3 (alpha=3) via Cardano's method.

    Returns (primal_out, theta) so the JVP can reuse theta.
    """
    n = values.shape[0]
    dtype = values.dtype
    u = jnp.sort(values)[::-1]
    u0 = u[0]
    u_shift = u - u0  # shift so max is 0; avoids catastrophic cancellation in mu3
    S = jnp.cumsum(u_shift)
    M2 = jnp.cumsum(u_shift**2)
    M3 = jnp.cumsum(u_shift**3)
    k_arr = jnp.arange(1, n + 1, dtype=dtype)

    # For each candidate k, solve sum_{i=1}^k (u_i - theta)^3 = 1
    # Depressed cubic after shifting by mean c = S_k / k
    c = S / k_arr
    mu2 = M2 - 2.0 * c * S + k_arr * c**2
    mu3 = M3 - 3.0 * c * M2 + 3.0 * c**2 * S - k_arr * c**3

    # u^3 + p_coeff*u + q_coeff = 0 where u = theta - c
    p_coeff = 3.0 * mu2 / k_arr  # >= 0
    q_coeff = (1.0 - mu3) / k_arr

    # Cardano's hyperbolic method (p >= 0 guarantees one real root)
    sp3 = jnp.sqrt(jnp.maximum(p_coeff / 3.0, 0.0))
    denom = 2.0 * jnp.maximum(p_coeff, jnp.finfo(dtype).tiny) * sp3
    A = 3.0 * jnp.abs(q_coeff) / denom
    u_hyp = -jnp.sign(q_coeff) * 2.0 * sp3 * jnp.sinh(jnp.arcsinh(A) / 3.0)
    u_cbrt = -jnp.sign(q_coeff) * jnp.abs(q_coeff) ** (1.0 / 3.0)
    u_root = jnp.where(
        p_coeff > jnp.finfo(dtype).eps * jnp.maximum(jnp.abs(q_coeff), 1.0),
        u_hyp,
        u_cbrt,
    )
    theta_k = u_root + c

    cond = u_shift > theta_k
    idx = jnp.count_nonzero(cond)
    theta = theta_k[idx - 1] + u0  # convert back to original coordinates
    y = jnp.maximum(values - theta, 0.0) ** 3
    return y / jnp.sum(y), theta


@jax.custom_jvp
def _proj_unit_simplex_pnorm_q4(values: jax.Array) -> jax.Array:
    return _proj_unit_simplex_pnorm_q4_impl(values)[0]


@_proj_unit_simplex_pnorm_q4.defjvp
def _proj_unit_simplex_pnorm_q4_jvp(
    primals: list[jax.Array], tangents: list[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    (values,) = primals
    (values_dot,) = tangents
    primal_out, theta = _proj_unit_simplex_pnorm_q4_impl(values)

    supp = (primal_out > 0).astype(values.dtype)
    t = jnp.maximum(values - theta, 0.0)
    w = t**2 * supp  # alpha=3: weight propto 3*t^2
    w_sum = jnp.sum(w)
    w_sum = jnp.where(w_sum > 0, w_sum, 1.0)

    raw_tangent = 3.0 * t**2 * (values_dot - jnp.dot(w, values_dot) / w_sum) * supp
    # y = t^3 / sum(t^3), quotient rule
    sum_t3 = jnp.sum(t**3)
    sum_t3 = jnp.where(sum_t3 > 0, sum_t3, 1.0)
    tangent_out = raw_tangent / sum_t3 - primal_out * jnp.sum(raw_tangent) / sum_t3
    return primal_out, tangent_out


@eqx.filter_jit
def _proj_simplex(
    x: Array,  # (..., n, ...)
    axis: int,
    softness: float | Array = 0.1,
    mode: Literal["smooth", "c0", "c1", "c2"] = "smooth",
    return_log_probs: bool = False,
) -> Array:  # (..., [n], ...)
    """Projects `x` onto the unit simplex along the specified axis.

    Solves the optimization problem along the specified axis:
        min_y <x, y> + softness * R(y)
        s.t. y >= 0, sum(y) = 1
    where R(y) is the regularizer determined by `mode`.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `axis`: Axis containing the simplex dimension.
    - `softness`: Controls the strength of the regularizer.
    - `mode`: Controls the type of regularizer:
        - `smooth`: C∞ smooth (based on logistic/softmax/entropic regularizer). Solved in closed form via softmax.
        - `c0`: C0 continuous (based on euclidean/L2 regularizer). Solved via the algorithm in https://arxiv.org/pdf/1309.1541.
        - `c1`: C1 differentiable (p=3/2 p-norm). P-norm regularizer is inspired by https://arxiv.org/abs/2302.01425. Solved in closed form via quadratic formula.
        - `c2`: C2 twice differentiable (p=4/3 p-norm). P-norm regularizer is inspired by https://arxiv.org/abs/2302.01425. Solved in closed form via Cardano's method.
    - `return_log_probs`: If True, returns log probabilities instead of probabilities. Exact zero probabilities are represented as `-inf`.

    **Returns:**

    An Array of shape (..., [n], ...) representing the projected values onto the unit simplex along the specified axis.
    """
    _validate_softness(softness)
    axis = _canonicalize_axis(axis, x.ndim)
    n = x.shape[axis]
    _x = x / softness
    if mode == "smooth":
        if return_log_probs:
            soft_index = jax.nn.log_softmax(_x, axis=axis)  # (..., [n], ...)
        else:
            soft_index = jax.nn.softmax(_x, axis=axis)  # (..., [n], ...)
    elif mode == "c0":
        _x = jnp.moveaxis(_x, axis, -1)  # (..., ..., n)
        *batch_sizes, n = _x.shape
        _x = _x.reshape(-1, n)  # (B, n)
        soft_index = jax.vmap(_proj_unit_simplex_pnorm_q2)(_x)
        soft_index = soft_index.reshape(*batch_sizes, n)  # (..., ..., [n])
        soft_index = jnp.moveaxis(soft_index, -1, axis)  # (..., [n], ...)
    else:
        if mode == "c1":
            proj = _proj_unit_simplex_pnorm_q3
        elif mode == "c2":
            proj = _proj_unit_simplex_pnorm_q4
        else:
            raise ValueError(f"Invalid mode: {mode}")

        _x = jnp.moveaxis(_x, axis, -1)
        *batch_sizes, n = _x.shape
        _x = _x.reshape(-1, n)
        soft_index = jax.vmap(proj)(_x)
        soft_index = soft_index.reshape(*batch_sizes, n)
        soft_index = jnp.moveaxis(soft_index, -1, axis)
    if return_log_probs and mode != "smooth":
        soft_index = jnp.log(soft_index)
    return soft_index
