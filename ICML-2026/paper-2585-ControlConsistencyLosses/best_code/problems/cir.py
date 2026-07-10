import jax.numpy as jnp
from jax import lax

EPS = 1e-12


def get_problem(cfg):
    a = cfg.get("a", 1.0)
    b = cfg.get("b", 1.0)
    sigma = cfg.get("sigma", 1.0)
    d = cfg.get("d", 1)
    T = cfg.get("T", 1.0)
    x_0 = jnp.array(cfg.get("x_0", [2.0]))
    x_T = jnp.array(cfg.get("x_T", [2.0]))

    def base_drift(x, t):
        return a * (b - x)

    def sigma_fn(x, t):
        return sigma * jnp.sqrt(jnp.maximum(x, 0.0))

    def cir_transition_params(x, y, s):
        s = jnp.maximum(s, EPS)
        exp_as = jnp.exp(-a * s)
        c = 2.0 * a / ((1.0 - exp_as) * sigma**2)
        x_clip = jnp.clip(x, 0.0)
        y_clip = jnp.clip(y, 0.0)
        u = c * x_clip * exp_as
        v = c * y_clip
        q = 2.0 * a * b / (sigma**2) - 1.0
        z = 2.0 * jnp.sqrt(jnp.maximum(u * v, EPS))
        return c, u, v, q, z, exp_as

    def I_ratio_cf(q, z, n_terms=500):
        z = jnp.maximum(z, EPS)
        z2 = z * z

        def body(i, r):
            k = n_terms - i
            return z2 / (2.0 * (q + k) + r)

        tail = lax.fori_loop(0, n_terms, body, jnp.zeros_like(z))
        return z / (2.0 * (q + 1.0) + tail)

    def Iprime_over_I(q, z):
        z = jnp.maximum(z, EPS)
        R_true = I_ratio_cf(q, z)
        R_asymp = -(q + 0.5) * (1.0 / z) + 0.5 * (q * q - 0.25) * (1.0 / z**2) + 1.0
        R = jnp.where(z > 1e4, R_asymp, R_true)
        return q / z + R

    def dlogp_dx_CIR(x, y, s):
        c, u, v, q, z, exp_as = cir_transition_params(x, y, s)
        du_dx = c * exp_as
        sqrt_v_over_u = jnp.sqrt(jnp.maximum(v, EPS) / jnp.maximum(u, EPS))
        dlogp_du = (
            -1.0
            - q / (2.0 * jnp.maximum(u, EPS))
            + Iprime_over_I(q, z) * sqrt_v_over_u
        )
        return du_dx * dlogp_du

    def true_drift_fn(x, t):
        return base_drift(x, t) + (sigma**2) * jnp.clip(x, 0.0) * dlogp_dx_CIR(x, x_T, T - t)

    return {
        "base_drift": base_drift,
        "sigma_fn": sigma_fn,
        "shape": (d,),
        "x_0": x_0,
        "x_T": x_T,
        "T": T,
        "true_drift_fn": true_drift_fn,
    }
