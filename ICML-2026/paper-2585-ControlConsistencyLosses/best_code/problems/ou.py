import jax.numpy as jnp

def get_problem(cfg):
    a = cfg.get("a", 0.0)
    b = cfg.get("b", 2.0)
    sigma = cfg.get("sigma", 0.1)
    d = cfg.get("d", 2)
    T = cfg.get("T", 1.0)
    x_0 = jnp.array(cfg.get("x_0", [1.0, 0.0]))
    x_T = jnp.array(cfg.get("x_T", [0.0, 1.0]))

    def base_drift(x, t):
        return a - b * x

    def OU_mean(x, s):
        return (a / b) + (x - (a / b)) * jnp.exp(-b * s)

    def OU_var(s):
        return (1 - jnp.exp(-2 * b * s)) / (2 * b) * sigma**2

    def true_h_transform(x, t):
        return (jnp.exp(-b * (T - t)) / OU_var(T - t)) * (x_T - OU_mean(x, T - t))

    def true_drift_fn(x, t):
        return base_drift(x, t) + sigma**2 * true_h_transform(x, t)

    return {
        "base_drift": base_drift,
        "sigma_fn": sigma,
        "shape": (d,),
        "x_0": x_0,
        "x_T": x_T,
        "T": T,
        "true_drift_fn": true_drift_fn,
    }
