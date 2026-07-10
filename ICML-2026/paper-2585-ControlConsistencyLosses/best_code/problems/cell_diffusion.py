import jax.numpy as jnp

def get_problem(cfg):
    sigma = cfg.get("sigma", 0.1)
    d = cfg.get("d", 2)
    T = cfg.get("T", 4.0)
    x_0 = jnp.array(cfg.get("x_0", [0.1, -0.1]))
    x_T = jnp.array(cfg.get("x_T", [2.0, -0.1]))

    c = 0.0625  # 2^{-4}

    def base_drift(x, t):
        vec = jnp.array([
            (x[0]**4 / (c + x[0]**4)) + (c / (c + x[1]**4)) - x[0],
            (x[1]**4 / (c + x[1]**4)) + (c / (c + x[0]**4)) - x[1],
        ])
        return vec

    return {
        "base_drift": base_drift,
        "sigma_fn": sigma,
        "shape": (d,),
        "x_0": x_0,
        "x_T": x_T,
        "T": T,
    }