import jax
import jax.numpy as jnp

def get_problem(cfg):
    beta = cfg.get("beta", 1.0)
    sigma = cfg.get("sigma", 3.0)
    d = cfg.get("d", 2)
    T = cfg.get("T", 0.05)
    x_0 = jnp.array(cfg.get("x_0", [-0.55828035, 1.44169]))
    x_T = jnp.array(cfg.get("x_T", [0.62361133, 0.02804632]))

    def muller_brown_potential(z):
        x, y = z[0], z[1]
        e1 = -200 * jnp.exp(-(x - 1)**2 - 10 * y**2)
        e2 = -100 * jnp.exp(-x**2 - 10 * (y - 0.5)**2)
        e3 = -170 * jnp.exp(-6.5*(0.5 + x)**2 + 11*(x + 0.5)*(y - 1.5) - 6.5*(y - 1.5)**2)
        e4 = 15.0 * jnp.exp(0.7*(1 + x)**2 + 0.6*(x + 1)*(y - 1) + 0.7*(y - 1)**2)
        return beta * (e1 + e2 + e3 + e4)

    def base_drift(x, t):
        return -jax.grad(muller_brown_potential)(x)

    return {
        "base_drift": base_drift,
        "sigma_fn": sigma,
        "shape": (d,),
        "x_0": x_0,
        "x_T": x_T,
        "T": T,
    }