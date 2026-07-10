import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def get_problem(cfg):
    a = jnp.sqrt(cfg.get("a_sq", 1.0))
    v = cfg.get("v", 3.0)
    sigma = cfg.get("sigma", 1.0)
    d = cfg.get("d", 1)
    T = cfg.get("T", 1.0)
    x_0 = jnp.array(cfg.get("x_0", [float(a)]))
    x_T = jnp.array(cfg.get("x_T", [float(-a)]))

    def double_well_potential(x):
        return (v * (x**2 - a**2)**2).sum()

    def base_drift(x, t):
        return -jax.grad(double_well_potential)(x)

    # ground-truth drift via backward-Kolmogorov on a (t, x) grid
    gt_xs_min = cfg.get("gt_xs_min", -4.0)
    gt_xs_max = cfg.get("gt_xs_max", 4.0)
    gt_xs_N = cfg.get("gt_xs_N", 10000)
    gt_ts_N = cfg.get("gt_ts_N", 1000)

    grid_xs = jnp.linspace(gt_xs_min, gt_xs_max, gt_xs_N)
    grid_ts = jnp.linspace(0.0, T, gt_ts_N)

    def _linear_delta_on_grid(xs, y):
        dx = xs[1] - xs[0]
        r = jnp.clip(jnp.searchsorted(xs, y, side="left"), 1, xs.shape[0] - 1)
        l = r - 1
        x_l, x_r = xs[l], xs[r]
        w_r = (y - x_l) / (x_r - x_l)
        w_l = 1.0 - w_r
        p = jnp.full_like(xs, 0.0)
        p = p.at[l].set(w_l / dx)
        p = p.at[r].set(w_r / dx)
        return p

    def _get_log_h(y, ts, xs):
        dx = xs[1] - xs[0]
        terminal_density = _linear_delta_on_grid(xs, y)
        terminal = jnp.log(terminal_density)

        ts_rev = ts[::-1]
        dts = jnp.abs(ts_rev[1:] - ts_rev[:-1])

        def one_step(log_p_next, inputs):
            t, dt = inputs

            def log_p_at_x(x):
                b = -jax.grad(double_well_potential)(jnp.array([x]))[0]
                mu = x + dt * b
                var = jnp.maximum(1e-16, dt * sigma**2)
                logK = -0.5 * jnp.log(2 * jnp.pi * var) - ((xs - mu)**2) / (2 * var) + jnp.log(dx)
                return logsumexp(logK + log_p_next)

            log_p = jax.vmap(log_p_at_x)(xs)
            log_p = log_p - jnp.max(log_p)
            return log_p, log_p

        _, log_ps_rev = jax.lax.scan(one_step, terminal, (ts_rev[1:], dts))
        return log_ps_rev[::-1]

    log_h = _get_log_h(x_T[0], grid_ts, grid_xs)
    dlogdx = (log_h[:, 2:] - log_h[:, :-2]) / (grid_xs[2:] - grid_xs[:-2])
    xs_mid = grid_xs[1:-1]
    ts_mid = grid_ts[:-1]

    def true_drift_fn(x, t):
        it = jnp.clip(jnp.searchsorted(ts_mid, t, side="right") - 1, 0, ts_mid.shape[0] - 1)
        ix = jnp.clip(jnp.argmin(jnp.abs(x[0] - xs_mid)), 0, xs_mid.shape[0] - 1)
        grad_log = dlogdx[it, ix]
        control = sigma**2 * grad_log
        return base_drift(x, t) + jnp.array([control])

    return {
        "base_drift": base_drift,
        "sigma_fn": sigma,
        "shape": (d,),
        "x_0": x_0,
        "x_T": x_T,
        "T": T,
        "true_drift_fn": true_drift_fn,
    }
