import jax
import jax.random as jr
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


def get_compute_m(g, theta_MAP, mu, batch_size):

    @jax.jit
    def compute_m(time_eval, x_eval):

        m = []
        for i in range(0, len(time_eval), batch_size):
            t_batch = time_eval[i : i + batch_size]
            x_batch = x_eval[i : i + batch_size]

            @jax.jit
            def G(theta):
                return jax.vmap(lambda t, x: g(t, x, theta))(t_batch, x_batch).squeeze()

            G_map, j_delta = jax.jvp(G, (theta_MAP,), (mu - theta_MAP,))
            m.append(G_map + j_delta)
        return jnp.concatenate(m)

    return compute_m


def get_compute_s(g, theta_MAP, time_all, x_all, w_B, mu, batch_size):

    @jax.jit
    def G(theta):
        return jax.vmap(lambda t, x: g(t, x, theta))(time_all, x_all).squeeze()

    # Pre‑build VJP closure at θ_map
    _, vjp_fun = jax.vjp(G, theta_MAP)

    def get_B_matvec(w_B):
        @jax.jit
        def B_matvec(v):
            # forward‐mode for J(θ_map)·v
            _, jvp = jax.jvp(G, (theta_MAP,), (v,))  # shape (R,)
            # weight and back‐propagate
            weighted = w_B * jvp  # shape (R,)
            (back,) = vjp_fun(weighted)  # shape (dθ,)
            return v / 2 + back

        return B_matvec

    @jax.jit
    def Sigma_mul(w_B, A):
        B_matvec = get_B_matvec(w_B)
        sol, _ = cg(B_matvec, A, maxiter=50, tol=1e-6)
        return sol / 2

    @jax.jit
    def batch_eval(t, x):

        f_g = lambda theta: g(t, x, theta)

        # JVP for mean shift
        g_map, j_delta = jax.jvp(f_g, (theta_MAP,), (mu - theta_MAP,))

        b = jax.jacfwd(f_g)(theta_MAP).squeeze()
        # Quadratic term

        quad = jnp.dot(b, Sigma_mul(w_B, b))
        # quad = jnp.vdot(b, Sigma_b)

        return jnp.sqrt((g_map + j_delta) ** 2 + quad) + 1e-6

    # s function
    @jax.jit
    def compute_s(time_eval, x_eval):
        s = []
        for i in range(0, len(time_eval), batch_size):
            t_batch = time_eval[i : i + batch_size]
            x_batch = x_eval[i : i + batch_size]
            s.append(jax.vmap(batch_eval, in_axes=(0, 0))(t_batch, x_batch).squeeze())
        return jnp.concatenate(s)

    return compute_s


def get_posterior_samples_phi(rng, num_samples, posterior_params):

    # Posterior samples phi and glin
    rng, subrng_phi = jr.split(rng, 2)
    return draw_samples_phi(
        subrng_phi,
        posterior_params["alpha"],
        posterior_params["beta"],
        num_samples,
    )


def get_prior_samples_glin_fn(
    rng, num_samples, batch_size, g, theta_MAP, posterior_params
):

    time_all = posterior_params["time_all"]
    x_all = posterior_params["x_all"]

    # m function
    compute_m = get_compute_m(g, theta_MAP, jnp.zeros_like(theta_MAP), batch_size)

    # s function
    compute_s = get_compute_s(
        g,
        theta_MAP,
        time_all,
        x_all,
        jnp.zeros_like(theta_MAP),
        jnp.zeros_like(theta_MAP),
        batch_size,
    )

    # Posterior samples phi and glin
    rng, subrng_glin_fn = jr.split(rng, 2)
    epsilon_samples = jr.normal(subrng_glin_fn, (num_samples,))

    @jax.jit
    def glin_fn(t, x):
        m = compute_m(t, x)
        s = compute_s(t, x)
        return m.reshape((1, -1)) + jnp.sqrt(s**2 - m**2).reshape(
            (1, -1)
        ) * epsilon_samples.reshape((-1, 1))

    return glin_fn


def get_posterior_samples_glin_fn(
    rng, num_samples, batch_size, g, theta_MAP, posterior_params
):
    w_B = posterior_params["w_B"]
    mu = posterior_params["mu"]

    time_all = posterior_params["time_all"]
    x_all = posterior_params["x_all"]

    # m function
    compute_m = get_compute_m(g, theta_MAP, mu, batch_size)

    # s function
    compute_s = get_compute_s(g, theta_MAP, time_all, x_all, w_B, mu, batch_size)

    # Posterior samples phi and glin
    rng, subrng_glin_fn = jr.split(rng, 2)
    epsilon_samples = jr.normal(subrng_glin_fn, (num_samples,))

    @jax.jit
    def glin_fn(t, x):
        m = compute_m(t, x)
        s = compute_s(t, x)
        return m.reshape((1, -1)) + jnp.sqrt(s**2 - m**2).reshape(
            (1, -1)
        ) * epsilon_samples.reshape((-1, 1))

    return glin_fn


def draw_samples_phi(rng, alpha, beta, num_samples):
    rng, step_rng = jr.split(rng)
    return jr.gamma(step_rng, alpha, (num_samples,)) / beta
