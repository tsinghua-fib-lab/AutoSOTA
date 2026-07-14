"""
Do MCMC to sample from a Torus Graph.

"""
__date__ = "May 2023 - September 2025"


import blackjax
import jax
import jax.random as jr
import jax.numpy as jnp

from .stats import get_stats



get_avg_stats = lambda x: jnp.mean(jax.vmap(get_stats)(x), axis=0)


def wrap_to_pi(a):
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def torus_graph_unnorm_log_density(x, phi):
    """
    Un-normalized torus graph log density (negative energy)

    Parameters
    ----------
    x : [d]
    phi : [d,d,2]

    Returns
    -------
    negative_energy : []
    """
    stats = get_stats(x)  # [d,d,2]
    return jnp.sum(stats * phi)


def short_adaptation_run(
    key,
    phi,
    initial_position=None,
    mode: str = "hmc",               # "nuts" or "hmc"
    num_warmup: int = 1000,
    target_accept: float = 0.8,
    mass_matrix: str = "diagonal",    # "diagonal" or "dense"
    num_integration_steps: int = 32,  # only for HMC
    progress_bar: bool = False,
):
    if isinstance(key, int):
        key = jr.PRNGKey(key)

    def logdensity(x):
        return torus_graph_unnorm_log_density(x, phi)

    d = phi.shape[0]
    if initial_position is None:
        initial_position = jnp.zeros(d)

    # vanilla algorithm for warmup
    algo = blackjax.nuts if mode == "nuts" else blackjax.hmc

    kwargs = dict(
        is_mass_matrix_diagonal=(mass_matrix == "diagonal"),
        target_acceptance_rate=target_accept,
        progress_bar=progress_bar,
    )
    if mode == "hmc":
        kwargs["num_integration_steps"] = num_integration_steps

    warmup = blackjax.window_adaptation(algo, logdensity, **kwargs)
    (final_state, parameters), _ = warmup.run(key, initial_position, num_steps=num_warmup)

    step_size = parameters["step_size"]
    inv_mass  = parameters["inverse_mass_matrix"]

    # Build a production kernel (unwrapped); we’ll add rotation in sampling
    if mode == "nuts":
        base_kernel = blackjax.nuts(logdensity, step_size, inv_mass)
    else:
        base_kernel = blackjax.hmc(logdensity, step_size, inv_mass, num_integration_steps)

    def build_kernel():
        return base_kernel

    return {
        "step_size": step_size,
        "inverse_mass_matrix": inv_mass,
        "final_state": final_state,
        "kernel": build_kernel,
        "parameters": parameters,
    }


def sample_torus_graph(
    key,
    n,
    phi,
    initial_position=None,
    step_size=None,
    inverse_mass_matrix=None,
    num_integration_steps=60,
    mode="hmc",
    warmup=1000,
    target_accept=0.8,
    mass_matrix="diagonal",
    global_rotation: bool = False,
    rotation_u_scale=jnp.pi/2,
):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    d = phi.shape[0]
    if initial_position is None:
        initial_position = jnp.zeros(d)

    logdensity = lambda x: torus_graph_unnorm_log_density(x, phi)

    # Tune if needed (no rotation inside warmup)
    if (step_size is None) or (inverse_mass_matrix is None):
        tuned = short_adaptation_run(
            key,
            phi,
            initial_position=initial_position,
            mode=mode,
            num_warmup=warmup,
            target_accept=target_accept,
            mass_matrix=mass_matrix,
            num_integration_steps=num_integration_steps,
        )
        kernel = tuned["kernel"]()
        state  = tuned["final_state"]
        step_size, inverse_mass_matrix = tuned["step_size"], tuned["inverse_mass_matrix"]
    else:
        if mode == "nuts":
            kernel = blackjax.nuts(logdensity, step_size, inverse_mass_matrix)
        else:
            kernel = blackjax.hmc(logdensity, step_size, inverse_mass_matrix, num_integration_steps)
        state = kernel.init(initial_position)

    step = jax.jit(kernel.step)

    def one_step(state, rng_key):
        rk0, rk1 = jr.split(rng_key)
        state, info = step(rk0, state)
        
        if global_rotation:
            u = jr.uniform(rk1, (), minval=-rotation_u_scale, maxval=rotation_u_scale)
            new_pos = wrap_to_pi(state.position + u)
            state = state._replace(position=new_pos)
        return state, state

    keys = jr.split(key, n)
    _, states = jax.lax.scan(one_step, state, keys)
    return states.position
