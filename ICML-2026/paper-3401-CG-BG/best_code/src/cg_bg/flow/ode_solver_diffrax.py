from functools import partial
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve
from flax.core import FrozenDict

from cg_bg.utils import get_progress
from cg_bg.utils.standarization import destandardize


class ODEState(eqx.Module):
    x: jnp.ndarray
    logp: jnp.ndarray


def gaussian_log_density(x: jnp.ndarray, mu: float = 0.0, sigma: float = 1.0) -> jnp.ndarray:
    dim = x.shape[-1]
    sq = jnp.sum((x - mu) ** 2, axis=-1)
    return -0.5 * (dim * jnp.log(2 * jnp.pi * sigma**2) + sq / sigma**2)


def com_energy_adjustment(x: jnp.ndarray) -> jnp.ndarray:
    num_atoms = x.shape[1]
    com_std = 1.0 / jnp.sqrt(num_atoms)
    com_norms = jnp.linalg.norm(jnp.mean(x, axis=1), axis=-1)
    log_term = jnp.log(com_norms**2) - jnp.log(jnp.sqrt(2.0) * com_std**3) - jsp.gammaln(1.5)
    return -(com_norms**2) / (2.0 * com_std**2) + log_term


def divergence(f: Callable[..., jnp.ndarray]) -> Callable[..., jnp.ndarray]:
    @partial(jax.vmap, in_axes=(None, 0, None, None))
    def div(params, x, input_args, _):
        def f_val(x_local):
            out = f(params, x_local, **input_args)
            out = jnp.atleast_1d(out)
            return out

        jac = jax.jacobian(f_val)(x)
        jac = jnp.squeeze(jac)
        if jac.ndim >= 2:
            return jnp.trace(jac)
        return jac

    return div


def get_dynamics(apply_fn: Callable):
    div_fn = divergence(apply_fn)

    def dynamics(t, state: ODEState, args):
        params, inputs, key = args
        x = state.x
        inputs["t"] = t
        v = jax.vmap(lambda p, x: apply_fn(p, x=x, **inputs), in_axes=(None, 0))(params, x)
        div = div_fn(params, x, inputs, key)

        return ODEState(x=v.reshape(state.x.shape), logp=-div.reshape(state.logp.shape))

    return dynamics


def make_diffrax_solver(method: str):
    if method.lower() == "tsit5":
        return Tsit5()
    if method.lower() == "dopri5":
        return Dopri5()
    raise ValueError(f"Unknown diffrax solver: {method}")


def batched_sampler(
    dt0: float,
    params: FrozenDict,
    apply_fn: Callable,
    method: str,
    num_batches: int,
    batch_size: int,
    n_nodes: int,
    n_dim: int,
    features: Optional[jnp.ndarray],
    rng: jax.random.PRNGKey,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    mu: float = 0.0,
    sigma: float = 1.0,
    com_center: bool = True,
    std: Optional[float] = None,
):
    if com_center and std is None:
        raise ValueError("std must be provided when com_center=True so samples and logp can be restored.")

    dynamics_fn = get_dynamics(apply_fn)
    solver = make_diffrax_solver(method)
    term = ODETerm(dynamics_fn)
    saveat = SaveAt(t1=True)
    controller = PIDController(rtol=rtol, atol=atol)
    params = eqx.tree_inference(params)
    inputs = {}
    if features is not None:
        inputs["features"] = features
        inputs["training"] = False

    def run_one_batch(rng):
        rng, rng_x0, rng_div = jax.random.split(rng, 3)

        x0 = jax.random.normal(rng_x0, (batch_size, n_nodes * n_dim)) * sigma + mu
        logp0 = gaussian_log_density(x0, mu=mu, sigma=sigma)

        state0 = ODEState(x=x0, logp=logp0)

        sol = diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=1.0,
            dt0=dt0,
            y0=state0,
            args=(params, inputs, rng_div),
            saveat=saveat,
            stepsize_controller=controller,
        )

        return rng, sol.ys

    run_one_batch = eqx.filter_jit(run_one_batch)

    states_x = []
    states_logp = []
    with get_progress() as progress:
        task_id = progress.add_task("Sampling", total=num_batches)
        for _ in range(num_batches):
            rng, state = run_one_batch(rng)
            states_x.append(state.x)
            states_logp.append(state.logp)
            progress.update(task_id, advance=1)

    x_all = jnp.concatenate(states_x, axis=0).reshape(-1, n_nodes, n_dim)
    logp_all = jnp.concatenate(states_logp, axis=0).reshape(-1)

    if com_center:
        # Change-of-variables corrections: undo COM removal, then undo the std rescaling.
        logp_all = logp_all - com_energy_adjustment(x_all)
        logp_all = logp_all - (n_nodes * n_dim) * jnp.log(std)
        x_all = destandardize(x_all, std)

    return x_all, logp_all
