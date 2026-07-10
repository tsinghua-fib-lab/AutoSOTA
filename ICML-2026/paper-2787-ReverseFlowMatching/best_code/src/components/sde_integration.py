import functools
import jax
import jax.numpy as jnp

from src.components.noise_schedule import noise_g


def grad_E(x, cond, energy_function, energy_params):
    def energy(_x):
        return jnp.sum(energy_function.apply(energy_params, _x, cond))

    return jax.grad(energy)(x)


def euler_maruyama_step(
    carry,
    t,
    score_module,
    score_params,
    dt,
    cond,
    sigma_min,
    sigma_diff,
    diffusion_scale=1.0,
    temperature=1.0,
):
    x, key = carry
    key, subkey = jax.random.split(key)
    t = t * jnp.ones(x.shape[0])

    # Calculate drift and diffusion terms
    g = jnp.expand_dims(noise_g(t, sigma_min, sigma_diff), -1)
    drift = (g**2) * score_module.apply(score_params, t, x, cond, temperature) * dt
    diffusion = (
        diffusion_scale * g * jnp.sqrt(dt) * jax.random.normal(key, shape=x.shape)
    )

    # Update the state
    x_next = x + drift + diffusion
    return (x_next, subkey), x_next


@functools.partial(
    jax.jit,
    static_argnames=(
        "score_module",
        "num_integration_steps",
        "sigma_min",
        "sigma_diff",
        "time_range",
    ),
)
def integrate_sde(
    key,
    score_module,
    score_params,
    x0,
    cond,
    num_integration_steps,
    sigma_min,
    sigma_diff,
    diffusion_scale=1.0,
    temperature=1.0,
    time_range=1.0,
):
    start_time = 1.0
    end_time = time_range - start_time

    times = jnp.linspace(start_time, end_time, num_integration_steps + 1)[:-1]

    x = x0
    loop_fn = functools.partial(
        euler_maruyama_step,
        score_module=score_module,
        score_params=score_params,
        dt=time_range / num_integration_steps,
        cond=cond,
        sigma_min=sigma_min,
        sigma_diff=sigma_diff,
        diffusion_scale=diffusion_scale,
        temperature=temperature,
    )
    (x, key), _ = jax.lax.scan(loop_fn, (x, key), times)

    return x, key
