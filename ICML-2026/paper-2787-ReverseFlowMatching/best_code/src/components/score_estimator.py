import functools
import jax
import jax.numpy as jnp

from src.components.noise_schedule import noise_h


def log_expectation_reward(
    key,
    t,
    actions,
    states,
    energy_function,
    energy_params,
    num_mc_samples,
    sigma_min,
    sigma_diff,
    temperature,
    clipper=None,
):
    repeated_t = jnp.expand_dims(t, 0).repeat(num_mc_samples, axis=0)
    repeated_actions = jnp.expand_dims(actions, 0).repeat(num_mc_samples, axis=0)
    repeated_states = jnp.expand_dims(states, 0).repeat(num_mc_samples, axis=0)

    h_t = jnp.expand_dims(noise_h(repeated_t, sigma_min, sigma_diff), -1)

    noised_actions = repeated_actions + (
        jax.random.normal(key, repeated_actions.shape) * jnp.sqrt(h_t)
    )

    log_rewards = energy_function.apply(energy_params, repeated_states, noised_actions)

    if clipper is not None and clipper.should_clip_log_rewards:
        log_rewards = clipper.clip_log_rewards(log_rewards)

    return jax.scipy.special.logsumexp(log_rewards / temperature, axis=-1) - jnp.log(
        num_mc_samples
    )


@functools.partial(
    jax.jit,
    static_argnames=("energy_function", "num_mc_samples", "sigma_min", "sigma_diff"),
)
def estimate_grad_Rt(
    key,
    t,
    actions,
    states,
    energy_function,
    energy_params,
    num_mc_samples,
    sigma_min,
    sigma_diff,
    temperature,
):
    subkeys = jax.random.split(key, actions.shape[0])
    if jnp.ndim(t) == 0:
        t = jnp.expand_dims(t, 0).tile(len(actions))

    grad_fxn = jax.grad(log_expectation_reward, argnums=2)
    vmapped_fxn = jax.vmap(
        grad_fxn, in_axes=(0, 0, 0, 0, None, None, None, None, None, 0)
    )

    return vmapped_fxn(
        subkeys,
        t,
        actions,
        states,
        energy_function,
        energy_params,
        num_mc_samples,
        sigma_min,
        sigma_diff,
        temperature,
    )
