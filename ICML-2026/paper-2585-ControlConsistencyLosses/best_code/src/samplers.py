'''
discretisation step functions, for different sampling methods
Take inputs, and return the next step
We compute JVPs and VJPs through these functions when computing training targets (for the single-step Jacobians)
'''

import jax
import jax.numpy as jnp

# Euler-Maruyama sampler
def euler_maruyama_sampler(
        x,
        t,
        dB_t,
        dt,
        drift_fn,
        sigma_fn,
):
    drift = drift_fn(x, t)
    sigma, apply_sigma_fn = sigma_fn(x, t)
    noise_term = apply_sigma_fn(dB_t)
    x_next = x + drift * dt + noise_term
    return x_next, drift


# Heun sampler
def heun_sampler(
        x,
        t,
        dB_t,
        dt,
        drift_fn,
        sigma_fn,
):
    # EM predictor
    drift = drift_fn(x, t)
    sigma, apply_sigma = sigma_fn(x, t)
    x_pred = x + drift * dt + apply_sigma(dB_t)

    # Heun
    drift_pred = drift_fn(x_pred, t + dt)
    sigma_pred, apply_sigma_pred = sigma_fn(x_pred, t + dt)

    x_next = (
        x
        + 0.5 * (drift + drift_pred) * dt
        + 0.5 * (apply_sigma(dB_t) + apply_sigma_pred(dB_t))
    )

    return x_next, drift

# TODO: add milstein sampler