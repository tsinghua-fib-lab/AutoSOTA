import os
import io

import numpy as np
import jax.numpy as jnp
import jax.random as jr

from simulation.time_to_censoring import get_time_to_censoring
from simulation.time_to_event import get_time_to_event


def simulate_data(config):

    # Create key
    rng = jr.PRNGKey(config.algorithm.seed_data_obs)

    # Get time-to-event and time-to-censoring distributions
    time_to_event = get_time_to_event(config)
    time_to_censoring = get_time_to_censoring(config)

    # Get number of samples, dim of covariates (x) and test ratio
    num_samples = config.data_obs.num_samples
    dim_x = config.data_obs.dim_x

    # Simulate covariates (x)
    rng, subrng = jr.split(rng)
    x = jr.normal(subrng, (num_samples, *dim_x))

    # Simulate time-to-event and time-to-censing
    rng, subrng_tte, subrng_ttc = jr.split(rng, 3)
    time_to_event_obs = time_to_event.sample(subrng_tte, num_samples, x)
    time_to_censoring_obs = time_to_censoring.sample(subrng_ttc, num_samples)

    # Generate time = minimum and event = event or censoring?
    time = jnp.minimum(time_to_event_obs, time_to_censoring_obs)
    event = time_to_event_obs <= time_to_censoring_obs

    # Randomly choose test, val and train indices
    rng, subrng = jr.split(rng)
    test_size = int(0.2 * num_samples)
    test_indices = jr.choice(subrng, num_samples, shape=(test_size,), replace=False)

    # Get remaining indices after test
    remaining_indices = jnp.setdiff1d(jnp.arange(num_samples), test_indices)

    # Sample validation indices (from remaining)
    subrng, val_rng = jr.split(subrng)
    val_size = int(0.2 * num_samples)
    val_indices = jr.choice(
        val_rng, remaining_indices, shape=(val_size,), replace=False
    )

    # Remaining are train indices
    train_indices = jnp.setdiff1d(remaining_indices, val_indices)

    # Create boolean masks
    data_obs_is_train = jnp.zeros(num_samples, dtype=bool).at[train_indices].set(True)
    data_obs_is_val = jnp.zeros(num_samples, dtype=bool).at[val_indices].set(True)
    data_obs_is_test = jnp.zeros(num_samples, dtype=bool).at[test_indices].set(True)

    # Keep track
    config.data_obs.data_train = {
        "time": time[data_obs_is_train].astype(jnp.float32),
        "event": event[data_obs_is_train],
        "x": x[data_obs_is_train, :].astype(jnp.float32),
    }
    config.data_obs.data_val = {
        "time": time[data_obs_is_val].astype(jnp.float32),
        "event": event[data_obs_is_val],
        "x": x[data_obs_is_val, :].astype(jnp.float32),
    }
    config.data_obs.data_test = {
        "time": time[data_obs_is_test].astype(jnp.float32),
        "event": event[data_obs_is_test],
        "x": x[data_obs_is_test, :].astype(jnp.float32),
    }
