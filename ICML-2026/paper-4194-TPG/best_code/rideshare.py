import jax
from jax.experimental import sparse
from functools import partial
from picard.rideshare_dispatch import (
    ManhattanRideshareDispatch,
    ManhattanRidesharePricing,
    GreedyPolicy,
    SimplePricingPolicy,
    EnvParams,
    obs_to_state,
    RideshareEvent,
)
from picard.nn import Policy
from jax import numpy as jnp
from typing import Dict, Callable, Tuple
import chex
from jax import Array
from jaxtyping import Integer, Float, Bool
from flax import struct
from sacred import Experiment
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import funcy as f
import pandas as pd
import haversine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ex = Experiment("rideshares")


@ex.config
def config():
    n_cars = 300  # Number of cars
    # Pricing choice model parameters
    w_price = -0.3
    w_eta = -0.005
    w_intercept = 4
    n_events = 10000  # Number of events to simulate per trial
    k = 10  # Total number of trials
    batch_size = 100  # Number of environments to run in parallel
    switch_every = 1000  # Switchback duration
    p = 0.5  # Treatment probability
    output = "results.csv"
    config_output = "config.csv"
    chunk_size = 1000  # Number of steps to process in each chunk
    naive_AB_design = True


@struct.dataclass
class ExperimentInfo:
    """
    Contains treatment assignment and cluster information for each step
    """

    t: Integer[Array, "n_steps"]
    is_treat: Bool[Array, "n_steps"]
    key: chex.PRNGKey


def stepper(
    env,
    env_params,
    A: Policy,
    B: Policy,
    carry: Tuple[Array, Array],  # (obs, state)
    info: ExperimentInfo,
):
    obs, state = carry
    key, policy_key = jax.random.split(info.key)
    action, action_info = jax.lax.cond(
        info.is_treat,
        lambda: B.apply(env_params, dict(), obs, policy_key),
        lambda: A.apply(env_params, dict(), obs, policy_key),
    )

    new_obs, new_state, reward, _, _ = env.step(key, state, action, env_params)
    
    # Get number of available drivers from state
    n_available = jnp.sum(new_state.times <= new_state.event.t)

    return ((new_obs, new_state), (reward, info.is_treat, n_available))


def run_trials(
    env,
    env_params,
    A,
    B,
    key,
    n_envs=10,
    n_steps=1000,
    p=0.5,
    chunk_size=1000,
    design=None,
):
    if design['name'] == "naive-AB":
        time_ids = env_params.events.t
        event_time_ids = time_ids
        ab_key, key = jax.random.split(key)
        is_treat = jax.random.bernoulli(ab_key, p, (n_envs, len(time_ids)))
    elif design['name'] == "pure-A":
        time_ids = env_params.events.t
        event_time_ids = time_ids
        is_treat = jnp.zeros((n_envs, len(time_ids)))
    elif design['name'] == "pure-B":
        time_ids = env_params.events.t
        event_time_ids = time_ids
        is_treat = jnp.ones((n_envs, len(time_ids)))
    elif design['name'] == "switchback":
        time_ids = (
            env_params.events.t // design['switch_every'] + 1
        ) * design['switch_every']  # Identifies the end of the period
        event_time_ids = env_params.events.t
        ab_key, key = jax.random.split(key)
        unq_times, unq_times_idx = jnp.unique(time_ids, return_inverse=True)
        is_treat = jax.random.bernoulli(ab_key, p, (n_envs, len(unq_times)))
        is_treat = is_treat[:, unq_times_idx]

    # Store time_ids for output
    time_ids_array = jnp.tile(time_ids.reshape(1, -1), (n_envs, 1))
    event_time_ids_array = jnp.tile(event_time_ids.reshape(1, -1), (n_envs, 1))

    reset_key, step_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_envs)
    step_keys = jax.random.split(step_key, (n_envs, n_steps))

    infos = ExperimentInfo(
        t=jnp.tile(env_params.events.t.reshape(1, -1), (n_envs, 1)),
        is_treat=is_treat,
        key=step_keys,
    )

    def scanner(obs_and_states_initial_batched_arg, global_batched_infos):
        
        # This function is vmapped. It processes one batch element over one chunk of steps.
        def scan_fn_for_one_batch_element_one_chunk(carry_one_batch_element_obs_state, infos_one_batch_element_one_chunk):
            # carry_one_batch_element_obs_state is (obs_i, state_i) for the i-th batch element.
            # infos_one_batch_element_one_chunk is an ExperimentInfo instance where each field
            # (e.g., .t, .key) has shape (current_chunk_length,) for the i-th batch element.
            
            # jax.lax.scan iterates over the leading axis of infos_one_batch_element_one_chunk.
            # stepper is partially applied with (env, env_params, A, B) from the outer scope.
            # stepper expects: carry=(obs, state), info=ExperimentInfo_for_one_step.
            final_carry_obs_state_one_batch_element, collected_outputs_one_batch_element_chunk = jax.lax.scan(
                partial(stepper, env, env_params, A, B),
                carry_one_batch_element_obs_state,
                infos_one_batch_element_one_chunk, # This is 'xs' for the inner scan.
            )
            # stepper returns ((new_obs, new_state), (reward, is_treat))
            # final_carry_obs_state_one_batch_element is (obs_final, state_final)
            # collected_outputs_one_batch_element_chunk is (rewards_chunk_array, is_treat_chunk_array)
            return final_carry_obs_state_one_batch_element, collected_outputs_one_batch_element_chunk

        # Vmap scan_fn_for_one_batch_element_one_chunk to run in parallel for all batch elements.
        vmapped_chunk_processor = jax.vmap(
            scan_fn_for_one_batch_element_one_chunk,
            in_axes=(0, 0), # Process 0-th axis of carry_batched and 0-th axis of infos_batched_for_chunk.
            out_axes=0      # The output is also batched along axis 0.
        )

        current_carry_batched_obs_state = obs_and_states_initial_batched_arg

        n_steps_total = global_batched_infos.t.shape[1] # Renamed from n_steps to avoid conflict
        # chunk_size is from the outer scope (run_trials parameter)
        n_chunks = (n_steps_total + chunk_size - 1) // chunk_size

        all_rewards_batched_chunks = []
        all_is_treat_batched_chunks = []
        all_n_available_batched_chunks = []

        for chunk_idx in tqdm(range(n_chunks), desc="Processing Chunks"):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n_steps_total)

            # Slice global_batched_infos to get data for the current chunk across all batch elements.
            # Each field in current_chunk_infos_batched will have shape (batch_size, current_chunk_length).
            current_chunk_infos_batched = ExperimentInfo(
                t=global_batched_infos.t[:, chunk_start:chunk_end],
                is_treat=global_batched_infos.is_treat[:, chunk_start:chunk_end],
                key=global_batched_infos.key[:, chunk_start:chunk_end],
            )
            
            # Apply the vmapped function to process the current chunk for all batch elements.
            current_carry_batched_obs_state, current_chunk_outputs_batched = vmapped_chunk_processor(
                current_carry_batched_obs_state,
                current_chunk_infos_batched
            )
            current_carry_batched_obs_state[0].block_until_ready() # obs part of the carry

            # current_chunk_outputs_batched is (rewards_batched_for_chunk, is_treat_batched_for_chunk)
            # each has shape (batch_size, current_chunk_length)
            all_rewards_batched_chunks.append(current_chunk_outputs_batched[0])
            all_is_treat_batched_chunks.append(current_chunk_outputs_batched[1])
            all_n_available_batched_chunks.append(current_chunk_outputs_batched[2])
            # Optional: Add current_full_carry_batched.block_until_ready() here if needed for debugging,
            # especially for memory profiling or ensuring sequential execution visibility.

        # Concatenate results from all chunks
        final_rewards_batched = jnp.concatenate(all_rewards_batched_chunks, axis=1)
        final_is_treat_batched = jnp.concatenate(all_is_treat_batched_chunks, axis=1)
        final_n_available_batched = jnp.concatenate(all_n_available_batched_chunks, axis=1)
        
        return final_rewards_batched, final_is_treat_batched, final_n_available_batched

    obs_and_states_initial_batched = jax.vmap(env.reset, in_axes=(0, None))(
        reset_keys, env_params
    )

    rewards_over_time, is_treat_over_time, n_available_over_time = scanner(obs_and_states_initial_batched, infos)

    # if design['name'] == "switchback":
    #     return rewards_over_time, is_treat_over_time, n_available_over_time, time_ids_array, event_time_ids_array
    
    return rewards_over_time, is_treat_over_time, n_available_over_time, time_ids_array, event_time_ids_array


def load_spatial_clusters():
    zones = pd.read_parquet("taxi-zones.parquet")
    unq_zones, unq_zone_ids = np.unique(zones["zone"], return_inverse=True)
    zones["zone_id"] = unq_zone_ids
    nodes = pd.read_parquet("manhattan-nodes.parquet")
    nodes["lng"] = nodes["lng"].astype(float)
    nodes["lat"] = nodes["lat"].astype(float)
    nodes_zones = nodes.merge(zones, on="osmid")

    centroids = nodes_zones.groupby("zone_id").aggregate(
        {"lat": "mean", "lng": "mean"}
    )
    dist = np.zeros((len(centroids), len(centroids)))
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            dist[i, j] = haversine.haversine(
                (centroids.iloc[i]["lat"], centroids.iloc[i]["lng"]),
                (centroids.iloc[j]["lat"], centroids.iloc[j]["lng"]),
            )
    return nodes_zones, dist


@ex.automain
def main(
    n_cars,
    w_price,
    w_eta,
    w_intercept,
    n_events,
    seed,
    k,
    batch_size,
    switch_every,
    p,
    chunk_size,
    design,
    output,
    config_output,
    _config,
):
    logging.info("Starting simulation with parameters: %s", _config)
    key = jax.random.PRNGKey(seed)
    env = ManhattanRidesharePricing(n_cars=n_cars, n_events=n_events)
    env_params = env.default_params
    env_params = env_params.replace(
        w_price=w_price, w_eta=w_eta, w_intercept=w_intercept
    )

    # print("First 10 event times (after normalization):", env_params.events.t[:10])
    # print("Max timestamp:", env_params.events.t.max())
    # print("Min timestamp:", env_params.events.t.min())
    # print("Number of unique timestamps:", jnp.unique(env_params.events.t).shape[0])


    nodes_zones, zone_dists = load_spatial_clusters()
    logging.info("Loaded spatial clusters and distances.")

    A = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.01)
    B = SimplePricingPolicy(n_cars=env.n_cars, price_per_distance=0.02)
    logging.info("Initialized pricing policies.")

    print(
        "Simulation time (mins)",
        (env_params.events.t.max() - env_params.events.t[5]) / 60,
    )
    print(
        "Simulation time (hrs)",
        (env_params.events.t.max() - env_params.events.t[5]) / 3600,
    )

    all_results = []
    keys = jax.random.split(key, (k-1) // batch_size + 1)
    for i, key_batch in enumerate(keys):
        logging.info("Starting batch %d/%d", i + 1, len(keys))
        rewards_batch, is_treat_batch, n_available_batch, time_ids_batch, event_time_ids_batch = run_trials(
            env,
            env_params,
            A,
            B,
            key_batch,
            n_envs=batch_size,
            n_steps=n_events,
            p=p,
            chunk_size=chunk_size,
            design=design,
        )
        
        # Convert JAX arrays to NumPy for pandas processing
        rewards_batch_np = np.array(rewards_batch)
        is_treat_batch_np = np.array(is_treat_batch)
        time_ids_batch_np = np.array(time_ids_batch)
        event_time_ids_batch_np = np.array(event_time_ids_batch)
        n_available_batch_np = np.array(n_available_batch)

        # Process results for this batch
        for env_idx in range(batch_size):
            for step_idx in range(n_events):
                all_results.append({
                    'global_trial_id': i * batch_size + env_idx,
                    'step_in_trial': step_idx,
                    'reward': rewards_batch_np[env_idx, step_idx],
                    'is_treat': bool(is_treat_batch_np[env_idx, step_idx]),
                    'time_id': time_ids_batch_np[env_idx, step_idx],
                    'event_time_id': event_time_ids_batch_np[env_idx, step_idx],
                    'n_available': n_available_batch_np[env_idx, step_idx]
                })
        logging.info("Completed batch %d/%d", i + 1, len(keys))

    ### mkdir
    import os
    os.makedirs(os.path.dirname(output), exist_ok=True)
    os.makedirs(os.path.dirname(config_output), exist_ok=True)

    if design["name"] == "switchback":
        config_output = config_output + f"_{design['switch_every']}"
        output = output + f"_{design['switch_every']}"

    pd.DataFrame.from_dict([_config]).to_csv(
        config_output + f"_{n_events}.csv", index=False,
    )
    # results_df = pd.concat(map(pd.DataFrame, all_results))
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output + f"_{n_events}.csv", index=False)
    logging.info("Simulation completed.")
