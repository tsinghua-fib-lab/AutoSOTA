import jax
import jax.numpy as jnp


def batch_drift_fn(drift_fn, xs, ts):
    
    batch_drift = jax.vmap(drift_fn, in_axes=(0, None))  # (B, d), scalar t -> (B, d)
    def step(carry, inputs):
        x_t, t = inputs              # x_t: (B, d), t: ()
        y_t = batch_drift(x_t, t)    # (B, d)
        return carry, y_t
    # Make time the leading axis for scanning
    xs_time_major = jnp.swapaxes(xs, 0, 1)  # (N, B, d)
    # Sequential over time via scan
    _, fn_evals = jax.lax.scan(step, None, (xs_time_major, ts))  # ys: (N, B, d)

    return jnp.swapaxes(fn_evals, 0, 1)

def compute_KL_to_ground_truth(key, bridge, true_drift_fn, learned_drift_fn, sigma, T, num_steps=1000, endpoint_idx = -1):
    num_trajs = 1000
    sample_keys = jax.random.split(key, num_trajs)
    ts = jnp.linspace(0, T, num_steps + 1)  # time steps for the trajectory
    dt = T / num_steps

    true_x_traj, true_drift_traj, dBt_traj = jax.vmap(
            bridge.sample_sde,
            in_axes=(0, None, None)
            )(sample_keys, true_drift_fn, num_steps) # (B,N+1,d), (B,N,d), (B,N,d)
    
    learned_drift_vals = batch_drift_fn(learned_drift_fn, true_x_traj, ts)  # (B,N,d)
    
    true_drift_traj = true_drift_traj[:,:endpoint_idx]
    learned_drift_vals = learned_drift_vals[:,:endpoint_idx]

    diffs = true_drift_traj - learned_drift_vals  # (B,N,d)

    sigma_us = diffs / sigma

    L2_per_step = 0.5 * jnp.sum(sigma_us**2, axis=(0,-1)) / num_trajs

    KL_per_traj = 0.5 * jnp.sum(sigma_us**2, axis=(-2,-1)) * dt
    KL_divergence = jnp.mean(KL_per_traj)
    return KL_divergence, L2_per_step

def compute_KL_to_reference(key, bridge, drift_fn, base_drift_fn, sigma, T, num_steps=1000, endpoint_idx=-1):
    num_trajs = 1000
    sample_keys = jax.random.split(key, num_trajs)
    ts = jnp.linspace(0, T, num_steps + 1)  # time steps for the trajectory
    dt = T / num_steps

    x_traj, drift_traj, dBt_traj = jax.vmap(
            bridge.sample_sde,
            in_axes=(0, None, None)
            )(sample_keys, drift_fn, num_steps) # (B,N+1,d), (B,N,d), (B,N,d)
    
    base_drift_vals = batch_drift_fn(base_drift_fn, x_traj, ts)  # (B,N,d)

    diffs = drift_traj[:,:endpoint_idx] - base_drift_vals[:,:endpoint_idx]  # (B,N,d)
    sigma_us = diffs / sigma

    L2_per_step = 0.5 * jnp.sum(sigma_us**2, axis=(0,-1)) / num_trajs

    KL_per_traj = 0.5 * jnp.sum(sigma_us**2, axis=(-2,-1)) * dt
    KL_divergence = jnp.mean(KL_per_traj)
    return KL_divergence, L2_per_step


def compute_KL_to_reference_2d(key, bridge, drift_fn, base_drift_fn, sigma, T, num_steps=1000, endpoint_idx=-1):
    num_trajs = 1000
    sample_keys = jax.random.split(key, num_trajs)
    ts = jnp.linspace(0, T, num_steps + 1)  # time steps for the trajectory
    dt = T / num_steps

    x_traj, drift_traj, dBt_traj = jax.vmap(
            bridge.sample_sde,
            in_axes=(0, None, None)
            )(sample_keys, drift_fn, num_steps) # (B,N+1,d), (B,N,d), (B,N,d)
    
    base_drift_vals = batch_drift_fn(base_drift_fn, x_traj, ts)  # (B,N,d)

    diffs = drift_traj[:,:endpoint_idx] - base_drift_vals[:,:endpoint_idx]  # (B,N,d)
    sigma_us = diffs / sigma

    L2_per_step = 0.5 * jnp.sum(sigma_us**2, axis=(0,-1)) / num_trajs

    KL_per_traj = 0.5 * jnp.sum(sigma_us[:,:,:2]**2, axis=(-2,-1)) * dt
    KL_divergence = jnp.mean(KL_per_traj)
    return KL_divergence, L2_per_step






def compute_KL_to_ground_truth_sigma_fn(key, bridge, true_drift_fn, learned_drift_fn, sigma_fn, T, num_steps=1000, endpoint_idx = -1):
    num_trajs = 1000
    sample_keys = jax.random.split(key, num_trajs)
    ts = jnp.linspace(0, T, num_steps + 1)  # time steps for the trajectory
    dt = T / num_steps

    true_x_traj, true_drift_traj, dBt_traj = jax.vmap(
            bridge.sample_sde,
            in_axes=(0, None, None)
            )(sample_keys, true_drift_fn, num_steps) # (B,N+1,d), (B,N,d), (B,N,d)
    
    learned_drift_vals = batch_drift_fn(learned_drift_fn, true_x_traj, ts)  # (B,N,d)
    
    true_drift_traj = true_drift_traj[:,:endpoint_idx]
    learned_drift_vals = learned_drift_vals[:,:endpoint_idx]

    diffs = true_drift_traj - learned_drift_vals  # (B,N,d)

    time_map_sigma  = jax.vmap(sigma_fn, in_axes=(0, 0))          # map over time
    batch_map_sigma = jax.vmap(time_map_sigma,    in_axes=(0, None))      # then over batch
    sigmas = batch_map_sigma(true_x_traj[:, :endpoint_idx], ts[:endpoint_idx])          # (B, N, d)

    sigma_us = diffs / sigmas

    L2_per_step = 0.5 * jnp.sum(sigma_us**2, axis=(0,-1)) / num_trajs

    KL_per_traj = 0.5 * jnp.sum(sigma_us**2, axis=(-2,-1)) * dt
    KL_divergence = jnp.mean(KL_per_traj)
    return KL_divergence, L2_per_step

def compute_KL_to_reference_sigma_fn(key, bridge, drift_fn, base_drift_fn, sigma_fn, T, num_steps=1000, endpoint_idx=-1):
    num_trajs = 1000
    sample_keys = jax.random.split(key, num_trajs)
    ts = jnp.linspace(0, T, num_steps + 1)  # time steps for the trajectory
    dt = T / num_steps

    x_traj, drift_traj, dBt_traj = jax.vmap(
            bridge.sample_sde,
            in_axes=(0, None, None)
            )(sample_keys, drift_fn, num_steps) # (B,N+1,d), (B,N,d), (B,N,d)
    
    base_drift_vals = batch_drift_fn(base_drift_fn, x_traj, ts)  # (B,N,d)

    diffs = drift_traj[:,:endpoint_idx] - base_drift_vals[:,:endpoint_idx]  # (B,N,d)
    
    time_map_sigma  = jax.vmap(sigma_fn, in_axes=(0, 0))          # map over time
    batch_map_sigma = jax.vmap(time_map_sigma,    in_axes=(0, None))      # then over batch
    sigmas = batch_map_sigma(x_traj[:, :endpoint_idx], ts[:endpoint_idx])          # (B, N, d)

    sigma_us = diffs / sigmas

    L2_per_step = 0.5 * jnp.sum(sigma_us**2, axis=(0,-1)) / num_trajs

    KL_per_traj = 0.5 * jnp.sum(sigma_us**2, axis=(-2,-1)) * dt
    KL_divergence = jnp.mean(KL_per_traj)
    return KL_divergence, L2_per_step