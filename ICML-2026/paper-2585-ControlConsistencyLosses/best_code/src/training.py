'''
Define required training functions as pure functions (for jax.jit)
'''

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import flax
import flax.linen as nn
from functools import partial

from .train_utils import _get_beta_scale, _get_coeff, _get_sigma_scale, _get_calculate_targets_fn



def _controlled_drift_fn(params, x, t, model, optional_base_drift_fn, guiding_drift_fn, coeff_fn, sigma_fn):
    """
    Returns the controlled drift. This can optionally include a base drift term, and a 'decay' multiplier on the neural adjustment.
    """
    base_drift_val = optional_base_drift_fn(x, t)
    guiding_drift_val = guiding_drift_fn(x, t)
    _, apply_sigma_fn = sigma_fn(x,t)
    model_output = model.apply(params, x, t)
    neural_adjustment = coeff_fn(t) * apply_sigma_fn(model_output)
    return base_drift_val + guiding_drift_val + neural_adjustment

def _sample_sde_fn(key, drift_fn, sigma_fn, sampler_fn, x_0, shape, ts):
    """
    Samples the SDE for the given drift, using the given sampler function and time discretisation.
    """
    def step(carry, t):
        x, key, prev_t = carry
        key, subkey = jax.random.split(key)
        dt = t - prev_t
        dB_t = jax.random.normal(subkey, shape=shape) * jnp.sqrt(dt)
        x_next, drift = sampler_fn(x, prev_t, dB_t, dt, drift_fn, sigma_fn)
        output = (x_next, drift, dB_t)
        return (x_next, key, t), output

    carry = (x_0, key, ts[0])
    _, outputs = jax.lax.scan(step, carry, ts[1:])
    traj, drift_traj, dBt_traj = outputs
    
    traj = jnp.vstack([x_0[None, :], traj])
    drift_traj = jnp.vstack([drift_traj, drift_traj[-1][None, :]])   # repeat the final one, to make a consistent size. It is not used in the computation.
    dBt_traj = jnp.vstack([dBt_traj, dBt_traj[-1][None, :]])

    return traj, drift_traj, dBt_traj

def _sample_controlled_sde_fn(key, params, model, optional_base_drift_fn, guiding_drift_fn, coeff_fn, sigma_fn, sampler_fn, x_0, shape, ts):
    """Sample trajectory using the controlled SDE."""
    drift_fn = partial(_controlled_drift_fn,
                       params,
                       model=model,
                       optional_base_drift_fn=optional_base_drift_fn,
                       guiding_drift_fn=guiding_drift_fn,
                       coeff_fn=coeff_fn,
                       sigma_fn=sigma_fn)
    traj, drift_traj, dBt_traj = _sample_sde_fn(key, drift_fn, sigma_fn, sampler_fn, x_0, shape, ts)
    return traj, drift_traj, dBt_traj


def _control_fn(params, xs, ts, base_drift_fn, model, optional_base_drift_fn, guiding_drift, coeff_fn, sigma_fn, a_inv_fn):
    """Compute control u for given states and times."""
    bts = jax.vmap(base_drift_fn)(xs, ts)
    return _get_ut_from_params_fn(params, xs, bts, ts, model, optional_base_drift_fn, guiding_drift, coeff_fn, sigma_fn, a_inv_fn)


def _get_ut_from_params_fn(params, xs, bts, ts, model, optional_base_drift_fn, guiding_drift_fn, coeff_fn, sigma_fn, a_inv_fn):
    """Compute control u for given states and times, given the base drift values."""
    controlled_drift_vmap = jax.vmap(partial(_controlled_drift_fn, model=model, optional_base_drift_fn=optional_base_drift_fn, guiding_drift_fn=guiding_drift_fn, coeff_fn=coeff_fn, sigma_fn=sigma_fn), in_axes=(None, 0, 0))
    controlled_drifts = controlled_drift_vmap(params, xs, ts)
    residuals = controlled_drifts - bts
    ut = jax.vmap(a_inv_fn, in_axes=(0, 0, 0))(xs, ts, residuals)  # (B, d)
    return ut

def _loss_fn(params, batch, model, optional_base_drift_fn, guiding_drift_fn, coeff_fn, sigma_fn, a_inv_fn, train_config):
    """Pure loss function."""
    xs, bts, targets, ts = batch
    ut_preds = _get_ut_from_params_fn(params, xs, bts, ts, model, optional_base_drift_fn=optional_base_drift_fn, guiding_drift_fn=guiding_drift_fn, coeff_fn=coeff_fn, sigma_fn=sigma_fn, a_inv_fn=a_inv_fn) # (B, d)

    sq_diffs = (ut_preds - targets) ** 2
    per_sample_loss = jnp.sum(sq_diffs, axis=-1)  # (B,)

    loss_clip = float(train_config.get('loss_clip', 1e6))
    per_sample_loss = jnp.minimum(per_sample_loss, loss_clip)  # clip each element
    loss = jnp.mean(per_sample_loss)
    return loss

def _train_step_fn(state, batch, model, optional_base_drift_fn, guiding_drift_fn, coeff_fn, sigma_fn, a_inv_fn, optimizer, train_config):
    """Pure training step function."""
    grad_fn = jax.value_and_grad(_loss_fn)
    loss, grads = grad_fn(state.params, batch, model, optional_base_drift_fn, guiding_drift_fn, coeff_fn, sigma_fn, a_inv_fn, train_config)
    
    updates, new_opt_state = optimizer.update(grads, state.opt_state)

    new_params = optax.apply_updates(state.params, updates)
    new_ema_params = optax.incremental_update(new_params, state.ema_params, state.ema_rate)

    # Also keep track of the exponential moving average of the gradients, for debugging
    is_first = (state.step == 0)
    new_ema_grads = jax.lax.cond(
        is_first,
        lambda _: grads,
        lambda _: optax.incremental_update(grads, state.ema_grads, state.grad_ema_rate),
        operand=None,
    )
    
    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        ema_params=new_ema_params,
        ema_grads=new_ema_grads,
        opt_state=new_opt_state,
    )
    return new_state, loss



# ==============================================================================
# Loops that can be scanned over
# ==============================================================================

def _inner_loop_body(carry, _, train_config, model, optional_base_drift_fn, guiding_drift_fn, coeff_fn, sigma_fn, a_inv_fn, optimizer):
    """
    Body of the inner training loop, performing multiple training steps on the same trajectories
    """
    state, traj_data, key = carry
    x_traj, bt_vals, targets, ts = traj_data

    step_key, next_key = jax.random.split(key)
    time_subkey, batch_subkey = jax.random.split(step_key)

    time_idxs = jax.random.choice(time_subkey, jnp.arange(train_config['num_steps']), shape=(train_config['train_batch_size'],), replace=True)

    batch_idxs = jax.random.choice(batch_subkey, jnp.arange(train_config['traj_batch_size']), shape=(train_config['train_batch_size'],), replace=True)

    batch = (
        x_traj[batch_idxs, time_idxs],
        bt_vals[batch_idxs, time_idxs],
        targets[batch_idxs, time_idxs],
        ts[time_idxs]
    )

    new_state, loss = _train_step_fn(state, batch, model, optional_base_drift_fn=optional_base_drift_fn, guiding_drift_fn=guiding_drift_fn, coeff_fn=coeff_fn, sigma_fn=sigma_fn, a_inv_fn=a_inv_fn, optimizer=optimizer, train_config=train_config)

    return (new_state, traj_data, next_key), loss

def _outer_loop_body(state, key, outer_step, train_config, static_bridge_data, optimizer):
    """Body of the outer training loop"""

    # =======================================================
    # Set up the required bridge objects
    # =======================================================
    model = static_bridge_data['model']
    x_0, x_T = static_bridge_data['x_0'], static_bridge_data['x_T']
    T = static_bridge_data['T']
    shape = static_bridge_data['shape']
    ts = state.ts  # Currently fixed time discretisation, but could be changed in future versions

    # Optionally anneal the drift coefficient - default to 1.0
    schedule_type = train_config.get('coeff_schedule', None)
    coeff = _get_coeff(schedule_type, outer_step, train_config['num_outer_iterations'])

    # Create a scaled version of the base drift function for this iteration
    unscaled_base_drift_fn = static_bridge_data['base_drift_fn']
    base_drift_fn = lambda x, t: coeff * unscaled_base_drift_fn(x, t)

    # construct objects used in the bridge parameterisation
    unscaled_optional_base_drift_fn = static_bridge_data['optional_base_drift_fn']
    optional_base_drift_fn = lambda x, t: coeff * unscaled_optional_base_drift_fn(x, t)
    guiding_drift_fn = static_bridge_data['guiding_drift_fn']
    coeff_fn = static_bridge_data['coeff_fn']
    base_sigma_fn = static_bridge_data['sigma_fn']
    base_a_inv_fn = static_bridge_data['a_inv_fn']
    sampler_fn = static_bridge_data['sampler_fn']

    # reference drift
    unscaled_reference_drift_fn = static_bridge_data['reference_drift_fn']
    reference_drift_fn = lambda x, t: coeff * unscaled_reference_drift_fn(x, t)

    # optionally anneal the noise level
    sigma_max_scale = train_config.get('sigma_max_scale', None)
    sigma_fn, a_inv_fn = _get_sigma_scale(sigma_max_scale, outer_step, train_config['num_outer_iterations'], base_sigma_fn, base_a_inv_fn)

    # set the beta schedule
    beta_schedule = train_config.get('beta_schedule', 'average')
    B_ratio_fn = _get_beta_scale(beta_schedule, T, train_config)
    
    # select which self-consistency property to use
    self_consistency = train_config.get('self_consistency', 'standard')
    _calculate_targets_fn = _get_calculate_targets_fn(self_consistency)


    # ====================================================
    # Sample the trajectories, compute the controls and correct terminal control
    # ====================================================
    sample_key, inner_loop_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, train_config['traj_batch_size'])
    vmap_sample_fn = jax.vmap(
        _sample_controlled_sde_fn, in_axes=(0, None, None, None, None, None, None, None, None, None, None)
    )
    x_traj, drift_traj, dBt_traj = vmap_sample_fn(
        sample_keys, state.ema_params, model, optional_base_drift_fn, guiding_drift_fn, coeff_fn, sigma_fn, sampler_fn, x_0, shape, ts
    )

    # calculate the controls
    vmap_base_drift = jax.vmap(jax.vmap(base_drift_fn, in_axes=(0, None)), in_axes=(1, 0))
    bt_vals = vmap_base_drift(x_traj, ts).transpose(1, 0, 2)    # (B, num_steps+1, d)
    residuals = drift_traj - bt_vals  # (B, num_steps+1, d)
    vmap_apply_inv_inner = jax.vmap(a_inv_fn, in_axes=(0, None, 0))   # (B,d) x scalar t -> (B,d)
    vmap_apply_inv = jax.vmap(vmap_apply_inv_inner, in_axes=(1, 0, 1))   # over time, feeds scalar t
    ut_traj = vmap_apply_inv(x_traj, ts, residuals).transpose(1, 0, 2)   # (B, T+1, d)

    # set terminal conditions - the final discretisation step is enforced exactly
    terminal_Brownian_drift = (x_T - x_traj[:, -2, :]) / (ts[-1] - ts[-2] + 1e-8)  # (B, d)
    terminal_residuals = terminal_Brownian_drift - bt_vals[:, -2, :]
    terminal_uts = jax.vmap(a_inv_fn, in_axes=(0, None, 0))(
            x_traj[:, -2, :], ts[-2], terminal_residuals
        )  # (B, d)
    # set the final two controls to the terminal control, only the penultimate is used for training
    ut_traj = ut_traj.at[:, -1, :].set(terminal_uts)
    ut_traj = ut_traj.at[:, -2, :].set(terminal_uts)

    # =========================================================
    # Compute the training targets by solving the backward equation
    # =========================================================

    control_fn = partial(_control_fn, params=state.ema_params, model=model, base_drift_fn=base_drift_fn,optional_base_drift_fn=optional_base_drift_fn, guiding_drift=guiding_drift_fn, coeff_fn=coeff_fn, sigma_fn=sigma_fn, a_inv_fn=a_inv_fn)

    vmap_targets_fn = jax.vmap(_calculate_targets_fn, in_axes=(0, 0, 0, None, None, None, None, None, None, None, None, None))
    training_targets, ratios = vmap_targets_fn(x_traj, ut_traj, dBt_traj, base_drift_fn, sigma_fn, sampler_fn, control_fn, reference_drift_fn, B_ratio_fn, shape, train_config, ts)
    
    traj_data = (x_traj, bt_vals, training_targets, ts)

    # =========================================================
    # Run inner training loops
    # =========================================================

    initial_inner_carry = (state, traj_data, inner_loop_key)

    partial_inner_loop_body = partial(
        _inner_loop_body, train_config=train_config, model=model, optional_base_drift_fn=optional_base_drift_fn, guiding_drift_fn=guiding_drift_fn, coeff_fn=coeff_fn, sigma_fn=sigma_fn, a_inv_fn=a_inv_fn, optimizer=optimizer
    )

    # Scan over the inner loop
    final_inner_carry, losses = jax.lax.scan(
        partial_inner_loop_body,
        init=initial_inner_carry,
        xs=None,
        length=train_config['num_inner_iterations']
    )

    # Unpack the final state from the carry
    final_state, _, _ = final_inner_carry
    
    outputs = {
        'ema_params': final_state.ema_params,
        'mean_loss': jnp.mean(losses),
        'x_traj': x_traj,
    }

    return final_state, outputs