'''
- Functions for computing the training targets backwards along the trajectory.
- Implemented for the standard self-consistency property (SC1), and the additive version (SC2).
'''

import jax
import jax.numpy as jnp

from functools import partial


def _calculate_targets_fn_standard(x_traj, ut_traj, dBt_traj, base_drift_fn, sigma_fn, sampler_fn, control_fn, reference_drift_fn, B_ratio_fn, shape, train_config, ts):
    """
    Computes the training targets by solving the backward equation. The B_ratio function determines the weighting over the timepoints (the beta schedule).
    """
    F1 = jnp.zeros(shape)
    ts = jnp.flip(ts, axis=0)

    partial_sampler_fn = partial(sampler_fn, drift_fn=base_drift_fn, sigma_fn=sigma_fn)

    def step(carry, inputs):
        F_t_plus_dt, t_plus_dt, x_t_plus_dt, u_t_plus_dt, dB_t_plus_t = carry
        t, x, u, dB_t = inputs
        dt = t_plus_dt - t

        if train_config.get('jacobian_method', 'euler') == 'euler':
            _, vjp_fun = jax.vjp(lambda x_arg: partial_sampler_fn(x_arg, t, dB_t, dt)[0], x)
            def right_multiply_jacobian(v):
                return vjp_fun(v)[0]

        if train_config['jacobian_method'] == 'exp':
            left_jac = jax.jacfwd(lambda x_arg: base_drift_fn(x_arg, t))(x)
            right_jac = jax.jacfwd(lambda x_arg: base_drift_fn(x_arg, t_plus_dt))(x_t_plus_dt)
            exp_jac = jax.scipy.linalg.expm(dt * left_jac)
            exp_jac = jax.scipy.linalg.expm(0.5 * dt * (left_jac + right_jac)) # trapezoidal
            def right_multiply_jacobian(v):
                return v @ exp_jac
        
        B_ratio = B_ratio_fn(x, t, x_t_plus_dt, t_plus_dt)
        F_t = right_multiply_jacobian(
            B_ratio* F_t_plus_dt + (1 - B_ratio) * u_t_plus_dt
        )

        if train_config.get('STL_adjustments', False) is True:
            # Applies the STL adjustments

            # (∇u) (sigma dB_t)
            def u_fun(x_arg):
                return control_fn(xs=x_arg[None], ts=t[None])[0]  # (d,)

            sigma_val, apply_sigma_fn = sigma_fn(x, t)
            noise_term = apply_sigma_fn(dB_t)  # = sigma(x,t) @ dB_t, shape (d,)
            _, STL_adjustment_first_term = jax.jvp(u_fun, (x,), (noise_term,))

            # (u) (∇sigma dB_t)
            def _sigma_times_dB(x_):
                _, apply_sigma_fn_ = sigma_fn(x_, t)
                return apply_sigma_fn_(dB_t)

            _, grad_sigma_vjp_fun = jax.vjp(_sigma_times_dB, x)
            STL_adjustment_second_term = grad_sigma_vjp_fun(u)[0]

            STL_adjustment = STL_adjustment_first_term + STL_adjustment_second_term
            F_t = F_t - STL_adjustment

        ratio = -1 # dummy value

        return (F_t, t, x, u, dB_t), (F_t, ratio)

    rev_uts = jnp.flip(ut_traj, axis=0)
    rev_xts = jnp.flip(x_traj, axis=0)
    rev_dB_ts = jnp.flip(dBt_traj, axis=0)

    terminal_ut = rev_uts[1]
    carry = (terminal_ut, ts[1], rev_xts[1], rev_uts[1], rev_dB_ts[1]) # the first ones aren't used (corresponding to endpoint)
    inputs = (ts[2:], rev_xts[2:], rev_uts[2:], rev_dB_ts[2:])
    _, (F_ts, ratios) = jax.lax.scan(step, carry, inputs)

    F_ts = jnp.vstack([F1[None, :], terminal_ut[None,:], F_ts])
    F_ts = jnp.flip(F_ts, axis=0)

    ratios = jnp.flip(ratios, axis=0)

    return F_ts, ratios



def _calculate_targets_fn_nodrift(x_traj, ut_traj, dBt_traj, base_drift_fn, sigma_fn, sampler_fn, control_fn, reference_drift_fn, B_ratio_fn, shape, train_config, ts):
    """
    Computes the training targets by solving the backward equation. The B_ratio function determines the weighting over the timepoints (the beta schedule).
    """
    F1 = jnp.zeros(shape)
    ts = jnp.flip(ts, axis=0)

    # perform the method using the combined drift
    def combined_drift_fn(x, t):
        return base_drift_fn(x, t) + reference_drift_fn(x, t)
    partial_sampler_fn = partial(sampler_fn, drift_fn=combined_drift_fn, sigma_fn=sigma_fn)

    def step(carry, inputs):
        F_t_plus_dt, t_plus_dt, x_t_plus_dt, u_t_plus_dt, dB_t_plus_t = carry
        t, x, u, dB_t = inputs
        dt = t_plus_dt - t
        
        B_ratio = B_ratio_fn(x, t, x_t_plus_dt, t_plus_dt)
        F_t =  B_ratio* F_t_plus_dt + (1 - B_ratio) * u_t_plus_dt

        if train_config.get('STL_adjustments', False) is True:
            # Applies the STL adjustments

            # (∇u) (sigma dB_t)
            def u_fun(x_arg):
                return control_fn(xs=x_arg[None], ts=t[None])[0]  # (d,)

            sigma_val, apply_sigma_fn = sigma_fn(x, t)
            noise_term = apply_sigma_fn(dB_t)  # = sigma(x,t) @ dB_t, shape (d,)
            _, STL_adjustment_first_term = jax.jvp(u_fun, (x,), (noise_term,))

            # (u) (∇sigma dB_t)
            def _sigma_times_dB(x_):
                _, apply_sigma_fn_ = sigma_fn(x_, t)
                return apply_sigma_fn_(dB_t)

            _, grad_sigma_vjp_fun = jax.vjp(_sigma_times_dB, x)
            STL_adjustment_second_term = grad_sigma_vjp_fun(u)[0]

            STL_adjustment = STL_adjustment_first_term + STL_adjustment_second_term
            F_t = F_t - STL_adjustment

        ratio = -1.0  # dummy value (not used in this version)

        # adjust for the running costs
        def f_fn(x_arg):
            return reference_drift_fn(x_arg,t)  # (d,)

        sigma_val, apply_sigma_fn = sigma_fn(x, t)
        _, f_vjp_fun = jax.vjp(f_fn, x)
        running_cost = f_vjp_fun( - u*dt)[0]
        F_t = F_t + running_cost


        return (F_t, t, x, u, dB_t), (F_t, ratio)

    rev_uts = jnp.flip(ut_traj, axis=0)
    rev_xts = jnp.flip(x_traj, axis=0)
    rev_dB_ts = jnp.flip(dBt_traj, axis=0)

    terminal_ut = rev_uts[1]
    carry = (terminal_ut, ts[1], rev_xts[1], rev_uts[1], rev_dB_ts[1]) # the first ones aren't used (corresponding to endpoint)
    inputs = (ts[2:], rev_xts[2:], rev_uts[2:], rev_dB_ts[2:])
    _, (F_ts, ratios) = jax.lax.scan(step, carry, inputs)

    F_ts = jnp.vstack([F1[None, :], terminal_ut[None,:], F_ts])
    F_ts = jnp.flip(F_ts, axis=0)

    ratios = jnp.flip(ratios, axis=0)

    return F_ts, ratios