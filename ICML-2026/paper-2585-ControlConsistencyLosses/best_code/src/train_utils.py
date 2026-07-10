import jax
import jax.numpy as jnp

from .compute_targets import _calculate_targets_fn_standard, _calculate_targets_fn_nodrift


def _get_sigma_fn(sigma, dim):
    """
    Returns a callable sigma_fn(x,t) -> (apply_sigma, sigma_val)
    where:
      - apply_sigma(dB_t) computes sigma @ dB_t efficiently
    """
    # --- normalize sigma to something callable
    if callable(sigma):
        def base_sigma_val(x,t):
            return sigma(x,t)
    else:
        def base_sigma_val(x,t):
            return sigma

    def sigma_fn(x, t):
        val = jnp.asarray(base_sigma_val(x,t))
        if val.ndim == 0:
            # scalar case: multiply directly
            s = val
            def apply_sigma(dB_t):
                return s * dB_t
            return s, apply_sigma

        elif val.ndim == 1:
            # diagonal case
            if val.shape[0] != dim:
                raise ValueError(f"sigma vector has shape {val.shape}, expected ({dim},)")
            s = val
            def apply_sigma(dB_t):
                return s * dB_t  # elementwise
            return s, apply_sigma

        elif val.ndim == 2:
            # full matrix
            if val.shape != (dim, dim):
                raise ValueError(f"sigma matrix has shape {val.shape}, expected ({dim}, {dim})")
            s = val
            def apply_sigma(dB_t):
                return s @ dB_t
            return s, apply_sigma

        else:
            raise ValueError("sigma must be scalar, vector (len d), or (d,d) matrix.")
        
    def a_inv_fn(x, t, v):
        # applies (sigma sigmaᵀ)^{-1} @ v. Handles scalar, diagonal, and full matrix cases.
        val = jnp.asarray(base_sigma_val(x, t))
        if val.ndim == 0:
            return v / (val**2 + 1e-8)
        elif val.ndim == 1:
            return v / (val**2 + 1e-8)
        else:
            a_matrix = val @ val.T
            eps = 1e-8
            return jnp.linalg.solve(a_matrix + eps*jnp.eye(a_matrix.shape[0]), v)
            # return jnp.linalg.solve(a_matrix, v)
    
    return sigma_fn, a_inv_fn


def _get_coeff(schedule_type, outer_step, num_outer_iterations):
    coeff = 1.0
    if schedule_type == 'linear':
        total_steps = num_outer_iterations
        # Linearly increase from near 0 to 1.0 over the outer iterations
        coeff = jnp.minimum(1.0, (outer_step + 1) / total_steps)
    elif schedule_type == 'sine':
        total_steps = num_outer_iterations
        # Cosine from near 0 to 1.0
        coeff = jnp.sin(0.5*jnp.pi * (outer_step + 1) / total_steps)
    elif schedule_type == 'linear_half':
        total_steps = num_outer_iterations
        half_steps = total_steps // 2
        # Linear increase from near 0 to 1.0 over the first half, then stay at 1.0
        coeff = jnp.where(
            outer_step < half_steps,
            (outer_step + 1) / half_steps,
            1.0
        )
    return coeff


def _get_sigma_scale(sigma_max_scale, outer_step, num_outer_iterations, base_sigma_fn, base_a_inv_fn):

    if sigma_max_scale is not None:
        # Decay from a scaled version of sigma to the base sigma, using a 'linear half' schedule
        total_steps = num_outer_iterations
        half_steps = total_steps // 2
        sigma_scale = jnp.where(
            outer_step < half_steps,
            sigma_max_scale + (1.0 - sigma_max_scale) * ((outer_step + 1) / half_steps),
            1.0
        )
        def sigma_fn(x, t):
            sigma_val, apply_sigma_fn = base_sigma_fn(x, t)
            return sigma_scale * sigma_val, lambda dB_t: sigma_scale * apply_sigma_fn(dB_t)
        
        def a_inv_fn(x, t, v):
            return base_a_inv_fn(x, t, v) / (sigma_scale ** 2)
    else:
        sigma_fn = base_sigma_fn
        a_inv_fn = base_a_inv_fn

    return sigma_fn, a_inv_fn


def _get_beta_scale(beta_schedule, T, train_config):
    if beta_schedule == 'average':
        B_ratio_fn = lambda x_t, t, x_t_plus_dt, t_plus_dt: (T - t_plus_dt) / (T - t)
    elif beta_schedule == 'endpoint':
        B_ratio_fn = lambda x_t, t, x_t_plus_dt, t_plus_dt: 1.0
    elif beta_schedule == 'next_step':
        B_ratio_fn = lambda x_t, t, x_t_plus_dt, t_plus_dt: 0.0
    elif beta_schedule == 'geom':
        B_ratio_fn = lambda x_t, t, x_t_plus_dt, t_plus_dt: train_config['B_ratio']
    elif beta_schedule == 'sqrt':
        B_ratio_fn = lambda x_t, t, x_t_plus_dt, t_plus_dt: jnp.power((T - t_plus_dt) / (T - t + 1e-8), 1.5)
    else:
        raise ValueError(f"Unknown beta_schedule {beta_schedule}")
    return B_ratio_fn

def _get_calculate_targets_fn(self_consistency_type):
    if self_consistency_type == 'standard':
        return _calculate_targets_fn_standard
    elif self_consistency_type == 'nodrift':
        return _calculate_targets_fn_nodrift
    else:
        raise ValueError(f"Unknown self_consistency type {self_consistency_type}")