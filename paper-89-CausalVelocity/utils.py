import numpy as np
import jax.numpy as jnp
import jax

from kernel import jnp_euclidean_cdist

def nullable_float(val):
    if not val:
        return None
    return float(val)

def nullable_int(val):
    if not val:
        return None
    return int(val)

def v_y(model, params, t):
    """
    returns v(y) for velocity model v(y,t; params)
    """
    return lambda _: model(_, t, params)

def gof_statistic(v_of_target, target_data, s_cause, sxy):
    """
    Computes the GoF statistic given the velocity (as a function of the target), data of the target, and the score estimates.
    """
    v_p, dvdy_p = jax.jvp(v_of_target, (target_data,), (jnp.ones_like(target_data),))
    gof = s_cause.squeeze() - dvdy_p - (sxy[:, 0] + v_p * sxy[:, 1])
    return gof

def score_loss(v_of_target, target_data, s_cause, sxy, loss_pos = "squared"):
    """
    Computes the loss corresponding to GoF statistic given the velocity (as a function of the target), data of the target, and the score estimates.
    """
    assert loss_pos in ["squared", "abs"], "loss_pos must be either 'squared' or 'abs'."
    gof = gof_statistic(v_of_target, target_data, s_cause, sxy)
    if loss_pos == "squared":
        return jnp.mean(gof ** 2)
    elif loss_pos == "abs":
        return jnp.mean(jnp.abs(gof))
    
def derivative_complexity(model, params, y_data, t_data, order = 1):
    """
    Computes the derivative complexity score.
    """
    v = lambda y, t: model(y, t, params)[0] ## v(y, t; params) = dy/dx 
    if order > 1:
        raise NotImplementedError("Higher order derivatives not yet implemented.")
    ddt = jax.grad(v, argnums = 1) ## dv/dt (y, t; params)
    ddy = jax.grad(v, argnums = 0) ## dv/dy (y, t; params)
    if y_data.ndim == 0:
        c = (ddt(y_data, t_data) + model(y_data, t_data, params) * ddy(y_data, t_data))**2
    else:
        ddt = jax.vmap(ddt)
        ddy = jax.vmap(ddy)
        c = jnp.mean((ddt(y_data, t_data) + model(y_data, t_data, params) * ddy(y_data, t_data))**2)
    return c
    

def standardize_data(x, return_statistics = False, trim_outliers = 0, trim_box = True, return_idx = False):
    """
    Standardize the data and trim outliers.
    If trim_box, trim according to statistics of x and y separately.
    Otherwise, trim according to pairwise distances. 
    """
    if x.shape[1] > 2:
        x = x[:, 0:2]

    if trim_outliers > 0:
        if trim_box:
            x_ = x[:, 0]
            y_ = x[:, 1]
            x_low, x_high = jnp.quantile(x_, jnp.array([trim_outliers/2, 1 - trim_outliers/2]))
            y_low, y_high = jnp.quantile(y_, jnp.array([trim_outliers/2, 1 - trim_outliers/2]))
            trim_idx = (x_ > x_low) & (x_ < x_high) & (y_ > y_low) & (y_ < y_high)
            x = x[trim_idx]
        else:
            dist = jnp_euclidean_cdist(x,x)
            dist = jnp.sum(dist, axis = 1) ## row sums
            dist_low, dist_high = jnp.quantile(dist, jnp.array([trim_outliers, 1 - trim_outliers]))
            trim_idx = (dist > dist_low) & (dist < dist_high)
            x = x[trim_idx]
    else:
        trim_idx = jnp.ones(x.shape[0], dtype = bool)

    mean = jnp.mean(x, axis = 0, keepdims = True)
    std = jnp.std(x, axis = 0, keepdims = True)
    if return_statistics:
        if return_idx:
            return (x - mean) / std, mean.squeeze(), std.squeeze(), trim_idx 
        else:
            return (x - mean) / std, mean.squeeze(), std.squeeze()
    else:
        if return_idx:
            return (x - mean) / std, trim_idx
        else:
            return (x - mean) / std

def l2_complexity(params, lam = 0.01):
    """
    L2 complexity of the parameters.
    """
    c = 0
    for param in params:
        if isinstance(params[param], dict):
            c += l2_complexity(params[param], lam = lam)
        else:
            c += lam * jnp.mean((params[param])**2)
    return c

def count_nparams(params):
    """
    Count the number of parameters in the model.
    """
    nparams = 0
    for param in params:
        if isinstance(params[param], dict):
            nparams += count_nparams(params[param])
        else:
            nparams += params[param].size
    return nparams