import jax 
import jax.numpy as jnp
import optax

from score import stein_score_all, kde_score_all, hybrid_score_all
from utils import v_y, gof_statistic, score_loss, derivative_complexity


def loss(params, model, cause_data, effect_data, s_cause, sxy, loss_pos = "squared", lam_l2 = 0.0001, lam_complexity = 0.0001, complexity_order = 1):
    """
    Computes the GoF score loss.
    model: callable, velocity model v(y, t; params)
    loss_pos: squared for squared loss, abs for absolute loss
    """
    v_ = v_y(model, params, cause_data)
    loss = score_loss(v_, effect_data, s_cause, sxy, loss_pos = loss_pos)
    for param in params:
        if "l2" in param:
            l = jnp.mean((params[param])**2)
            loss += lam_l2 * l
    dc = derivative_complexity(model, params, y_data = effect_data, t_data = cause_data, order = complexity_order)
    loss += lam_complexity * dc
    return loss

def fit(params_init: optax.Params, model, 
        cause_data, effect_data, s_cause, sxy, 
        cond_data_val, target_data_val, s_cause_val, sxy_val,
        loss_pos = "squared", 
        lam_l2 = 0.0001,
        lam_complexity = 0.0001,
        complexity_order = 1,
        verbose = False,
        lr = 0.1,
        n_steps = 100,
        optimizer = "adam",
        tol = 1e-6,
        **kwargs) -> optax.Params:
    """
    Fit velocity model.
    """
    assert optimizer in ["adam", "sgd"], f"Optimizer must be either 'adam' or 'sgd', got {optimizer}."
    if optimizer == "adam":
        optimizer = optax.adam(lr)
    elif optimizer == "sgd":
        optimizer = optax.sgd(lr)
    
    opt_state = optimizer.init(params_init)

    loss_train = lambda params: loss(params, model, cause_data, effect_data, s_cause, sxy, loss_pos, lam_l2, lam_complexity, complexity_order) 
    loss_train = jax.jit(loss_train)
    loss_val = lambda params: loss(params, model, cond_data_val, target_data_val, s_cause_val, sxy_val, loss_pos, lam_l2, lam_complexity, complexity_order)
    loss_val = jax.jit(loss_val)

    def step(params, opt, opt_state):
        _, grads = jax.value_and_grad(loss_train)(params)#, model, cause_data, effect_data, s_cause, sxy, loss_pos, lam_l2, lam_complexity)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        val_loss = loss_val(params)#, model, cond_data_val, target_data_val, s_cause_val, sxy_val, loss_pos, lam_l2, lam_complexity)
        return params, opt_state, val_loss
    
    best_loss = float('inf')
    loss_prev = float('inf')
    params = params_init.copy()
    best_params = params.copy()
    final_steps = n_steps
    
    for _ in (range(n_steps)):
        params, opt_state, val_loss = step(params, optimizer, opt_state)
        if verbose:
            print(f'step {_}, validation loss: {val_loss}', end="\r",flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params.copy()
            if verbose:
                print(f"Saved parameters at step {_} with validation loss {best_loss}", end="\r",flush=True)
        increment = abs(loss_prev - val_loss)
        loss_prev = val_loss
        if increment < tol:
            final_steps = _ + 1
            if verbose:
                print(f"Converged at step {_} with validation loss {val_loss}")
            break


    return best_loss, best_params, final_steps

def fit_sm_bd(params_init: optax.Params, model, x_data, y_data, 
              reg = 0.1, 
              score = "stein", 
              score_kernel = "gauss",
              gof = "sq", 
              sx_data = None,
              sy_data = None,
              sxy_data = None,
              x_data_test = None,
              y_data_test = None,
              sx_data_test = None,
              sy_data_test = None,
              sxy_data_test = None,
              **kwargs):
    """
    Optimize the parameters of the velocity with score matching in both directions
    return scores, complexity, and best parameters of both directions. 
    """

    if kwargs.get("bandwidth_factor") is not None:
        bandwidth_factor = kwargs.get("bandwidth_factor")
    else:
        bandwidth_factor = 1.0

    assert score in ["stein", "hybrid", "kde"], f"Score must be 'stein' 'hybrid' or 'kde', got {score}."
    assert gof in ["sq", "raw"], f"GoF must be either 'sq' or 'raw', got {gof}."    

    if any (x is None for x in [sx_data, sy_data, sxy_data]):
        if score == "stein":
            sx_data, sy_data, sxy_data = stein_score_all(x_data, y_data, reg = reg, score_kernel = score_kernel, bandwidth_factor = bandwidth_factor)
        elif score == "kde":
            sx_data, sy_data, sxy_data = kde_score_all(x_data, y_data, reg = reg, score_kernel = score_kernel, bandwidth_factor = bandwidth_factor)
        elif score == "hybrid":
            sx_data, sy_data, sxy_data = hybrid_score_all(x_data, y_data, reg = reg, score_kernel = score_kernel, bandwidth_factor = bandwidth_factor)

    n_samples = x_data.shape[0]

    x_data_train, x_data_val = x_data, x_data
    y_data_train, y_data_val = y_data, y_data
    sx_data_train, sx_data_val = sx_data, sx_data
    sy_data_train, sy_data_val = sy_data, sy_data
    sxy_data_train, sxy_data_val = sxy_data, sxy_data
    if x_data_test is None:
        x_data_test = x_data 
    if y_data_test is None:
        y_data_test = y_data
    if sx_data_test is None:
        sx_data_test = sx_data
    if sy_data_test is None:
        sy_data_test = sy_data
    if sxy_data_test is None:
        sxy_data_test = sxy_data

    ## forward model

    params_xy = params_init.copy()
    params_yx = params_init.copy()

    best_loss_xy, best_params_xy, steps_xy = fit(params_xy, model,
                                       x_data_train, y_data_train, sx_data_train, sxy_data_train,
                                       x_data_val, y_data_val, sx_data_val, sxy_data_val, 
                                       **kwargs)
    best_loss_yx, best_params_yx, steps_yx = fit(params_yx, model, 
                                       y_data_train, x_data_train, sy_data_train, sxy_data_train[:, [1,0]], 
                                       y_data_val, x_data_val, sy_data_val, sxy_data_val[:, [1,0]], 
                                       **kwargs)
    
    v_xy = v_y(model, best_params_xy, x_data_test)
    v_yx = v_y(model, best_params_yx, y_data_test)  

    gof_xy = gof_statistic(v_xy, y_data_test, sx_data_test, sxy_data_test)
    gof_yx = gof_statistic(v_yx, x_data_test, sy_data_test, sxy_data_test[:, [1,0]])

    ## evaluate 2nd derivative complexity

    complexity_xy = derivative_complexity(model, best_params_xy, y_data = y_data_test, t_data = x_data_test)
    complexity_yx = derivative_complexity(model, best_params_yx, y_data = x_data_test, t_data = y_data_test)

    if gof == "sq":
        score_xy = jnp.mean(gof_xy**2)
        score_yx = jnp.mean(gof_yx**2)
    elif gof == "raw": 
        score_xy  = jnp.mean(jnp.abs(gof_xy))
        score_yx = jnp.mean(jnp.abs(gof_yx))

    return score_xy, score_yx, complexity_xy, complexity_yx, best_params_xy, best_params_yx, steps_xy, steps_yx

