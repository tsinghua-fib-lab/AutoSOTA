"""
Run stochastic score matching to fit a Torus Graph model.

"""
__date__ = "January 2024 - September 2025"

import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, value_and_grad, vjp
import optax
from tqdm import tqdm

from .stats import get_H_hat, flat_stats, get_stats


@jit
def get_group_l1(phi):
    d = int((len(phi) // 2) ** 0.5)
    phi = phi.reshape(d, d, 2)
    l1 = jnp.power(phi, 2)
    l1 = l1 + jnp.transpose(l1, (1, 0, 2))
    l1 = l1[jnp.tril_indices(d, -1)]
    l1 = jnp.sqrt(jnp.sum(l1, axis=1))
    return l1


@jit
def loss_func(x, phi, H_hat, l2_reg, l1_reg):
    linear_term = -jnp.inner(phi, H_hat)
    _, f_vjp = vjp(flat_stats, x)
    (jac_term,) = f_vjp(phi)
    jac_term = 0.5 * jnp.sum(jnp.power(jac_term, 2))
    l2_loss = 0.5 * l2_reg * jnp.sum(jnp.power(phi, 2))
    l1_loss = l1_reg * jnp.sum(get_group_l1(phi))
    return linear_term + jac_term + l2_loss + l1_loss


vec_loss_func = vmap(loss_func, in_axes=[0, None, None, None, None])
"""Vectorized loss function"""

@jit
def loss_func_no_h(x, phi, l2_reg, l1_reg):
    d = x.shape[0]
    coeff = (2 * jnp.ones((d, d)) - jnp.eye(d)).reshape(d, d, 1)
    H_hat = (get_stats(x) * coeff).flatten()
    linear_term = -jnp.inner(phi, H_hat)
    _, f_vjp = vjp(flat_stats, x)
    (jac_term,) = f_vjp(phi)
    jac_term = 0.5 * jnp.sum(jnp.power(jac_term, 2))
    l2_loss = 0.5 * l2_reg * jnp.sum(jnp.power(phi, 2))
    l1_loss = l1_reg * jnp.sum(get_group_l1(phi))
    return linear_term + jac_term + l2_loss + l1_loss

vec_loss_func_no_h = vmap(loss_func_no_h, in_axes=[0, None, None, None])
"""Vectorized loss function"""


def estimate_params_ssm(
    key,
    X,
    H_hat=None,
    weights=None,
    phi=None,
    batch_size=64,
    n_iter=100,
    alpha=0.99,
    opt_state=None,
    l2_reg=0.0,
    l1_reg=0.0,
    replace=True,
    lr=3e-2,
    mode: str = "adam",
    no_improvement: int = None,
):
    """
    Run stochastic score matching to estimate the Torus Graph parameters.

    Parameters
    ----------
    key : jr.PRNGKey
    X : [n,d]
    H_Hat : None or jnp.ndarray, optional
    phi : None or jnp.ndarray, optional
        Shape : (2 d^2,)
    batch_size : int, optional
    n_iter : int, optional
    alpha : float, optional
    opt_state : None or Optax optimizer state, optional
    l2_reg : float, optional
    l1_reg : float, optional
    replace : bool, optional
    lr : float, optional

    Returns
    -------
    phi : jnp.ndarray
        Shape: (d,d,2)
    opt_state : Optax optimizer state
    """
    n, d = X.shape
    
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    
    if weights is None:
        weights = jnp.ones(n) / n
    else:
        weights = weights / jnp.sum(weights)

    if H_hat is None:
        H_hat = get_H_hat(X, weights=weights)
    
    if phi is None:
        phi = 1e-3 * jnp.ones(2 * X.shape[1] ** 2)
    else:
        phi = phi.reshape(-1)
    
    if mode == "adam":
        optimizer = optax.adam(lr)
    elif mode == "adamw":
        optimizer = optax.adamw(lr)
    elif mode == "sgd":
        optimizer = optax.sgd(lr)   
    elif mode == "sgd_momentum":
        optimizer = optax.sgd(lr, momentum=0.9, nesterov=False)
    elif mode == "adam_cosine":
        warmup = max(1, n_iter // 20) if n_iter > 20 else 0
        if warmup > 0 and warmup < n_iter:
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=lr,
                warmup_steps=warmup,
                decay_steps=n_iter,
                end_value=lr * 1e-3
            )
            optimizer = optax.adam(learning_rate=schedule)
        else:
            optimizer = optax.adam(lr)
    else:
        raise NotImplementedError(mode)

    if opt_state is None:
        opt_state = optimizer.init(phi)

    def batch_loss_func(Xb, phi, H_hat, w):
        w = w / (jnp.sum(w) + 1e-12)
        return jnp.sum(w * vec_loss_func(Xb, phi, H_hat, l2_reg, l1_reg))
    
    batch_val_grad = value_and_grad(batch_loss_func, argnums=1)

    # Loop with minibatches ...
    pbar = tqdm(range(1, n_iter + 1), desc=f"Optimizing phi: {np.nan}")
    smooth_val = None
    best_loss, last_improvement = jnp.inf, 0
    for i in pbar:
        key, subkey = jr.split(key)
        idx = jr.choice(subkey, n, shape=(batch_size,), replace=replace)
        Xb, wb = X[idx], weights[idx] # (b,d) (b,)

        val, grads = batch_val_grad(Xb, phi, H_hat, wb)
        if i == 1:
            smooth_val = val
        else:
            smooth_val = alpha * smooth_val + (1.0 - alpha) * val
        pbar.set_description(f"Optimizing phi: {smooth_val:.2f}")
        if mode == "adamw":
            updates, opt_state = optimizer.update(grads, opt_state, phi)
        else:
            updates, opt_state = optimizer.update(grads, opt_state)
        phi = optax.apply_updates(phi, updates)

        # Early stopping.
        if no_improvement is not None:
            if smooth_val < best_loss:
                best_loss = smooth_val
                last_improvement = i
            elif i - last_improvement > no_improvement:
                break
            

    return phi.reshape(d,d,2), opt_state



def estimate_params_ssm_with_loader(
    key,
    dataloader,
    phi=None,
    batch_size=64,
    n_iter=1000,
    alpha=0.99,
    opt_state=None,
    l2_reg=0.0,
    l1_reg=0.0,
    replace=True,
    lr=3e-2,
    transition_steps=250,
    true_phi = None,
):
    """
    Run stochastic score matching to estimate the Torus Graph parameters.

    Parameters
    ----------
    key : jr.PRNGKey
    dataloader: src.CWTPhaseLoader
    phi : None or jnp.ndarray, optional
    batch_size : int, optional
    n_iter : int, optional
    alpha : float, optional
    opt_state : None or Optax optimizer state, optional
    l2_reg : float, optional
    l1_reg : float, optional
    replace : bool, optional
    lr : float, optional

    Returns
    -------
    phi : jnp.ndarray
        Shape: (d,d,2)
    opt_state : Optax optimizer state
    """
    d = dataloader.C * len(dataloader.freqs)
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    if phi is None:
        phi = 1e-2 * jnp.ones(2 * d ** 2)
    else:
        phi = phi.flatten()

    if transition_steps is None:
        lr_schedule = None
    else:
        lr_schedule = optax.exponential_decay(
            init_value=lr,                      # initial learning rate
            transition_steps=transition_steps,  # decay every 1000 steps
            decay_rate=0.9,                     # decay by 10%
            staircase=True                      # use step-wise decay
        )

    optimizer = optax.adam(learning_rate=lr_schedule)
    if opt_state is None:
        opt_state = optimizer.init(phi)

    batch_loss_func = lambda Xb, phi: jnp.mean(
        vec_loss_func_no_h(Xb, phi, l2_reg, l1_reg)
    )
    batch_val_grad = value_and_grad(batch_loss_func, argnums=1)

    # Loop with minibatches ...
    pbar = tqdm(range(1, n_iter + 1), desc=f"Optimizing phi: {np.nan}")
    smooth_val = None
    losses = []
    for i in pbar:
        key, subkey = jr.split(key)
        Xb = next(dataloader)
        Xb = Xb.reshape(len(Xb), -1)
        val, grads = batch_val_grad(Xb, phi)
        if i == 1:
            smooth_val = val
        else:
            smooth_val = alpha * smooth_val + (1.0 - alpha) * val
        pbar.set_description(f"Optimizing phi: {smooth_val:.2f}")
        updates, opt_state = optimizer.update(grads, opt_state)
        phi = optax.apply_updates(phi, updates)

        losses.append(val)

    return phi.reshape(d,d,2), opt_state, losses
