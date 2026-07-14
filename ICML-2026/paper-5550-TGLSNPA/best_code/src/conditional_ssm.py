"""
A simple conditional model.

"""

__date__ = "February 2024"


from jax import jit, vmap, value_and_grad, vjp
import jax.numpy as jnp
import jax.random as jr
import optax
from tqdm import tqdm


from .stats import get_H_hat, flat_stats


@jit
def get_group_l1(phi):
    """[d,d,2]"""
    l1 = jnp.concatenate([phi, jnp.transpose(phi, (1, 0, 2))], 2)
    l1 = jnp.linalg.norm(l1, axis=2)  # [d,d]
    l1 = 0.5 * (jnp.sum(l1) - jnp.trace(l1))
    return l1


@jit
def loss_func(x, y, params, l2_reg, l1_reg):
    """
    Calculate the score matching loss.

    Parameters
    ----------
    x : [m]
    y : [n]
    params : (W, B)
    l2_reg : float

    Returns
    -------
    loss : []
    """
    assert x.ndim == y.ndim == 1
    # Map x to y parameters.
    W, B = params  # [2m, 2n], [n,n,2]
    X_in = jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=0)  # [2m]
    X_out = X_in @ W  # [2n]
    alpha, beta = jnp.split(X_out, 2, axis=0)  # [n] [n]
    B = B.at[..., 0].add(jnp.diag(alpha))  # [n,n,2]
    B = B.at[..., 1].add(jnp.diag(beta))  # [n,n,2]
    phi = B.flatten()

    # Calculate the loss.
    H_hat = get_H_hat(y[None])
    linear_term = -jnp.inner(phi, H_hat)
    _, f_vjp = vjp(flat_stats, y)
    (jac_term,) = f_vjp(phi)
    jac_term = jnp.sum(jnp.power(jac_term, 2))
    l2_loss = l2_reg * jnp.sum(jnp.power(phi, 2))
    # l1_loss = l1_reg * jnp.sum(get_group_l1(B))
    return linear_term + 0.5 * (jac_term + l2_loss)  # + l1_loss


vec_loss_func = vmap(loss_func, in_axes=[0, 0, None, None, None])
"""Vectorized loss function"""


eval_loss_func = lambda Xb, Yb, params: jnp.mean(
    vec_loss_func(Xb, Yb, params, 0.0, 0.0)
)
"""No regularization"""


def fit_conditional_ssm(
    key,
    X,
    Y,
    batch_size=64,
    n_iter=100,
    alpha=0.99,
    opt_state=None,
    replace=True,
    lr=3e-2,
    l2_reg=0.0,
    l1_reg=0.0,
    verbose=False,
):
    """
    X : [w,m]
    Y : [w,n]

    """
    assert X.ndim == Y.ndim == 2
    assert X.shape[0] == Y.shape[0]
    assert batch_size <= X.shape[0], f"{batch_size} >= {X.shape[0]}"
    m, n = X.shape[1], Y.shape[1]
    w = X.shape[0]

    # Initialize the weights.
    W = jnp.zeros((2 * m, 2 * n))
    B = jnp.zeros((n, n, 2))
    params = (W, B)
    optimizer = optax.adam(lr)
    if opt_state is None:
        opt_state = optimizer.init(params)

    batch_loss_func = lambda Xb, Yb, params, l2_reg, l1_reg: jnp.mean(
        vec_loss_func(Xb, Yb, params, l2_reg, l1_reg)
    )
    batch_val_grad = value_and_grad(batch_loss_func, argnums=2)

    # Loop with minibatches ...
    if verbose:
        pbar = tqdm(range(1, n_iter + 1), desc=f"Optimizing phi: {jnp.nan}")
        smooth_val = None
    else:
        pbar = range(1, n_iter + 1)
    for i in pbar:
        key, subkey = jr.split(key)
        idx = jr.choice(subkey, w, shape=(batch_size,), replace=replace)
        Xb, Yb = X[idx], Y[idx]
        val, grads = batch_val_grad(Xb, Yb, params, l2_reg, l1_reg)
        if verbose:
            if i == 1:
                smooth_val = val
            else:
                smooth_val = alpha * smooth_val + (1.0 - alpha) * val
            pbar.set_description(f"Optimizing params: {smooth_val:.3f}")
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    return params, opt_state
