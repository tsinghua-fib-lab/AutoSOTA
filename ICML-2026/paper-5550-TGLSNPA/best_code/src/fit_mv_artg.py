"""
Fit multivariate AR-TG models to data.

"""
__date__ = "August 2025"

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from .von_mises import vm_log_pdf


def _sliding_window(x: jnp.ndarray, w: int) -> jnp.ndarray:
    # Returns shape (N - w + 1, w, *)
    starts = jnp.arange(x.shape[0] - w + 1)
    return jax.vmap(lambda s: jax.lax.dynamic_slice_in_dim(x, s, w))(starts)

sliding_window = jax.jit(_sliding_window, static_argnames=("w",))


def fit_multivariate_artg_ssm(
        key,
        loader,
        F,
        R,
        L,
        W = None,
        opt_state = None,
        lr: float = 1e-2,
        num_steps: int = 5000,
        alpha=0.99,
    ):
    """

    Parameters
    ----------
    key : jax.random.PRNGKey
    loader : SequentialPhaseLoader

    Returns
    -------
    W : jnp.ndarray
        Shape: [F, L*R*2, R*2]
    opt_state : 
    losses : list
    """

    if W is None:
        W = 1e-3 * jax.random.normal(key, (F, R*L*2, R*2))

    def loss_fn(W, batch):
        """
        W : [F, L*R*2, R*2]
        batch : [B, L + 1, R, F] angles
        """
        targets = batch[:, -1] # [B, R, F]
        targets = jnp.transpose(targets, (0,2,1)) # [B, F, R]

        windows = jnp.stack([jnp.cos(batch[:,:-1]), jnp.sin(batch[:,:-1])], -1) # [B, L, R, F, 2]
        windows = jnp.transpose(windows, (0,3,1,2,4)) # [B, F, L, R, 2]
        windows = windows.reshape(windows.shape[0], F, -1) # [B, F, L*R*2]

        pred = jnp.einsum('fio,bfi->bfo', W, windows) # [B, F, R*2]
        pred = pred.reshape(-1, F, R, 2) # [B, F, R, 2]
        a, b = pred[..., 0], pred[..., 1] # both [B, F, R]
        
        # independent von Mises on each channel:
        # vm_log_pdf(a, b, x) returns log p(x|a,b)
        logp = vm_log_pdf(a, b, targets) # [B, F, R]

        # average negative log-likelihood
        loss = -jnp.mean(logp)
        return loss

    # set up Optax Adam
    optimizer = optax.adam(lr)
    if opt_state is None:
        opt_state = optimizer.init(W)

    @jax.jit
    def step(W, batch, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(W, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        W = optax.apply_updates(W, updates)
        return W, opt_state, loss

    # run the loop
    pbar = tqdm(range(num_steps))
    smooth_loss = None
    losses = []
    for _ in pbar:
        batch = next(loader)
        W, opt_state, loss = step(W, batch, opt_state)
        if smooth_loss is None:
            smooth_loss = loss
        else:
            smooth_loss = alpha * smooth_loss + (1-alpha) * loss
        pbar.set_description(f"NLL: {smooth_loss:.5f}")
        losses.append(loss)

    return W, opt_state, losses
