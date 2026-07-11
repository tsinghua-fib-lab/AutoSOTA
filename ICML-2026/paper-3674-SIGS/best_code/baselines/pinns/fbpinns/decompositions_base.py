"""
Contains fast memory-limited functions for computing whether a set of points are inside a set of models

We batch computation across the points dimension to limit memory usage

Notes:
    we want to avoid many dynamic shapes where possible, to avoid lots of implicit compilation
    below the only dynamic shape is the size of the global n_take, m_take (and final reduction of inside_ims/inside_ips)
    which is precomputed using _inside_sum_batch
    we avoid using (dynamic) nonzero in the inner batch loop by instead using a gather operation on n_take, m_take
    lax.scan and lax.map need static batch shapes, so masking is used for remainders
    eventually this could be batched across model dimension too

This module is used by decompositions.py
"""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(3,4))
def _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn):

    def batch_step(x):
        i, mask = x
        x_batch_ = jax.lax.dynamic_slice(x_batch, (i,0), (batch_size, x_batch.shape[1]))# (n, xd)
        inside_ = jnp.expand_dims(mask,1)*inside_fn(all_params, x_batch_, ims)# (n, m)
        s1, s2 = jnp.any(inside_, axis=1), inside_.sum(0)
        return (s1, s2)# (n), (m)

    # get fully-populated batches by shifting last value of irange
    r = x_batch.shape[0]%batch_size
    shift = batch_size-r if r else 0
    irange = jnp.arange(0, x_batch.shape[0], batch_size)# (k)
    mask = jnp.ones((len(irange), batch_size), dtype=bool)# (k, n)
    irange = irange.at[-1].add(-shift)
    mask = mask.at[-1,:shift].set(False)
    s1, s2 = jax.lax.map(batch_step, (irange, mask))

    # parse ims and ips
    inside_ips = jnp.concatenate([s1[:-1].ravel(), s1[-1][shift:]], axis=0)# (n)
    inside_ims = s2.sum(0)# (m)
    d = (inside_ims.mean()**(1/x_batch.shape[1]))# average number of points per model
    s = inside_ims.sum()
    inside_ims = inside_ims.astype(bool)
    return (s, inside_ips, inside_ims, d), irange, mask

@partial(jax.jit, static_argnums=(3,4,5))
def _inside_take_batch(all_params, x_batch, ims, batch_size, inside_fn, s, irange, mask):

    def batch_step(carry, x):
        i, mask = x
        x_batch_ = jax.lax.dynamic_slice(x_batch, (i,0), (batch_size, x_batch.shape[1]))# (n, xd)
        inside_ = jnp.expand_dims(mask,1)*inside_fn(all_params, x_batch_, ims)# (n, m)
        inside_ = inside_.ravel()# (n*m)
        itake = jnp.cumsum(inside_)-1# (n*m)
        ii_ = jnp.expand_dims(inside_,1)*ii.at[:,0].add(i)# (n*m, 2)
        take, s = carry
        take = take.at[s+itake].add(ii_)# (s, 2)
        return (take, s+itake[-1]+1), None

    ix,iy = jnp.meshgrid(jnp.arange(batch_size), jnp.arange(ims.shape[0]), indexing="ij")# (n, m)
    ii = jnp.stack([ix.ravel(), iy.ravel()], axis=1)# (n*m, 2)
    take = jnp.zeros((s,2), dtype=int)# (s, 2)
    (take, _), _ = jax.lax.scan(batch_step, (take, 0), (irange, mask))
    return take

def inside_points_batch(all_params, x_batch, ims, batch_size, inside_fn):
    assert batch_size <= x_batch.shape[0]
    (s, inside_ips, inside_ims, d), irange, mask = _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn)
    inside_ims = jnp.arange(ims.shape[0])[inside_ims]
    s = s.item()
    take = _inside_take_batch(all_params, x_batch, ims, batch_size, inside_fn, s, irange, mask)
    return take[:,0], take[:,1], inside_ims

def inside_models_batch(all_params, x_batch, ims, batch_size, inside_fn):
    assert batch_size <= x_batch.shape[0]
    (s, inside_ips, inside_ims, d), irange, mask = _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn)
    inside_ips = jnp.arange(x_batch.shape[0])[inside_ips]
    return inside_ips, d




if __name__ == "__main__":

    import jax.random as random

    def inside_fn(all_params, x_batch, ims):
        "Code for assessing if point is in ND hyperrectangle"
        x_batch = jnp.expand_dims(x_batch, 1)# (n,1,xd)
        xmins = jnp.expand_dims(all_params[0][ims], 0)# (1,mc,xd)
        xmaxs = jnp.expand_dims(all_params[1][ims], 0)# (1,mc,xd)
        inside = (x_batch >= xmins) & (x_batch <= xmaxs)# (n,mc,xd)
        inside = jnp.all(inside, -1)# (n,mc) keep as bool to reduce memory
        return inside

    def inside(all_params, x_batch, ims, inside_fn):
        "full batch code to compare to"
        inside = inside_fn(all_params, x_batch, ims)# (n, m)
        n_take, m_take = jnp.nonzero(inside)
        inside_ims = jnp.nonzero(jnp.any(inside, axis=0))[0]
        inside_ips = jnp.nonzero(jnp.any(inside, axis=1))[0]
        return n_take, m_take, inside_ims, inside_ips

    n,m = 10000, 1000
    x_batch = random.uniform(random.PRNGKey(0), (n,2), minval=0, maxval=2)
    c = random.uniform(random.PRNGKey(0), (m,2), minval=1, maxval=3)
    xmin, xmax = c.copy(), c.copy()
    xmin -= 0.1
    xmax += 0.1
    all_params = [xmin, xmax]
    ims = jnp.arange(m)

    n_take_true, m_take_true, inside_ims_true, inside_ips_true = inside(all_params, x_batch, ims, inside_fn)

    for batch_size in [1, 9, 10, 128, n, n+1]:
        print(batch_size)

        n_take, m_take, inside_ims = inside_points_batch(all_params, x_batch, ims, batch_size, inside_fn)
        inside_ips, d = inside_models_batch(all_params, x_batch, ims, batch_size, inside_fn)

        assert (n_take_true==n_take).all()
        assert (m_take_true==m_take).all()
        assert (inside_ims_true==inside_ims).all()
        assert (inside_ips_true==inside_ips).all()