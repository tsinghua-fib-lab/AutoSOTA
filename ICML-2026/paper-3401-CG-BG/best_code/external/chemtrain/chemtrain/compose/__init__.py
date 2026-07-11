import jax
import jax.numpy as jnp

from mace_jax.adapters.cuequivariance import symmetric_contraction as sc

# This patch is necessary, as the original function contains a lax.cond for
# error checking. However, JAX transformations might replace lax.cond by
# lax.select, such that the error is always raised.

def _select_weights_no_raise(weight_flat, selector, *, dtype, num_elements):
    selector = jnp.asarray(selector)

    if selector.ndim == 1:
        idx = selector.astype(jnp.int32)
        invalid_mask = (idx < 0) | (idx >= num_elements)

        safe_idx = jnp.where(invalid_mask, jnp.int32(0), idx)
        return weight_flat[safe_idx]

    if selector.ndim == 2:
        if selector.shape[1] != num_elements:
            raise ValueError("Mixing matrix must have second dimension num_elements")
        return jnp.asarray(selector, dtype=dtype) @ weight_flat

    raise ValueError("indices must be rank-1 (element ids) or rank-2 (mixing matrix)")

sc._select_weights = _select_weights_no_raise
