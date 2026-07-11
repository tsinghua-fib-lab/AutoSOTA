#from __future__ import annotations

import jax
import jax.numpy as jnp

from flax import nnx

Array = jax.Array

# ------------------------------------------------------------
# Utils & inits
# ------------------------------------------------------------

def variance_scaling_for_layers(num_layers: int) -> nnx.Initializer:
    scale = 2.0 / max(1, num_layers)
    return nnx.initializers.variance_scaling(
        scale=scale, mode="fan_in", distribution="truncated_normal"
    )

def ensure_mask_shape(mask: Array | None, B: int, H: int, Tq: int, Tk: int) -> Array | None:
    """Broadcast mask to [B, H, Tq, Tk] with True=keep, False=mask out."""
    if mask is None:
        return None
    # Accept [T, T], [B, T, T], [B, 1, T, T], [B, H, T, T]
    if mask.ndim == 2:
        mask = mask[None, None, :, :]
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]
    elif mask.ndim == 4:
        pass
    else:
        raise ValueError(f"Mask must have ndim 2, 3, or 4, got {mask.ndim}.")
    # Broadcast to [B, H, Tq, Tk]
    mask = jnp.broadcast_to(mask, (B, H, Tq, Tk))
    return mask