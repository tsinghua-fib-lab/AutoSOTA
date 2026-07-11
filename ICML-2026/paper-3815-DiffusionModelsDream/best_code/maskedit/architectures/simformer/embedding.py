from __future__ import annotations

import jax
import jax.numpy as jnp

from flax import nnx

class GaussianFourierEmbedding(nnx.Module):
    """Gaussian Fourier Embedding module"""
    def __init__(self, embed_dim: int, scale: float = 1.0, *, rngs: nnx.Rngs):
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim
        self.scale = scale
        half = embed_dim // 2+ 1
        key = rngs()
        B = jax.random.normal(key, (half,)) * scale
        self.B = nnx.Param(B)

    def __call__(self, t: Array) -> Array:
        arg = t * self.B.value[None, None, :]
        emb = jnp.concatenate([jnp.sin(arg), jnp.cos(arg)], axis=-1)
        return emb[..., : self.embed_dim]