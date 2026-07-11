from dataclasses import dataclass

import jax.numpy as jnp
from flax import linen as nn


@dataclass
class TimeEmbedding(nn.Module):
    embedding_layers: int

    @nn.compact
    def __call__(self, t: jnp.ndarray):

        h_t = nn.Dense(self.embedding_layers)(t)  # (B, embed_dim)
        h_t = nn.swish(h_t)
        t_embed = nn.Dense(3)(h_t)  # (B, 3)
        t_features = jnp.concatenate([t, t_embed], axis=-1)  # (B, 4)

        return t_features  # (B, 4)


@dataclass
class MLP(nn.Module):
    hidden_layers: int
    embedding_layers: int
    n_layers: int

    def reshape_inputs(self, x: jnp.ndarray, t: jnp.ndarray):

        # Reshape x to shape (BS, N_ATOMs==1, DIM==1)
        assert x.ndim in (0, 1, 2, 3), "Input x should be of shape (), (BS, N_ATOMs), (BS, N_ATOMs, DIM) or (BS,)"
        if x.ndim == 0:
            x = x.reshape(1, 1)  # () -> (1, 1)
        elif x.ndim == 1:
            x = x.reshape(x.shape[0], 1)  # (BS,) -> (BS, 1)
        elif x.ndim == 3:
            x = x.reshape(x.shape[0], 1)  # (BS, N_ATOMs, DIM) -> (BS, 1)

        # Reshape t to shape (BS, 1)
        assert t.ndim in (0, 1, 2), "Input t should be of shape (), (BS, 1) or (BS,)"
        if t.ndim == 1:
            t = t[:, None]  # (BS) -> (BS, 1)
        elif t.ndim == 0:
            t = t[None, None]  # () -> (1, 1)

        return x, t

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray):
        x, t = self.reshape_inputs(x, t)
        output = self.forward(x, t)
        return output

    @nn.compact
    def forward(self, x: jnp.ndarray, t: jnp.ndarray):

        t_emb = TimeEmbedding(self.embedding_layers)(t)  # (B, time_features)
        h = jnp.concatenate([x, t_emb], axis=-1)  # (B, 5)

        for _ in range(self.n_layers):
            h = nn.Dense(self.hidden_layers)(h)  # (B, hidden_dim)
            h = nn.swish(h)

        output = nn.Dense(x.shape[-1])(h)

        return output
