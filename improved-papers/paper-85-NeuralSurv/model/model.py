from typing import Dict
from dataclasses import field
import flax.linen as nn
import jax.numpy as jnp


def get_model(config):
    if config.model.name == "mlp":
        mlp_main = MLP(
            n_hidden=config.model.n_hidden,
            n_layers=config.model.n_layers,
            activation=config.model.activation,
        )
        return NMLP(
            mlp_main=mlp_main,
        )
    else:
        raise NotImplementedError(f"Model {config.score_network.name} not implemented.")


class NMLP(nn.Module):
    mlp_main: nn

    @nn.compact
    def __call__(self, t, x):
        if len(t.shape) == 0:  # when batch size == 1
            t = t.reshape(1, -1)
        if len(x.shape) == 1:  # when batch size == 1
            x = x.reshape(1, -1)
        return self.mlp_main(x, t)


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    half_dim = embedding_dim // 2
    emb = jnp.log(max_positions) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = timesteps.ravel().astype(jnp.float32)[:, None] * emb
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    return emb


class MLP(nn.Module):
    n_hidden: int
    n_layers: int
    activation: nn
    use_dropout: bool = False
    use_batch_norm: bool = False
    use_residual: bool = False
    dropout_rate: float = 0.1
    _derivatives: Dict[int, callable] = field(default_factory=dict, init=False)

    time_embed_dim: int = 8

    @nn.compact
    def __call__(self, x, t, training=False):

        if len(t.shape) == 1:  # when batch size == 1
            t = t.reshape(-1, 1)

        t_emb = get_timestep_embedding(t, embedding_dim=self.time_embed_dim)
        x = jnp.concatenate([x, t_emb], axis=-1)

        for _ in range(self.n_layers - 1):
            y = self.activation(nn.Dense(self.n_hidden)(x))

            # Batch normalization if enabled
            if self.use_batch_norm:
                y = nn.BatchNorm()(y, use_running_average=not training)

            # Dropout if enabled
            if self.use_dropout:
                y = nn.Dropout(self.dropout_rate)(y, deterministic=False)

            # Residual connection if enabled
            if self.use_residual and x.shape == y.shape:
                x = x + y  # Add skip connection
            else:
                x = y  # Just update x

        return nn.Dense(1)(x)
