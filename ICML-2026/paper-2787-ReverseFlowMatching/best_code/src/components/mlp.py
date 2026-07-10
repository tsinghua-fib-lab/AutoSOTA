from typing import Optional

import jax.numpy as jnp
import flax.linen as nn


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


def zeros_init():
    return nn.initializers.zeros_init()


def sinusoidal_embedding(x, size, scale):
    x = x * scale
    half_size = size // 2
    emb = jnp.log(jnp.array([1000.0])) / (half_size - 1)
    emb = jnp.exp(-emb * jnp.arange(half_size))
    emb = jnp.expand_dims(x, -1) * jnp.expand_dims(emb, 0)
    emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
    return emb


class NormedLinear(nn.Module):
    size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.size, kernel_init=default_init())(x)
        x = nn.relu(x)
        return x


class Block(nn.Module):
    size: int

    @nn.compact
    def __call__(self, x):
        return x + nn.gelu((nn.Dense(self.size, kernel_init=default_init())(x)))


class ScoreMLP(nn.Module):
    hidden_size: int
    hidden_layers: int
    emb_size: int
    out_dim: int

    @nn.compact
    def __call__(self, t, x, s, temp):
        time_emb = sinusoidal_embedding(t.squeeze(), self.emb_size, scale=1.0)
        temp_emb = sinusoidal_embedding(temp.squeeze(), self.emb_size, scale=1.0)
        x = jnp.concatenate((s, x, time_emb, temp_emb), axis=-1)

        x = nn.Dense(self.hidden_size, kernel_init=default_init())(x)
        x = nn.gelu(x)
        for _ in range(self.hidden_layers):
            x = Block(self.hidden_size)(x)
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        return x


class ActorMLP(nn.Module):
    hidden_size: int
    hidden_layers: int
    out_dim: int

    @nn.compact
    def __call__(self, x):

        x = nn.Dense(self.hidden_size, kernel_init=default_init())(x)
        x = nn.gelu(x)
        for _ in range(self.hidden_layers):
            x = Block(self.hidden_size)(x)
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        return x


class MLP(nn.Module):
    hidden_size: int
    hidden_layers: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.hidden_layers):
            x = NormedLinear(self.hidden_size)(x)
        x = nn.Dense(1, kernel_init=default_init())(x)
        return x.squeeze(-1)
