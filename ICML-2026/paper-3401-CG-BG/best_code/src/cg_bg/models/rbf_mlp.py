from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass
class RBF(nn.Module):
    num_centers: int
    sigma: float = 1.0
    learnable_centers: bool = True

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]

        # Initialize RBF centers
        centers = self.param(
            "centers",
            lambda rng: jax.random.uniform(
                rng,
                shape=(self.num_centers, input_dim),
                minval=10.0,
                maxval=50.0,
            ),
        )

        if not self.learnable_centers:
            centers = jax.lax.stop_gradient(centers)

        x = x[:, None, :]  # (B, 1, D)
        centers = centers[None, :, :]  # (1, K, D)
        dists = jnp.sum((x - centers) ** 2, axis=-1)  # (B, K)

        return jnp.exp(-dists / (2 * self.sigma**2))  # (B, K)


@dataclass
class RBFMLP(nn.Module):
    hidden_dim: int
    n_layers: int
    num_rbf_centers: int = 100
    sigma: float = 5.0

    def reshape_inputs(self, x: jnp.ndarray):

        # Reshape x to shape (B, D)
        assert x.ndim in (0, 1, 2), "Input x should be of shape (BS, D) or (BS,) or ()"
        if x.ndim == 0:
            x = x[None, None]  # () -> (1, 1)
        elif x.ndim == 1:
            x = x[:, None]  # (BS,) -> (BS, 1)

        return x

    def __call__(self, x):
        x = self.reshape_inputs(x)
        output = self.forward(x)
        return output

    @nn.compact
    def forward(self, x):

        h = RBF(num_centers=self.num_rbf_centers, sigma=self.sigma)(x)

        for _ in range(self.n_layers):
            h = nn.Dense(self.hidden_dim)(h)
            h = nn.softplus(h)
        output = nn.Dense(1)(h)

        return output
