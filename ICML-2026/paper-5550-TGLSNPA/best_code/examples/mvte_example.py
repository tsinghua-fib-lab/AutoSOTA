"""
Example demonstrating estimation of multivariate transfer entropy on synthetic data.

"""
__date__ = "August - September 2025"

import jax
import jax.numpy as jnp
import numpy as np
import os

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fit_mv_artg import fit_multivariate_artg_sgd
from src.multivariate_transfer_entropy import (
    draw_random_process,
    sample_process,
    estimate_mv_te,
)


class ArrayRandomWindowLoader:
    """Yield random windows from an array of phase angles."""

    def __init__(self, data, batch_size, L, seed=0):
        self.data = np.asarray(data)
        self.batch_size = int(batch_size)
        self.L = int(L)
        self.T = self.data.shape[0]
        self.R = self.data.shape[1]
        self.F = 1
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self):
        starts = self.rng.integers(0, self.T - self.L, size=self.batch_size)
        windows = np.stack([
            self.data[s : s + self.L + 1] for s in starts
        ], axis=0)  # (B, L+1, R)
        return jnp.asarray(windows)[..., None]  # (B, L+1, R, 1)


def lag_statistics_from_array(data, L):
    """Compute mean and covariance of lagged cos/sin features from data."""
    data = jnp.asarray(data) # (T, R)
    T, R = data.shape
    W = T - L
    wins = jnp.stack([data[t : t + L] for t in range(W)], axis=0)  # (W, L, R)
    cosv = jnp.cos(wins)
    sinv = jnp.sin(wins)
    z = jnp.stack([cosv, sinv], axis=-1)  # (W, L, R, 2)
    z = jnp.transpose(z, (0, 2, 1, 3)).reshape(W, R * L * 2) # (W, RL2)
    mean = jnp.mean(z, axis=0) # (RL2,)
    centered = z - mean[None]
    cov = centered.T @ centered / W # (RL2, RL2)
    return mean[None], cov[None] # (1, RL2), (1, RL2, RL2)


def main():
    key = jax.random.PRNGKey(0)
    R = 3
    L = 5
    F = 1
    T = 10000

    pattern = jnp.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ]
    )

    key, sub = jax.random.split(key)
    W_true = draw_random_process(sub, pattern, L)

    key, sub = jax.random.split(key)
    samples = sample_process(sub, W_true, T)  # (T, R)

    means, covars = lag_statistics_from_array(samples, L) # (1, RL2), (1, RL2, RL2)
    covars = covars + 1e-1 * jnp.eye(R * L * 2)[None] # regularize the covariance

    loader = ArrayRandomWindowLoader(samples, batch_size=128, L=L, seed=0)

    key, sub = jax.random.split(key)
    W_hat, _, _ = fit_multivariate_artg_sgd(
        sub,
        loader,
        F,
        R,
        L,
        lr=3e-3,
        num_steps=5000,
    )
    key, sub = jax.random.split(key)
    te = estimate_mv_te(
        key,
        loader,
        W_hat,
        means,
        covars,
        R,
        L,
        F,
        max_num_batches=2000,
        show_progress=True,
    )

    print("Estimated transfer entropy matrix:")
    print(np.asarray(te[0]))


if __name__ == "__main__":
    main()
