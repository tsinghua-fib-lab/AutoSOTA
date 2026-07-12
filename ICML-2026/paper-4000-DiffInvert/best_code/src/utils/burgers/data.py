from typing import Tuple
from functools import partial
import numpy as np
from torch.utils import data
import jax.numpy as jnp
from jax.numpy import ndarray
from jax import random, jit

KeyArray = random.PRNGKey


def generate_one_ics_training_data(
    key: KeyArray,
    u0: ndarray,
    P: int = 101
) -> Tuple[ndarray, ndarray, ndarray]:
    """Geneate ics training data corresponding to one input sample"""
    t_0 = jnp.zeros((P, 1))
    x_0 = jnp.linspace(0, 1, P)[:, None]
    y = jnp.hstack([t_0, x_0])
    u = jnp.tile(u0, (P, 1))
    s = u0
    return u, y, s


def generate_one_bcs_training_data(
    key: KeyArray,
    u0: ndarray,
    P: int = 101
) -> Tuple[ndarray, ndarray, ndarray]:
    """Geneate bcs training data corresponding to one input sample"""
    t_bc = random.uniform(key, (P, 1))
    x_bc1 = jnp.zeros((P, 1))
    x_bc2 = jnp.ones((P, 1))
    y1 = jnp.hstack([t_bc, x_bc1])  # shape = (P, 2)
    y2 = jnp.hstack([t_bc, x_bc2])  # shape = (P, 2)
    u = jnp.tile(u0, (P, 1))
    y =  jnp.hstack([y1, y2])  # shape = (P, 4)
    s = jnp.zeros((P, 1))
    return u, y, s


def generate_one_res_training_data(
    key: KeyArray,
    u0: ndarray,
    P: int = 101
) -> Tuple[ndarray, ndarray, ndarray]:
    """Geneate res training data corresponding to one input sample"""
    subkeys = random.split(key, 2)
    t_res = random.uniform(subkeys[0], (P, 1))
    x_res = random.uniform(subkeys[1], (P, 1))
    u = jnp.tile(u0, (P, 1))
    y =  jnp.hstack([t_res, x_res])
    s = jnp.zeros((P, 1))
    return u, y, s


def generate_one_test_data(
    idx: np.ndarray,
    usol: ndarray,
    P: int = 101
) -> Tuple[ndarray, ndarray, ndarray]:
    """Geneate test data corresponding to one input sample"""
    u = usol[idx]
    u0 = u[0, :]
    t = jnp.linspace(0, 1, P)
    x = jnp.linspace(0, 1, P)
    T, X = jnp.meshgrid(t, x)
    s = u.T.flatten()
    u = jnp.tile(u0, (P ** 2, 1))
    y = jnp.hstack([T.flatten()[:, None], X.flatten()[:, None]])
    return u, y, s


class DataGenerator(data.Dataset):
    """Data generator for training and testing"""
    def __init__(
        self,
        u: ndarray,
        y: ndarray,
        s: ndarray,
        batch_size: int = 64,
        rng_key: KeyArray = random.PRNGKey(1234)
    ) -> None:
        'Initialization'
        self.u = u
        self.y = y
        self.s = s

        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index: int) -> Tuple[Tuple[ndarray, ndarray], ndarray]:
        """Generate one batch of data"""
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key: KeyArray) -> Tuple[Tuple[ndarray, ndarray], ndarray]:
        """Generates data containing batch_size samples"""
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx, :]
        y = self.y[idx, :]
        u = self.u[idx, :]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs
