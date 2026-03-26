import numpy as np
import scipy.special as scsp
from jax import numpy as jnp


def theta_epsilon(epsilon: np.float64, r: np.float64) -> np.float64:
    """
    Calculating theta_epsilon function from the notes
    """

    def Q(x):
        return (1 - scsp.erf(x / np.sqrt(2))) / 2

    return Q(epsilon / r - r / 2) - np.exp(epsilon) * Q(epsilon / r + r / 2)


def clamp_matrix(matrix, min_val, max_val):
    return jnp.clip(matrix, min_val, max_val)
