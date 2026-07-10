import math

import numpy as np


class IndependenceCopula:
    """
    Independence copula implemented with numpy.
    """

    def __init__(self):
        pass

    def cdf(self, u: np.ndarray) -> np.ndarray:
        """
        Compute cumulative distribution function.

        Parameters
        ----------
        u : ndarray (float)
            ndarray of shape [batch_size, 2]. Each element should be in [0, 1].

        Returns
        -------
        probability : ndarray (float)
            ndarray of shape [batch_size].
        """
        if u.ndim != 2:
            raise ValueError("u must be 2-dimensional array.")
        if u.shape[1] != 2:
            raise ValueError("u must have shape [batch_size, 2].")

        return np.prod(u, axis=1)


class FrankCopula:
    """
    Frank copula implemented with numpy.
    """

    def __init__(self, theta: float):
        """
        Initialization of Frank copula.

        Parameters
        ----------
        theta : float
            parameter of Frank copula.
        """

        self.theta = theta
        self.denominator = math.exp(-theta) - 1.0

    def cdf(self, u: np.ndarray) -> np.ndarray:
        """
        Compute cumulative distribution function.

        Parameters
        ----------
        u : ndarray (float)
            ndarray of shape [batch_size, 2].
            Each element should be in [0, 1].

        Returns
        -------
        cumulative probability : ndarray (float)
            ndarray of shape [batch_size].
        """

        temp_0 = np.exp(-self.theta * u[:, 0]) - 1
        temp_1 = np.exp(-self.theta * u[:, 1]) - 1
        return -(1.0 / self.theta) * np.log(1 + temp_0 * temp_1 / self.denominator)


class SurvivalCopula:
    """
    Survival copula implemented with pytorch.
    """

    def __init__(self, copula):
        self.copula = copula

    def cdf(self, u: np.ndarray) -> np.ndarray:
        if u.ndim != 2:
            raise ValueError("u must be 2-dimensional array.")
        return u[:, 0] + u[:, 1] - 1 + self.copula.cdf(1.0 - u)


def create(name: str, theta: float):
    if name == "independence":
        return IndependenceCopula()
    elif name == "frank":
        return FrankCopula(theta)
    else:
        raise ValueError(f"Invalid copula name: {name}")
