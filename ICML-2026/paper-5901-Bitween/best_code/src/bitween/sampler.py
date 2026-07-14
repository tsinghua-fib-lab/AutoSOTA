from enum import StrEnum
from typing import Any, Callable

import numpy as np
from sympy import Symbol


class Domain(StrEnum):
    """
    Domain of the variables.
    """

    Real = "Real"
    Positive_Real = "Positive Real"
    Integer = "Integer"
    Positive_Integer = "Positive Integer"

    def latex(self):
        """
        :return: latex representation of the domain
        """
        if self == Domain.Real:
            return "\\mathbb{R}"
        elif self == Domain.Positive_Real:
            return "\\mathbb{R}^+"
        elif self == Domain.Integer:
            return "\\mathbb{Z}"
        elif self == Domain.Positive_Integer:
            return "\\mathbb{Z}^+"
        else:
            raise ValueError("Invalid domain")


class Distribution:
    """
    A class to represent a distribution with NumPy's random functions.

    Attributes:
        distribution (Callable): A callable NumPy distribution function.
        kwargs (dict): Parameters to initialize the distribution.
    """

    def __init__(self, distribution: Callable[..., np.ndarray], **kwargs: Any):
        """
        Initializes the Distribution with a callable and its parameters.

        Args:
            distribution (Callable): A callable NumPy distribution function.
            **kwargs: Parameters for the distribution function.
        """
        self.distribution = distribution
        self.kwargs = kwargs

    def generate_samples(self, sample_size: int) -> np.ndarray:
        """
        Generates samples using the specified distribution.

        Args:
            sample_size (int): The number of samples to generate.

        Returns:
            np.ndarray: An array of generated samples.
        """
        return self.distribution(size=sample_size, **self.kwargs)

    def __repr__(self) -> str:
        """
        Provides a string representation of the Distribution.

        Returns:
            str: A string describing the distribution and its parameters.
        """
        return f"Distribution(distribution={self.distribution.__name__}, kwargs={self.kwargs})"

    def latex(self) -> str:
        """
        :return: latex representation of the distribution
        """
        return f"{self.distribution.__name__}({', '.join([f'{key}={value}' for key, value in self.kwargs.items()])})"


def sample(distribution: Distribution, variables: list[str | Symbol]):
    # standard range
    values = {}
    for var in variables:
        values[str(var)] = distribution.generate_samples(1)[0]
    return values


if __name__ == "__main__":  # noqa E123
    # Example usage:

    # Creating a uniform distribution
    uniform_dist = Distribution(np.random.uniform, low=0.0, high=10.0)
    print(uniform_dist)
    # Output: Distribution(distribution=uniform, kwargs={'low': 0.0, 'high': 10.0})
    samples_uniform = uniform_dist.generate_samples(sample_size=5)
    print("Uniform Samples:", samples_uniform)

    # Creating a normal distribution
    normal_dist = Distribution(np.random.normal, loc=0.0, scale=1.0)
    print(normal_dist)
    # Output: Distribution(distribution=normal, kwargs={'loc': 0.0, 'scale': 1.0})
    samples_normal = normal_dist.generate_samples(sample_size=5)
    print("Normal Samples:", samples_normal)

    # Creating an integer distribution
    integer_dist = Distribution(np.random.randint, low=1, high=100)
    print(integer_dist)
    # Output: Distribution(distribution=randint, kwargs={'low': 1, 'high': 100})
    samples_integers = integer_dist.generate_samples(sample_size=5)
    print("Integer Samples:", samples_integers)

    # Select random reals from a list
    custom_dist = Distribution(np.random.choice, a=[1.0, 2.0, 3.0, 4.0, 5.0])
    print(custom_dist)
    # Output: Distribution(distribution=choice, kwargs={'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
    samples_custom = custom_dist.generate_samples(sample_size=5)
    print("Custom Samples:", samples_custom)
