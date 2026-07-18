"""
TSP Solver
---------------------------------

Main module for the TSP solver. This module contains the main entry point
for the TSP solver. To use it, first import it

.. code-block:: python

    import fast_tsp


.. currentmodule:: fast_tsp

.. autosummary::
    :toctree: _generate

    find_tour
    greedy_nearest_neighbor
    solve_tsp_exact
    is_valid_tour
    compute_cost
    score_tour
"""

from __future__ import annotations

import warnings
from typing import List
from typing import Union

import numpy as np

from ._core import __version__  # type: ignore
from ._core import compute_cost as __compute_cost  # type: ignore
from ._core import find_tour as __find_tour  # type: ignore
from ._core import greedy_nearest_neighbor as __greedy_nearest_neighbor  # type: ignore
from ._core import is_valid_tour as __is_valid_tour  # type: ignore
from ._core import score_tour as __score_tour  # type: ignore
from ._core import solve_tsp_exact as __solve_tsp_exact  # type: ignore

Tour = List[int]
DistMatrix = Union[List[List[int]], np.ndarray]

_UINT16_MAX = 2**16 - 1


def is_valid_dist_matrix(dists: DistMatrix) -> str | None:
    """
    Returns a string describing the error if the distance matrix is invalid,
    otherwise returns None.
    Checks the following:
    - The distance matrix is non-empty.
    - The distance matrix is 2D and square.
    - The distance matrix contains only non-negative integers.
    - The distance matrix contains only integers that fit in a uint16.
    - The diagonal is 0.
    - The distance matrix is symmetric.
    - The distance matrix satisfies the triangle inequality.
    """
    try:
        dists = np.array(dists)
    except Exception as e:
        return f'Could not convert the distance matrix to a numpy array: {e}'

    if len(dists.shape) != 2:
        return 'Distance matrix must be 2D'

    n, m = dists.shape
    if n != m:
        return f'Distance matrix must be square, but got ({n}, {m})'

    if n == 0:
        return 'Distance matrix must be non-empty'

    if isinstance(dists[0][0], float):
        typ = type(dists[0][0])
        return f'Distance matrix must contain integers, but found {typ}'

    min_val: int = np.min(dists)  # type: ignore
    max_val: int = np.max(dists)  # type: ignore
    if min_val < 0:
        return f'All distances must be non-negative, but found {min_val:,}'
    if max_val > _UINT16_MAX:
        return f'All distances must be <= {_UINT16_MAX:,}, but found {max_val:,}'

    if not np.all(np.diag(dists) == 0):
        return 'Distance matrix diagonal must be 0'

    if not np.all(dists == dists.T):
        return 'Distance matrix must be symmetric'

    # Check that the distance matrix satisfies the triangle inequality
    for i in range(n):
        for j in range(i):
            if not all(dists[i, j] <= dists[i, :] + dists[:, j]):
                return f'Distance matrix does not satisfy the triangle inequality for cities {i} and {j}'

    return None


def find_tour(dists: DistMatrix, duration_seconds: float = 2.0) -> Tour:
    """Find a tour using the fast heuristic.

    Run a local solver to find a near-optimal TSP tour. For small
    problems, the exact solution is returned.

    Args:
        dists: A distance matrix.
        duration_seconds: The maximum duration of the tour.

    Returns:
        A tour that is ideally near-optimal.

    .. code-block:: python

      dists = [[0, 1], [1, 0]]
      tour = fast_tsp.find_tour(dists)
      print(tour)
      # [0, 1]
    """
    try:
        return __find_tour(dists, duration_seconds)
    except Exception as e:
        dist_matrix_error_msg = is_valid_dist_matrix(dists)
        if dist_matrix_error_msg is None:
            raise
        else:
            raise ValueError(
                f'An exception occurred while running the solver: {e}\nInvalid distance matrix: {dist_matrix_error_msg}'
            ) from e


def greedy_nearest_neighbor(dists: DistMatrix) -> Tour:
    """Solve the TSP using the greedy nearest neighbor algorithm.

    Solve TSP using the greedy nearest neighbor heuristic. This is a very
    rough approximation to the exact solution. The preferred way is to
    use `find_tour`.

    Args:
        dists: The distance matrix.

    Returns:
        The tour computed by the nearest neighbor algorithm.
    """
    return __greedy_nearest_neighbor(dists)


def solve_tsp_exact(dists: DistMatrix) -> Tour:
    """Solve the TSP using the exact algorithm.

    Solves TSP optimally using the bottom-up Held-Karp's algorithm in
    :math:`O(2^n * n^2)` time. This is tractable for small n but quickly becomes
    untractable for n even of medium size.

    Args:
        dists: The distance matrix.

    Returns:
        An optimal tour for the problem instance.
    """
    if len(dists) > 20:
        warnings.warn(
            'The exact algorithm is not recommended for n > 20. Consider using the fast heuristic instead.',
            stacklevel=2,
        )
    return __solve_tsp_exact(dists)


def is_valid_tour(n: int, tour: Tour) -> bool:
    """Check if a tour is valid.

    Args:
        n: The number of cities.
        tour: The tour to test whether it is valid.

    Returns:
        True if the tour is valid, False otherwise.
    """
    return __is_valid_tour(n, tour)


def compute_cost(tour: Tour, dists: DistMatrix) -> float:
    """Returns the cost of a tour.

    Computes the cost of a tour on a distance matrix. No validation on
    whether the tour is valid is performed.

    Args:
        tour: The tour.
        dists: The distance matrix.

    Returns:
        The cost of the tour.
    """
    return __compute_cost(tour, dists)


def score_tour(tour: Tour, dists: DistMatrix, opt_cost: int) -> float:
    """Return the score of a tour, where 1 is better and 0 is worst.

    Returns a double in the range 0..1, where 0 denotes a solution that
    is worse than or equivalent to the naive solution and 1 denotes a
    solution that is equivalent to the optimal solution.

    Args:
        tour: The tour.
        dists: The distance matrix.
        opt_cost: The cost of the optimal tour.

    Returns:
        The score of the tour.
    """
    return __score_tour(tour, dists, opt_cost)


__all__ = [
    '__version__',
    'compute_cost',
    'find_tour',
    'greedy_nearest_neighbor',
    'is_valid_tour',
    'score_tour',
    'solve_tsp_exact',
]