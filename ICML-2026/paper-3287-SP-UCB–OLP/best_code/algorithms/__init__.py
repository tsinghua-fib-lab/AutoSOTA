"""
Algorithms for Alibaba SP-UCB-OLP Experiment

Available algorithms:
- SPUCBOLP: Main SP-UCB-OLP algorithm
- SPGreedyOLP: Greedy (alpha=0)
- OraclePolicy: Perfect information upper bound
- RandomPolicy: Uniform random lower bound
"""

from .base import BaseAlgorithm
from .sp_ucb_olp import SPUCBOLP
from .baselines import SPGreedyOLP, OraclePolicy, RandomPolicy

__all__ = [
    'BaseAlgorithm',
    'SPUCBOLP',
    'SPGreedyOLP',
    'OraclePolicy',
    'RandomPolicy',
]


def get_algorithm(name: str, K: int, d: int, T: int, B, config=None):
    """
    Factory function to create algorithm by name.

    Parameters
    ----------
    name : str
        Algorithm name: 'SP-UCB-OLP', 'Greedy', 'Oracle', 'Random'
    K : int
        Number of configurations
    d : int
        Number of resource dimensions
    T : int
        Time horizon
    B : np.ndarray
        Total budget vector
    config : dict, optional
        Algorithm configuration

    Returns
    -------
    BaseAlgorithm
        Instantiated algorithm
    """
    import numpy as np
    B = np.asarray(B)

    algorithms = {
        'SP-UCB-OLP': SPUCBOLP,
        'SPUCBOLP': SPUCBOLP,
        'Greedy': SPGreedyOLP,
        'SPGreedyOLP': SPGreedyOLP,
        'Oracle': OraclePolicy,
        'OraclePolicy': OraclePolicy,
        'Random': RandomPolicy,
        'RandomPolicy': RandomPolicy,
    }

    if name not in algorithms:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(algorithms.keys())}")

    return algorithms[name](K, d, T, B, config)
