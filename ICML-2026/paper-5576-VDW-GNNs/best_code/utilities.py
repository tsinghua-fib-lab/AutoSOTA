"""
General utility classes and functions.
"""
import os
import time
import numpy as np
from numpy.random import RandomState
import yaml
from pathlib import Path
from typing import (
    Tuple,
    List,
    Any,
    Optional
)

def read_yaml(path: Path) -> dict:
    try:
        with open(path) as _f:
            data = yaml.safe_load(_f)
            if isinstance(data, dict):
                return data
            else:
                raise ValueError(f"[CONFIG] YAML file {path} is not a dictionary")
    except FileNotFoundError:
        print(f"[CONFIG] YAML file not found at {path}")
        return {}

def merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries.
    The 'override' dictionary overrides the 'base' dictionary.
    If a key is present in both dictionaries, the value in the 'override'
    dictionary is used.
    If a key is present in the 'base' dictionary but not in the 'override'
    dictionary, the value in the 'base' dictionary is used.
    """
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out

def generate_random_integers(
    n: int, 
    min_val: int = 0, 
    max_val: int = 100, 
    replace: bool = False,
    random_state: Optional[RandomState] = None, 
    seed: Optional[int] = None
) -> List[int]:
    """
    Generates a list of random integers.

    Args:
        n: number of integers to generate.
        min_val: minimum integer value.
        max_val: maximum integer value.
        replace: whether to sample with
            replacement.
        random_state: numpy RandomState
            instance. If None, a new one
            will be instantiated.
        seed: seed for random_state.
    Returns:
        List of random integers of length n.
    """
    if n > (max_val - min_val + 1):
        raise ValueError(
            f"Cannot sample {n} unique integers from " 
            f"the range {min_val} to {max_val}."
        )

    # create new RandomState if needed (i.e. if not passed
    # a RandomState instance, or passed a new seed)
    if (random_state is None) or (seed is not None):
        random_state = RandomState(seed)
    sample = random_state.choice(
        np.arange(min_val, max_val + 1), 
        size=n, 
        replace=replace
    ).tolist()
    
    return sample 