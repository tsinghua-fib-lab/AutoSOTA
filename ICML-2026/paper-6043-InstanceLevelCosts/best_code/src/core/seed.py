"""
Centralized seed management for reproducible experiments.

This module provides utilities for setting random seeds across all major
ML libraries (numpy, random, torch) to ensure reproducible results.

Usage:
    from core.seed import set_seed

    set_seed(42)  # Sets all random seeds to 42
"""

import os
import random
import numpy as np
from typing import Optional

# Optional torch import (may not be installed)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Sets seeds for:
    - Python's built-in random module
    - Python's hash randomization (PYTHONHASHSEED)
    - NumPy
    - PyTorch (if available)

    Args:
        seed: Random seed value (typically 42, 123, or 456)
        deterministic: If True and torch is available, enables deterministic
                      CUDA operations. May reduce performance but ensures
                      reproducibility on GPU.

    Example:
        >>> set_seed(42)
        >>> # All random operations now use seed 42
        >>> np.random.rand(3)
        array([0.37454012, 0.95071431, 0.73199394])

    Note:
        PYTHONHASHSEED must be set before Python starts to be fully effective.
        For complete reproducibility, set it in your environment or at program startup.
    """
    # Python hash seed (for dict/set iteration order)
    # Note: This only affects the current process; won't affect subprocesses
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        if deterministic:
            # Enable deterministic mode for reproducibility
            # Note: This may impact performance
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Get a NumPy random number generator with optional seed.
    
    This is the modern NumPy approach (preferred over np.random.seed).
    Returns a Generator that can be passed to functions expecting an RNG.
    
    Args:
        seed: Optional seed value. If None, uses system entropy.
    
    Returns:
        NumPy Generator instance
        
    Example:
        >>> rng = get_rng(42)
        >>> rng.random(3)
        array([0.77395605, 0.43887844, 0.85859792])
    """
    return np.random.default_rng(seed)


def seed_worker(worker_id: int, base_seed: int = 42) -> None:
    """
    Seed worker processes for PyTorch DataLoader reproducibility.
    
    Use as worker_init_fn in DataLoader to ensure each worker has
    a different but reproducible seed.
    
    Args:
        worker_id: Worker ID (provided by DataLoader)
        base_seed: Base seed to derive worker seeds from
        
    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=lambda wid: seed_worker(wid, base_seed=42)
        ... )
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(worker_seed)
