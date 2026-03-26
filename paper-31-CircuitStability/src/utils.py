"""Utility functions for the circuit stability framework.

This module provides common utility functions used throughout the framework,
including random seed management, data processing, and other helper functions.
"""

import os
import math
import torch
import random
import transformers
import numpy as np


def seed_everything(seed: int = 42):
    """Set random seeds for all libraries to ensure reproducibility.
    
    This function sets the random seed for Python's random module, NumPy,
    PyTorch (both CPU and CUDA), and the Transformers library. This ensures
    that experiments are reproducible across different runs.
    
    Args:
        seed: Random seed value to use for all random number generators.
             Default is 42.
    
    Example:
        >>> seed_everything(42)
        >>> # All subsequent random operations will be deterministic
    """
    random.seed(seed)
    transformers.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
