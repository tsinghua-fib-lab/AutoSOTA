# utils/seed.py

import random
import numpy as np
import torch


def set_global_seed(seed: int):
    """
    Set Python, NumPy, and PyTorch random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
