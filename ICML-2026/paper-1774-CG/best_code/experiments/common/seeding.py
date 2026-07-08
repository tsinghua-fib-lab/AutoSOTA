"""Deterministic seeding shared across experiments."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy and torch (CPU + CUDA) from a single integer.

    Returns the seed so it can be logged. Kept dependency-light: NumPy and torch
    are imported lazily so the helper also works in minimal environments.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a core dep
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a core dep
        pass
    return seed
