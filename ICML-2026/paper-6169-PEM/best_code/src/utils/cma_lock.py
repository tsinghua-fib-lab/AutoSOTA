from __future__ import annotations

import contextlib
import threading

import numpy as np

# pycma may touch NumPy's global RNG; threaded runs can introduce cross-talk.
# Use a single shared lock across all pycma-based optimizers in this project.
CMA_GLOBAL_LOCK = threading.Lock()


@contextlib.contextmanager
def cma_locked(*, seed: int | None = None):
    """
    Context for pycma calls in threaded sweeps:
    - serialize pycma usage to avoid cross-talk,
    - optionally reset NumPy global RNG to a deterministic seed.
    """

    with CMA_GLOBAL_LOCK:
        if seed is not None:
            np.random.seed(int(seed) & 0xFFFFFFFF)
        yield
