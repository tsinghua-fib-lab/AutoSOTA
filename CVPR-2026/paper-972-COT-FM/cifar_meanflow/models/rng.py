"""
This module provides utilities to mimic JAX's random number generator (RNG) control in PyTorch,
enabling reproducible and distributed-safe random operations.
However, GPU computation is not strictly deterministic, so full numerical reproducibility is not guaranteed.
"""

import torch
import hashlib
import random
import numpy as np


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def fold_in(seed: int, *args) -> int:
    """
    Simulate jax.random.fold_in via SHA256 hashing.
    Args can be anything hashable: step, rank, etc.
    """
    data = str((seed,) + args)
    h = hashlib.sha256(data.encode("utf-8")).hexdigest()
    folded_seed = int(h, 16) % (2**63)  # Safe for torch.manual_seed()
    return folded_seed


def train_step_with_rng_control(train_step_fn, model_without_ddp, step: int, base_seed: int, *args, **kwargs):
    rank = get_rank()
    seed = fold_in(base_seed, step, rank, "train_step")
    input_device = args[0].device if len(args) > 0 and torch.is_tensor(args[0]) else "cpu"

    with torch.random.fork_rng(devices=[input_device], enabled=True):
        torch.manual_seed(seed)
        if torch.cuda.is_available() and "cuda" in str(input_device):
            torch.cuda.manual_seed(seed)
        return train_step_fn(model_without_ddp ,*args, **kwargs)


def augment_with_rng_control(augment_pipe, samples, base_seed: int, steps: int):
    rank = get_rank()

    # Derive a unique, safe seed per rank and step
    # seed = fold_in(base_seed, steps, rank, "augment")

    # Use fork_rng to isolate the seed change
    with torch.random.fork_rng(devices=[samples.device], enabled=True):
        # torch.manual_seed(seed)
        # if samples.is_cuda:
        #     torch.cuda.manual_seed(seed)
        output = augment_pipe(samples)

    return output


def worker_init_fn(worker_id, rank, base_seed=0):
    seed = fold_in(base_seed, worker_id, rank, "worker_init")
    torch.manual_seed(fold_in(seed, 'torch'))
    random.seed(fold_in(seed, 'python'))
    np.random.seed(fold_in(seed, 'numpy') % 2**32)
