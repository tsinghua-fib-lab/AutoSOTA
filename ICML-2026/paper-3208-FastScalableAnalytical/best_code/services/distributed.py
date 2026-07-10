"""Entry-point script for generating samples with analytical diffusion models.
Multi-GPU (torchrun) version: each GPU samples a shard of num_samples in parallel.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Tuple


import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Distributed helpers
# -----------------------------------------------------------------------------

def init_distributed() -> Tuple[bool, int, int, int, torch.device]:
    """Init torch.distributed from torchrun env. Falls back to single-process."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return True, rank, world_size, local_rank, device


def is_main(rank: int) -> bool:
    return rank == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_object_from_main(obj: Any, rank: int) -> Any:
    """Broadcast a python object from rank0 to all ranks."""
    if not (dist.is_available() and dist.is_initialized()):
        return obj
    buf = [obj if rank == 0 else None]
    dist.broadcast_object_list(buf, src=0)
    return buf[0]


def split_count(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    """Evenly split `total` items across ranks. Returns (count, global_offset)."""
    base = total // world_size
    rem = total % world_size
    count = base + (1 if rank < rem else 0)
    offset = base * rank + min(rank, rem)
    return count, offset


class RunPathsProxy:
    """Duck-typed RunPaths for non-main ranks (avoid depending on RunPaths ctor signature)."""
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.logs = run_dir / "logs"
        self.images = run_dir / "images"
        self.intermediate_images = run_dir / "intermediate_images"
        self.config = run_dir / "config.yaml"