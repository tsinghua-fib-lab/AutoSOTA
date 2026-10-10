from __future__ import annotations

import os

import torch

DEFAULT_NUM_WORKERS = 8
DEFAULT_PREFETCH_FACTOR = 2


def resolve_num_workers(requested_num_workers: int | None) -> int:
    if requested_num_workers is not None:
        if requested_num_workers < 0:
            raise ValueError(
                f"num_workers must be non-negative, got {requested_num_workers}"
            )
        return requested_num_workers

    cpu_count = os.cpu_count()
    if cpu_count is None:
        return DEFAULT_NUM_WORKERS
    return max(1, min(DEFAULT_NUM_WORKERS, cpu_count))


def build_dataloader_kwargs(
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> dict:
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = DEFAULT_PREFETCH_FACTOR
    return dataloader_kwargs
