from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch.optim as optim


def _normalize_base_lrs(optimizer: optim.Optimizer, base_lrs: float | list[float]) -> list[float]:
    if not isinstance(base_lrs, list):
        return [float(base_lrs) for _ in optimizer.param_groups]
    if len(base_lrs) != len(optimizer.param_groups):
        raise ValueError("base_lrs length must match optimizer.param_groups.")
    return [float(lr) for lr in base_lrs]


def cosine_lr(optimizer: optim.Optimizer, base_lrs: float | list[float], warmup_length: int, steps: int) -> Callable[[int], None]:
    base_lrs_list = _normalize_base_lrs(optimizer, base_lrs)

    def _lr_adjuster(step: int) -> None:
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs_list, strict=True):
            if step < warmup_length:
                lr = base_lr * (step + 1) / max(1, warmup_length)
            else:
                e = step - warmup_length
                es = max(1, steps - warmup_length)
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            param_group["lr"] = lr

    return _lr_adjuster


def linear_lr(optimizer: optim.Optimizer, base_lrs: float | list[float], warmup_length: int, steps: int) -> Callable[[int], None]:
    base_lrs_list = _normalize_base_lrs(optimizer, base_lrs)

    def _lr_adjuster(step: int) -> None:
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs_list, strict=True):
            if step < warmup_length:
                lr = base_lr * (step + 1) / max(1, warmup_length)
            else:
                e = step - warmup_length
                es = max(1, steps - warmup_length)
                lr = max(0.0, 1.0 - (e / es)) * base_lr
            param_group["lr"] = lr

    return _lr_adjuster


def constant_lr(optimizer: optim.Optimizer, base_lrs: float | list[float], warmup_length: int, steps: int) -> Callable[[int], None]:
    del steps
    base_lrs_list = _normalize_base_lrs(optimizer, base_lrs)

    def _lr_adjuster(step: int) -> None:
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs_list, strict=True):
            if step < warmup_length:
                lr = base_lr * (step + 1) / max(1, warmup_length)
            else:
                lr = base_lr
            param_group["lr"] = lr

    return _lr_adjuster


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    *,
    name: str,
    base_lrs: float | list[float],
    warmup_length: int,
    steps: int,
) -> Callable[[int], None]:
    scheduler_name = str(name).strip().lower()
    if scheduler_name == "cosine":
        return cosine_lr(optimizer, base_lrs, warmup_length, steps)
    if scheduler_name == "linear":
        return linear_lr(optimizer, base_lrs, warmup_length, steps)
    if scheduler_name == "constant":
        return constant_lr(optimizer, base_lrs, warmup_length, steps)
    raise ValueError(f"Unknown lr scheduler: {name}")
