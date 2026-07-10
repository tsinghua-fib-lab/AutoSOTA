from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.optim as optim


def build_optimizer(
    params: Iterable[nn.Parameter],
    opt: str,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    opt_name = opt.lower()
    if opt_name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if opt_name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "muon":
        muon_cls = getattr(optim, "Muon", None)
        if muon_cls is None:
            raise ValueError("Optimizer 'muon' is not available in this torch build.")
        return muon_cls(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {opt}")


@runtime_checkable
class Strategy(Protocol):
    name: str

    def configure(
        self,
        *,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        device: torch.device,
        **kwargs,
    ) -> tuple[optim.Optimizer, Callable[[int], None], dict[str, int]]:
        """
        Must:
          - set requires_grad appropriately
          - return optimizer, scheduler(step), and info dict
        """
