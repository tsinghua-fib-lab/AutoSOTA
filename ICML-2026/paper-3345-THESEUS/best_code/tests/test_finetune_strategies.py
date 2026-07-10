from __future__ import annotations

import torch
import torch.nn as nn

from merge_and_rebase.finetune.strategies.base import build_optimizer
from merge_and_rebase.finetune.strategies.full import FullFinetune


def test_build_optimizer_selects_requested_optimizer() -> None:
    model = nn.Linear(4, 2)

    optimizer = build_optimizer(model.parameters(), "adamw", lr=1e-3, weight_decay=0.01)

    assert isinstance(optimizer, torch.optim.AdamW)


def test_full_finetune_configure_uses_shared_optimizer_and_counts_params() -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    expected_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer, scheduler, info = FullFinetune().configure(
        model=model,
        lr=1e-3,
        weight_decay=0.01,
        warmup_length=0,
        optimizer="adamw",
        steps=4,
        device=torch.device("cpu"),
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert callable(scheduler)
    assert info["trainable_params"] == expected_params
