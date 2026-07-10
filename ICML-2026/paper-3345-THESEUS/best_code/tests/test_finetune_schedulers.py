from __future__ import annotations

import math

import pytest
import torch

from merge_and_rebase.finetune.schedulers import build_lr_scheduler
from merge_and_rebase.finetune.strategies.registry import get_strategy


def test_build_lr_scheduler_rejects_unknown_name() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.SGD([param], lr=0.2)
    with pytest.raises(ValueError, match="Unknown lr scheduler"):
        build_lr_scheduler(opt, name="triangle", base_lrs=0.2, warmup_length=0, steps=4)


def test_build_lr_scheduler_uses_configured_curve_and_warmup() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))

    opt_constant = torch.optim.SGD([param], lr=0.2)
    sched_constant = build_lr_scheduler(opt_constant, name="constant", base_lrs=0.2, warmup_length=2, steps=6)
    sched_constant(0)
    assert math.isclose(opt_constant.param_groups[0]["lr"], 0.1, rel_tol=1e-6)
    sched_constant(4)
    assert math.isclose(opt_constant.param_groups[0]["lr"], 0.2, rel_tol=1e-6)

    opt_linear = torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.2)
    sched_linear = build_lr_scheduler(opt_linear, name="linear", base_lrs=0.2, warmup_length=2, steps=6)
    sched_linear(4)
    assert math.isclose(opt_linear.param_groups[0]["lr"], 0.1, rel_tol=1e-6)
    sched_linear(6)
    assert math.isclose(opt_linear.param_groups[0]["lr"], 0.0, abs_tol=1e-9)

    opt_cosine = torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.2)
    sched_cosine = build_lr_scheduler(opt_cosine, name="cosine", base_lrs=0.2, warmup_length=2, steps=6)
    sched_cosine(4)
    assert math.isclose(opt_cosine.param_groups[0]["lr"], 0.1, rel_tol=1e-6)


def test_build_lr_scheduler_updates_multiple_param_groups() -> None:
    p1 = torch.nn.Parameter(torch.tensor([1.0]))
    p2 = torch.nn.Parameter(torch.tensor([2.0]))
    opt = torch.optim.SGD(
        [
            {"params": [p1], "lr": 0.2},
            {"params": [p2], "lr": 0.05},
        ]
    )

    scheduler = build_lr_scheduler(opt, name="constant", base_lrs=[0.2, 0.05], warmup_length=2, steps=6)
    scheduler(0)
    assert math.isclose(opt.param_groups[0]["lr"], 0.1, rel_tol=1e-6)
    assert math.isclose(opt.param_groups[1]["lr"], 0.025, rel_tol=1e-6)
    scheduler(4)
    assert math.isclose(opt.param_groups[0]["lr"], 0.2, rel_tol=1e-6)
    assert math.isclose(opt.param_groups[1]["lr"], 0.05, rel_tol=1e-6)


def test_full_strategy_honors_scheduler_name() -> None:
    model = torch.nn.Linear(2, 1)
    strategy = get_strategy("full")
    opt, scheduler, info = strategy.configure(
        model=model,
        lr=0.2,
        weight_decay=0.0,
        warmup_length=2,
        scheduler_name="constant",
        optimizer="sgd",
        steps=6,
        device=torch.device("cpu"),
    )

    scheduler(0)
    assert math.isclose(opt.param_groups[0]["lr"], 0.1, rel_tol=1e-6)
    scheduler(5)
    assert math.isclose(opt.param_groups[0]["lr"], 0.2, rel_tol=1e-6)
    assert info["scheduler_name"] == "constant"
