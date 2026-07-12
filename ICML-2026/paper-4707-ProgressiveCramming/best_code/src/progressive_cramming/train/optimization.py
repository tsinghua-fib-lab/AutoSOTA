from typing import Any

import torch
from torch.optim import SGD, AdamW, Optimizer
from transformers import get_scheduler


def build_optimizer_and_scheduler(
    args,
    parameters: list[torch.nn.Parameter],
    num_training_steps: int | None = None,
    num_processes: int = 1,
) -> tuple[Optimizer, Any]:
    """Build the AdamW/SGD optimizer + LR scheduler from training args."""
    if args.optim == "adamw_torch":
        optimizer = AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.adam_beta1, args.adam_beta2),
        )
    elif args.optim == "sgd":
        optimizer = SGD(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError("Only AdamW and SGD are supported!")

    lr_scheduler = None
    if num_training_steps is not None:
        if args.lr_scheduler_kwargs is not None:
            assert args.lr_scheduler_kwargs["min_lr"] < args.learning_rate, (
                f"min_lr must be lower than regular LR, " f"{args.lr_scheduler_kwargs['min_lr']} < {args.learning_rate}!"
            )
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps * num_processes,
            num_training_steps=num_training_steps * num_processes,
            scheduler_specific_kwargs=args.lr_scheduler_kwargs,
        )
    return optimizer, lr_scheduler
