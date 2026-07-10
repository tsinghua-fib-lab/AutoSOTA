from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from .base import build_optimizer
from merge_and_rebase.finetune.schedulers import build_lr_scheduler
from .registry import register


@dataclass(frozen=True)
class LinearProbe:
    name: str = "linear_probe"

    def configure(
        self,
        *,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        warmup_length: int,
        scheduler_name: str = "cosine",
        optimizer: str = "adamw",
        steps: int,
        device: torch.device,
        **kwargs,
    ) -> tuple[optim.Optimizer, Callable[[int], None], dict[str, int]]:
        if not hasattr(model, "clip_model"):
            raise ValueError("LinearProbe expects an ImageEncoder-style model with `clip_model`.")

        for p in model.parameters():
            p.requires_grad = False

        text_features = getattr(model.clip_model, "_zs_text_features", None)
        if not isinstance(text_features, torch.Tensor) or text_features.ndim != 2 or text_features.numel() == 0:
            raise RuntimeError("LinearProbe requires zero-shot text features to be built before strategy.configure().")

        num_classes, feat_dim = int(text_features.shape[0]), int(text_features.shape[1])
        head = nn.Linear(feat_dim, num_classes, bias=False)
        head.weight.data.copy_(text_features.detach().to(dtype=head.weight.dtype, device=head.weight.device))
        head = head.to(device=device, dtype=text_features.dtype)
        head.weight.requires_grad_(True)
        model.head = head
        model.to(device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        opt = build_optimizer(trainable_params, optimizer, lr, weight_decay)
        scheduler = build_lr_scheduler(
            opt,
            name=scheduler_name,
            base_lrs=lr,
            warmup_length=warmup_length,
            steps=steps,
        )
        info = {
            "trainable_params": sum(p.numel() for p in trainable_params),
            "head_params": int(model.head.weight.numel()),
            "scheduler_name": scheduler_name,
        }
        return opt, scheduler, info


register(LinearProbe())
