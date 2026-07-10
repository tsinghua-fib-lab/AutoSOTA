from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from merge_and_rebase.finetune.schedulers import build_lr_scheduler, cosine_lr
from merge_and_rebase.models.parameter_subsets import resolve_parameter_subset
from ._delta_parameterization import bind_delta_parameterization
from .base import build_optimizer
from .registry import register


@dataclass(frozen=True)
class FullFinetune:
    name: str = "full"

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
        strategy_cfg: dict | None = None,
        **kwargs,
    ) -> tuple[optim.Optimizer, Callable[[int], None], dict[str, int]]:
        del kwargs

        parameterization = str(_cfg_value(strategy_cfg, "parameterization", "weights")).strip().lower()
        trainable_params_mode = str(_cfg_value(strategy_cfg, "trainable_params", "all_trainable")).strip().lower()
        if parameterization not in {"weights", "delta"}:
            raise ValueError("strategy.params.parameterization must be one of: weights, delta")
        if trainable_params_mode not in {"all_trainable", "regularized_only"}:
            raise ValueError("strategy.params.trainable_params must be one of: all_trainable, regularized_only")

        named_params = list(model.named_parameters())
        initial_trainables = [(name, param) for name, param in named_params if param.requires_grad]
        initial_trainable_names = {name for name, _ in initial_trainables}

        effective_trainable_names = set(initial_trainable_names)
        params_mode_supported = trainable_params_mode == "all_trainable"
        if trainable_params_mode == "regularized_only":
            resolution = resolve_parameter_subset(model, "regularized_only")
            if resolution.supported:
                effective_trainable_names &= set(resolution.parameter_names)
                params_mode_supported = True

        if trainable_params_mode == "regularized_only":
            for name, param in named_params:
                param.requires_grad_(name in effective_trainable_names)
        else:
            for name, param in named_params:
                param.requires_grad_(name in initial_trainable_names)

        if parameterization == "weights":
            opt_params = [param for _, param in named_params if param.requires_grad]
        else:
            opt_params = bind_delta_parameterization(
                model=model,
                named_params=named_params,
                target_names=effective_trainable_names,
                device=device,
            )

        if not opt_params:
            raise RuntimeError("full strategy produced zero trainable parameters.")

        opt = build_optimizer(opt_params, optimizer, lr, weight_decay)
        scheduler = build_lr_scheduler(
            opt,
            name=scheduler_name,
            base_lrs=lr,
            warmup_length=warmup_length,
            steps=steps,
        )
        info = {
            "trainable_params": int(sum(p.numel() for p in opt_params)),
            "delta_params": int(sum(p.numel() for p in opt_params)) if parameterization == "delta" else 0,
            "trainable_params_fallback": int(trainable_params_mode == "regularized_only" and not params_mode_supported),
        }
        info["parameterization"] = parameterization
        info["trainable_params_mode"] = trainable_params_mode
        info["trainable_params_effective"] = "all_trainable" if not params_mode_supported else trainable_params_mode
        info["scheduler_name"] = scheduler_name

        return opt, scheduler, info


def _cfg_value(strategy_cfg: dict | None, key: str, default: Any) -> Any:
    if not isinstance(strategy_cfg, dict):
        return default
    params = strategy_cfg.get("params", None)
    if isinstance(params, dict) and key in params:
        return params[key]
    return strategy_cfg.get(key, default)


register(FullFinetune())
