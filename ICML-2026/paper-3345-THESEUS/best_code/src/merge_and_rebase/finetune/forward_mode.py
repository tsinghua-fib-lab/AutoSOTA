from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from merge_and_rebase.models.forward_modes import list_forward_modes
from merge_and_rebase.utils.linearization import LinearizedModule
from merge_and_rebase.utils.peft_materialization import (
    materialized_peft_param_map,
    training_linearization_param_names,
)


def resolve_training_forward_mode(strategy_cfg: dict[str, Any] | None) -> str:
    cfg = dict(strategy_cfg or {})
    name = str(cfg.get("forward_mode", "standard")).strip()
    if name not in list_forward_modes():
        raise ValueError(f"strategy.forward_mode must be one of: {list_forward_modes()}")
    return name


def apply_training_forward_mode(
    *,
    model: nn.Module,
    forward_mode: str,
    device: torch.device,
    output_transform: Callable[[Any], torch.Tensor] | None = None,
    output_builder: Callable[[torch.Tensor], Any] | None = None,
) -> dict[str, int]:
    if forward_mode == "standard":
        model.forward_mode_name = forward_mode  # type: ignore[attr-defined]
        return {"linearized_params": 0, "linearized_buffers": 0}

    if forward_mode != "linearized_ntk":
        raise ValueError(f"Unsupported training forward mode: {forward_mode}")

    param_names = training_linearization_param_names(model, trainable_only=True)
    if not param_names:
        raise RuntimeError("No trainable parameters found for linearized_ntk forward mode.")

    linearized = LinearizedModule.from_module(
        model,
        device=device,
        copy_module=True,
        param_names=param_names,
    )

    def _current_param_map() -> dict[str, torch.Tensor]:
        getter = getattr(model, "_current_param_map", None)
        raw = getter() if callable(getter) else None
        current_raw = None if raw is None else dict(raw)
        return materialized_peft_param_map(model, raw_current_params=current_raw)

    def _linearized_forward(*args: Any, **kwargs: Any) -> Any:
        out = linearized.forward(
            current_module=model,
            current_params=_current_param_map(),
            args=args,
            kwargs=kwargs,
            output_transform=output_transform,
        )
        return output_builder(out) if output_builder is not None else out

    model.forward = _linearized_forward  # type: ignore[method-assign]
    model.forward_mode_name = forward_mode  # type: ignore[attr-defined]
    model._ntk_linearized = True  # type: ignore[attr-defined]
    model._linearized_module = linearized  # type: ignore[attr-defined]
    return {
        "linearized_params": len(linearized.param_names),
        "linearized_buffers": len(linearized.buffer_names),
    }
