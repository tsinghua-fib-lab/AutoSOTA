from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FixedPointConfig:
    n_bits: int
    frac_bits: int
    signed: bool = True
    rounding: str = "round"  # round | trunc | floor

    @property
    def scale(self) -> float:
        return float(1 << self.frac_bits)


def _wrap_signed(q: torch.Tensor, n_bits: int) -> torch.Tensor:
    mod = 1 << n_bits
    max_pos = 1 << (n_bits - 1)
    q = torch.remainder(q, mod)
    return torch.where(q >= max_pos, q - mod, q)


def _apply_rounding(x: torch.Tensor, mode: str) -> torch.Tensor:
    mode = mode.lower()
    if mode == "round":
        return torch.round(x)
    if mode == "trunc":
        return torch.trunc(x)
    if mode == "floor":
        return torch.floor(x)
    raise ValueError(f"Unsupported rounding mode: {mode}")


def quantize(x: torch.Tensor, cfg: FixedPointConfig) -> torch.Tensor:
    if not torch.is_floating_point(x):
        x = x.float()
    q = _apply_rounding(x * cfg.scale, cfg.rounding).to(torch.int64)
    if cfg.signed:
        q = _wrap_signed(q, cfg.n_bits)
    else:
        q = torch.remainder(q, 1 << cfg.n_bits)
    return q.to(torch.float32) / cfg.scale


def _quantize_nested(obj: Any, cfg: FixedPointConfig) -> Any:
    if torch.is_tensor(obj):
        return quantize(obj, cfg)
    if isinstance(obj, tuple):
        return tuple(_quantize_nested(v, cfg) for v in obj)
    if isinstance(obj, list):
        return [_quantize_nested(v, cfg) for v in obj]
    if isinstance(obj, dict):
        return {k: _quantize_nested(v, cfg) for k, v in obj.items()}
    return obj


def quantize_module_params(module: nn.Module, cfg: FixedPointConfig) -> None:
    for name, param in module.named_parameters(recurse=False):
        if param is None:
            continue
        with torch.no_grad():
            param.copy_(quantize(param.data, cfg))
    for name, buffer in module.named_buffers(recurse=False):
        if buffer is None:
            continue
        with torch.no_grad():
            module._buffers[name] = quantize(buffer, cfg)


def apply_fixed_point_emulation(
    model: nn.Module,
    cfg: FixedPointConfig,
    quantize_weights: bool = True,
) -> Tuple[nn.Module, Iterable[torch.utils.hooks.RemovableHandle]]:
    handles = []

    if quantize_weights:
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                quantize_module_params(module, cfg)

    hook_classes = (
        nn.Linear,
        nn.Embedding,
        nn.LayerNorm,
        nn.Softmax,
        nn.GELU,
        nn.SiLU,
    )
    extra_class_names = {
        "GELUActivation",
        "NewGELUActivation",
        "FastGELUActivation",
        "QuickGELUActivation",
        "SiLUActivation",
        "GPT2Block",
        "GPTNeoBlock",
        "BertLayer",
        "BertOutput",
        "BertSelfOutput",
        "GPT2MLP",
        "GPT2Attention",
        "GPTNeoMLP",
        "GPTNeoAttention",
    }

    def hook_fn(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        return _quantize_nested(output, cfg)

    for module in model.modules():
        if isinstance(module, hook_classes) or module.__class__.__name__ in extra_class_names:
            handles.append(module.register_forward_hook(hook_fn))

    return model, handles
