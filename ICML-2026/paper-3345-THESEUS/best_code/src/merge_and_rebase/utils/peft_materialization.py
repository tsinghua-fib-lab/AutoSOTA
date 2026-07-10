from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn


def is_lora_parameter_name(name: str) -> bool:
    return "lora_" in str(name)


def is_peft_linear_module(module: nn.Module) -> bool:
    return hasattr(module, "base_layer") and isinstance(getattr(module, "base_layer"), nn.Linear)


def materialize_peft_lora_weight(module: nn.Module) -> torch.Tensor:
    if not is_peft_linear_module(module):
        raise TypeError("Expected a PEFT LoRA-wrapped linear module.")

    base_layer = getattr(module, "base_layer")
    weight = base_layer.weight
    delta = torch.zeros_like(weight)
    lora_a = getattr(module, "lora_A", {})
    lora_b = getattr(module, "lora_B", {})
    scaling = getattr(module, "scaling", {})

    for adapter_name, a_mod in lora_a.items():
        b_mod = lora_b[adapter_name] if adapter_name in lora_b else None
        if b_mod is None:
            continue
        scale = float(scaling.get(adapter_name, 1.0))
        delta = delta + (b_mod.weight @ a_mod.weight).to(device=weight.device, dtype=weight.dtype) * scale

    return weight + delta


def _param_from_map_or_module(
    *,
    param_map: Mapping[str, torch.Tensor] | None,
    full_name: str,
    fallback: torch.Tensor,
) -> torch.Tensor:
    if param_map is None:
        return fallback
    value = param_map.get(full_name, None)
    if isinstance(value, torch.Tensor):
        return value.to(device=fallback.device, dtype=fallback.dtype)
    return fallback


def lora_factor_parameter_names(module_name: str, module: nn.Module) -> tuple[str, ...]:
    if not is_peft_linear_module(module):
        raise TypeError("Expected a PEFT LoRA-wrapped linear module.")

    names: list[str] = []
    prefix = f"{module_name}." if module_name else ""
    lora_a = getattr(module, "lora_A", {})
    lora_b = getattr(module, "lora_B", {})
    for adapter_name in lora_a:
        if adapter_name not in lora_b:
            continue
        names.append(f"{prefix}lora_A.{adapter_name}.weight")
        names.append(f"{prefix}lora_B.{adapter_name}.weight")
    return tuple(names)


def materialize_peft_lora_weight_from_param_map(
    module_name: str,
    module: nn.Module,
    *,
    param_map: Mapping[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if not is_peft_linear_module(module):
        raise TypeError("Expected a PEFT LoRA-wrapped linear module.")

    base_layer = getattr(module, "base_layer")
    prefix = f"{module_name}." if module_name else ""
    weight_name = f"{prefix}base_layer.weight"
    weight = _param_from_map_or_module(
        param_map=param_map,
        full_name=weight_name,
        fallback=base_layer.weight,
    )
    delta = torch.zeros_like(weight)
    lora_a = getattr(module, "lora_A", {})
    lora_b = getattr(module, "lora_B", {})
    scaling = getattr(module, "scaling", {})

    for adapter_name, a_mod in lora_a.items():
        b_mod = lora_b[adapter_name] if adapter_name in lora_b else None
        if b_mod is None:
            continue
        a = _param_from_map_or_module(
            param_map=param_map,
            full_name=f"{prefix}lora_A.{adapter_name}.weight",
            fallback=a_mod.weight,
        )
        b = _param_from_map_or_module(
            param_map=param_map,
            full_name=f"{prefix}lora_B.{adapter_name}.weight",
            fallback=b_mod.weight,
        )
        scale = float(scaling.get(adapter_name, 1.0))
        delta = delta + (b @ a).to(device=weight.device, dtype=weight.dtype) * scale

    return weight + delta


def materialized_linearization_param_names(
    module: nn.Module,
    *,
    trainable_only: bool,
) -> list[str]:
    lora_host_weight_names: set[str] = set()
    for module_name, submodule in module.named_modules():
        if not is_peft_linear_module(submodule):
            continue
        weight_name = f"{module_name}.base_layer.weight" if module_name else "base_layer.weight"
        if trainable_only:
            has_trainable_lora = any(
                param.requires_grad and is_lora_parameter_name(name)
                for name, param in submodule.named_parameters()
            )
            if has_trainable_lora:
                lora_host_weight_names.add(weight_name)
        else:
            lora_host_weight_names.add(weight_name)

    out: list[str] = []
    seen: set[str] = set()
    for name, param in module.named_parameters():
        if is_lora_parameter_name(name):
            continue
        if trainable_only and not param.requires_grad and name not in lora_host_weight_names:
            continue
        if name in seen:
            continue
        out.append(name)
        seen.add(name)
    return out


def training_linearization_param_names(
    module: nn.Module,
    *,
    trainable_only: bool,
    extra_param_names: Iterable[str] | None = None,
) -> list[str]:
    out = materialized_linearization_param_names(module, trainable_only=trainable_only)
    seen = set(out)
    module_param_names = {name for name, _ in module.named_parameters()}

    delta_module = getattr(module, "_delta_module", None)
    delta_names = getattr(delta_module, "names", ()) if delta_module is not None else ()
    for name in tuple(delta_names) + tuple(extra_param_names or ()):
        if name not in module_param_names or name in seen:
            continue
        out.append(name)
        seen.add(name)
    return out


def materialized_peft_param_map(
    module: nn.Module,
    *,
    raw_current_params: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    raw = dict(raw_current_params or {})
    out: dict[str, torch.Tensor] = {}

    for name, param in module.named_parameters():
        if is_lora_parameter_name(name):
            continue
        out[name] = raw.get(name, param)

    for module_name, submodule in module.named_modules():
        if not is_peft_linear_module(submodule):
            continue
        weight_name = f"{module_name}.base_layer.weight" if module_name else "base_layer.weight"
        out[weight_name] = materialize_peft_lora_weight_from_param_map(
            module_name,
            submodule,
            param_map=raw,
        )

    return out
