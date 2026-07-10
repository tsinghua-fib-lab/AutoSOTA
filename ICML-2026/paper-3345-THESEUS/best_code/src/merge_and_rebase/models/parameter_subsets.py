from __future__ import annotations

import re
from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class ParameterSubsetResolution:
    name: str
    parameter_names: tuple[str, ...]
    supported: bool


@dataclass(frozen=True)
class VisualParameterPlan:
    name: str
    parameter_names: tuple[str, ...]
    lora_parameter_names: tuple[str, ...]
    dense_parameter_names: tuple[str, ...]
    lora_target_modules: tuple[str, ...]
    supported: bool


_VISUAL_PREFIX_RE = re.compile(r"^(clip_model\.model\.visual|model\.visual|visual)\.(.+)$")
_ATTN_SPLIT_RE = re.compile(
    r"transformer\.resblocks\.\d+\.attn\.(q_proj|k_proj|v_proj|out_proj)\.(weight|bias)$"
)
_ATTN_FUSED_RE = re.compile(r"transformer\.resblocks\.\d+\.attn\.(in_proj_weight|in_proj_bias)$")
_MLP_RE = re.compile(r"transformer\.resblocks\.\d+\.mlp\.(c_fc|c_proj)\.(weight|bias)$")
_LORA_CAPABLE_SPLIT_WEIGHT_RE = re.compile(
    r"transformer\.resblocks\.\d+\.attn\.(q_proj|k_proj|v_proj|out_proj)\.weight$"
)
_LORA_CAPABLE_MLP_WEIGHT_RE = re.compile(r"transformer\.resblocks\.\d+\.mlp\.(c_fc|c_proj)\.weight$")


def _normalize_visual_local_name(local_name: str) -> str:
    out = str(local_name)
    while out.startswith("base_model.model."):
        out = out[len("base_model.model.") :]
    out = out.replace(".base_layer.", ".")
    return out


def _visual_local_name(full_name: str) -> str | None:
    match = _VISUAL_PREFIX_RE.match(full_name)
    if match is not None:
        local_name = match.group(2)
    elif full_name.startswith("base_model.model."):
        local_name = full_name
    elif full_name.startswith(("transformer.", "class_embedding", "proj", "lin_proj.", "ln_pre.", "ln_post.")):
        local_name = full_name
    else:
        return None
    return _normalize_visual_local_name(local_name)


def _is_layer_norm_param(model: nn.Module, full_name: str) -> bool:
    module_name, _, param_name = full_name.rpartition(".")
    if param_name not in {"weight", "bias"}:
        return False
    module = dict(model.named_modules()).get(module_name, None)
    return isinstance(module, nn.LayerNorm)


def _is_regularized_only_name(model: nn.Module, full_name: str) -> bool:
    local_name = _visual_local_name(full_name)
    if local_name is None:
        return False
    if local_name == "class_embedding":
        return True
    if local_name in {"proj", "lin_proj.weight"}:
        return True
    if _ATTN_SPLIT_RE.fullmatch(local_name) is not None:
        return True
    if _ATTN_FUSED_RE.fullmatch(local_name) is not None:
        return True
    if _MLP_RE.fullmatch(local_name) is not None:
        return True
    return _is_layer_norm_param(model, full_name)


def _is_lora_capable_local_name(local_name: str) -> bool:
    if local_name in {"proj", "lin_proj.weight"}:
        return True
    if _ATTN_FUSED_RE.fullmatch(local_name) is not None:
        return local_name.endswith("weight")
    if _LORA_CAPABLE_SPLIT_WEIGHT_RE.fullmatch(local_name) is not None:
        return True
    return _LORA_CAPABLE_MLP_WEIGHT_RE.fullmatch(local_name) is not None


def _lora_target_modules_for_local_name(local_name: str) -> set[str]:
    if local_name in {"proj", "lin_proj.weight"}:
        return {"lin_proj"}
    if _ATTN_FUSED_RE.fullmatch(local_name) is not None:
        return {"q_proj", "k_proj", "v_proj", "out_proj"}
    attn_match = _LORA_CAPABLE_SPLIT_WEIGHT_RE.fullmatch(local_name)
    if attn_match is not None:
        return {str(attn_match.group(1))}
    mlp_match = _LORA_CAPABLE_MLP_WEIGHT_RE.fullmatch(local_name)
    if mlp_match is not None:
        return {str(mlp_match.group(1))}
    return set()


def resolve_visual_parameter_plan(model: nn.Module, name: str) -> VisualParameterPlan:
    subset_name = str(name).strip().lower()
    if subset_name != "regularized_only":
        return VisualParameterPlan(
            name=subset_name,
            parameter_names=(),
            lora_parameter_names=(),
            dense_parameter_names=(),
            lora_target_modules=(),
            supported=False,
        )

    parameter_names: list[str] = []
    lora_parameter_names: list[str] = []
    dense_parameter_names: list[str] = []
    lora_target_modules: set[str] = set()

    for full_name, _param in model.named_parameters():
        local_name = _visual_local_name(full_name)
        if local_name is None or not _is_regularized_only_name(model, full_name):
            continue
        parameter_names.append(full_name)
        if _is_lora_capable_local_name(local_name):
            lora_parameter_names.append(full_name)
            lora_target_modules.update(_lora_target_modules_for_local_name(local_name))
        else:
            dense_parameter_names.append(full_name)

    return VisualParameterPlan(
        name=subset_name,
        parameter_names=tuple(parameter_names),
        lora_parameter_names=tuple(lora_parameter_names),
        dense_parameter_names=tuple(dense_parameter_names),
        lora_target_modules=tuple(sorted(lora_target_modules)),
        supported=bool(parameter_names),
    )


def resolve_parameter_subset(model: nn.Module, name: str) -> ParameterSubsetResolution:
    plan = resolve_visual_parameter_plan(model, name)
    return ParameterSubsetResolution(name=plan.name, parameter_names=plan.parameter_names, supported=plan.supported)
