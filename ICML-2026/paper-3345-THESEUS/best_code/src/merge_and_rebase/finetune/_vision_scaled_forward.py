from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.func import functional_call

from merge_and_rebase.models.openclip_classifier import normalize_features, zero_shot_logits_from_features
from merge_and_rebase.utils.peft_materialization import (
    is_peft_linear_module,
    lora_factor_parameter_names,
    materialize_peft_lora_weight_from_param_map,
)

if TYPE_CHECKING:
    from merge_and_rebase.finetune._vision_runtime import ImageEncoder


def effective_parameter_map(model: nn.Module) -> dict[str, torch.Tensor]:
    getter = getattr(model, "_current_param_map", None)
    if callable(getter):
        raw = getter()
        if isinstance(raw, dict):
            return {str(k): v for k, v in raw.items() if isinstance(v, torch.Tensor)}
    return {name: param for name, param in model.named_parameters()}


def snapshot_parameter_map(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in effective_parameter_map(model).items()}


def parameter_maps_compatible(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> bool:
    if set(lhs) != set(rhs):
        return False
    return all(tuple(lhs[key].shape) == tuple(rhs[key].shape) for key in lhs)


def scaled_parameter_map(
    *,
    model: nn.Module,
    base_params: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    current = effective_parameter_map(model)
    out: dict[str, torch.Tensor] = {}
    for key, value in current.items():
        if key in base_params and tuple(base_params[key].shape) == tuple(value.shape):
            base = base_params[key].to(device=value.device, dtype=value.dtype)
            out[key] = base + float(alpha) * (value - base)
        else:
            out[key] = value

    # PEFT LoRA modules need along-path scaling in effective weight space.
    # Scaling A and B factors independently changes the path to alpha^2 terms.
    for module_name, module in model.named_modules():
        if not is_peft_linear_module(module):
            continue
        weight_name = f"{module_name}.base_layer.weight" if module_name else "base_layer.weight"
        if weight_name not in current or weight_name not in base_params:
            continue
        current_weight = materialize_peft_lora_weight_from_param_map(module_name, module, param_map=current)
        base_weight = materialize_peft_lora_weight_from_param_map(module_name, module, param_map=base_params)
        out[weight_name] = base_weight + float(alpha) * (current_weight - base_weight)
        for factor_name in lora_factor_parameter_names(module_name, module):
            if factor_name not in base_params:
                continue
            base_factor = base_params[factor_name]
            ref = out.get(factor_name, current.get(factor_name, base_factor))
            out[factor_name] = base_factor.to(device=ref.device, dtype=ref.dtype)
    return out


def _prefixed_tensor_map(named_tensors, prefix: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    prefix_dot = f"{prefix}."
    for name, tensor in named_tensors:
        if name.startswith(prefix_dot):
            out[name[len(prefix_dot) :]] = tensor
    return out


def _stash_last_features(
    model: ImageEncoder,
    *,
    visual_features: torch.Tensor,
    image_features: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    model._last_visual_features = visual_features
    model._last_image_features = image_features
    model._last_logits = logits
    classifier = model.clip_model
    setattr(classifier, "_last_visual_features", visual_features)
    setattr(classifier, "_last_image_features", image_features)
    setattr(classifier, "_last_logits", logits)


def run_scaled_image_encoder(
    *,
    model: ImageEncoder,
    images: torch.Tensor,
    alpha: float,
    base_params: dict[str, torch.Tensor],
) -> torch.Tensor:
    scaled_params = scaled_parameter_map(model=model, base_params=base_params, alpha=alpha)
    classifier = model.clip_model
    forward_mode = str(getattr(model, "forward_mode_name", "standard")).strip().lower()

    if forward_mode == "linearized_ntk":
        linearized = getattr(model, "_linearized_module", None)
        if linearized is None:
            raise RuntimeError("linearized_ntk model is missing _linearized_module.")
        visual_params = {}
        for prefix in ("clip_model.model.visual.", "model.visual.", "visual."):
            visual_params = {k[len(prefix) :]: v for k, v in scaled_params.items() if k.startswith(prefix)}
            if visual_params:
                break
        output_transform = normalize_features if bool(getattr(classifier, "normalize", False)) else None
        mode_params = dict(getattr(model, "forward_mode_params", {}) or {})
        post_transform = None
        if bool(getattr(classifier, "normalize", False)) and bool(mode_params.get("linearized_feature_normalization", True)):
            post_transform = normalize_features
        visual_features = linearized.forward(
            current_module=classifier.model.visual,
            current_params=visual_params,
            args=(images,),
        )
        if output_transform is None and post_transform is None:
            image_features = visual_features
        else:
            image_features = linearized.forward(
                current_module=classifier.model.visual,
                current_params=visual_params,
                args=(images,),
                output_transform=output_transform,
                post_transform=post_transform,
            )
        if model.head is not None:
            head_params = _prefixed_tensor_map(scaled_params.items(), "head")
            head_buffers = dict(model.head.named_buffers())
            logits = classifier.logit_scale * functional_call(
                model.head,
                (head_params, head_buffers),
                args=(image_features,),
                strict=False,
            )
        else:
            logits = zero_shot_logits_from_features(
                classifier,
                image_features,
                normalize_image_features=bool(mode_params.get("linearized_logit_normalization", True)),
            )
        _stash_last_features(model, visual_features=visual_features, image_features=image_features, logits=logits)
        return logits

    visual_params = _prefixed_tensor_map(scaled_params.items(), "clip_model.model.visual")
    visual_buffers = dict(classifier.model.visual.named_buffers())
    visual_features = functional_call(
        classifier.model.visual,
        (visual_params, visual_buffers),
        args=(images,),
        strict=False,
    )
    image_features = normalize_features(visual_features) if classifier.normalize else visual_features
    if model.head is not None:
        head_params = _prefixed_tensor_map(scaled_params.items(), "head")
        head_buffers = dict(model.head.named_buffers())
        logits = classifier.logit_scale * functional_call(
            model.head,
            (head_params, head_buffers),
            args=(image_features,),
            strict=False,
        )
    else:
        logits = classifier.logit_scale * (image_features @ classifier._zs_text_features.t())
    _stash_last_features(model, visual_features=visual_features, image_features=image_features, logits=logits)
    return logits
