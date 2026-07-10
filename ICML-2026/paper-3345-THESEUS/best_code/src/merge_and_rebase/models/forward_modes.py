from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MethodType
from typing import Any, Protocol

import torch
import torch.nn as nn

from merge_and_rebase.models.openclip_classifier import (
    OpenClipClassifier,
    normalize_features,
    zero_shot_logits_from_features,
)
from merge_and_rebase.utils.linearization import LinearizedModule
from merge_and_rebase.utils.peft_materialization import (
    materialized_peft_param_map,
    training_linearization_param_names,
)


def _as_forward_mode_params(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        params: dict[str, Any] = {}
    elif isinstance(raw, Mapping):
        params = dict(raw)
    else:
        raise TypeError("forward_mode_params must be a mapping when provided.")
    params["linearized_feature_normalization"] = bool(params.get("linearized_feature_normalization", True))
    params["linearized_logit_normalization"] = bool(params.get("linearized_logit_normalization", True))
    return params


def normalize_forward_mode_params(forward_mode: str, params: Mapping[str, Any] | None) -> dict[str, Any]:
    mode_name = str(forward_mode).strip().lower()
    if mode_name == "linearized_ntk":
        return _as_forward_mode_params(params)
    return {}


def resolve_shared_forward_mode_params(
    forward_mode: str,
    params_list: Sequence[Mapping[str, Any] | None],
) -> dict[str, Any]:
    mode_name = str(forward_mode).strip().lower()
    if mode_name != "linearized_ntk":
        return {}
    if not params_list:
        return _as_forward_mode_params(None)
    normalized = [normalize_forward_mode_params(mode_name, params) for params in params_list]
    first = normalized[0]
    for other in normalized[1:]:
        if other != first:
            raise ValueError("Inconsistent forward_mode_params across checkpoints for linearized_ntk.")
    return first


class ForwardMode(Protocol):
    name: str

    def bind(
        self,
        *,
        clf: OpenClipClassifier,
        base_sd: dict[str, torch.Tensor],
        strict_load: bool,
        params: Mapping[str, Any] | None = None,
    ) -> None: ...


def _visual_output_transform(classifier: OpenClipClassifier) -> Callable[[torch.Tensor], torch.Tensor] | None:
    if not bool(getattr(classifier, "normalize", False)):
        return None
    return normalize_features


def _linearized_post_transform(
    classifier: OpenClipClassifier,
    *,
    params: Mapping[str, Any] | None,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    if not bool(getattr(classifier, "normalize", False)):
        return None
    if not bool(normalize_forward_mode_params("linearized_ntk", params).get("linearized_feature_normalization", True)):
        return None
    return normalize_features


def _build_linearized_visual(
    *,
    current_visual: nn.Module,
    base_sd: dict[str, torch.Tensor],
    strict_load: bool,
    param_names: list[str] | None = None,
) -> LinearizedModule:
    device = next(current_visual.parameters()).device
    ref_visual = deepcopy(current_visual).to(device)
    ref_visual.eval()
    for p in ref_visual.parameters():
        p.requires_grad = False

    visual_base_sd = {
        key[len("visual.") :]: value.to(device=device) for key, value in base_sd.items() if key.startswith("visual.")
    }
    if not visual_base_sd:
        raise ValueError("linearized_ntk forward mode requires base state_dict keys prefixed with 'visual.'.")

    miss, unexp = ref_visual.load_state_dict(visual_base_sd, strict=strict_load)
    if strict_load and (miss or unexp):
        raise RuntimeError(
            f"Failed to load base visual weights for linearized mode. missing={len(miss)}, unexpected={len(unexp)}"
        )
    return LinearizedModule.from_module(ref_visual, copy_module=False, param_names=param_names)


def _prefixed_local_param_map(
    raw: Mapping[str, torch.Tensor],
    *,
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            continue
        for prefix in prefixes:
            if key.startswith(prefix):
                out[key[len(prefix) :]] = value
                break
    return out


def _visual_param_map_from_model(model: nn.Module, current_visual: nn.Module) -> dict[str, torch.Tensor]:
    getter = getattr(model, "_current_param_map", None)
    prefixes = ("clip_model.model.visual.", "model.visual.", "visual.")
    local_map: dict[str, torch.Tensor] | None = None
    if callable(getter):
        raw = getter()
        if isinstance(raw, Mapping):
            local = _prefixed_local_param_map(raw, prefixes=prefixes)
            if local:
                local_map = local
    return materialized_peft_param_map(current_visual, raw_current_params=local_map)


def _visual_delta_param_names_from_model(model: nn.Module) -> list[str]:
    delta_module = getattr(model, "_delta_module", None)
    raw_names = tuple(getattr(delta_module, "names", ())) if delta_module is not None else ()
    prefixes = ("clip_model.model.visual.", "model.visual.", "visual.")
    out: list[str] = []
    seen: set[str] = set()
    for name in raw_names:
        for prefix in prefixes:
            if not str(name).startswith(prefix):
                continue
            local_name = str(name)[len(prefix) :]
            if local_name not in seen:
                out.append(local_name)
                seen.add(local_name)
            break
    return out


def _project_linearized_features(
    *,
    target: Any,
    classifier: OpenClipClassifier,
    image_features: torch.Tensor,
    params: Mapping[str, Any] | None,
) -> torch.Tensor:
    head = getattr(target, "head", None)
    if isinstance(head, nn.Module):
        return classifier.logit_scale * head(image_features)
    mode_params = normalize_forward_mode_params("linearized_ntk", params)
    return zero_shot_logits_from_features(
        classifier,
        image_features,
        normalize_image_features=bool(mode_params.get("linearized_logit_normalization", True)),
    )


def _stash_last_features(*, target: Any, classifier: OpenClipClassifier, visual_features: torch.Tensor, image_features: torch.Tensor, logits: torch.Tensor) -> None:
    for obj in {id(target): target, id(classifier): classifier}.values():
        setattr(obj, "_last_visual_features", visual_features)
        setattr(obj, "_last_image_features", image_features)
        setattr(obj, "_last_logits", logits)


def _bind_linearized_classifier_forward(
    *,
    target: Any,
    classifier: OpenClipClassifier,
    current_visual: nn.Module,
    base_sd: dict[str, torch.Tensor],
    strict_load: bool,
    params: Mapping[str, Any] | None,
    current_params_getter: Callable[[], Mapping[str, torch.Tensor]] | None = None,
    param_names: list[str] | None = None,
) -> LinearizedModule:
    linearized = _build_linearized_visual(
        current_visual=current_visual,
        base_sd=base_sd,
        strict_load=strict_load,
        param_names=param_names,
    )
    output_transform = _visual_output_transform(classifier)
    post_transform = _linearized_post_transform(classifier, params=params)
    normalized_params = normalize_forward_mode_params("linearized_ntk", params)

    def _linearized_forward(self, images: torch.Tensor) -> torch.Tensor:
        current_params = current_params_getter() if callable(current_params_getter) else None
        visual_features = linearized.forward(
            current_module=current_visual,
            current_params=current_params,
            args=(images,),
        )
        if output_transform is None and post_transform is None:
            image_features = visual_features
        else:
            image_features = linearized.forward(
                current_module=current_visual,
                current_params=current_params,
                args=(images,),
                output_transform=output_transform,
                post_transform=post_transform,
            )
        logits = _project_linearized_features(
            target=self,
            classifier=classifier,
            image_features=image_features,
            params=normalized_params,
        )
        _stash_last_features(
            target=self,
            classifier=classifier,
            visual_features=visual_features,
            image_features=image_features,
            logits=logits,
        )
        return logits

    target.forward = MethodType(_linearized_forward, target)  # type: ignore[method-assign]
    target.forward_mode_name = "linearized_ntk"  # type: ignore[attr-defined]
    target.forward_mode_params = normalized_params  # type: ignore[attr-defined]
    target._linearized_visual_ref = linearized.ref_module  # type: ignore[attr-defined]
    target._linearized_module = linearized  # type: ignore[attr-defined]
    return linearized


@dataclass(frozen=True)
class StandardForwardMode:
    name: str = "standard"

    def bind(
        self,
        *,
        clf: OpenClipClassifier,
        base_sd: dict[str, torch.Tensor],
        strict_load: bool,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        del base_sd, strict_load, params
        clf.forward = clf.__class__.forward.__get__(clf, clf.__class__)  # type: ignore[method-assign]
        clf.forward_mode_name = self.name  # type: ignore[attr-defined]
        clf.forward_mode_params = {}  # type: ignore[attr-defined]


@dataclass(frozen=True)
class LinearizedNtkForwardMode:
    name: str = "linearized_ntk"

    def bind(
        self,
        *,
        clf: OpenClipClassifier,
        base_sd: dict[str, torch.Tensor],
        strict_load: bool,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        clf.forward = clf.__class__.forward.__get__(clf, clf.__class__)  # type: ignore[method-assign]
        _bind_linearized_classifier_forward(
            target=clf,
            classifier=clf,
            current_visual=clf.model.visual,
            base_sd=base_sd,
            strict_load=strict_load,
            params=params,
            param_names=training_linearization_param_names(clf.model.visual, trainable_only=False),
        )


_FORWARD_MODES: dict[str, ForwardMode] = {
    "standard": StandardForwardMode(),
    "linearized_ntk": LinearizedNtkForwardMode(),
}


def list_forward_modes() -> list[str]:
    return sorted(_FORWARD_MODES.keys())


def get_forward_mode(name: str) -> ForwardMode:
    if name not in _FORWARD_MODES:
        raise KeyError(f"Unknown forward mode '{name}'. Available: {sorted(_FORWARD_MODES)}")
    return _FORWARD_MODES[name]


def bind_training_forward_mode(
    *,
    model: nn.Module,
    forward_mode: str,
    base_sd: dict[str, torch.Tensor],
    strict_load: bool,
    params: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    mode_name = str(forward_mode).strip().lower()
    if mode_name == "standard":
        model.forward_mode_name = "standard"  # type: ignore[attr-defined]
        model.forward_mode_params = {}  # type: ignore[attr-defined]
        return {"linearized_params": 0, "linearized_buffers": 0}
    if mode_name != "linearized_ntk":
        raise KeyError(f"Unknown training forward mode '{forward_mode}'. Available: {sorted(_FORWARD_MODES)}")

    classifier = getattr(model, "clip_model", None)
    if (
        classifier is None
        or not hasattr(classifier, "model")
        or not hasattr(classifier, "_zs_text_features")
        or not hasattr(classifier, "normalize")
        or not hasattr(classifier, "logit_scale")
    ):
        raise TypeError("training forward mode binding expects model.clip_model to expose classifier-style fields")
    linearized = _bind_linearized_classifier_forward(
        target=model,
        classifier=classifier,
        current_visual=classifier.model.visual,
        base_sd=base_sd,
        strict_load=strict_load,
        params=params,
        current_params_getter=lambda: _visual_param_map_from_model(model, classifier.model.visual),
        param_names=training_linearization_param_names(
            classifier.model.visual,
            trainable_only=True,
            extra_param_names=_visual_delta_param_names_from_model(model),
        ),
    )
    model._ntk_linearized = True  # type: ignore[attr-defined]
    return {
        "linearized_params": len(linearized.param_names),
        "linearized_buffers": len(linearized.buffer_names),
    }


def resolve_auto_forward_mode(forward_modes: list[str | None]) -> str:
    if forward_modes and all(mode == "linearized_ntk" for mode in forward_modes):
        return "linearized_ntk"
    return "standard"
