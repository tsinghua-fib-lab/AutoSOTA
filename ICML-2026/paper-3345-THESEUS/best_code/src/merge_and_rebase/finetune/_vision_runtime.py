from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from merge_and_rebase.io.ckpt import align_to_base_keys, normalize_common_prefixes, unwrap_state_dict
from merge_and_rebase.io.peft_helpers import state_dict_looks_patched_attn
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier

from ._vision_scaled_forward import (
    effective_parameter_map,
    parameter_maps_compatible,
    run_scaled_image_encoder,
    scaled_parameter_map,
    snapshot_parameter_map,
)


class ImageEncoder(nn.Module):
    """
    Wraps an OpenCLIP image encoder + optional linear head.
    Forward: images -> logits [B, C]
    """

    def __init__(self, classifier: OpenClipClassifier) -> None:
        super().__init__()
        self.clip_model = classifier
        self.head: nn.Linear | None = None
        self._last_visual_features: torch.Tensor | None = None
        self._last_image_features: torch.Tensor | None = None
        self._last_logits: torch.Tensor | None = None
        for param in self.clip_model.model.transformer.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        visual_features = self.clip_model.model.visual(images)
        image_features = visual_features
        if self.clip_model.normalize:
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-12)

        if self.clip_model._zs_text_features.numel() == 0:
            raise RuntimeError("Call build_zeroshot_text_features() before forward in zero-shot mode.")

        if self.head is not None:
            logits = self.clip_model.logit_scale * self.head(image_features)
        else:
            logits = self.clip_model.logit_scale * (image_features @ self.clip_model._zs_text_features.t())
        self._last_visual_features = visual_features
        self._last_image_features = image_features
        self._last_logits = logits
        return logits

    @torch.no_grad()
    def top1(self, loader, device: str) -> float:
        dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.to(dev)
        self.eval()

        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            logits = self(x)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        return float(correct / max(1, total))


def build_image_encoder(*, build_cfg: OpenClipBuildConfig, device: torch.device | None = None) -> ImageEncoder:
    classifier = OpenClipClassifier.build(build_cfg)
    model = ImageEncoder(classifier)
    if device is not None:
        model = model.to(device)
    return model


def _select_compatible_state_dict(
    sd: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k not in base:
            continue
        if tuple(v.shape) != tuple(base[k].shape):
            continue
        out[k] = v
    return out


def _load_compatible_state_dict(
    target: nn.Module,
    sd: dict[str, torch.Tensor],
    *,
    allow_align: bool,
) -> tuple[int, str]:
    base = target.state_dict()
    exact = _select_compatible_state_dict(sd, base)
    if exact:
        target.load_state_dict(exact, strict=False)
        return len(exact), "exact"

    if not allow_align:
        return 0, "exact"

    aligned = align_to_base_keys(normalize_common_prefixes(sd), base)
    if aligned:
        target.load_state_dict(aligned, strict=False)
        return len(aligned), "aligned"
    return 0, "aligned"


def load_model_init_checkpoint(
    *,
    model: ImageEncoder,
    ckpt_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and str(obj.get("format", "")).strip().lower() == "peft":
        raise ValueError(
            f"Checkpoint '{ckpt_path}' is a PEFT adapter checkpoint. "
            "train_vision initialization currently supports full checkpoints or raw state_dicts only."
        )

    raw_sd = unwrap_state_dict(obj)
    wrapper_loaded, wrapper_mode = _load_compatible_state_dict(model, raw_sd, allow_align=False)
    if wrapper_loaded > 0:
        tuned_text = OpenClipClassifier.extract_tuned_text_features_from_checkpoint(obj=obj, ckpt_path=ckpt_path)
        summary = {
            "checkpoint": str(ckpt_path),
            "loaded_tensors": int(wrapper_loaded),
            "load_target": "wrapper",
            "load_mode": wrapper_mode,
            "checkpoint_format": obj.get("format") if isinstance(obj, dict) else None,
            "checkpoint_task": obj.get("task") if isinstance(obj, dict) else None,
            "has_tuned_text_features": bool(isinstance(tuned_text, torch.Tensor)),
        }
        return obj, summary

    inner_target = model.clip_model.model
    inner_loaded, inner_mode = _load_compatible_state_dict(inner_target, raw_sd, allow_align=True)
    if inner_loaded > 0:
        tuned_text = OpenClipClassifier.extract_tuned_text_features_from_checkpoint(obj=obj, ckpt_path=ckpt_path)
        summary = {
            "checkpoint": str(ckpt_path),
            "loaded_tensors": int(inner_loaded),
            "load_target": "clip_model.model",
            "load_mode": inner_mode,
            "checkpoint_format": obj.get("format") if isinstance(obj, dict) else None,
            "checkpoint_task": obj.get("task") if isinstance(obj, dict) else None,
            "has_tuned_text_features": bool(isinstance(tuned_text, torch.Tensor)),
        }
        return obj, summary

    raise ValueError(
        f"Unable to load any compatible tensors from checkpoint '{ckpt_path}'. "
        "Expected a full train_vision checkpoint payload or a raw OpenCLIP-compatible state_dict."
    )


def initialize_task_text_features(
    *,
    model: ImageEncoder,
    classnames: list[str],
    build_cfg: OpenClipBuildConfig,
    device: torch.device,
    ckpt_obj: Any = None,
    ckpt_path: str | None = None,
    text_features_source: str = "zero_shot",
) -> str:
    model.clip_model.build_zeroshot_text_features(classnames, build_cfg)
    zero_shot_text = model.clip_model._zs_text_features.detach().clone()
    source = str(text_features_source).strip().lower()
    if source not in {"auto", "zero_shot", "tuned_ckpt"}:
        raise ValueError("initialization.text_features_source must be one of: auto, zero_shot, tuned_ckpt")

    tuned_text = OpenClipClassifier.extract_tuned_text_features_from_checkpoint(
        obj=ckpt_obj,
        ckpt_path=(ckpt_path or "<unknown>"),
    )
    selected = "zero_shot"
    if source == "tuned_ckpt":
        if tuned_text is None:
            raise ValueError(
                f"Checkpoint '{ckpt_path or '<unknown>'}' has no tuned_text_features. "
                "Use initialization.text_features_source='zero_shot' or 'auto', "
                "or point to a two-stage checkpoint that saved tuned text features."
            )
        selected = "tuned_ckpt"
    elif source == "auto" and tuned_text is not None:
        selected = "tuned_ckpt"

    if selected == "tuned_ckpt":
        assert isinstance(tuned_text, torch.Tensor)
        if tuple(tuned_text.shape) != tuple(zero_shot_text.shape):
            raise ValueError(
                f"Checkpoint '{ckpt_path or '<unknown>'}' tuned_text_features shape={tuple(tuned_text.shape)} "
                f"does not match current task zero-shot text shape={tuple(zero_shot_text.shape)}."
            )
        model.clip_model._zs_text_features = tuned_text.to(device=device)
        model.clip_model._zs_text_fingerprint = None
        return selected

    model.clip_model._zs_text_features = zero_shot_text.to(device=device)
    model.clip_model._zs_text_fingerprint = None
    return selected


def materialized_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    materialized_state_getter = getattr(model, "_materialized_state_dict", None)
    if callable(materialized_state_getter):
        return materialized_state_getter()
    return model.state_dict()


def build_vision_model_payload(
    *,
    model: ImageEncoder,
    build_cfg: OpenClipBuildConfig,
    forward_mode: str,
    forward_mode_params: dict[str, Any],
    strategy: str | None = None,
    task: str | None = None,
    classnames: list[str] | None = None,
    num_classes: int | None = None,
    checkpoint_init_summary: dict[str, Any] | None = None,
    text_features_init_source: str | None = None,
    include_weights: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backbone": {
            "kind": "openclip",
            "model_name": build_cfg.model_name,
            "pretrained": build_cfg.pretrained,
            "dtype": build_cfg.dtype,
        },
        "forward_mode": str(forward_mode),
        "forward_mode_params": dict(forward_mode_params),
        "patched_attn": False,
    }
    if strategy is not None:
        payload["strategy"] = str(strategy)
    if task is not None:
        payload["task"] = str(task)
    if classnames is not None:
        payload["classnames"] = list(classnames)
    if num_classes is not None:
        payload["num_classes"] = int(num_classes)
    if checkpoint_init_summary is not None:
        payload["initialization"] = dict(checkpoint_init_summary)
    if text_features_init_source is not None:
        payload["text_features_init_source"] = str(text_features_init_source)
    tuned_text = getattr(model.clip_model, "_zs_text_features", None)
    if isinstance(tuned_text, torch.Tensor) and tuned_text.ndim == 2 and tuned_text.numel() > 0:
        payload["tuned_text_features"] = tuned_text.detach().cpu()

    model_sd = materialized_model_state_dict(model)
    payload["patched_attn"] = bool(getattr(model, "peft_patched_attn", False)) or state_dict_looks_patched_attn(model_sd)
    attn_patch_cfg_raw = getattr(model, "peft_attn_patch_cfg", None)
    if isinstance(attn_patch_cfg_raw, dict):
        payload["attn_patch_cfg"] = dict(attn_patch_cfg_raw)
    if include_weights:
        payload["state_dict"] = {k: v.detach().cpu() for k, v in model_sd.items()}
        payload["format"] = "full"
    return payload
