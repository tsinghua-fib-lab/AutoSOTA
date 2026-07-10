from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from ..models.grad_recipes import clip_contrastive_recipe
from ..models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier
from ..rebase.base import TensorDict
from .utils import build_grad_dataloader


def resolve_rebase_method_config(cfg: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    method_name = str(cfg.get("method", "gradfix"))
    method_params = cfg.get("method_params", {})
    if method_params is None:
        method_params = {}
    if not isinstance(method_params, dict):
        raise ValueError("config['method_params'] must be a dict when provided.")

    out = dict(method_params)
    if method_name == "gradfix":
        for legacy_key in ("mask_mode", "vote"):
            if legacy_key not in out and cfg.get(legacy_key) is not None:
                out[legacy_key] = cfg[legacy_key]
    return method_name, out


def format_rebase_method_label(method_name: str, method_params: dict[str, Any]) -> str:
    if method_name != "gradfix":
        return method_name
    mask_mode = str(method_params.get("mask_mode", "normal"))
    vote = str(method_params.get("vote", "mean"))
    return f"gradfix(mask={mask_mode}, vote={vote})"


def transport_vision_task_vector(
    *,
    method: Any,
    source_base: dict[str, torch.Tensor],
    target_base: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
    clf_target: OpenClipClassifier,
    clf_source: OpenClipClassifier | None,
    task_name: str,
    loaders: Any,
    source_loaders: Any | None,
    classnames: list[str],
    build_cfg_task: OpenClipBuildConfig,
    device: str,
    strict: bool,
    method_params: dict[str, Any],
    grad_batch_size: int | None,
    grad_imgs_per_class: int | None,
    grad_num_batches: int | None,
    num_workers: int,
    seed: int,
) -> TensorDict:
    method_name = getattr(method, "name", None)

    if method_name != "gradfix":
        method_kwargs = dict(method_params)
        extra_kwargs: dict[str, Any] = {}

        if method_name == "theseus":
            if clf_source is None or source_loaders is None:
                raise ValueError(
                    f"Method '{method_name}' requires source model and source dataloader, but they were not provided."
                )
            extra_kwargs = {
                "source_model": clf_source.model,
                "target_model": clf_target.model,
                "source_dataloader": source_loaders.train,
                "target_dataloader": loaders.train,
            }
            method_kwargs.setdefault("device", device)
            method_kwargs.setdefault("seed", int(seed))

        return method.transport(
            source_base=source_base,
            target_base=target_base,
            delta=delta,
            strict=strict,
            **extra_kwargs,
            **method_kwargs,
        )

    vote = str(method_params.get("vote", "mean"))
    grad_loader = build_grad_dataloader(
        loaders.train,
        loaders.train.dataset,
        grad_batch_size=grad_batch_size,
        grad_imgs_per_class=grad_imgs_per_class,
        grad_num_batches=grad_num_batches,
        num_workers=num_workers,
        seed=seed,
    )
    n_grad_samples = len(grad_loader.dataset) if hasattr(grad_loader, "dataset") else "?"
    print(
        f"  {task_name}: grad dataloader - {n_grad_samples} samples ({grad_imgs_per_class} per class), "
        f"batch_size={getattr(grad_loader, 'batch_size', '?')}"
    )

    recipe = clip_contrastive_recipe(
        clf_target,
        classnames,
        build_cfg_task,
        device=device,
        reduction="none" if vote in {"majority", "max"} else "mean",
    )
    target_model_copy = deepcopy(clf_target.model)
    try:
        return method.transport(
            source_base=source_base,
            target_base=target_base,
            delta=delta,
            strict=strict,
            target_model=target_model_copy,
            target_dataloader=grad_loader,
            recipe=recipe,
            device=device,
            **method_params,
        )
    finally:
        del target_model_copy
