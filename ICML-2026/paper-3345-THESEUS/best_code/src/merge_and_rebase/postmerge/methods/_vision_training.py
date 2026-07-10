from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch.func import functional_call

from ..base import PostMergeContext
from ..task_delta_bank import TaskDeltaBank


def prediction_entropy(logits: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    temp = max(1e-12, float(temperature))
    scaled = logits / temp
    probs = torch.softmax(scaled, dim=-1)
    log_probs = torch.log_softmax(scaled, dim=-1)
    return -(probs * log_probs).sum(dim=-1).mean()


class VisionPostmergeTrainer:
    def __init__(self, context: PostMergeContext, cfg: dict[str, Any]):
        resources = dict(context.resources)
        self.clf = resources.get("classifier", resources.get("clf", None))
        self.per_task = list(resources.get("per_task", []))
        if self.clf is None or not self.per_task:
            raise ValueError("Vision postmerge methods require classifier and per_task resources.")

        self.device = str(cfg.get("device", resources.get("device", next(context.model.parameters()).device)))
        self.loss_kind = str(cfg.get("loss", "ce")).strip().lower()
        if self.loss_kind not in {"ce", "entropy"}:
            raise ValueError("postmerge.loss must be one of: ce, entropy")
        self.entropy_temperature = float(cfg.get("entropy_temperature", 1.0))
        self.batches_per_task = int(cfg.get("batches_per_task", 2))
        if self.batches_per_task <= 0:
            raise ValueError("postmerge.batches_per_task must be > 0.")
        max_batches = cfg.get("max_batches_per_task", None)
        self.max_batches_per_task = None if max_batches is None else int(max_batches)

        self.text_features_by_task: dict[str, torch.Tensor] = {}
        self.iters_by_task: dict[str, Any] = {}
        self.batches_seen_by_task: dict[str, int] = {}
        for item in self.per_task:
            task_name = str(item["task"])
            text_features = item.get("text_features", None)
            if text_features is None:
                self.clf.build_zeroshot_text_features(
                    list(item["classnames"]),
                    item["build_cfg_task"],
                    cache_dir="src/.cache/zs_cache",
                    force_rebuild=False,
                )
                text_features = self.clf._zs_text_features.detach().cpu()
            self.text_features_by_task[task_name] = text_features.detach().cpu()
            self.iters_by_task[task_name] = iter(item["loaders"].train)
            self.batches_seen_by_task[task_name] = 0

    def iter_batches(self):
        for item in self.per_task:
            for _ in range(self.batches_per_task):
                yield item, *self._next_batch(item)

    def _next_batch(self, item: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        task_name = str(item["task"])
        if self.max_batches_per_task is not None and self.batches_seen_by_task[task_name] >= self.max_batches_per_task:
            raise RuntimeError(
                f"Postmerge reached max_batches_per_task={self.max_batches_per_task} for task '{task_name}' "
                "before completing all optimizer steps. Increase/remove the cap or reduce postmerge.steps."
            )
        try:
            images, labels = next(self.iters_by_task[task_name])
        except StopIteration:
            self.iters_by_task[task_name] = iter(item["loaders"].train)
            images, labels = next(self.iters_by_task[task_name])
        self.batches_seen_by_task[task_name] += 1
        return images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

    def text_features_for(self, item: dict[str, Any], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        task_name = str(item["task"])
        return self.text_features_by_task[task_name].to(device=device, dtype=dtype)

    def logits_from_image_features(self, item: dict[str, Any], image_features: torch.Tensor) -> torch.Tensor:
        if isinstance(image_features, (tuple, list)):
            image_features = image_features[0]
        if self.clf.normalize:
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-12)
        task_text = self.text_features_for(item, dtype=image_features.dtype, device=image_features.device)
        return self.clf.logit_scale * (image_features @ task_text.t())

    def visual_logits_from_params(
        self,
        item: dict[str, Any],
        visual_params: dict[str, torch.Tensor],
        images: torch.Tensor,
    ) -> torch.Tensor:
        image_features = functional_call(self.clf.model.visual, visual_params, (images,))
        return self.logits_from_image_features(item, image_features)

    def loss_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.loss_kind == "ce":
            return F.cross_entropy(logits, labels)
        return prediction_entropy(logits, temperature=self.entropy_temperature)

    def iter_losses(self, compute_logits: Callable[[dict[str, Any], torch.Tensor], torch.Tensor]):
        for item, images, labels in self.iter_batches():
            logits = compute_logits(item, images)
            loss = self.loss_from_logits(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Postmerge loss became non-finite for task '{item['task']}': {float(loss.detach())}")
            yield loss

    def summed_loss(self, compute_logits: Callable[[dict[str, Any], torch.Tensor], torch.Tensor]) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for loss in self.iter_losses(compute_logits):
            losses.append(loss)
        if not losses:
            raise RuntimeError("Postmerge did not receive any training batches.")
        return torch.stack(losses).sum()

    def backward_summed_loss(self, compute_logits: Callable[[dict[str, Any], torch.Tensor], torch.Tensor]) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for loss in self.iter_losses(compute_logits):
            loss.backward()
            losses.append(loss.detach())
        if not losses:
            raise RuntimeError("Postmerge did not receive any training batches.")
        return torch.stack(losses).sum()


def visual_params_from_bank(
    *,
    bank: TaskDeltaBank,
    model: torch.nn.Module,
    alpha_values: torch.Tensor,
    alpha_mode: str,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    dev = torch.device(device)
    visual_params: dict[str, torch.Tensor] = {}
    for name, param in model.visual.named_parameters():
        full_key = f"visual.{name}"
        if full_key not in bank.base or full_key not in bank.layer_for_key:
            continue
        base_tensor = bank.base[full_key].to(device=dev, dtype=param.dtype)
        acc = torch.zeros_like(base_tensor)
        for task_idx, deltas in enumerate(bank.deltas_by_task):
            alpha = bank.alpha_for(alpha_values, task_index=task_idx, key=full_key, mode=alpha_mode)
            delta = deltas[full_key].to(device=dev, dtype=param.dtype)
            acc = acc + float(bank.weights[task_idx]) * alpha.to(device=dev, dtype=param.dtype) * delta
        visual_params[name] = base_tensor + acc
    if not visual_params:
        raise RuntimeError("Postmerge vision loss found no visual parameters in the delta bank.")
    return visual_params


def visual_params_from_trainable_deltas(
    *,
    bank: TaskDeltaBank,
    model: torch.nn.Module,
    trainable_deltas: list[dict[str, torch.nn.Parameter]],
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    dev = torch.device(device)
    visual_params: dict[str, torch.Tensor] = {}
    for name, param in model.visual.named_parameters():
        full_key = f"visual.{name}"
        if full_key not in bank.base or full_key not in bank.layer_for_key:
            continue
        base_tensor = bank.base[full_key].to(device=dev, dtype=param.dtype)
        acc = torch.zeros_like(base_tensor)
        for task_idx, deltas in enumerate(trainable_deltas):
            delta = deltas[full_key].to(device=dev, dtype=param.dtype)
            acc = acc + float(bank.weights[task_idx]) * delta
        visual_params[name] = base_tensor + acc
    if not visual_params:
        raise RuntimeError("Postmerge vision loss found no visual parameters in the trainable delta bank.")
    return visual_params


def visual_params_from_trainable_merged_delta(
    *,
    bank: TaskDeltaBank,
    model: torch.nn.Module,
    trainable_merged_delta: dict[str, torch.Tensor],
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    dev = torch.device(device)
    visual_params: dict[str, torch.Tensor] = {}
    for name, param in model.visual.named_parameters():
        full_key = f"visual.{name}"
        if full_key not in bank.base or full_key not in bank.layer_for_key:
            continue
        base_tensor = bank.base[full_key].to(device=dev, dtype=param.dtype)
        delta = trainable_merged_delta[full_key].to(device=dev, dtype=param.dtype)
        visual_params[name] = base_tensor + delta
    if not visual_params:
        raise RuntimeError("Postmerge vision loss found no visual parameters in the trainable merged delta.")
    return visual_params
