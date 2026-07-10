from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from merge_and_rebase.data.vision_loaders import build_vision_loaders, load_hf_splits
from merge_and_rebase.eval.datasets.vision8_14_20 import _vision_spec
from merge_and_rebase.models.openclip_classifier import OpenClipBuildConfig, OpenClipClassifier


class RegularizerImageEncoder(nn.Module):
    def __init__(self, classifier: OpenClipClassifier) -> None:
        super().__init__()
        self.clip_model = classifier

    def forward(self, images):
        return self.clip_model.model.visual(images)


@dataclass(frozen=True)
class VisionRegularizerTaskContext:
    task: str
    build_cfg: OpenClipBuildConfig
    loader: Any
    model: nn.Module
    attn_patch_cfg: dict[str, Any] | None = None


def build_vision_regularizer_task_context(
    *,
    task: str,
    build_cfg: OpenClipBuildConfig,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    seed: int,
) -> VisionRegularizerTaskContext:
    hf_path, hf_config, split_map = _vision_spec(task)
    hf_ds = load_hf_splits(
        hf_path,
        config=hf_config,
        requested_splits=tuple(dict.fromkeys(split_map.values())),
    )
    classifier = OpenClipClassifier.build(build_cfg)
    loaders = build_vision_loaders(
        hf_ds=hf_ds,
        hf_path=hf_path,
        preprocess=classifier.preprocess,
        ft_epochs=1,
        split_map=split_map,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        val_fraction=val_fraction,
        seed=seed,
    )
    model = RegularizerImageEncoder(classifier)
    return VisionRegularizerTaskContext(
        task=task,
        build_cfg=build_cfg,
        loader=loaders.train,
        model=model,
    )
