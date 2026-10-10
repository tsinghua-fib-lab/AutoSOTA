from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class SampleTransform3D(Protocol):
    def __call__(self, sample: dict) -> dict: ...


def _normalize_patch_size(patch_size: Sequence[int]) -> tuple[int, int, int]:
    if len(patch_size) != 3:
        raise ValueError(
            f"Patch size must contain exactly 3 spatial dimensions, got {tuple(patch_size)!r}"
        )
    return tuple(int(axis) for axis in patch_size)


def validate_patch_size(sample: dict, patch_size: Sequence[int]) -> None:
    if "image" not in sample:
        raise KeyError("Augmentation sample is missing required key 'image'")

    image = sample["image"]
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Augmentation sample 'image' must be a numpy.ndarray, got {type(image).__name__}"
        )
    if image.ndim != 4:
        raise ValueError(
            f"Augmentation sample 'image' must be channel-first 4D (C, D, H, W), got shape {tuple(image.shape)!r}"
        )

    expected_shape = _normalize_patch_size(patch_size)
    actual_shape = tuple(int(axis) for axis in image.shape[1:])
    if actual_shape != expected_shape:
        raise ValueError(
            "Augmentation pipeline must preserve the designated patch size: "
            f"expected spatial shape {expected_shape}, got {actual_shape}"
        )


class Compose3D:
    def __init__(
        self, augmentations: Sequence[SampleTransform3D] | None = None
    ) -> None:
        self.augmentations = list(augmentations or [])

    def __call__(self, sample: dict) -> dict:
        augmented_sample = sample
        for augmentation in self.augmentations:
            augmented_sample = augmentation(augmented_sample)
        return augmented_sample


def apply_augmentations(
    sample: dict,
    pipeline: Compose3D | None,
    patch_size: Sequence[int],
) -> dict:
    augmented_sample = sample if pipeline is None else pipeline(sample)
    validate_patch_size(augmented_sample, patch_size)
    return augmented_sample
