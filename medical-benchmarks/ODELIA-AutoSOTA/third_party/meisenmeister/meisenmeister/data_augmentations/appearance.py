from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class MultiplicativeBrightness3D:
    def __init__(
        self,
        probability: float,
        multiplier_range: Sequence[float] = (0.5, 2.0),
        p_per_channel: float = 1.0,
        synchronize_channels: bool = False,
    ) -> None:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                f"MultiplicativeBrightness3D probability must be between 0 and 1, got {probability!r}"
            )
        if not 0.0 <= float(p_per_channel) <= 1.0:
            raise ValueError(
                f"MultiplicativeBrightness3D p_per_channel must be between 0 and 1, got {p_per_channel!r}"
            )
        if len(multiplier_range) != 2:
            raise ValueError(
                "MultiplicativeBrightness3D multiplier_range must contain exactly two values"
            )

        multiplier_min = float(multiplier_range[0])
        multiplier_max = float(multiplier_range[1])
        if multiplier_min > multiplier_max:
            raise ValueError(
                "MultiplicativeBrightness3D multiplier_range must be ordered as (min, max)"
            )

        self.probability = float(probability)
        self.multiplier_range = (multiplier_min, multiplier_max)
        self.p_per_channel = float(p_per_channel)
        self.synchronize_channels = bool(synchronize_channels)

    def _sample_multiplier(self) -> float:
        low, high = self.multiplier_range
        if low == high:
            return low
        return float(np.random.uniform(low, high))

    def __call__(self, sample: dict) -> dict:
        if self.probability == 0.0:
            return sample
        if np.random.random() >= self.probability:
            return sample

        image = sample["image"]
        brightened_image = image.copy()

        synchronized_multiplier = (
            self._sample_multiplier() if self.synchronize_channels else None
        )
        for channel_index in range(brightened_image.shape[0]):
            if np.random.random() >= self.p_per_channel:
                continue

            multiplier = (
                synchronized_multiplier
                if synchronized_multiplier is not None
                else self._sample_multiplier()
            )
            brightened_image[channel_index] = (
                brightened_image[channel_index] * multiplier
            )

        return {**sample, "image": brightened_image.astype(image.dtype, copy=False)}


class Contrast3D:
    def __init__(
        self,
        probability: float,
        contrast_range: Sequence[float] = (0.75, 1.25),
        preserve_range: bool = True,
        p_per_channel: float = 1.0,
        synchronize_channels: bool = False,
    ) -> None:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                f"Contrast3D probability must be between 0 and 1, got {probability!r}"
            )
        if not 0.0 <= float(p_per_channel) <= 1.0:
            raise ValueError(
                f"Contrast3D p_per_channel must be between 0 and 1, got {p_per_channel!r}"
            )
        if len(contrast_range) != 2:
            raise ValueError(
                "Contrast3D contrast_range must contain exactly two values"
            )

        contrast_min = float(contrast_range[0])
        contrast_max = float(contrast_range[1])
        if contrast_min > contrast_max:
            raise ValueError("Contrast3D contrast_range must be ordered as (min, max)")

        self.probability = float(probability)
        self.contrast_range = (contrast_min, contrast_max)
        self.p_per_channel = float(p_per_channel)
        self.preserve_range = bool(preserve_range)
        self.synchronize_channels = bool(synchronize_channels)

    def _sample_contrast_factor(self) -> float:
        low, high = self.contrast_range
        if low == high:
            return low
        return float(np.random.uniform(low, high))

    def __call__(self, sample: dict) -> dict:
        if self.probability == 0.0:
            return sample
        if np.random.random() >= self.probability:
            return sample

        image = sample["image"]
        contrasted_image = image.copy()

        synchronized_factor = (
            self._sample_contrast_factor() if self.synchronize_channels else None
        )
        for channel_index in range(contrasted_image.shape[0]):
            if np.random.random() >= self.p_per_channel:
                continue

            factor = (
                synchronized_factor
                if synchronized_factor is not None
                else self._sample_contrast_factor()
            )
            channel = contrasted_image[channel_index]
            channel_mean = float(channel.mean())
            adjusted = (channel - channel_mean) * factor + channel_mean
            if self.preserve_range:
                adjusted = np.clip(adjusted, channel.min(), channel.max())
            contrasted_image[channel_index] = adjusted

        return {**sample, "image": contrasted_image.astype(image.dtype, copy=False)}
