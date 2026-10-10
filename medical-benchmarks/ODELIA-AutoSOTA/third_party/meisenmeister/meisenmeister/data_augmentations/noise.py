from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class GaussianNoise3D:
    def __init__(
        self,
        probability: float,
        noise_variance: Sequence[float] = (0.0, 0.1),
        p_per_channel: float = 1.0,
        synchronize_channels: bool = True,
    ) -> None:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                f"GaussianNoise3D probability must be between 0 and 1, got {probability!r}"
            )
        if not 0.0 <= float(p_per_channel) <= 1.0:
            raise ValueError(
                f"GaussianNoise3D p_per_channel must be between 0 and 1, got {p_per_channel!r}"
            )
        if len(noise_variance) != 2:
            raise ValueError(
                "GaussianNoise3D noise_variance must contain exactly two values"
            )

        variance_min = float(noise_variance[0])
        variance_max = float(noise_variance[1])
        if variance_min > variance_max:
            raise ValueError(
                "GaussianNoise3D noise_variance must be ordered as (min, max)"
            )
        if variance_min < 0.0:
            raise ValueError(
                "GaussianNoise3D noise_variance values must be non-negative"
            )

        self.probability = float(probability)
        self.noise_variance = (variance_min, variance_max)
        self.p_per_channel = float(p_per_channel)
        self.synchronize_channels = bool(synchronize_channels)

    def _sample_noise_scale(self) -> float:
        low, high = self.noise_variance
        if low == high:
            return low
        return float(np.random.uniform(low, high))

    def __call__(self, sample: dict) -> dict:
        if self.probability == 0.0:
            return sample
        if np.random.random() >= self.probability:
            return sample

        image = sample["image"]
        noisy_image = image.copy()

        synchronized_scale = (
            self._sample_noise_scale() if self.synchronize_channels else None
        )
        for channel_index in range(noisy_image.shape[0]):
            if np.random.random() >= self.p_per_channel:
                continue

            noise_scale = (
                synchronized_scale
                if synchronized_scale is not None
                else self._sample_noise_scale()
            )

            noisy_image[channel_index] = noisy_image[channel_index] + np.random.normal(
                0.0,
                noise_scale,
                size=noisy_image[channel_index].shape,
            )

        return {**sample, "image": noisy_image.astype(image.dtype, copy=False)}
