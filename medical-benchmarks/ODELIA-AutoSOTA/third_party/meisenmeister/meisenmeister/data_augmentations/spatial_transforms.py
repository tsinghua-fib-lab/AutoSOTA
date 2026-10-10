from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import affine_transform


class FlipAxes3D:
    def __init__(self, probability: float, axes: Sequence[int]) -> None:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                f"FlipAxes3D probability must be between 0 and 1, got {probability!r}"
            )

        normalized_axes = tuple(int(axis) for axis in axes)
        invalid_axes = [axis for axis in normalized_axes if axis not in (0, 1, 2)]
        if invalid_axes:
            raise ValueError(
                f"FlipAxes3D axes must be chosen from (0, 1, 2), got {normalized_axes!r}"
            )

        self.probability = float(probability)
        self.axes = normalized_axes

    def __call__(self, sample: dict) -> dict:
        if self.probability == 0.0 or not self.axes:
            return sample
        if np.random.random() >= self.probability:
            return sample

        image = sample["image"]
        flipped_image = np.flip(
            image, axis=tuple(axis + 1 for axis in self.axes)
        ).copy()
        return {**sample, "image": flipped_image}


class RandomShiftWithinMargin3D:
    def __init__(
        self,
        probability: float,
        max_shift_voxels: Sequence[int],
    ) -> None:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                f"RandomShiftWithinMargin3D probability must be between 0 and 1, got {probability!r}"
            )
        if len(max_shift_voxels) != 3:
            raise ValueError(
                "RandomShiftWithinMargin3D max_shift_voxels must contain exactly three values"
            )

        normalized_shifts = tuple(int(axis) for axis in max_shift_voxels)
        if any(axis < 0 for axis in normalized_shifts):
            raise ValueError(
                "RandomShiftWithinMargin3D max_shift_voxels values must be non-negative"
            )

        self.probability = float(probability)
        self.max_shift_voxels = normalized_shifts

    def _sample_shift(self) -> tuple[int, int, int]:
        sampled_shift = []
        for max_shift in self.max_shift_voxels:
            if max_shift == 0:
                sampled_shift.append(0)
                continue
            sampled_shift.append(int(np.random.randint(-max_shift, max_shift + 1)))
        return tuple(sampled_shift)

    def __call__(self, sample: dict) -> dict:
        if self.probability == 0.0:
            return sample
        if np.random.random() >= self.probability:
            return sample

        image = sample["image"]
        shifted_image = np.zeros_like(image)
        spatial_shape = image.shape[1:]
        sampled_shift = self._sample_shift()

        source_slices = []
        destination_slices = []
        for size, shift in zip(spatial_shape, sampled_shift, strict=True):
            src_start = max(0, -shift)
            src_stop = min(size, size - shift) if shift >= 0 else size
            dst_start = max(0, shift)
            dst_stop = dst_start + max(0, src_stop - src_start)
            source_slices.append(slice(src_start, src_stop))
            destination_slices.append(slice(dst_start, dst_stop))

        shifted_image[
            :,
            destination_slices[0],
            destination_slices[1],
            destination_slices[2],
        ] = image[
            :,
            source_slices[0],
            source_slices[1],
            source_slices[2],
        ]
        return {**sample, "image": shifted_image}


class RemoveMargin3D:
    def __init__(self, margin_voxels: Sequence[int]) -> None:
        if len(margin_voxels) != 3:
            raise ValueError(
                "RemoveMargin3D margin_voxels must contain exactly three values"
            )

        normalized_margins = tuple(int(axis) for axis in margin_voxels)
        if any(axis < 0 for axis in normalized_margins):
            raise ValueError("RemoveMargin3D margin_voxels values must be non-negative")

        self.margin_voxels = normalized_margins

    def __call__(self, sample: dict) -> dict:
        image = sample["image"]
        masked_image = image.copy()
        for axis, (axis_size, margin) in enumerate(
            zip(image.shape[1:], self.margin_voxels, strict=True),
            start=1,
        ):
            if margin <= 0:
                continue
            margin = min(margin, axis_size)
            leading_slice = [slice(None)] * masked_image.ndim
            leading_slice[axis] = slice(0, margin)
            masked_image[tuple(leading_slice)] = 0
            trailing_slice = [slice(None)] * masked_image.ndim
            trailing_slice[axis] = slice(axis_size - margin, axis_size)
            masked_image[tuple(trailing_slice)] = 0

        return {**sample, "image": masked_image}


def _resample_channel_with_scale(channel: np.ndarray, scale: float) -> np.ndarray:
    center_index = _get_image_center_index(channel.shape)
    matrix = np.eye(3, dtype=np.float32) / scale
    offset = center_index - matrix @ center_index
    scaled = affine_transform(
        channel,
        matrix=matrix,
        offset=offset,
        output_shape=channel.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return scaled.astype(channel.dtype, copy=False)


def _get_image_center_index(
    shape: Sequence[int] | tuple[int, int, int],
) -> np.ndarray:
    center_index = np.asarray(shape, dtype=np.float32)
    return (center_index - 1.0) / 2.0


def _get_image_center_point(image: sitk.Image) -> tuple[float, float, float]:
    center_index = _get_image_center_index(image.GetSize()).tolist()
    return tuple(image.TransformContinuousIndexToPhysicalPoint(center_index))


def _rotate_channel_with_angles(
    channel: np.ndarray,
    angles_degrees: tuple[float, float, float],
) -> np.ndarray:
    rotation_z, rotation_y, rotation_x = angles_degrees
    image = sitk.GetImageFromArray(channel.astype(np.float32, copy=False))
    center_point = _get_image_center_point(image)

    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_point)
    # Public augmentation parameters use NumPy spatial axis order (z, y, x).
    # SimpleITK Euler3DTransform expects rotations around physical axes (x, y, z).
    transform.SetRotation(
        float(np.deg2rad(rotation_x)),
        float(np.deg2rad(rotation_y)),
        float(np.deg2rad(rotation_z)),
    )

    rotated = sitk.Resample(
        image,
        image,
        transform,
        sitk.sitkLinear,
        0.0,
        image.GetPixelID(),
    )
    return sitk.GetArrayFromImage(rotated).astype(channel.dtype, copy=False)


class RandomScaling3D:
    def __init__(
        self,
        probability: float,
        scaling: Sequence[float],
    ) -> None:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                f"RandomScaling3D probability must be between 0 and 1, got {probability!r}"
            )
        if len(scaling) != 2:
            raise ValueError("RandomScaling3D scaling must contain exactly two values")

        scaling_min = float(scaling[0])
        scaling_max = float(scaling[1])
        if scaling_min <= 0.0 or scaling_max <= 0.0:
            raise ValueError("RandomScaling3D scaling values must be positive")
        if scaling_min > scaling_max:
            raise ValueError("RandomScaling3D scaling must be ordered as (min, max)")

        self.probability = float(probability)
        self.scaling = (scaling_min, scaling_max)

    def _sample_scale(self) -> float:
        low, high = self.scaling
        if low == high:
            return low
        return float(np.random.uniform(low, high))

    def __call__(self, sample: dict) -> dict:
        if self.probability == 0.0:
            return sample
        if np.random.random() >= self.probability:
            return sample

        image = sample["image"]
        scale = self._sample_scale()
        scaled_image = np.stack(
            [_resample_channel_with_scale(channel, scale) for channel in image],
            axis=0,
        )
        return {**sample, "image": scaled_image.astype(image.dtype, copy=False)}


class RandomRotation3D:
    def __init__(
        self,
        probability: float,
        max_rotation_degrees: Sequence[float],
    ) -> None:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                f"RandomRotation3D probability must be between 0 and 1, got {probability!r}"
            )
        if len(max_rotation_degrees) != 3:
            raise ValueError(
                "RandomRotation3D max_rotation_degrees must contain exactly three values"
            )

        normalized_angles = tuple(float(angle) for angle in max_rotation_degrees)
        if any(angle < 0.0 for angle in normalized_angles):
            raise ValueError(
                "RandomRotation3D max_rotation_degrees values must be non-negative"
            )

        self.probability = float(probability)
        self.max_rotation_degrees = normalized_angles

    def _sample_angles(self) -> tuple[float, float, float]:
        sampled_angles = []
        for max_angle in self.max_rotation_degrees:
            if max_angle == 0.0:
                sampled_angles.append(0.0)
                continue
            sampled_angles.append(float(np.random.uniform(-max_angle, max_angle)))
        return tuple(sampled_angles)

    def __call__(self, sample: dict) -> dict:
        if self.probability == 0.0:
            return sample
        if np.random.random() >= self.probability:
            return sample

        image = sample["image"]
        sampled_angles = self._sample_angles()
        rotated_image = np.stack(
            [_rotate_channel_with_angles(channel, sampled_angles) for channel in image],
            axis=0,
        )
        return {**sample, "image": rotated_image.astype(image.dtype, copy=False)}
