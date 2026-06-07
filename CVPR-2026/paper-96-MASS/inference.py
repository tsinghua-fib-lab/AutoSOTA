#!/usr/bin/env python3
"""
Standalone in-context MASS/Iris segmentation inference from raw NIfTI images.

Inputs:
  - one or more testing images
  - one or more reference images
  - matching reference segmentation masks

The script follows the training preprocessing as closely as possible for raw
NIfTI deployment: reorientation, optional resampling, body/foreground crop,
intensity normalization, reference-centered task crops, sliding-window
inference, and resampling the final mask back to the original image space.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from tqdm import tqdm

import models
from utils.registry import get_model


Array3D = np.ndarray
BBoxZYX = Tuple[int, int, int, int, int, int]


def load_config_from_checkpoint(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Load model/evaluation config from a checkpoint or sibling config file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "config" in checkpoint:
        return checkpoint["config"]

    config_json = checkpoint_path.parent / "config.json"
    if config_json.exists():
        with open(config_json, "r") as f:
            return json.load(f)

    config_yaml = checkpoint_path.parent / "config.yaml"
    if config_yaml.exists():
        import yaml

        with open(config_yaml, "r") as f:
            return yaml.safe_load(f)

    raise RuntimeError("No configuration found in checkpoint or checkpoint directory")


def load_model(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    use_ema: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Instantiate the model and load regular or EMA weights."""
    config = load_config_from_checkpoint(checkpoint_path)
    model_config = config.get("model", {}).copy()
    model_name = model_config.pop("type", "iris")

    model_cls = get_model(model_name)
    model = model_cls(**model_config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if use_ema and "ema_model" in checkpoint:
        state_dict = checkpoint["ema_model"]
        logging.info("Loading EMA model weights from key 'ema_model'")
    elif use_ema and "ema_state_dict" in checkpoint:
        state_dict = checkpoint["ema_state_dict"]
        logging.info("Loading EMA model weights from key 'ema_state_dict'")
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        logging.info("Loading regular model weights from key 'model'")
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        logging.info("Loading regular model weights from key 'model_state_dict'")
    else:
        state_dict = checkpoint
        logging.info("Loading checkpoint as a raw state_dict")

    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_state_dict[key[7:] if key.startswith("module.") else key] = value

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    if missing:
        logging.warning("Missing checkpoint keys: %s", missing[:20])
    if unexpected:
        logging.warning("Unexpected checkpoint keys: %s", unexpected[:20])

    model.eval()
    logging.info("Model loaded from %s", checkpoint_path)
    logging.info("Checkpoint epoch: %s", checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown")
    return model, config


def strip_nii_suffix(path: Union[str, Path]) -> str:
    """Return a filename stem while treating .nii.gz as one suffix."""
    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def read_nifti(path: Union[str, Path]) -> sitk.Image:
    """Read a NIfTI image and raise a clear error if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return sitk.ReadImage(str(path))


def reorient_image(image: sitk.Image, orientation: str) -> sitk.Image:
    """Reorient an image with SimpleITK/DICOM orientation codes."""
    if orientation.lower() in ("none", "native", "original"):
        return image
    return sitk.DICOMOrient(image, orientation)


def resample_to_spacing(
    image: sitk.Image,
    spacing_xyz: Sequence[float],
    interpolator: int,
    default_value: float = 0.0,
    pixel_id: Optional[int] = None,
) -> sitk.Image:
    """Resample a SimpleITK image to spacing in x, y, z order."""
    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    new_size = [
        max(1, int(round(old_size[i] * old_spacing[i] / float(spacing_xyz[i]))))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(tuple(float(x) for x in spacing_xyz))
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(default_value))
    if pixel_id is not None:
        resampler.SetOutputPixelType(pixel_id)
    return resampler.Execute(image)


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    interpolator: int,
    default_value: float = 0.0,
    pixel_id: Optional[int] = None,
) -> sitk.Image:
    """Resample an image onto another image's grid."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(float(default_value))
    if pixel_id is not None:
        resampler.SetOutputPixelType(pixel_id)
    return resampler.Execute(image)


def image_to_array(image: sitk.Image, dtype=np.float32) -> Array3D:
    """Convert a SimpleITK image to a numpy array in z, y, x order."""
    return sitk.GetArrayFromImage(image).astype(dtype, copy=False)


def array_to_image(array: Array3D, reference: sitk.Image, pixel_id: Optional[int] = None) -> sitk.Image:
    """Convert a z, y, x numpy array back to a SimpleITK image grid."""
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(reference)
    if pixel_id is not None:
        image = sitk.Cast(image, pixel_id)
    return image


def guess_modality(image: Array3D) -> str:
    """Heuristically choose CT, PET, or MR intensity handling."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return "ct"
    low, high = np.percentile(finite, [0.5, 99.5])
    if low < -200 and high > 300:
        return "ct"
    if low >= 0 and high > 5:
        return "pet"
    return "mr"


def normalize_image(
    image: Array3D,
    modality: str,
    ct_clip: Tuple[float, float],
    eps: float = 1e-6,
) -> Array3D:
    """Apply modality-specific clipping followed by per-volume z-score."""
    image = image.astype(np.float32, copy=False)
    modality = modality.lower()

    if modality == "auto":
        modality = guess_modality(image)
        logging.info("Auto-detected modality: %s", modality)

    if modality == "ct":
        # Match training: CT uses a fixed HU window before z-score.
        image = np.clip(image, ct_clip[0], ct_clip[1])
    else:
        finite = image[np.isfinite(image)]
        if finite.size:
            p2, p98 = np.percentile(finite, [2, 98])
            if p98 > p2:
                # MR/PET use per-volume percentile clipping before z-score.
                image = np.clip(image, p2, p98)

    mean = float(np.mean(image))
    std = float(np.std(image))
    if std > eps:
        image = (image - mean) / std
    else:
        image = image - mean

    return image.astype(np.float32, copy=False)


def largest_connected_component(mask: Array3D) -> Array3D:
    """Return the largest connected foreground component of a binary mask."""
    if not np.any(mask):
        return mask.astype(bool)
    labeled, num = ndimage.label(mask)
    if num <= 1:
        return mask.astype(bool)
    sizes = np.bincount(labeled.reshape(-1))
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def keep_largest_component_per_label(label_map: Array3D) -> Array3D:
    """Keep the largest connected component independently for each label."""
    processed = np.zeros_like(label_map)
    for label in np.unique(label_map):
        label = int(label)
        if label == 0:
            continue
        component = largest_connected_component(label_map == label)
        processed[component] = label
    return processed


def make_body_mask(
    image: Array3D,
    modality: str,
    ct_body_threshold: float,
) -> Array3D:
    """Build a conservative foreground/body mask for unlabeled target crops."""
    modality = guess_modality(image) if modality == "auto" else modality.lower()
    finite = np.isfinite(image)

    if modality == "ct":
        mask = finite & (image > ct_body_threshold)
    else:
        values = image[finite]
        if values.size == 0:
            return np.zeros(image.shape, dtype=bool)
        abs_threshold = max(float(np.percentile(np.abs(values), 99)) * 1e-3, 1e-6)
        mask = finite & (np.abs(image) > abs_threshold)

    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    return largest_connected_component(mask)


def bbox_from_mask(mask: Array3D, margin_zyx: Sequence[int]) -> Optional[BBoxZYX]:
    """Convert a binary mask to a z, y, x bounding box with margin."""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None

    z0, z1 = int(coords[0].min()), int(coords[0].max()) + 1
    y0, y1 = int(coords[1].min()), int(coords[1].max()) + 1
    x0, x1 = int(coords[2].min()), int(coords[2].max()) + 1
    dz, dy, dx = [int(v) for v in margin_zyx]
    d, h, w = mask.shape

    z0 = max(0, z0 - dz)
    y0 = max(0, y0 - dy)
    x0 = max(0, x0 - dx)
    z1 = min(d, z1 + dz)
    y1 = min(h, y1 + dy)
    x1 = min(w, x1 + dx)
    return z0, z1, y0, y1, x0, x1


def crop_sitk_zyx(image: sitk.Image, bbox: BBoxZYX) -> sitk.Image:
    """Crop a SimpleITK image using a bbox expressed in z, y, x array order."""
    z0, z1, y0, y1, x0, x1 = bbox
    index_xyz = [int(x0), int(y0), int(z0)]
    size_xyz = [int(x1 - x0), int(y1 - y0), int(z1 - z0)]
    return sitk.RegionOfInterest(image, size=size_xyz, index=index_xyz)


def maybe_crop_body(
    image: sitk.Image,
    modality: str,
    method: str,
    margin_zyx: Sequence[int],
    ct_body_threshold: float,
) -> Tuple[sitk.Image, Optional[BBoxZYX]]:
    """Optionally crop target images when GT labels are unavailable."""
    if method == "none":
        return image, None
    if method != "threshold":
        raise ValueError(f"Unsupported body crop method: {method}")

    array = image_to_array(image, dtype=np.float32)
    # Inference has no GT crop, so use a conservative foreground/body crop.
    mask = make_body_mask(array, modality, ct_body_threshold)
    bbox = bbox_from_mask(mask, margin_zyx)
    if bbox is None:
        logging.warning("Body crop requested but no body mask was found; using full image")
        return image, None

    logging.info("Body crop bbox zyx: %s", bbox)
    return crop_sitk_zyx(image, bbox), bbox


def crop_arrays_centered(
    image: Array3D,
    mask: Array3D,
    crop_size_zyx: Sequence[int],
) -> Tuple[Array3D, Array3D]:
    """Crop/pad image and mask around the reference foreground center."""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        raise ValueError("Reference mask has no foreground voxels")

    center = [int(round((int(axis.min()) + int(axis.max())) / 2.0)) for axis in coords]
    crop_size = [int(v) for v in crop_size_zyx]
    shape = image.shape

    src_slices = []
    dst_slices = []
    pad_shape = []
    for axis, size in enumerate(crop_size):
        start = center[axis] - size // 2
        end = start + size
        src_start = max(0, start)
        src_end = min(shape[axis], end)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
        pad_shape.append(size)

    image_crop = np.zeros(pad_shape, dtype=image.dtype)
    mask_crop = np.zeros(pad_shape, dtype=mask.dtype)
    image_crop[tuple(dst_slices)] = image[tuple(src_slices)]
    mask_crop[tuple(dst_slices)] = mask[tuple(src_slices)]
    return image_crop, mask_crop


def preprocess_target_image(
    image_path: Union[str, Path],
    orientation: str,
    target_spacing_xyz: Sequence[float],
    image_interpolator: int,
    modality: str,
    body_method: str,
    body_margin_zyx: Sequence[int],
    ct_body_threshold: float,
    ct_clip: Tuple[float, float],
) -> Tuple[sitk.Image, sitk.Image, torch.Tensor]:
    """Prepare a target image for sliding-window inference."""
    original = read_nifti(image_path)
    oriented = reorient_image(original, orientation)
    resampled = resample_to_spacing(
        oriented,
        target_spacing_xyz,
        image_interpolator,
        default_value=0.0,
        pixel_id=sitk.sitkFloat32,
    )
    cropped, _ = maybe_crop_body(
        resampled,
        modality=modality,
        method=body_method,
        margin_zyx=body_margin_zyx,
        ct_body_threshold=ct_body_threshold,
    )

    array = image_to_array(cropped, dtype=np.float32)
    normalized = normalize_image(array, modality, ct_clip)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float()
    logging.info("Target tensor shape: %s", tuple(tensor.shape))
    return original, cropped, tensor


def preprocess_reference_pair(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    orientation: str,
    target_spacing_xyz: Sequence[float],
    image_interpolator: int,
    modality: str,
    reference_label: int,
    crop_size_zyx: Sequence[int],
    ct_clip: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare one reference image/mask pair for task embedding extraction."""
    image = reorient_image(read_nifti(image_path), orientation)
    mask = reorient_image(read_nifti(mask_path), orientation)

    image = resample_to_spacing(
        image,
        target_spacing_xyz,
        image_interpolator,
        default_value=0.0,
        pixel_id=sitk.sitkFloat32,
    )
    mask = resample_to_reference(
        mask,
        image,
        sitk.sitkNearestNeighbor,
        default_value=0.0,
        pixel_id=sitk.sitkUInt16,
    )

    image_array = image_to_array(image, dtype=np.float32)
    mask_array = image_to_array(mask, dtype=np.int32)

    foreground = mask_array == int(reference_label)
    if not np.any(foreground):
        raise ValueError(f"Reference mask {mask_path} has no voxels for label {reference_label}")

    # References are cropped around the requested label to match training crops.
    image_crop, mask_crop = crop_arrays_centered(image_array, foreground.astype(np.uint8), crop_size_zyx)
    image_crop = normalize_image(image_crop, modality, ct_clip)
    mask_crop = (mask_crop > 0).astype(np.float32)

    image_tensor = torch.from_numpy(image_crop).unsqueeze(0).unsqueeze(0).float()
    mask_tensor = torch.from_numpy(mask_crop).unsqueeze(0).unsqueeze(0).float()
    logging.info(
        "Reference %s tensor shape: %s, foreground voxels: %d",
        image_path,
        tuple(image_tensor.shape),
        int(mask_crop.sum()),
    )
    return image_tensor, mask_tensor


def average_task_embeddings(embeddings: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """Average task embeddings from multiple reference examples."""
    if not embeddings:
        raise ValueError("No task embeddings to average")

    averaged = []
    for level in range(len(embeddings[0])):
        tensors = [embedding[level] for embedding in embeddings]
        shapes = [tuple(t.shape) for t in tensors]
        if any(shape != shapes[0] for shape in shapes):
            raise ValueError(f"Reference embedding shapes do not match at level {level}: {shapes}")
        averaged.append(sum(tensors) / len(tensors))
    return averaged


def collect_reference_labels(
    reference_masks: Sequence[Union[str, Path]],
    requested_labels: Optional[Sequence[int]],
) -> List[int]:
    """Collect reference labels from CLI args or all nonzero mask values."""
    if requested_labels:
        return sorted({int(label) for label in requested_labels})

    labels = set()
    for mask_path in reference_masks:
        mask = read_nifti(mask_path)
        mask_array = image_to_array(mask, dtype=np.int32)
        labels.update(int(label) for label in np.unique(mask_array) if int(label) != 0)

    if not labels:
        raise ValueError("No nonzero labels found in reference masks")

    return sorted(labels)


def encode_reference_embeddings_by_label(
    model: nn.Module,
    reference_images: Sequence[Union[str, Path]],
    reference_masks: Sequence[Union[str, Path]],
    device: torch.device,
    orientation: str,
    target_spacing_xyz: Sequence[float],
    image_interpolator: int,
    modality: str,
    reference_labels: Sequence[int],
    crop_size_zyx: Sequence[int],
    ct_clip: Tuple[float, float],
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[int, List[torch.Tensor]]:
    if len(reference_images) != len(reference_masks):
        raise ValueError(
            f"reference-images and reference-masks must have the same length, got "
            f"{len(reference_images)} and {len(reference_masks)}"
        )

    embeddings_by_label = {}
    for label in reference_labels:
        # Build one binary reference task per foreground label value.
        embeddings = []
        for image_path, mask_path in zip(reference_images, reference_masks):
            try:
                ref_img, ref_mask = preprocess_reference_pair(
                    image_path=image_path,
                    mask_path=mask_path,
                    orientation=orientation,
                    target_spacing_xyz=target_spacing_xyz,
                    image_interpolator=image_interpolator,
                    modality=modality,
                    reference_label=int(label),
                    crop_size_zyx=crop_size_zyx,
                    ct_clip=ct_clip,
                )
            except ValueError as exc:
                if "has no voxels for label" in str(exc):
                    logging.info("Skipping reference %s for absent label %s", mask_path, label)
                    continue
                raise

            ref_img = ref_img.to(device)
            ref_mask = ref_mask.to(device)

            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        ref_feat = model.encode_image_feature(ref_img)
                        embedding = model.encode_visual_prior(ref_feat, ref_mask)
                else:
                    ref_feat = model.encode_image_feature(ref_img)
                    embedding = model.encode_visual_prior(ref_feat, ref_mask)

            embeddings.append(embedding)

        if not embeddings:
            raise ValueError(f"No usable references found for label {label}")

        # Multiple reference examples define an ensemble for the same label.
        embeddings_by_label[int(label)] = average_task_embeddings(embeddings)
        logging.info("Encoded label %s from %d reference(s)", label, len(embeddings))

    return embeddings_by_label


def get_sliding_window_coords(
    image_shape_zyx: Sequence[int],
    window_size_zyx: Sequence[int],
    overlap: float,
) -> List[BBoxZYX]:
    """Generate z, y, x sliding-window boxes that cover the whole volume."""
    steps = [max(1, int(round(size * (1.0 - overlap)))) for size in window_size_zyx]
    coords = []

    for axis_size, window, step in zip(image_shape_zyx, window_size_zyx, steps):
        if axis_size <= window:
            coords.append([0])
            continue
        starts = list(range(0, axis_size - window + 1, step))
        if starts[-1] != axis_size - window:
            starts.append(axis_size - window)
        coords.append(starts)

    windows = []
    for z in coords[0]:
        for y in coords[1]:
            for x in coords[2]:
                windows.append((z, z + window_size_zyx[0], y, y + window_size_zyx[1], x, x + window_size_zyx[2]))
    return windows


def create_gaussian_window(window_size_zyx: Sequence[int], sigma_scale: float = 0.125) -> torch.Tensor:
    """Create blending weights for overlapping sliding-window predictions."""
    gaussians = []
    for size in window_size_zyx:
        sigma = max(float(size) * sigma_scale, 1e-6)
        x = np.arange(size)
        gaussians.append(np.exp(-((x - (size - 1) / 2) ** 2) / (2 * sigma**2)))
    window = gaussians[0][:, None, None] * gaussians[1][None, :, None] * gaussians[2][None, None, :]
    window = window / np.max(window)
    return torch.from_numpy(window.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def pad_to_min_size(tensor: torch.Tensor, min_size_zyx: Sequence[int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Pad a 5D tensor so each spatial axis is at least the window size."""
    _, _, d, h, w = tensor.shape
    pad_d = max(0, int(min_size_zyx[0]) - d)
    pad_h = max(0, int(min_size_zyx[1]) - h)
    pad_w = max(0, int(min_size_zyx[2]) - w)
    if pad_d or pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d))
    return tensor, (d, h, w)


def sliding_window_inference(
    model: nn.Module,
    image: torch.Tensor,
    encoded_prior: List[torch.Tensor],
    device: torch.device,
    window_size_zyx: Sequence[int],
    overlap: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    image = image.to(device)
    # Pad once before tiling so every window matches the trained patch size.
    image, original_shape = pad_to_min_size(image, window_size_zyx)
    _, _, d, h, w = image.shape

    windows = get_sliding_window_coords((d, h, w), window_size_zyx, overlap)
    logging.info("Running sliding-window inference with %d windows", len(windows))

    gaussian = create_gaussian_window(window_size_zyx).to(device)
    num_classes = int(encoded_prior[0].shape[1])
    output = torch.zeros((1, num_classes, d, h, w), dtype=torch.float32, device=device)
    weight = torch.zeros((1, 1, d, h, w), dtype=torch.float32, device=device)

    with torch.no_grad():
        for z0, z1, y0, y1, x0, x1 in tqdm(windows, desc="Inference windows"):
            patch = image[:, :, z0:z1, y0:y1, x0:x1]
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model.forward_with_encoded_prior(patch, encoded_prior)
                    prob = torch.sigmoid(logits).float()
            else:
                logits = model.forward_with_encoded_prior(patch, encoded_prior)
                prob = torch.sigmoid(logits).float()

            output[:, :, z0:z1, y0:y1, x0:x1] += prob * gaussian
            weight[:, :, z0:z1, y0:y1, x0:x1] += gaussian

    # Gaussian weights soften seams between overlapping windows.
    output = output / torch.clamp(weight, min=1e-8)
    d0, h0, w0 = original_shape
    return output[:, :, :d0, :h0, :w0].cpu()


def concat_encoded_priors(
    encoded_priors_by_label: Dict[int, List[torch.Tensor]],
    labels: Sequence[int],
) -> List[torch.Tensor]:
    """Concatenate class priors at each decoder scale for multi-class output."""
    num_scales = len(encoded_priors_by_label[int(labels[0])])
    concatenated = []
    for scale_idx in range(num_scales):
        # The model expects priors concatenated per decoder scale.
        concatenated.append(
            torch.cat(
                [encoded_priors_by_label[int(label)][scale_idx] for label in labels],
                dim=1,
            )
        )
    return concatenated


def multiclass_inference(
    model: nn.Module,
    image: torch.Tensor,
    encoded_priors_by_label: Dict[int, List[torch.Tensor]],
    output_labels_by_label: Dict[int, int],
    device: torch.device,
    window_size_zyx: Sequence[int],
    overlap: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    threshold: float,
    max_classes_per_forward: int = 0,
    keep_probability_maps: bool = False,
    keep_largest_component: bool = True,
) -> Tuple[Array3D, Optional[Dict[int, Array3D]]]:
    """Run multi-class in-context inference and convert probabilities to labels."""
    best_probability = None
    label_map = None
    probability_maps = {} if keep_probability_maps else None
    labels = sorted(int(label) for label in encoded_priors_by_label)
    chunk_size = len(labels) if max_classes_per_forward <= 0 else int(max_classes_per_forward)

    for start in range(0, len(labels), chunk_size):
        current_labels = labels[start:start + chunk_size]
        encoded_prior = concat_encoded_priors(encoded_priors_by_label, current_labels)
        logging.info("Running inference for labels %s", current_labels)

        probability = sliding_window_inference(
            model=model,
            image=image,
            encoded_prior=encoded_prior,
            device=device,
            window_size_zyx=window_size_zyx,
            overlap=overlap,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        probability_array = probability.squeeze(0).numpy().astype(np.float32)

        if best_probability is None:
            best_probability = np.zeros_like(probability_array[0], dtype=np.float32)
            label_map = np.zeros_like(probability_array[0], dtype=np.uint16)

        for channel_idx, label in enumerate(current_labels):
            class_probability = probability_array[channel_idx]
            output_label = int(output_labels_by_label[label])
            # Multi-class output is built by assigning each voxel to the most
            # confident class above threshold.
            update_mask = (class_probability >= threshold) & (class_probability > best_probability)
            label_map[update_mask] = output_label
            best_probability[update_mask] = class_probability[update_mask]

            if probability_maps is not None:
                probability_maps[int(label)] = class_probability

    if keep_largest_component:
        label_map = keep_largest_component_per_label(label_map)

    return label_map, probability_maps


def save_label_map_in_original_space(
    label_map: Array3D,
    cropped_reference: sitk.Image,
    original_reference: sitk.Image,
    output_path: Union[str, Path],
    probability_maps: Optional[Dict[int, Array3D]] = None,
) -> None:
    """Write cropped-space predictions back onto the original NIfTI geometry."""
    seg_crop = array_to_image(label_map.astype(np.uint16), cropped_reference, pixel_id=sitk.sitkUInt16)
    seg_original = resample_to_reference(
        seg_crop,
        original_reference,
        sitk.sitkNearestNeighbor,
        default_value=0.0,
        pixel_id=sitk.sitkUInt16,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(seg_original, str(output_path))
    logging.info("Saved segmentation: %s", output_path)

    if probability_maps:
        for label, probability_array in probability_maps.items():
            prob_crop = array_to_image(probability_array.astype(np.float32), cropped_reference, pixel_id=sitk.sitkFloat32)
            prob_original = resample_to_reference(
                prob_crop,
                original_reference,
                sitk.sitkLinear,
                default_value=0.0,
                pixel_id=sitk.sitkFloat32,
            )
            prob_path = output_path.with_name(f"{strip_nii_suffix(output_path)}_label{label}_prob.nii.gz")
            sitk.WriteImage(prob_original, str(prob_path))
            logging.info("Saved probability for label %s: %s", label, prob_path)


def resolve_output_paths(
    test_images: Sequence[Union[str, Path]],
    output: Optional[str],
    output_dir: Optional[str],
    postfix: str,
) -> List[Path]:
    if output and len(test_images) > 1:
        raise ValueError("--output can only be used with one --test-image. Use --output-dir for batches.")

    if output:
        return [Path(output)]

    out_dir = Path(output_dir or "inference_outputs")
    return [out_dir / f"{strip_nii_suffix(path)}{postfix}.nii.gz" for path in test_images]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MASS/Iris in-context segmentation inference from raw NIfTI")

    parser.add_argument("--checkpoint", required=True, help="Path to MASS/Iris checkpoint")
    parser.add_argument("--test-image", nargs="+", required=True, help="Testing image(s), nii or nii.gz")
    parser.add_argument("--reference-image", nargs="+", required=True, help="Reference image(s), nii or nii.gz")
    parser.add_argument("--reference-mask", nargs="+", required=True, help="Reference segmentation mask(s), nii or nii.gz")
    parser.add_argument("--output", default=None, help="Output segmentation path for a single test image")
    parser.add_argument("--output-dir", default=None, help="Output directory for one or more test images")
    parser.add_argument("--output-postfix", default="_seg", help="Postfix used with --output-dir")

    parser.add_argument("--gpu", default="0", help="GPU id to use")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA checkpoint weights if available")
    parser.add_argument("--disable-amp", action="store_true", help="Disable AMP during inference")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default=None, help="AMP dtype override")

    parser.add_argument("--orientation", default="RAS", help="Training orientation, e.g. RAS. Use 'none' to keep native.")
    # Raw NIfTI inference uses SimpleITK image spacing order: x y z.
    parser.add_argument("--target-spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5], help="Resampling spacing in x y z order")
    parser.add_argument("--image-interp", choices=["linear", "bspline"], default="linear", help="Image interpolation for resampling")
    parser.add_argument("--modality", choices=["auto", "ct", "mr", "pet"], default="auto", help="Controls intensity clipping and body thresholding")
    parser.add_argument("--ct-clip", type=float, nargs=2, default=[-991.0, 500.0], help="CT clip range before z-score normalization")

    parser.add_argument("--body-method", choices=["threshold", "none"], default="threshold", help="Target body crop method")
    parser.add_argument("--body-margin", type=int, nargs=3, default=[16, 32, 32], help="Body crop margin in z y x voxels")
    parser.add_argument("--ct-body-threshold", type=float, default=-500.0, help="Threshold for CT body crop")

    parser.add_argument("--window-size", type=int, nargs=3, default=None, help="Sliding/reference crop size in z y x order")
    parser.add_argument("--overlap", type=float, default=None, help="Sliding-window overlap")
    parser.add_argument("--max-classes-per-forward", type=int, default=0, help="Class chunk size for one target forward; 0 means all labels at once")
    parser.add_argument("--reference-label", type=int, nargs="+", default=None, help="Reference label(s) to segment; default uses all nonzero labels")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum probability for assigning a foreground label")
    parser.add_argument("--output-label", type=int, default=None, help="Optional output label override, only valid when segmenting one reference label")
    parser.add_argument("--save-probability", action="store_true", help="Also save one probability map per reference label")
    parser.add_argument("--no-largest-component", action="store_true", help="Disable keeping only the largest connected component per output label")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    model, config = load_model(args.checkpoint, device, use_ema=args.use_ema)

    # Prefer checkpoint evaluation settings unless the CLI explicitly overrides them.
    config_window = config.get("evaluation", {}).get("window_size", [128, 128, 128])
    window_size = tuple(args.window_size or config_window)
    overlap = float(args.overlap if args.overlap is not None else config.get("evaluation", {}).get("overlap", 0.5))

    image_interpolator = sitk.sitkLinear if args.image_interp == "linear" else sitk.sitkBSpline
    amp_config = config.get("amp", {})
    use_amp = bool(amp_config.get("enabled", False)) and not args.disable_amp and device.type == "cuda"
    dtype_name = args.amp_dtype or amp_config.get("dtype", "float16")
    amp_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16

    reference_labels = collect_reference_labels(args.reference_mask, args.reference_label)
    logging.info("Reference labels to segment: %s", reference_labels)
    if args.output_label is not None and len(reference_labels) != 1:
        raise ValueError("--output-label can only be used when exactly one --reference-label is selected")

    # Reference images are encoded once and reused for every testing image.
    encoded_priors_by_label = encode_reference_embeddings_by_label(
        model=model,
        reference_images=args.reference_image,
        reference_masks=args.reference_mask,
        device=device,
        orientation=args.orientation,
        target_spacing_xyz=args.target_spacing,
        image_interpolator=image_interpolator,
        modality=args.modality,
        reference_labels=reference_labels,
        crop_size_zyx=window_size,
        ct_clip=(float(args.ct_clip[0]), float(args.ct_clip[1])),
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    output_labels_by_label = {
        int(label): int(args.output_label if args.output_label is not None else label)
        for label in reference_labels
    }
    output_paths = resolve_output_paths(args.test_image, args.output, args.output_dir, args.output_postfix)

    for test_image, output_path in zip(args.test_image, output_paths):
        logging.info("Processing target image: %s", test_image)
        # Target preprocessing mirrors training spacing/orientation, then the
        # final mask is resampled back to the original NIfTI space.
        original, cropped, target_tensor = preprocess_target_image(
            image_path=test_image,
            orientation=args.orientation,
            target_spacing_xyz=args.target_spacing,
            image_interpolator=image_interpolator,
            modality=args.modality,
            body_method=args.body_method,
            body_margin_zyx=args.body_margin,
            ct_body_threshold=args.ct_body_threshold,
            ct_clip=(float(args.ct_clip[0]), float(args.ct_clip[1])),
        )

        label_map, probability_maps = multiclass_inference(
            model=model,
            image=target_tensor,
            encoded_priors_by_label=encoded_priors_by_label,
            output_labels_by_label=output_labels_by_label,
            device=device,
            window_size_zyx=window_size,
            overlap=overlap,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            threshold=args.threshold,
            max_classes_per_forward=args.max_classes_per_forward,
            keep_probability_maps=args.save_probability,
            keep_largest_component=not args.no_largest_component,
        )

        save_label_map_in_original_space(
            label_map=label_map,
            cropped_reference=cropped,
            original_reference=original,
            output_path=output_path,
            probability_maps=probability_maps,
        )


if __name__ == "__main__":
    main()
