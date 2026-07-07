"""Overlap-tile inference utilities for FLAME.

Tiling policy:
  - Tile size: 256x256
  - Valid output: center 192x192
  - Stride: 192
  - Padding: cv2.BORDER_REFLECT_101 to avoid black borders
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch


def _compute_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _compute_reflect_padding(
    height: int,
    width: int,
    tile_size: int,
    valid_size: int,
) -> Tuple[int, int, int, int, int]:
    if tile_size <= 0 or valid_size <= 0:
        raise ValueError("tile_size and valid_size must be positive")
    if valid_size > tile_size:
        raise ValueError("valid_size must be <= tile_size")
    if (tile_size - valid_size) % 2 != 0:
        raise ValueError("tile_size - valid_size must be even to form a centered valid region")

    margin = (tile_size - valid_size) // 2
    pad_top = pad_bottom = margin
    pad_left = pad_right = margin

    padded_h = height + pad_top + pad_bottom
    if padded_h < tile_size:
        extra = tile_size - padded_h
        pad_top += extra // 2
        pad_bottom += extra - extra // 2

    padded_w = width + pad_left + pad_right
    if padded_w < tile_size:
        extra = tile_size - padded_w
        pad_left += extra // 2
        pad_right += extra - extra // 2

    return pad_top, pad_bottom, pad_left, pad_right, margin


@torch.no_grad()
def overlap_tile_predict_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    *,
    tile_size: int = 256,
    valid_size: int = 192,
    stride: int = 192,
    return_detection_logit: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Predict per-pixel logits on an arbitrary-size image via overlap-tiling.

    Args:
        model: ForgeryLocalizer-like module. Must expose `transforms.transforms`.
        image: Float tensor in [0,1] with shape [3, H, W] on CPU or GPU.
        device: Device to run inference on.
        tile_size: Input tile size (square).
        valid_size: Central valid region size (square).
        stride: Sliding stride (square).
        return_detection_logit: If True, returns a global detection logit aggregated by max over tiles.

    Returns:
        logits: Tensor of shape [1, 1, H, W] on `device`.
        detection_logit: Optional tensor of shape [1, 1] on `device`.
    """
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError("image must be a [3, H, W] tensor")

    image_cpu = image.detach().to("cpu", dtype=torch.float32)
    img_np = image_cpu.permute(1, 2, 0).numpy()
    height, width = int(img_np.shape[0]), int(img_np.shape[1])

    pad_top, pad_bottom, pad_left, pad_right, margin = _compute_reflect_padding(
        height, width, tile_size, valid_size
    )

    padded = cv2.copyMakeBorder(
        img_np,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )
    padded_h, padded_w = int(padded.shape[0]), int(padded.shape[1])

    ys = _compute_starts(padded_h, tile_size, stride)
    xs = _compute_starts(padded_w, tile_size, stride)

    accum = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight = np.zeros((padded_h, padded_w), dtype=np.float32)

    max_detection_logit: Optional[float] = None
    amp_enabled = device.type == "cuda"

    for y0 in ys:
        for x0 in xs:
            tile = padded[y0 : y0 + tile_size, x0 : x0 + tile_size, :]
            tile_t = torch.from_numpy(tile).permute(2, 0, 1).contiguous()

            if not hasattr(model, "transforms") or not hasattr(model.transforms, "transforms"):
                raise AttributeError("model must expose `model.transforms.transforms` for SAM2 preprocessing")

            tile_t = model.transforms.transforms(tile_t)
            tile_t = tile_t.unsqueeze(0).to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                if return_detection_logit:
                    tile_logits, extras = model(tile_t, streams=None, output_extras=True)
                    det_logit = extras.get("detection_logit", None)
                    if det_logit is not None:
                        det_val = float(det_logit.detach().float().max().cpu().item())
                        max_detection_logit = det_val if max_detection_logit is None else max(max_detection_logit, det_val)
                else:
                    tile_logits = model(tile_t, streams=None, output_extras=False)

            logits_np = tile_logits.detach().float().cpu().squeeze(0).squeeze(0).numpy()
            center = logits_np[margin : margin + valid_size, margin : margin + valid_size]

            y1 = y0 + margin
            x1 = x0 + margin
            accum[y1 : y1 + valid_size, x1 : x1 + valid_size] += center
            weight[y1 : y1 + valid_size, x1 : x1 + valid_size] += 1.0

    merged = accum / np.maximum(weight, 1e-6)
    merged = merged[pad_top : pad_top + height, pad_left : pad_left + width]

    logits = torch.from_numpy(merged).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    detection_logit = None
    if return_detection_logit and max_detection_logit is not None:
        detection_logit = torch.tensor([[max_detection_logit]], device=device, dtype=torch.float32)

    return logits, detection_logit


@torch.no_grad()
def overlap_tile_predict_logits_global_guided(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    *,
    tile_size: int = 256,
    valid_size: int = 192,
    stride: int = 192,
    return_detection_logit: bool = False,
    tile_batch_size: int = 8,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Predict per-pixel logits via overlap-tiling using a global semantic context.

    This implements a Global-Guided Local Refinement strategy:
      1) Resize full image to model.global_transforms resolution and run the (frozen) SAM encoder once.
      2) Slide local patches over the original image and run the forensic stream per patch.
      3) ROI Align global semantic features to each patch region and decode a refined local mask.
      4) Merge patch logits via overlap-tiling.
    """
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError("image must be a [3, H, W] tensor")

    if not hasattr(model, "global_transforms") or not hasattr(model, "encode_global"):
        raise AttributeError("model must expose `global_transforms` and `encode_global` for global-guided tiling")

    image_cpu = image.detach().to("cpu", dtype=torch.float32)
    img_np = image_cpu.permute(1, 2, 0).numpy()
    height, width = int(img_np.shape[0]), int(img_np.shape[1])

    pad_top, pad_bottom, pad_left, pad_right, margin = _compute_reflect_padding(
        height, width, tile_size, valid_size
    )

    padded = cv2.copyMakeBorder(
        img_np,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )
    padded_h, padded_w = int(padded.shape[0]), int(padded.shape[1])

    ys = _compute_starts(padded_h, tile_size, stride)
    xs = _compute_starts(padded_w, tile_size, stride)

    accum = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight = np.zeros((padded_h, padded_w), dtype=np.float32)

    max_detection_logit: Optional[float] = None
    amp_enabled = device.type == "cuda"

    # Encode global semantic context once.
    global_t = model.global_transforms.transforms(image_cpu)
    global_t = global_t.unsqueeze(0).to(device, non_blocking=True)
    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
        global_context = model.encode_global(global_t)

    # Process tiles in mini-batches.
    tile_tensors = []
    tile_coords = []
    tile_positions = []

    def _flush_batch():
        nonlocal max_detection_logit, tile_tensors, tile_coords, tile_positions, accum, weight
        if not tile_tensors:
            return
        tiles_b = torch.stack(tile_tensors, dim=0).to(device, non_blocking=True)
        coords_b = torch.stack(tile_coords, dim=0).to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            tile_logits, extras = model(
                tiles_b,
                streams=None,
                output_extras=True,
                global_context=global_context,
                norm_coords=coords_b,
            )

        if return_detection_logit:
            det_logit = extras.get("detection_logit", None) if isinstance(extras, dict) else None
            if det_logit is not None:
                det_vals = det_logit.detach().float().flatten().cpu().numpy().tolist()
                for det_val in det_vals:
                    max_detection_logit = det_val if max_detection_logit is None else max(max_detection_logit, det_val)

        logits_np = tile_logits.detach().float().cpu().squeeze(1).numpy()  # [B, tile, tile]
        for logits_tile, (y0, x0) in zip(logits_np, tile_positions):
            center = logits_tile[margin : margin + valid_size, margin : margin + valid_size]
            y1 = y0 + margin
            x1 = x0 + margin
            accum[y1 : y1 + valid_size, x1 : x1 + valid_size] += center
            weight[y1 : y1 + valid_size, x1 : x1 + valid_size] += 1.0

        tile_tensors = []
        tile_coords = []
        tile_positions = []

    for y0 in ys:
        for x0 in xs:
            tile = padded[y0 : y0 + tile_size, x0 : x0 + tile_size, :]
            tile_t = torch.from_numpy(tile).permute(2, 0, 1).contiguous()

            if not hasattr(model, "transforms") or not hasattr(model.transforms, "transforms"):
                raise AttributeError("model must expose `model.transforms.transforms` for local preprocessing")

            tile_t = model.transforms.transforms(tile_t)

            # Compute normalized coords in the original (unpadded) image space.
            x1_orig = x0 - pad_left
            y1_orig = y0 - pad_top
            x2_orig = x1_orig + tile_size
            y2_orig = y1_orig + tile_size

            x1c = float(max(0, x1_orig))
            y1c = float(max(0, y1_orig))
            x2c = float(min(width, x2_orig))
            y2c = float(min(height, y2_orig))

            norm = torch.tensor(
                [
                    x1c / float(max(width, 1)),
                    y1c / float(max(height, 1)),
                    x2c / float(max(width, 1)),
                    y2c / float(max(height, 1)),
                ],
                dtype=torch.float32,
            )

            tile_tensors.append(tile_t)
            tile_coords.append(norm)
            tile_positions.append((y0, x0))

            if len(tile_tensors) >= tile_batch_size:
                _flush_batch()

    _flush_batch()

    merged = accum / np.maximum(weight, 1e-6)
    merged = merged[pad_top : pad_top + height, pad_left : pad_left + width]

    logits = torch.from_numpy(merged).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    detection_logit = None
    if return_detection_logit and max_detection_logit is not None:
        detection_logit = torch.tensor([[max_detection_logit]], device=device, dtype=torch.float32)

    return logits, detection_logit
