from __future__ import annotations

from typing import Any

import numpy as np
import torch


def build_time_grid(num_frames: int, fps: float) -> np.ndarray:
    if fps <= 0:
        return np.zeros((num_frames,), dtype=np.float64)
    return np.arange(num_frames, dtype=np.float64) / float(fps)


def interpolate_bbox_at_time(segment: dict[str, Any], t_query: float) -> tuple[float, float, float, float] | None:
    t_list = segment.get("t") or []
    if not t_list:
        return None

    t_start = float(segment.get("t_start", t_list[0]))
    t_end = float(segment.get("t_end", t_list[-1]))
    if t_query < t_start or t_query > t_end:
        return None

    cx_list = segment["cx"]
    cy_list = segment["cy"]
    w_list = segment["w"]
    h_list = segment["h"]
    if len(t_list) == 1:
        return float(cx_list[0]), float(cy_list[0]), float(w_list[0]), float(h_list[0])

    right = int(np.searchsorted(np.asarray(t_list, dtype=np.float64), t_query, side="left"))
    right = min(max(right, 0), len(t_list) - 1)
    left = max(right - 1, 0)
    if t_query >= t_list[-1]:
        left = right = len(t_list) - 1

    if left == right:
        return float(cx_list[left]), float(cy_list[left]), float(w_list[left]), float(h_list[left])

    t_left = float(t_list[left])
    t_right = float(t_list[right])
    alpha = 0.0 if abs(t_right - t_left) < 1e-8 else (t_query - t_left) / (t_right - t_left)
    cx = (1 - alpha) * float(cx_list[left]) + alpha * float(cx_list[right])
    cy = (1 - alpha) * float(cy_list[left]) + alpha * float(cy_list[right])
    width = (1 - alpha) * float(w_list[left]) + alpha * float(w_list[right])
    height = (1 - alpha) * float(h_list[left]) + alpha * float(h_list[right])
    return cx, cy, width, height


def rasterize_bbox_to_mask(cx: float, cy: float, width: float, height: float, H: int, W: int) -> torch.Tensor:
    mask = torch.zeros((H, W), dtype=torch.float32)
    if H <= 0 or W <= 0:
        return mask

    x1 = int(np.floor((cx - width / 2.0) * W))
    x2 = int(np.ceil((cx + width / 2.0) * W))
    y1 = int(np.floor((cy - height / 2.0) * H))
    y2 = int(np.ceil((cy + height / 2.0) * H))
    x1, x2 = max(0, min(W, x1)), max(0, min(W, x2))
    y1, y2 = max(0, min(H, y1)), max(0, min(H, y2))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    return mask


def build_object_masks(
    objects: list[dict[str, Any]],
    t_sample: np.ndarray,
    H: int,
    W: int,
    object_ids: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    masks: dict[str, torch.Tensor] = {}
    T = len(t_sample)
    for obj in objects:
        object_id = str(obj.get("object_id", ""))
        if object_ids is not None and object_id not in object_ids:
            continue

        mask = torch.zeros((1, 1, T, H, W), dtype=torch.float32)
        for segment in obj.get("segments", []):
            for ti, t in enumerate(t_sample):
                bbox = interpolate_bbox_at_time(segment, float(t))
                if bbox is None:
                    continue
                mask[0, 0, ti] = torch.maximum(mask[0, 0, ti], rasterize_bbox_to_mask(*bbox, H=H, W=W))
        masks[object_id] = mask
    return masks


def aggregate_masks(
    masks: dict[str, torch.Tensor],
    r_by_object: dict[str, float],
    mode: str = "max",
) -> torch.Tensor:
    if not masks:
        raise ValueError("No masks were provided")

    first = next(iter(masks.values()))
    aggregate = torch.zeros_like(first)
    if mode not in {"max", "sum"}:
        raise ValueError(f"Unsupported mask aggregation mode: {mode}")

    for object_id, mask in masks.items():
        strength = float(r_by_object.get(object_id, 0.0))
        weighted = strength * mask
        aggregate = torch.maximum(aggregate, weighted) if mode == "max" else aggregate + weighted
    return aggregate.clamp(0.0, 1.0)
