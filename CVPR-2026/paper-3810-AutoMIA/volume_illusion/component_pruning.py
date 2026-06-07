"""Projection-aligned component (PAC) selector for binary volumes."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List

try:
    from scipy.ndimage import label as cc_label
except Exception as _e:  # pragma: no cover
    cc_label = None

from pytorch3d.structures import Volumes


def _compute_view_weight_for_camera(volume_model, camera) -> float:
    """Reproduce the dichromatic blend logic used in the forward pass."""
    centers = camera.get_camera_center()
    view_dir = torch.nn.functional.normalize(-centers, dim=-1)
    axis = volume_model.axis_dir.to(view_dir.device)
    dot = torch.matmul(view_dir, axis)
    w = (dot >= 0).float().item()
    return float(w)


def _soft_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)


def _outside_loss(pred: torch.Tensor, target: torch.Tensor, bg_far_w: Optional[torch.Tensor]) -> torch.Tensor:
    if bg_far_w is None:
        return (pred * (1 - target)).mean()
    if bg_far_w.dim() == 3 and bg_far_w.shape[0] == 1:
        w = bg_far_w.squeeze(0)
    else:
        w = bg_far_w
    num = (pred * (1 - target) * w).sum()
    den = w.sum().clamp_min(1e-6)
    return num / den


def _fill_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    fg = (target > 0.5).float()
    if fg.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return (((pred - 1.0) ** 2) * fg).sum() / fg.sum()


@torch.no_grad()
def select_components_by_projection(
    volume_model,
    camera_view1,
    camera_view2,
    target_alpha1: torch.Tensor,
    target_alpha2: torch.Tensor,
    density_threshold: float,
    bg_far_w1: Optional[torch.Tensor] = None,
    bg_far_w2: Optional[torch.Tensor] = None,
    k_max: int = 4,
    score_thresh: float = 0.1,
    min_ratio: float = 0.001,
) -> torch.Tensor:
    """Select 3D components whose projections best explain both silhouettes."""

    if cc_label is None:
        return torch.ones_like(volume_model.log_densities[0], dtype=torch.bool)

    device = volume_model.log_densities.device

    dens_raw = torch.sigmoid(volume_model.inner_temperature * volume_model.log_densities - volume_model.density_bias)
    dens_raw = torch.clamp(dens_raw * volume_model.outer_scale, 0.0, 1.0)
    binary = (dens_raw[0] > float(density_threshold)).to(torch.uint8).cpu().numpy()

    labels, nlabel = cc_label(binary, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if nlabel == 0:
        return torch.zeros_like(volume_model.log_densities[0], dtype=torch.bool)

    sizes = np.bincount(labels.flatten())
    if len(sizes) == 0:
        return torch.zeros_like(volume_model.log_densities[0], dtype=torch.bool)
    sizes[0] = 0

    total_vox = binary.size
    cand_ids: List[int] = [i for i in range(1, nlabel + 1) if sizes[i] / total_vox >= float(min_ratio)]
    if not cand_ids:
        cand_ids = [int(np.argmax(sizes))]

    D = volume_model.get_densities()
    C1 = torch.sigmoid(volume_model.log_colors1).permute(0, 2, 3, 1).unsqueeze(0)
    C2 = torch.sigmoid(volume_model.log_colors2).permute(0, 2, 3, 1).unsqueeze(0)

    w1 = _compute_view_weight_for_camera(volume_model, camera_view1)
    w2 = _compute_view_weight_for_camera(volume_model, camera_view2)
    C_cam1 = w1 * C1 + (1.0 - w1) * C2
    C_cam2 = w2 * C1 + (1.0 - w2) * C2

    ta1 = target_alpha1.squeeze()
    ta2 = target_alpha2.squeeze()
    if ta1.dim() != 2 or ta2.dim() != 2:
        raise ValueError("target_alpha must be (H,W) or (1,H,W)")
    ta1 = ta1.to(device)
    ta2 = ta2.to(device)
    if bg_far_w1 is not None:
        bg_far_w1 = bg_far_w1.squeeze().to(device)
    if bg_far_w2 is not None:
        bg_far_w2 = bg_far_w2.squeeze().to(device)

    scores: List[Tuple[float, int]] = []
    max_size = float(sizes.max()) if sizes.max() > 0 else 1.0

    for lid in cand_ids:
        M_np = (labels == lid)
        M = torch.from_numpy(M_np).to(device)
        M_t = M.permute(1, 2, 0).unsqueeze(0).unsqueeze(0).float()

        D_i = D * M_t

        vol1 = Volumes(densities=D_i, features=C_cam1 * M_t, voxel_size=volume_model._voxel_size)
        vol2 = Volumes(densities=D_i, features=C_cam2 * M_t, voxel_size=volume_model._voxel_size)

        rgba1 = volume_model._renderer(cameras=camera_view1, volumes=vol1)[0]
        rgba2 = volume_model._renderer(cameras=camera_view2, volumes=vol2)[0]
        a1 = rgba1[0, ..., 3].clamp(0.0, 1.0)
        a2 = rgba2[0, ..., 3].clamp(0.0, 1.0)

        iou1 = _soft_iou(a1, ta1)
        iou2 = _soft_iou(a2, ta2)
        out1 = _outside_loss(a1, ta1, bg_far_w1)
        out2 = _outside_loss(a2, ta2, bg_far_w2)
        fill1 = _fill_loss(a1, ta1)
        fill2 = _fill_loss(a2, ta2)

        size_norm = float(sizes[lid]) / max_size

        w_iou = 1.0
        w_out = 1.0
        w_fill = 0.3
        w_size = 0.05

        score = (w_iou * (iou1 + iou2) / 2.0
                 - w_out * (out1 + out2) / 2.0
                 - w_fill * (fill1 + fill2) / 2.0
                 + w_size * float(np.log(max(size_norm, 1e-6))))
        scores.append((float(score.item() if isinstance(score, torch.Tensor) else score), int(lid)))

    scores.sort(key=lambda x: x[0], reverse=True)
    selected: List[int] = [lid for (s, lid) in scores if s >= float(score_thresh)][: int(k_max)]
    if not selected:
        selected = [scores[0][1]]

    keep_mask_np = np.isin(labels, np.array(selected, dtype=np.int64))
    keep_mask = torch.from_numpy(keep_mask_np).to(device)
    return keep_mask


