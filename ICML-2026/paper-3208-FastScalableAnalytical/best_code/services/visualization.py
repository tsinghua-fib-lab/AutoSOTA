# generate/visualization.py
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torchvision.utils import make_grid, save_image


def save_intermediates(
    model_name,
    dataset,
    result,
    run_paths,
    *,
    rank: int = 0,
    sample_offset: int = 0,
):
    xt_dir = Path(run_paths.intermediate_images) / f"rank{rank:02d}" / "x_t"
    x0_dir = Path(run_paths.intermediate_images) / f"rank{rank:02d}" / "x0_pred"
    xt_dir.mkdir(parents=True, exist_ok=True)
    x0_dir.mkdir(parents=True, exist_ok=True)

    for i, (xt, x0) in enumerate(zip(result.trajectory_xt, result.trajectory_x0)):
        t_label = result.timesteps[i] if result.timesteps else i

        if model_name != "ours":
            xt = dataset.postprocess(xt)
            x0 = dataset.postprocess(x0)

        for j, img in enumerate(xt):
            global_idx = sample_offset + j
            save_image(
                img.detach().cpu(),
                xt_dir / f"step_{t_label:04d}_sample_{global_idx:06d}.png",
            )

        for j, img in enumerate(x0):
            global_idx = sample_offset + j
            save_image(
                img.detach().cpu(),
                x0_dir / f"step_{t_label:04d}_sample_{global_idx:06d}.png",
            )


def save_comparison_step_grid(
    model_name,
    dataset,
    trajectories: List[torch.Tensor],
    t: int,
    save_dir,
    filename_suffix: str = "comparison",
):
    if not trajectories:
        raise ValueError("trajectories list cannot be empty")

    n = min(8, trajectories[0].shape[0])

    combined_list = []
    for traj in trajectories:
        x = traj[:n].detach().cpu().clamp(0, 1)
        combined_list.extend([img for img in x])

    grid = make_grid(
        combined_list,
        nrow=n,
         padding=0,
        normalize=False,
    )
    save_image(
        grid,
        Path(save_dir) / f"step_{t:04d}_{filename_suffix}.png",
        "png"
    )
    return grid
