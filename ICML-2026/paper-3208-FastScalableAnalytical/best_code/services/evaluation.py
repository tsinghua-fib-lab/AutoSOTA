# generate/evaluation.py
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from typing import Any, Dict, List

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image

from services.visualization import (
    save_intermediates,
    save_comparison_step_grid,
)

LOGGER = logging.getLogger("local_diffusion.generate")

## metrics
def calculate_l1_score(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Mean absolute error (L1) between two tensors.
    Returns python float.
    """
    # assumes matching dtype/shape; takes elementwise abs then mean
    return torch.mean(torch.abs(a - b)).item()


def calculate_r2_score(x: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate R² score between two sets of images.
    
    Computes the coefficient of determination R² = 1 - (SS_res / SS_tot)
    averaged over the batch.
    
    Args:
        x: Predicted images, shape (N, ...)
        y: Ground truth images (or reference), shape (N, ...)
        
    Returns:
        float: R² score
    """
    # Ensure tensors are on CPU and flattened properly
    x_flat = x.detach().reshape(x.size(0), -1).cpu()
    y_flat = y.detach().reshape(y.size(0), -1).cpu()

    # Calculate R² score for each sample pair
    var_y = torch.var(y_flat, dim=1)
    ss_res = torch.sum((x_flat - y_flat) ** 2, dim=1)
    
    # Avoid division by zero
    var_y = torch.where(var_y == 0, torch.ones_like(var_y), var_y)
    
    r2 = 1 - (ss_res / (var_y * x_flat.size(1)))
    return r2.mean().item()


def calculate_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate Mean Squared Error between two sets of images.
    
    Args:
        x: First set of images, shape (N, ...)
        y: Second set of images, shape (N, ...)
        
    Returns:
        float: Mean Squared Error
    """
    x_flat = x.detach().reshape(x.size(0), -1).cpu()
    y_flat = y.detach().reshape(y.size(0), -1).cpu()
    mse = torch.mean((x_flat - y_flat) ** 2, dim=1)
    return mse.mean().item()


def calculate_l2_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate L2 Euclidean distance between two sets of images.
    
    Args:
        x: First set of images, shape (N, ...)
        y: Second set of images, shape (N, ...)
        
    Returns:
        float: Mean L2 distance
    """
    x_flat = x.detach().reshape(x.size(0), -1).cpu()
    y_flat = y.detach().reshape(y.size(0), -1).cpu()
    dist = torch.norm(x_flat - y_flat, p=2, dim=1)
    return dist.mean().item()


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------

def gather_and_compute_global_metrics(
    main_x0_pp: torch.Tensor,
    base_x0_pp: torch.Tensor,
    *,
    rank: int,
    world_size: int,
):
    """Gather variable-length tensors and compute global metrics on rank0."""

    if world_size == 1:
        return {
            "mse": float(calculate_mse(main_x0_pp, base_x0_pp)),
            "l1": float(calculate_l1_score(main_x0_pp, base_x0_pp)),
            "r2": float(calculate_r2_score(main_x0_pp, base_x0_pp)),
        }

    assert dist.is_available() and dist.is_initialized()

    device = torch.device("cuda", torch.cuda.current_device())
    main_x0_pp = main_x0_pp.to(device, non_blocking=True)
    base_x0_pp = base_x0_pp.to(device, non_blocking=True)

    local_n = torch.tensor(
        [main_x0_pp.shape[0]],
        device=device,
        dtype=torch.long,
    )

    sizes = [
        torch.zeros(1, device=device, dtype=torch.long)
        for _ in range(world_size)
    ]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)

    def _pad(x, n):
        if x.shape[0] == n:
            return x
        pad_shape = (n - x.shape[0],) + x.shape[1:]
        pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=0)

    main_pad = _pad(main_x0_pp, max_n)
    base_pad = _pad(base_x0_pp, max_n)

    main_pad = main_pad.contiguous()

    main_list = [torch.zeros_like(main_pad) for _ in range(world_size)]
    base_list = [torch.zeros_like(base_pad) for _ in range(world_size)]

    dist.all_gather(main_list, main_pad)
    dist.all_gather(base_list, base_pad)

    if rank != 0:
        return None

    main_all = torch.cat(
        [m[:n] for m, n in zip(main_list, sizes)],
        dim=0,
    ).cpu()

    base_all = torch.cat(
        [b[:n] for b, n in zip(base_list, sizes)],
        dim=0,
    ).cpu()

    return {
        "mse": float(calculate_mse(main_all, base_all)),
        "l1": float(calculate_l1_score(main_all, base_all)),
        "r2": float(calculate_r2_score(main_all, base_all)),
    }


# -----------------------------------------------------------------------------
# Evaluation logic
# -----------------------------------------------------------------------------
def extract_experiment_config(cfg: OmegaConf) -> Dict[str, Any]:
    return {
        "seed": cfg.experiment.seed,
        "dataset":  cfg.dataset.name,
        "resolution": cfg.dataset.resolution,
        "sampling": {
            "num_samples": cfg.sampling.num_samples,
            "batch_size": cfg.sampling.batch_size,
            "num_inference_steps": cfg.sampling.num_inference_steps,
        },
        "baseline_name": cfg.metrics.baseline_name,
        "method": {
            "name": cfg.model.name,
            "k_min": cfg.model.params.get("k_min", cfg.model.params.get("k_neighbors", None)),
            "k_max": cfg.model.params.get("k_max", cfg.model.params.get("m_coarse_candidates", None)),
            "weight_method": cfg.model.params.get("weight_method", None)
        },
    }

def evaluate_main_model(
    model,
    dataset_bundle,
    result,
    cfg: OmegaConf,
    run_paths: Any,
    sampling_time_total: float,
    *,
    rank: int = 0,
    world_size: int = 1,
    sample_offset: int = 0,
    local_batch: int = 1,
) -> None:
    LOGGER.info("Evaluating main model results...")

    metrics: Dict[str, Any] = {}

    local_num_samples = (
        int(result.images.shape[0])
        if hasattr(result, "images") and result.images is not None
        else 0
    )

    num_batches = math.ceil(local_num_samples / max(1, local_batch))
    total_steps = num_batches * int(cfg.sampling.num_inference_steps)
    avg_step_time = sampling_time_total / total_steps if total_steps > 0 else 0.0

    if rank == 0:
        metrics.update(
            dict(
                experiment_config=extract_experiment_config(cfg),
                main_sampling_time_total_wall_max=float(sampling_time_total),
                main_sampling_time_per_step_est=float(avg_step_time),
            )
        )

    images_tensor = result.images
    if images_tensor is None:
        _log_metrics(metrics, run_paths, rank)
        return

    if hasattr(model, "device") and images_tensor.device != model.device:
        images_tensor = images_tensor.to(model.device)

    if cfg.model.name in ["gaussian", "pca_locality_channel_wise"]:
        processed_images = images_tensor.detach().cpu().clamp(0, 1)
    else:
        processed_images = dataset_bundle.postprocess(images_tensor).detach().cpu()

    if cfg.metrics.output.save_final_images and run_paths.images is not None:
        out_dir = Path(run_paths.images) / f"rank{rank:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, image in enumerate(processed_images):
            global_idx = sample_offset + idx
            save_image(image, out_dir / f"sample_{global_idx:06d}.png")

    if cfg.metrics.output.save_image_grid and rank == 0:
        n = min(processed_images.size(0), 16)
        if n > 0:
            grid = make_grid(processed_images[:n], nrow=4, normalize=False)
            save_image(grid, Path(run_paths.run_dir) / "grid.png")

    # if (
    #     getattr(result, "trajectory_xt", None)
    #     and getattr(result, "trajectory_x0", None)
    #     and cfg.metrics.output.save_intermediate_images
    # ):
    #     save_intermediates(
    #         cfg.model.name,
    #         dataset_bundle,
    #         result,
    #         run_paths,
    #         rank=rank,
    #         sample_offset=sample_offset,
    #     )
    if rank == 0:
        _log_metrics(metrics, run_paths, rank)

from pathlib import Path
from typing import List
import torch
import torchvision.utils as vutils


def save_comparison_trajectory_grids(
    main_traj: List[torch.Tensor],   # list length=steps, each [B,C,H,W]
    base_traj: List[torch.Tensor],   # list length=steps, each [B,C,H,W]
    timesteps: List[int],            # length=steps
    out_dir: Path,
    filename_prefix: str = "traj_x0_comparison",
) -> None:
    """
    Save per-sample trajectory comparison:
    A grid with 2 rows (main on first row, baseline on second row),
    and `steps` columns (each column is one timestep).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = min(len(main_traj), len(base_traj), len(timesteps))
    if steps == 0:
        return

    B = main_traj[0].shape[0]

    for b in range(B):
        # order: main_t0..main_tS-1, base_t0..base_tS-1
        imgs = []
        for i in range(steps):
            imgs.append(main_traj[i][b])  # [C,H,W]
        for i in range(steps):
            imgs.append(base_traj[i][b])

        # nrow=steps => first row is main (steps images), second row is baseline (steps images)
        grid = vutils.make_grid(
            imgs,
            nrow=steps,
            padding=0,
            normalize=False,
        )

        # include timestep range in filename (optional)
        t0, t1 = timesteps[0], timesteps[steps - 1]
        out_path = out_dir / f"{filename_prefix}_sample{b:03d}_t{t0}_to_t{t1}.png"
        vutils.save_image(grid, out_path)



def evaluate_comparison(
    dataset_bundle,
    main_result,
    baseline_result,
    cfg: OmegaConf,
    run_paths: Any,
    *,
    rank: int = 0,
    return_intermediates
) -> None:
    LOGGER.info("Evaluating comparison metrics...")

    if not (
        getattr(main_result, "trajectory_x0", None)
        and getattr(baseline_result, "trajectory_x0", None)
    ):
        return

    comp_dir = Path(run_paths.run_dir) / "comparison" / f"rank{rank:02d}"
    comp_dir.mkdir(parents=True, exist_ok=True)

    steps = min(
        len(main_result.trajectory_x0),
        len(baseline_result.trajectory_x0),
    )

    step_metrics_list: Dict[str, List[float]] = {
        "mse_list": [],
        "l1_list": [],
        "r2_list": [],
    }

     # NEW: whether to save the per-image full trajectory comparison
    # default: mirror save_intermediate_images
    save_traj_imgs = True
    # NEW: cache postprocessed step outputs for later per-image trajectory grids
    main_traj_pp = []  # List[Tensor [B,C,H,W]]
    base_traj_pp = []  # List[Tensor [B,C,H,W]]
    timesteps_used = []  # List[int]



    for i in range(steps):
        is_last = i == steps - 1
        # if not cfg.metrics.output.save_intermediate_images and not is_last:
        #     continue

        should_save_step_img = cfg.metrics.output.save_intermediate_images
        should_do_this_step = cfg.metrics.output.save_intermediate_images or is_last or save_traj_imgs or return_intermediates

        if not should_do_this_step:
            continue

        t = main_result.timesteps[i]
        main_x0 = main_result.trajectory_x0[i]
        base_x0 = baseline_result.trajectory_x0[i]

        if cfg.model.name in ["gaussian", "pca_locality_channel_wise"]:
            main_x0_pp = main_x0.clamp(0, 1)
        else:
            main_x0_pp = dataset_bundle.postprocess(main_x0)

        base_x0_pp = dataset_bundle.postprocess(base_x0)

        # NEW: cache for per-image full trajectory comparison
        if save_traj_imgs:
            main_traj_pp.append(main_x0_pp.detach().cpu())
            base_traj_pp.append(base_x0_pp.detach().cpu())
            timesteps_used.append(int(t))


        if cfg.metrics.output.save_intermediate_images:
            save_comparison_step_grid(
                cfg.model.name,
                dataset_bundle,
                [main_x0_pp, base_x0_pp],
                t,
                comp_dir,
                filename_suffix="comparison_x0",
            )
        if return_intermediates: 
            global_metrics = gather_and_compute_global_metrics(
            main_x0_pp,
            base_x0_pp,
            rank=rank,
            world_size=dist.get_world_size()
            if dist.is_initialized()
            else 1,
        )
        else:
            if is_last: 
                global_metrics = gather_and_compute_global_metrics(
                main_x0_pp,
                base_x0_pp,
                rank=rank,
                world_size=dist.get_world_size()
                if dist.is_initialized()
                else 1,
                )

        if rank == 0 and global_metrics is not None:
            step_metrics_list["mse_list"].append(float(global_metrics["mse"]))
            step_metrics_list["l1_list"].append(float(global_metrics["l1"]))
            step_metrics_list["r2_list"].append(float(global_metrics["r2"]))

            if is_last:
                _log_metrics(
                    {
                        "mse_score": global_metrics["mse"],
                        "l1_score": global_metrics["l1"],
                        "r2_score": global_metrics["r2"],
                        **step_metrics_list,
                    },
                    run_paths,
                    rank,
                )
        # NEW: after the loop, save per-image trajectory grid across all x_t(step)
        if save_traj_imgs and len(main_traj_pp) > 0:
            traj_dir = comp_dir / "trajectory_grids"
            save_comparison_trajectory_grids(
                main_traj=main_traj_pp,
                base_traj=base_traj_pp,
                timesteps=timesteps_used,
                out_dir=traj_dir,
                filename_prefix="traj_x0_comparison",
            )
        
# -----------------------------------------------------------------------------
# Metric logging
# -----------------------------------------------------------------------------
def _round_metrics(metrics: Any, ndigits: int = 8):
    # ✅ float：round
    if isinstance(metrics, float):
        return round(metrics, ndigits)

    # dict: recurse
    if isinstance(metrics, dict):
        return {k: _round_metrics(v, ndigits) for k, v in metrics.items()}

    # list/tuple: recurse on each element, preserving container type
    if isinstance(metrics, (list, tuple)):
        rounded = [_round_metrics(x, ndigits) for x in metrics]
        return type(metrics)(rounded)

    # other types (int/str/None/torch.Tensor/...) are returned as-is
    return metrics

def _log_metrics(metrics: Dict[str, Any], run_paths: Any, rank: int):
    if not metrics or rank != 0:
        return

    metrics = _round_metrics(metrics)
    path = Path(run_paths.run_dir) / "metrics.json"

    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            pass

    existing.update(metrics)
    path.write_text(json.dumps(existing, indent=2))

    if rank == 0:
        print("\n=== Run Metrics ===")
        for k, v in sorted(metrics.items()):
            print(f"{k}: {v}")
        print("===================\n")
