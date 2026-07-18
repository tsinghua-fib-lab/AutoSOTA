#!/usr/bin/env python3
"""
Simplified in-memory training script for Mip-NeRF 360 scenes.
Uses TideGaussianModel but runs entirely in GPU memory (no SSD offload).
Based on the standard 3D Gaussian Splatting training loop with evaluation.
"""

import os
import sys
import math
import json
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from argparse import ArgumentParser

from arguments import (
    AuxiliaryParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    BenchmarkParams,
    DebugParams,
    init_args,
)
from scene import Scene
from scene.cameras import Camera
from strategies.tide_engine.gaussian_model import TideGaussianModel
from strategies.base_engine import (
    calculate_filters,
    pipeline_forward_one_step,
    torch_compiled_loss,
)
from clm_kernels import fused_ssim
from utils.general_utils import (
    safe_state,
    prepare_output_and_logger,
    get_cur_iter,
    get_args,
    set_cur_iter,
    set_log_file,
    get_img_width,
    get_img_height,
    get_log_file,
    print_rank_0,
    check_initial_gpu_memory_usage,
    log_cpu_memory_usage,
)
from utils.camera_utils import loadCam, loadCam_raw_from_disk
from utils.loss_utils import l1_loss, ssim as py_ssim


def compute_psnr(img1, img2):
    """Compute PSNR between two images (3, H, W) in [0, 1]."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse.item()))


def compute_ssim(img1, img2):
    """Compute SSIM using fused CUDA kernel. Images in (3, H, W), [0, 1]."""
    return fused_ssim(img1.unsqueeze(0), img2.unsqueeze(0), train=False).item()


@torch.no_grad()
def evaluate_scene(args, gaussians, scene, background, pipe_args, log_file):
    """Evaluate PSNR/SSIM on all test cameras."""
    test_cameras_info = scene.getTestCamerasInfo()
    if not test_cameras_info:
        print_rank_0("No test cameras available for evaluation.")
        return {"PSNR": 0.0, "SSIM": 0.0}

    psnr_list = []
    ssim_list = []

    for cam_info in test_cameras_info:
        # Load camera from pre-decoded raw files (fast)
        cam = loadCam_raw_from_disk(args, 0, cam_info, to_gpu=True)
        cam.original_image = cam.original_image_backup.cuda()
        cam.world_view_transform = cam.world_view_transform.cuda()
        cam.full_proj_transform = cam.full_proj_transform.cuda()

        opacity_gpu = gaussians.get_opacity
        scaling_gpu = gaussians.get_scaling
        rotation_gpu = gaussians.get_rotation
        xyz_gpu = gaussians.get_xyz

        features = gaussians.get_features
        if not features.is_cuda:
            features = features.cuda()

        filters, _, _ = calculate_filters(
            [cam], xyz_gpu, opacity_gpu, scaling_gpu, rotation_gpu
        )
        visible_mask = filters[0]

        if visible_mask.numel() == 0:
            continue

        filtered_xyz = xyz_gpu[visible_mask]
        filtered_opacity = opacity_gpu[visible_mask]
        filtered_scaling = scaling_gpu[visible_mask]
        filtered_rotation = rotation_gpu[visible_mask]
        filtered_shs = features[visible_mask]

        rendered_image, _, _ = pipeline_forward_one_step(
            filtered_opacity,
            filtered_scaling,
            filtered_rotation,
            filtered_xyz,
            filtered_shs,
            cam,
            scene,
            gaussians,
            background,
            pipe_args,
            eval=True,
        )

        gt_image = cam.original_image.float() / 255.0
        gt_image = torch.clamp(gt_image, 0.0, 1.0)
        rendered_image = torch.clamp(rendered_image, 0.0, 1.0)

        psnr_val = compute_psnr(rendered_image, gt_image)
        ssim_val = compute_ssim(rendered_image, gt_image)

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0

    return {"PSNR": avg_psnr, "SSIM": avg_ssim}


def training(dataset_args, opt_args, pipe_args, args, log_file):
    """Main GPU-only training loop."""

    torch.cuda.set_device(args.gpu)
    prepare_output_and_logger(dataset_args)
    log_cpu_memory_usage("at the beginning of training")

    print_rank_0("Creating TideGaussianModel (GPU-only / no-offload mode)")
    gaussians = TideGaussianModel(sh_degree=dataset_args.sh_degree)
    gaussians.args = args

    with torch.no_grad():
        scene = Scene(args, gaussians)
        n_train = len(scene.getTrainCamerasInfo()) if scene.getTrainCamerasInfo() else 0
        n_test = len(scene.getTestCamerasInfo()) if scene.getTestCamerasInfo() else 0
        print_rank_0(
            "Scene loaded: %d train, %d test cameras" % (n_train, n_test)
        )

    print_rank_0("Moving parameters to GPU for in-memory training...")
    gaussians._xyz = nn.Parameter(gaussians._xyz.cuda())
    gaussians._opacity = nn.Parameter(gaussians._opacity.cuda())
    gaussians._scaling = nn.Parameter(gaussians._scaling.cuda())
    gaussians._rotation = nn.Parameter(gaussians._rotation.cuda())

    if gaussians._features_dc is not None:
        gaussians._features_dc = nn.Parameter(gaussians._features_dc.cuda())
    if gaussians._features_rest is not None:
        gaussians._features_rest = nn.Parameter(gaussians._features_rest.cuda())

    N = gaussians._xyz.shape[0]
    print_rank_0("Total Gaussians: %d" % N)

    # Set up optimizer
    lr_scale = 1.0
    if opt_args.lr_scale_mode == "linear":
        lr_scale = args.bsz
    elif opt_args.lr_scale_mode == "sqrt":
        lr_scale = np.sqrt(args.bsz)

    param_groups = [
        {
            "params": [gaussians._xyz],
            "lr": opt_args.position_lr_init * gaussians.spatial_lr_scale * lr_scale * args.lr_scale_pos_and_scale,
            "name": "xyz",
        },
        {
            "params": [gaussians._opacity],
            "lr": opt_args.opacity_lr,
            "name": "opacity",
        },
        {
            "params": [gaussians._scaling],
            "lr": opt_args.scaling_lr * args.lr_scale_pos_and_scale * lr_scale,
            "name": "scaling",
        },
        {
            "params": [gaussians._rotation],
            "lr": opt_args.rotation_lr,
            "name": "rotation",
        },
    ]

    if gaussians._features_dc is not None:
        param_groups.append({
            "params": [gaussians._features_dc],
            "lr": opt_args.feature_lr,
            "name": "features_dc",
        })
    if gaussians._features_rest is not None:
        param_groups.append({
            "params": [gaussians._features_rest],
            "lr": opt_args.feature_lr / 20.0,
            "name": "features_rest",
        })

    optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

    from utils.general_utils import get_expon_lr_func
    xyz_scheduler_args = get_expon_lr_func(
        lr_init=opt_args.position_lr_init * gaussians.spatial_lr_scale * lr_scale * args.lr_scale_pos_and_scale * args.position_lr_mult,
        lr_final=opt_args.position_lr_final * gaussians.spatial_lr_scale * lr_scale * args.lr_scale_pos_and_scale * args.position_lr_mult,
        lr_delay_mult=opt_args.position_lr_delay_mult,
        max_steps=opt_args.position_lr_max_steps,
    )

    # Scale LR scheduling
    exp_scale_lr = getattr(args, "exp_scale_lr", False)
    scale_lr_init = getattr(args, "scale_lr_init", 0.020)
    scale_lr_final = getattr(args, "scale_lr_final", 0.005)
    cosine_lr = getattr(args, "cosine_lr", False)
    warmup_steps = getattr(args, "warmup_steps", 500)
    adaptive_ssim = getattr(args, "adaptive_ssim", False)
    grad_clip_val = getattr(args, "grad_clip", 0.0)
    opacity_l1_weight = getattr(args, "opacity_l1_weight", 0.0)
    total_iters = opt_args.iterations

    scale_lr_tau = total_iters / 3.0
    if exp_scale_lr:
        print_rank_0("Exp scale LR: init=%.4f, final=%.4f, tau=%.1f" % (scale_lr_init, scale_lr_final, scale_lr_tau))

    if cosine_lr:
        print_rank_0("Cosine LR schedule with %d warmup steps" % warmup_steps)

    background = None
    bg_color = [1, 1, 1] if dataset_args.white_background else None
    if bg_color is not None:
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_cameras_info = scene.getTrainCamerasInfo()
    n_train = len(train_cameras_info) if train_cameras_info else 0
    print_rank_0("Train cameras: %d" % n_train)

    if not getattr(args, "disable_auto_densification", False):
        gaussians.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        gaussians.denom = torch.zeros((N, 1), device="cuda")
    else:
        gaussians.xyz_gradient_accum = torch.empty((0, 1), device="cuda")
        gaussians.denom = torch.empty((0, 1), device="cuda")

    check_initial_gpu_memory_usage("after training setup")

    progress_bar = tqdm(range(1, opt_args.iterations + 1), desc="Training progress")
    ema_loss_for_log = 0.0
    iteration_times = []

    for iteration in range(1, opt_args.iterations + 1):
        iter_start = time.time()
        set_cur_iter(iteration)

        for param_group in optimizer.param_groups:
            if param_group["name"] == "xyz":
                if cosine_lr:
                    # Cosine annealing with warmup
                    if iteration <= warmup_steps:
                        progress = iteration / warmup_steps
                    else:
                        progress = (iteration - warmup_steps) / max(1, total_iters - warmup_steps)
                        progress = 0.5 * (1.0 + math.cos(math.pi * progress))
                    param_group["lr"] = xyz_scheduler_args(0) * progress
                else:
                    param_group["lr"] = xyz_scheduler_args(iteration - 1)
            elif param_group["name"] == "scaling":
                if exp_scale_lr:
                    progress = (iteration - 1) / max(1, total_iters - 1)
                    param_group["lr"] = scale_lr_final + (scale_lr_init - scale_lr_final) * math.exp(-progress * total_iters / scale_lr_tau)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick camera (sequential / trajectory ordering)
        cam_idx = iteration % n_train
        cam_info = train_cameras_info[cam_idx]
        viewpoint_cam = loadCam_raw_from_disk(args, cam_idx, cam_info, to_gpu=True)
        viewpoint_cam.original_image = viewpoint_cam.original_image_backup.cuda()
        # Camera transforms are on CPU (offload=True), move to GPU
        viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
        viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()

        filters, _, _ = calculate_filters(
            [viewpoint_cam],
            gaussians.get_xyz,
            gaussians.get_opacity,
            gaussians.get_scaling,
            gaussians.get_rotation,
        )
        visible_mask = filters[0]

        if visible_mask.numel() == 0:
            continue

        filtered_xyz = gaussians.get_xyz[visible_mask]
        filtered_opacity = gaussians.get_opacity[visible_mask]
        filtered_scaling = gaussians.get_scaling[visible_mask]
        filtered_rotation = gaussians.get_rotation[visible_mask]

        features = gaussians.get_features
        if not features.is_cuda:
            features = features.cuda()
        filtered_shs = features[visible_mask]

        rendered_image, viewspace_point_tensor, _ = pipeline_forward_one_step(
            filtered_opacity,
            filtered_scaling,
            filtered_rotation,
            filtered_xyz,
            filtered_shs,
            viewpoint_cam,
            scene,
            gaussians,
            background,
            pipe_args,
            eval=False,
        )

        gt_image = viewpoint_cam.original_image.float().contiguous()

        # Adaptive SSIM weight
        if adaptive_ssim:
            lambda_ssim = 0.1 + 0.2 * min(1.0, (iteration - 1) / max(1, 0.7 * total_iters))
        else:
            lambda_ssim = 0.2

        # Compute L1 loss
        Ll1 = torch.abs(rendered_image - torch.clamp(gt_image / 255.0, 0.0, 1.0)).mean()
        # Compute SSIM loss
        ssim_loss = fused_ssim(rendered_image.unsqueeze(0), torch.clamp(gt_image / 255.0, 0.0, 1.0).unsqueeze(0))
        loss = (1.0 - lambda_ssim) * Ll1 + lambda_ssim * (1.0 - ssim_loss)

        # Opacity L1 regularization
        if opacity_l1_weight > 0:
            opacity_l1 = gaussians.get_opacity.abs().mean()
            loss = loss + opacity_l1_weight * opacity_l1

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(gaussians._xyz, grad_clip_val)
            torch.nn.utils.clip_grad_norm_(gaussians._scaling, grad_clip_val)

        optimizer.step()

        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": "%.6f" % ema_loss_for_log})
        progress_bar.update(1)

        # Evaluation at test iterations
        if iteration in args.test_iterations:
            msg = "\n[EVAL] Running evaluation at iteration %d..." % iteration
            print_rank_0(msg)
            eval_start = time.time()
            metrics = evaluate_scene(args, gaussians, scene, background, pipe_args, log_file)
            eval_time = time.time() - eval_start

            msg = (
                "[EVAL] Iteration %d: PSNR=%.4f, SSIM=%.4f, eval_time=%.1fs"
                % (iteration, metrics["PSNR"], metrics["SSIM"], eval_time)
            )
            print_rank_0(msg)
            log_file.write(msg + "\n")
            log_file.flush()

            metrics_path = os.path.join(args.model_path, "eval_iter_%d.json" % iteration)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

    # Final evaluation
    print_rank_0("\n[FINAL] Computing final metrics...")
    final_metrics = evaluate_scene(args, gaussians, scene, background, pipe_args, log_file)

    tail_n = min(100, len(iteration_times))
    if iteration_times:
        avg_iter_ms = float(np.mean(iteration_times[-tail_n:]) * 1000.0)
        img_per_s = float(1.0 / np.mean(iteration_times[-tail_n:]))
    else:
        avg_iter_ms = 0.0
        img_per_s = 0.0

    final_output = {
        "PSNR": final_metrics["PSNR"],
        "SSIM": final_metrics["SSIM"],
        "LPIPS": 0.0,
        "Iter (ms)": avg_iter_ms,
        "Img/s": img_per_s,
        "GPU Util. (%)": 0.0,
    }

    final_path = os.path.join(args.model_path, "final_metrics.json")
    with open(final_path, "w") as f:
        json.dump(final_output, f, indent=2)

    msg = (
        "\n[FINAL] Results: PSNR=%.4f, SSIM=%.4f, Iter (ms)=%.1f, Img/s=%.2f"
        % (final_output["PSNR"], final_output["SSIM"],
           final_output["Iter (ms)"], final_output["Img/s"])
    )
    print_rank_0(msg)

    return final_output


def main():
    parser = ArgumentParser(description="TideGS Mip-NeRF 360 Training (GPU-only)")
    ap = AuxiliaryParams(parser)
    mp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    bp = BenchmarkParams(parser)
    dp = DebugParams(parser)

    # SOTA optimization flags
    parser.add_argument("--exp_scale_lr", action="store_true",
                        help="Enable exponential scale LR schedule (ImprovedGS+)")
    parser.add_argument("--scale_lr_init", type=float, default=0.020,
                        help="Initial scale LR for exponential schedule")
    parser.add_argument("--scale_lr_final", type=float, default=0.002,
                        help="Final scale LR for exponential schedule")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="Gradient max norm clipping (0=disabled)")
    parser.add_argument("--adaptive_ssim", action="store_true",
                        help="Enable adaptive SSIM weight schedule (0.1->0.3)")
    parser.add_argument("--cosine_lr", action="store_true",
                        help="Use cosine LR schedule instead of exponential")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="LR warmup steps for cosine schedule")
    parser.add_argument("--opacity_l1_weight", type=float, default=0.0,
                        help="L1 regularization weight on opacity (0=disabled)")
    parser.add_argument("--position_lr_mult", type=float, default=1.0,
                        help="Multiplier for position learning rate")

    args = parser.parse_args()

    # Force no-offload mode
    args.no_offload = True
    args.clm_offload = False
    args.naive_offload = False
    args.use_ssd_offload = False
    args.pure_ssd_offload = False

    init_args(args)
    safe_state(args.quiet)

    os.makedirs(args.model_path, exist_ok=True)
    log_file = open(os.path.join(args.model_path, "python.log"), "w", buffering=1)
    set_log_file(log_file)

    print_rank_0("TideGS Mip-NeRF 360 (GPU-only) Training")
    print_rank_0("  Source: %s" % args.source_path)
    print_rank_0("  Output: %s" % args.model_path)
    print_rank_0("  Iterations: %d" % args.iterations)
    print_rank_0("  Batch size: %d" % args.bsz)
    print_rank_0("  Eval: %s" % str(args.eval))
    print_rank_0("  Test iterations: %s" % str(args.test_iterations))
    print_rank_0("  Disable densification: %s" % str(args.disable_auto_densification))

    log_file.write("TideGS Mip-NeRF 360 (GPU-only) Training\n")
    log_file.write("  Source: %s\n" % args.source_path)
    log_file.write("  Output: %s\n" % args.model_path)
    log_file.flush()

    final_output = training(
        mp.extract(args),
        op.extract(args),
        pp.extract(args),
        args,
        log_file,
    )
    log_file.close()


if __name__ == "__main__":
    main()
