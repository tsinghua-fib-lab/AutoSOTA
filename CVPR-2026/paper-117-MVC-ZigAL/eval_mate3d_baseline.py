#!/usr/bin/env python3
"""Evaluate baseline (no LoRA) MV-Adapter LCM-SDXL on MATE-3D benchmark."""

import os
import sys
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from safetensors.torch import load_file

# Force HF mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import (
    get_orthogonal_camera,
    get_plucker_embeds_from_cameras_ortho,
    make_image_grid,
)
from mvczigal.diffusers_patch.lcm_scheduler import LCMScheduler
from mvczigal.rewards import rewards


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="/tmp/mate3d_results")
    parser.add_argument("--limit_prompts", type=int, default=0, help="0=all prompts")
    parser.add_argument("--metrics", type=str, default="pick_score,hpsv2,image_reward",
                       help="Comma-separated metrics")
    return parser.parse_args()


def load_pipeline(args):
    """Load the baseline MV-Adapter LCM-SDXL pipeline."""
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = args.device

    print("Loading models...")

    # Load VAE
    print("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=dtype,
    )

    # Load LCM-SDXL UNet
    print("  Loading LCM-SDXL UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        "latent-consistency/lcm-sdxl",
        torch_dtype=dtype,
    )

    # Load pipeline
    print("  Loading SDXL pipeline...")
    pipeline = MVAdapterT2MVSDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        vae=vae,
        unet=unet,
    )

    # Set up LCM scheduler
    scheduler = ShiftSNRScheduler.from_scheduler(
        pipeline.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=LCMScheduler,
    )
    pipeline.scheduler = scheduler

    # Move to device
    pipeline.vae.to(device, dtype=dtype)
    pipeline.unet.to(device, dtype=dtype)
    pipeline.text_encoder.to(device, dtype=dtype)
    pipeline.text_encoder_2.to(device, dtype=dtype)

    # Load MV-Adapter
    print("  Loading MV-Adapter...")
    pipeline.init_custom_adapter(num_views=args.num_views)
    pipeline.load_custom_adapter("huanngzh/mv-adapter", "mvadapter_t2mv_sdxl.safetensors")
    pipeline.cond_encoder.to(device, dtype=dtype)

    # Disable safety checker
    pipeline.safety_checker = None

    # Enable VAE slicing for memory
    pipeline.enable_vae_slicing()

    print("Pipeline loaded successfully!")
    return pipeline


def generate_multiview(pipeline, prompt, args):
    """Generate multiview images for a single prompt."""
    num_views = args.num_views
    device = args.device

    # Prepare cameras
    cameras = get_orthogonal_camera(
        elevation_deg=[0] * num_views,
        distance=[1.8] * num_views,
        left=-0.55, right=0.55,
        bottom=-0.55, top=0.55,
        azimuth_deg=[x - 90 for x in [0, 45, 90, 180, 270, 315]],
        device=device,
    )

    plucker_embeds = get_plucker_embeds_from_cameras_ortho(
        cameras.c2w, [1.1] * num_views, 768  # width = 768
    )
    control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    negative_prompt = "watermark, ugly, deformed, noisy, blurry, low contrast"

    images = pipeline(
        prompt,
        height=768,
        width=768,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=num_views,
        control_image=control_images,
        control_conditioning_scale=1.0,
        negative_prompt=negative_prompt,
        generator=generator,
        eta=1.0,
    ).images

    return images


def load_prompts(limit=0):
    """Load MATE-3D prompts."""
    repo_prompts_path = "/repo/mvczigal/data/MATE_3D.txt"
    local_prompts_path = os.path.join(os.path.dirname(__file__), "mvczigal", "data", "MATE_3D.txt")

    for path in [repo_prompts_path, local_prompts_path]:
        if os.path.exists(path):
            with open(path, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
            if limit > 0:
                prompts = prompts[:limit]
            return prompts

    raise FileNotFoundError("MATE_3D.txt not found")


def evaluate_metric(metric_name, images, prompts, dtype, device):
    """Evaluate a single metric."""
    scorer_fn = getattr(rewards, metric_name)(dtype, device)

    if metric_name == "hyper_score":
        scores = scorer_fn(images, prompts, len(prompts))
        # Returns list of [alignment, geometry, texture, overall] scores
        return scores
    else:
        scores = scorer_fn(images, prompts)
        return scores


def main():
    args = parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = args.device

    # Load pipeline
    pipeline = load_pipeline(args)

    # Load prompts
    prompts = load_prompts(args.limit_prompts)
    print(f"\nLoaded {len(prompts)} MATE-3D prompts")

    # Parse metrics
    metric_names = [m.strip() for m in args.metrics.split(",")]
    print(f"Metrics to evaluate: {metric_names}")

    # Load scorers
    scorers = {}
    for name in metric_names:
        print(f"  Loading {name} scorer...")
        scorers[name] = getattr(rewards, name)(torch.float32, device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate and evaluate
    all_results = {name: [] for name in metric_names}

    # For HyperScore, collect per-dimension results
    if "hyper_score" in metric_names:
        hyper_dims = ["alignment", "geometry", "texture", "overall"]
        for dim in hyper_dims:
            all_results[f"hyper_score_{dim}"] = []

    for i, prompt in enumerate(tqdm(prompts, desc="Evaluating prompts")):
        try:
            # Generate multiview images
            images = generate_multiview(pipeline, prompt, args)

            # Convert PIL images to tensors for scoring
            image_tensors = []
            for img in images:
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                image_tensors.append(img_tensor)
            image_batch = torch.stack(image_tensors).to(device)

            # Evaluate each metric
            for name in metric_names:
                if name == "hyper_score":
                    dim_scores = scorers[name](image_batch, [prompt] * args.num_views, args.num_views)
                    for j, dim in enumerate(hyper_dims):
                        all_results[f"hyper_score_{dim}"].append(dim_scores[j].cpu().item())
                    # Average across dimensions for "hyper_score"
                    avg_score = np.mean([s.cpu().item() for s in dim_scores])
                    all_results[name].append(avg_score)
                else:
                    scores = scorers[name](image_batch, [prompt] * args.num_views)
                    if isinstance(scores, torch.Tensor):
                        scores = scores.cpu().numpy()
                    # Average across views for single-view metrics
                    avg_score = float(np.mean(scores))
                    all_results[name].append(avg_score)

            # Save first few generated images
            if i < 3:
                grid = make_image_grid(images, rows=1)
                grid.save(os.path.join(args.output_dir, f"sample_{i:03d}.png"))

        except Exception as e:
            print(f"\nError on prompt '{prompt}': {e}")
            continue

    # Compute final results
    print("\n" + "=" * 60)
    print("=== REPRODUCTION RESULTS ===")
    for name in sorted(all_results.keys()):
        values = all_results[name]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"Metric: {name}, Mean: {mean_val:.4f}, Std: {std_val:.4f}, N={len(values)}")

    # Save raw results
    results_path = os.path.join(args.output_dir, "results.npz")
    np.savez(results_path, **all_results)
    print(f"\nRaw results saved to {results_path}")

    print("\nREPRODUCTION COMPLETED")


if __name__ == "__main__":
    main()
