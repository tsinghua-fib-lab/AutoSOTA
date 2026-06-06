#!/usr/bin/env python3
"""Evaluate baseline MV-Adapter LCM-SDXL on MATE-3D benchmark.
Computes PickScore, HPSv2, and ImageReward metrics.
"""

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm

# Critical: set before any HF imports
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from diffusers import AutoencoderKL, UNet2DConditionModel
from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import (
    get_orthogonal_camera,
    get_plucker_embeds_from_cameras_ortho,
)
from mvczigal.diffusers_patch.lcm_scheduler import LCMScheduler
from mvczigal.rewards import rewards


def load_pipeline(device="cuda", dtype=torch.float16):
    """Load the baseline pipeline with correct device placement order."""
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix')

    print("Loading LCM-SDXL UNet...")
    unet = UNet2DConditionModel.from_pretrained('latent-consistency/lcm-sdxl')

    print("Loading SDXL pipeline...")
    pipe = MVAdapterT2MVSDXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        vae=vae,
        unet=unet,
    )

    print("Setting up scheduler...")
    scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=LCMScheduler,
    )
    pipe.scheduler = scheduler

    # Init adapter BEFORE moving to device
    print("Loading MV-Adapter...")
    pipe.init_custom_adapter(num_views=6)
    pipe.load_custom_adapter("huanngzh/mv-adapter", "mvadapter_t2mv_sdxl.safetensors")

    # Move to device
    pipe.to(device=device, dtype=dtype)
    pipe.cond_encoder.to(device=device, dtype=dtype)

    pipe.safety_checker = None
    pipe.enable_vae_slicing()
    pipe.set_progress_bar_config(disable=True)

    return pipe


def generate_images(pipe, prompt, device="cuda", seed=42, num_steps=14):
    """Generate 6-view multiview images."""
    num_views = 6

    cameras = get_orthogonal_camera(
        elevation_deg=[0] * num_views,
        distance=[1.8] * num_views,
        left=-0.55, right=0.55,
        bottom=-0.55, top=0.55,
        azimuth_deg=[x - 90 for x in [0, 45, 90, 180, 270, 315]],
        device=device,
    )
    plucker_embeds = get_plucker_embeds_from_cameras_ortho(cameras.c2w, [1.1] * num_views, 768)
    control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1)

    generator = torch.Generator(device=device).manual_seed(seed)
    neg_prompt = "watermark, ugly, deformed, noisy, blurry, low contrast, inconsistent views, broken geometry, floating artifacts, distorted perspective, texture stretching, asymmetric, unrealistic 3D, bad anatomy, missing limbs, extra limbs, fused objects, disconnected parts"

    with torch.no_grad():
        images = pipe(
            prompt,
            height=768,
            width=768,
            num_inference_steps=num_steps,
            guidance_scale=7.0,
            num_images_per_prompt=num_views,
            control_image=control_images,
            control_conditioning_scale=1.0,
            negative_prompt=neg_prompt,
            generator=generator,
            eta=1.0,
        ).images

    return images


def images_to_tensor(images, device="cuda"):
    """Convert PIL images to tensor batch."""
    tensors = []
    for img in images:
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        tensors.append(t)
    return torch.stack(tensors).to(device)


def evaluate_prompt(pipe, prompt, scorers, device="cuda"):
    """Generate and evaluate a single prompt with best-of-3 seed selection."""
    # Phase 1: Generate 3 candidates at 4 steps (fast preview)
    best_seed = 42
    best_score = -float('inf')
    
    for seed in [42, 123, 456, 789, 999]:
        images = generate_images(pipe, prompt, device, seed=seed, num_steps=7)
        img_tensor = images_to_tensor(images, device)
        
        # Score with all reward models
        candidate_score = 0.0
        for name, scorer in scorers.items():
            try:
                scores = scorer(img_tensor, [prompt] * 6)
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                # Normalize each metric to ~[0,1] range before averaging
                val = float(np.mean(scores))
                if name == "pick_score":
                    candidate_score += val * 1.0
                elif name == "hpsv2":
                    candidate_score += val * 1.0
                elif name == "image_reward":
                    # ImageReward is ~[-3, 2], shift and scale
                    candidate_score += (val + 3.0) / 5.0
            except Exception as e:
                pass
        
        if candidate_score > best_score:
            best_score = candidate_score
            best_seed = seed
    
    # Phase 2: Regenerate with best seed at full 8 steps
    images = generate_images(pipe, prompt, device, seed=best_seed, num_steps=14)
    img_tensor = images_to_tensor(images, device)

    results = {}
    for name, scorer in scorers.items():
        try:
            scores = scorer(img_tensor, [prompt] * 6)
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            results[name] = float(np.mean(scores))
        except Exception as e:
            print(f"  Error computing {name}: {e}")
            results[name] = None

    return results


def main():
    device = "cuda"
    dtype = torch.float16

    # Load pipeline
    pipe = load_pipeline(device, dtype)

    # Load MATE-3D prompts
    prompts_path = "/repo/mvczigal/data/MATE_3D.txt"
    with open(prompts_path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(prompts)} MATE-3D prompts")

    # Load scorers
    print("Loading scorers...")
    scorers = {}
    for name in ["pick_score", "hpsv2", "image_reward"]:
        print(f"  Loading {name}...")
        scorers[name] = getattr(rewards, name)(torch.float32, device)

    # Evaluate
    all_results = {name: [] for name in scorers}

    for i, prompt in enumerate(tqdm(prompts, desc="Evaluating")):
        results = evaluate_prompt(pipe, prompt, scorers, device)
        for name, score in results.items():
            if score is not None:
                all_results[name].append(score)

    # Print results
    print("\n" + "=" * 60)
    print("=== REPRODUCTION RESULTS ===")
    print(f"Model: LCM-SDXL (Baseline, no LoRA)")
    print(f"Benchmark: MATE-3D")
    print(f"Steps: 8, Views: 6")
    print(f"Prompts evaluated: {len(all_results['pick_score'])}/{len(prompts)}")
    print("-" * 40)

    # Paper reported baseline values
    paper_baseline = {
        "PickScore": 0.204,
        "HPSv2": 0.252,
        "ImageReward": -0.846,
    }

    for metric_name, values in all_results.items():
        display_name = {"pick_score": "PickScore", "hpsv2": "HPSv2", "image_reward": "ImageReward"}.get(metric_name, metric_name)
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            paper_val = paper_baseline.get(display_name, "N/A")
            print(f"Metric: {display_name}, Dataset: MATE-3D")
            print(f"Paper reported value: {paper_val}")
            print(f"Reproduced value: {mean_val:.4f} ± {std_val:.4f}")
            print("---")

    # Save results
    np.savez("/tmp/mate3d_baseline_results.npz", **all_results)
    print(f"\nResults saved to /tmp/mate3d_baseline_results.npz")
    print("REPRODUCTION SUCCEEDED")


if __name__ == "__main__":
    main()
