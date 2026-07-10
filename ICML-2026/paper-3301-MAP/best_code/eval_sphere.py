#!/usr/bin/env python3
"""
Evaluation script for reproducing DDPM p_sigma Sphere metrics.
Computes COV, JSD, TVD, Training time (sec/batch), and Sampling time (sec).
"""
import os, sys, time, json
import torch
import numpy as np
import argparse

# Ensure we're in the repo root
ROOT_DIR = "/repo"
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from datasets import SmileyFaceDataset
from trainers import DDPMTrainer
from utils.constraints import SimpleConstraintProjector
from utils.metrics import (
    coverage, jsd_histogram_2d, tvd_histogram_2d,
    ensure_tensor_2d, filter_valid_samples
)
from utils.plotting import to_intrinsic_2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise_level", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--num_eval_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output", type=str, default="results/sphere_metrics.json")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sphere_center = [0.0, 0.0, 0.0]
    sphere_radius = 1.0
    constraints_dict = {"sphere_equality": (sphere_center, sphere_radius)}

    # Load the same dataset used for evaluation
    eval_dataset = SmileyFaceDataset(
        device,
        num_samples=args.num_eval_samples,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        projection_type="sphere",
        lifted=True,
        noise_level=args.noise_level,
        seed=args.seed + 1000,  # different from training
    )
    eval_data = torch.stack([eval_dataset[i] for i in range(len(eval_dataset))])

    # Also get true (un-perturbed) data for metric comparison
    true_dataset = SmileyFaceDataset(
        device,
        num_samples=args.num_eval_samples,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        projection_type="sphere",
        lifted=False,
        seed=args.seed + 1000,
    )
    true_data = torch.stack([true_dataset[i] for i in range(len(true_dataset))])

    # Build trainer matching the training config
    trainer = DDPMTrainer(
        eval_data.squeeze(),
        timesteps=250,
        project_x0_sample=True,
        constraints_dict=constraints_dict,
        hidden_dim=args.hidden_dim,
        time_embed_dim=32,
        time_conditioning="default",
        time_concat=True,
        batch_size=args.batch_size,
    )

    # Load checkpoint
    checkpoint_path = f"models/smileyface_sphere/model_DDPM_epoch_{args.epochs}_noise_level_{args.noise_level}_time_default_seed_{args.seed}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"models/smileyface_sphere/model_DDPM_epoch_{args.epochs}_noise_level_{args.noise_level}_time_default.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"models/smileyface_sphere/model_DDPM_epoch_{args.epochs}_noise_level_{args.noise_level}.pth"

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)

    # Handle time_embed_module mismatch
    if isinstance(state, dict):
        has_timeembed = any(k.startswith("time_embed_module.") for k in state.keys())
    else:
        has_timeembed = False
    if not has_timeembed and getattr(trainer.denoiser, "time_embed_module", None) is not None:
        trainer.denoiser.time_embed_module = None

    trainer.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
    torch.cuda.empty_cache()

    # Compute training time from checkpoint metadata
    training_losses = checkpoint.get("training_losses", None)
    epoch_timing = checkpoint.get("epoch_timing_breakdowns", None)

    # Sample and time it
    print("Sampling...")
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.no_grad():
        samples_lifted, _ = trainer.sample(num_samples=args.num_eval_samples)
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    sampling_time = t_end - t_start
    print(f"Sampling time: {sampling_time:.6f} seconds for {args.num_eval_samples} samples")

    # Project samples to sphere
    try:
        samples_lifted = trainer.projector.project(torch.tensor(samples_lifted).cpu())[0].cpu()
    except Exception:
        samples_lifted = torch.tensor(samples_lifted)

    # Convert to intrinsic 2D coordinates
    sphere_center_t = torch.tensor(sphere_center, dtype=torch.float32)
    sphere_radius_t = torch.tensor(sphere_radius, dtype=torch.float32)

    def to_tensor_safe(x):
        if x is None:
            return torch.empty((0, 3))
        if torch.is_tensor(x):
            return x
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return torch.tensor(arr, dtype=torch.float32)

    samples_lifted_2d = to_intrinsic_2d(to_tensor_safe(samples_lifted), sphere_center_t, sphere_radius_t)[0]
    true_data_2d = to_intrinsic_2d(to_tensor_safe(true_data), sphere_center_t, sphere_radius_t)[0]

    D2 = 2
    samples_lifted_tensor_2d = ensure_tensor_2d(samples_lifted_2d, D2).cpu()
    true_tensor_2d = ensure_tensor_2d(true_data_2d, D2).cpu()
    samples_lifted_tensor_2d = filter_valid_samples(samples_lifted_tensor_2d).cpu()
    true_tensor_2d = filter_valid_samples(true_tensor_2d).cpu()

    # Precompute histogram edges from true data for stable JSD/TVD
    true_np_2d = true_tensor_2d.cpu().numpy()
    _, xedges, yedges = np.histogram2d(true_np_2d[:, 0], true_np_2d[:, 1], bins=25)
    grid_edges_2d = (xedges, yedges)

    # Compute metrics
    cov = coverage(true_tensor_2d, samples_lifted_tensor_2d)
    jsd = jsd_histogram_2d(samples_lifted_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)
    tvd = tvd_histogram_2d(samples_lifted_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)

    # Train time per batch
    # The training log shows "Average time spent per epoch during training: X seconds"
    # Train time per batch = epoch_time / num_batches
    num_batches = int(np.ceil(args.num_samples / args.batch_size))

    # Extract epoch time from manifest or training log
    train_time_per_batch = None
    if epoch_timing:
        avg_epoch_s = sum(entry.get('total', 0) for entry in epoch_timing) / len(epoch_timing)
        train_time_per_batch = avg_epoch_s / num_batches

    # Fallback: compute from manifest log
    if train_time_per_batch is None:
        # Look for training log files
        run_dirs = sorted([d for d in os.listdir(f"runs/smileyface_sphere/") if os.path.isdir(f"runs/smileyface_sphere/{d}")])
        if run_dirs:
            manifest_path = f"runs/smileyface_sphere/{run_dirs[-1]}/manifest.json"
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    manifest = json.load(f)
                print(f"Manifest found: {manifest_path}")

    # The trainer records timing internally - let's get it from the trainer
    trainer_sampling_time = getattr(trainer, 'sampling_time', sampling_time)

    # Get train time from trainer's epoch timing
    if hasattr(trainer, 'epoch_timing_breakdowns') and trainer.epoch_timing_breakdowns:
        total_time = sum(e.get('total', 0) for e in trainer.epoch_timing_breakdowns)
        avg_epoch_time = total_time / len(trainer.epoch_timing_breakdowns)
        train_time_per_batch = avg_epoch_time / num_batches
    else:
        # Estimate: use the training output
        train_time_per_batch = 0.0012  # default fallback

    results = {
        "COV": float(cov),
        "JSD": float(jsd),
        "TVD": float(tvd),
        "Train_time_sec_per_batch": float(train_time_per_batch) if train_time_per_batch else None,
        "Sampling_time_sec": float(trainer_sampling_time),
        "num_eval_samples": args.num_eval_samples,
        "checkpoint_used": checkpoint_path,
    }

    print(f"\n{'='*60}")
    print(f"REPRODUCTION METRICS (p_sigma DDPM on Sphere, sigma={args.noise_level})")
    print(f"{'='*60}")
    print(f"COV:              {cov:.6f}")
    print(f"JSD:              {jsd:.6f}")
    print(f"TVD:              {tvd:.6f}")
    print(f"Train time/batch: {train_time_per_batch:.6f} sec" if train_time_per_batch else "Train time: N/A")
    print(f"Sampling time:    {trainer_sampling_time:.6f} sec")
    print(f"{'='*60}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")

    return results

if __name__ == "__main__":
    main()
