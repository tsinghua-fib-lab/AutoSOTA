#!/usr/bin/env python3
"""Evaluation script for DASBS trained model on 2D Ising model.

Loads a checkpoint, generates samples, and computes ∆Mag, ∆Corr, EW2
against ground truth MCMC reference samples.

Usage:
    python eval.py --ckpt outputs/ising_L24_beta0.28/ckpt5.pth \
                   --gt_samples gt_samples_ising_L24_beta0.28.npy \
                   --n_samples 10000 --batch_size 512 --steps 200
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Register eval resolver required by config
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

from noise import get_noise
from model import get_model, EMA
from energy.ising import ising2d_mag, ising2d_2pt_corr, ising2d_ham
from utils import compute_energy_w2_distance
from sampling import batched_sample_tau_leaping


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DASBS model on 2D Ising")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--gt_samples", type=str, required=True, help="Path to ground truth .npy samples")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for sampling")
    parser.add_argument("--steps", type=int, default=200, help="Number of tau-leaping steps")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    return parser.parse_args()


def main():
    opt = parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {opt.ckpt}")
    ckpt = torch.load(opt.ckpt, weights_only=False, map_location=opt.device)
    args = ckpt["args"]
    OmegaConf.set_struct(args, False)
    args.device = opt.device

    print(f"Config: L={args.L}, beta={args.beta}, model={args.model.name}")
    print(f"  loss.controller={args.loss.controller}, loss.corrector={args.loss.corrector}")
    print(f"  noise={args.noise.type}, alpha={args.noise.alpha}, gamma={args.noise.gamma}")

    # Load ground truth
    print(f"Loading ground truth: {opt.gt_samples}")
    gt_samples = torch.from_numpy(np.load(opt.gt_samples)).to(
        dtype=torch.long, device=args.device
    )
    gt_spin = gt_samples * 2 - 1  # {0,1} -> {-1,1}
    L = args.L

    gt_abs_mag = ising2d_mag(gt_spin).abs().mean().item()
    gt_corrs = {r: ising2d_2pt_corr(gt_spin, r) for r in range(1, L // 2 + 1)}

    print(f"Ground truth |m| = {gt_abs_mag:.6f}")

    # Build model and load weights (use EMA if available)
    noise = get_noise(args)
    controller = get_model(args, require_time=True).to(args.device)
    corrector = get_model(args, require_time=False).to(args.device)

    controller.load_state_dict(ckpt["controller_state_dict"])
    corrector.load_state_dict(ckpt["corrector_state_dict"])

    if "ema_controller_state_dict" in ckpt:
        ema_c = EMA(controller.parameters(), decay=args.ema.decay)
        ema_cr = EMA(corrector.parameters(), decay=args.ema.decay)
        ema_c.load_state_dict(ckpt["ema_controller_state_dict"])
        ema_cr.load_state_dict(ckpt["ema_corrector_state_dict"])
        ema_c.copy_to(controller.parameters())
        ema_cr.copy_to(corrector.parameters())
        print("Applied EMA weights")

    controller.eval()
    corrector.eval()

    # Generate samples
    print(f"Generating {opt.n_samples} samples with {opt.steps} steps...")
    all_samples = []
    for r in range((opt.n_samples + opt.batch_size - 1) // opt.batch_size):
        bs = min(opt.batch_size, opt.n_samples - r * opt.batch_size)
        samples, _ = batched_sample_tau_leaping(
            rounds=1,
            batch_size=bs,
            steps=opt.steps,
            args=args,
            controller=controller,
            noise=noise,
            cond=None,
        )
        all_samples.append(samples[1].cpu())

    all_samples = torch.cat(all_samples, dim=0)[: opt.n_samples]
    model_spin = all_samples * 2 - 1  # {0,1} -> {-1,1}
    print(f"Generated {all_samples.shape[0]} samples")

    # Compute ∆Mag
    model_abs_mag = ising2d_mag(model_spin).abs().mean().item()
    delta_mag = abs(model_abs_mag - gt_abs_mag)
    print(f"Model |m| = {model_abs_mag:.6f}, ∆Mag = {delta_mag:.6f}")

    # Compute ∆Corr
    delta_corr_vals = []
    for r in range(1, L // 2 + 1):
        mc = ising2d_2pt_corr(model_spin, r)
        delta_corr_vals.append(abs(mc - gt_corrs[r]))
    delta_corr = np.mean(delta_corr_vals)
    print(f"∆Corr = {delta_corr:.6f}")

    # Compute EW2
    ew2 = compute_energy_w2_distance(
        model_spin, gt_spin, energy_fn=lambda s: ising2d_ham(s, J=args.J, h=args.h)
    )
    print(f"EW2 = {ew2:.4f}")

    # Final output
    print("=" * 50)
    print("METRICS:")
    print(f"  ∆Mag  = {delta_mag:.6f}")
    print(f"  ∆Corr = {delta_corr:.6f}")
    print(f"  EW2   = {ew2:.4f}")
    print("=" * 50)

    return {
        "delta_mag": delta_mag,
        "delta_corr": delta_corr,
        "ew2": ew2,
        "model_abs_mag": model_abs_mag,
        "gt_abs_mag": gt_abs_mag,
    }


if __name__ == "__main__":
    main()
