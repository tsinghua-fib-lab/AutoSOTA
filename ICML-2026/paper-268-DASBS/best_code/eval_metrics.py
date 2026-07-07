import sys
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from noise import get_noise
from model import get_model, EMA
from energy.ising import ising2d_mag, ising2d_2pt_corr, ising2d_ham
from utils import compute_energy_w2_distance
from sampling import batched_sample_tau_leaping
import os

# Load args from checkpoint
ckpt_path = "/repo/outputs/ising_L24_beta0.28/ckpt5.pth"
ckpt = torch.load(ckpt_path, weights_only=False, map_location="cuda:0")
args = ckpt["args"]

# Set device
args.device = "cuda:0"

# Load ground truth samples
gt_samples = torch.from_numpy(np.load("/repo/gt_samples_ising_L24_beta0.28.npy")).to(
    dtype=torch.long, device=args.device)

# Convert gt samples to {-1, 1} for ising functions
gt_samples_spin = gt_samples * 2 - 1  # {0,1} -> {-1,1}

# Compute ground truth statistics
gt_abs_mag = ising2d_mag(gt_samples_spin).abs().mean().item()
print(f"Ground truth ⟨|m|⟩: {gt_abs_mag:.6f}")

# Ground truth 2-point correlation
L = 24
gt_corrs = {}
for r in range(-L//2, L//2 + 1):
    gt_corrs[r] = ising2d_2pt_corr(gt_samples_spin, r)
print("Ground truth 2pt correlations:", {k: f"{v:.6f}" for k, v in gt_corrs.items()})

# Load model
noise = get_noise(args)
controller = get_model(args, require_time=True).to(args.device)
corrector = get_model(args, require_time=False).to(args.device)

controller.load_state_dict(ckpt["controller_state_dict"])
corrector.load_state_dict(ckpt["corrector_state_dict"])

# Load EMA
ema_controller = EMA(controller.parameters(), decay=args.ema.decay)
ema_corrector = EMA(corrector.parameters(), decay=args.ema.decay)
if "ema_controller_state_dict" in ckpt:
    ema_controller.load_state_dict(ckpt["ema_controller_state_dict"])
    ema_corrector.load_state_dict(ckpt["ema_corrector_state_dict"])
    ema_controller.copy_to(controller.parameters())
    ema_corrector.copy_to(corrector.parameters())
    print("Loaded EMA weights")

controller.eval()
corrector.eval()

# Generate samples (large batch for good statistics)
num_samples = 10000
batch_size = 512
sampling_steps = 200  # Same as training

print(f"\nGenerating {num_samples} samples with {sampling_steps} steps...")
all_samples = []
rounds = (num_samples + batch_size - 1) // batch_size
for r in range(rounds):
    actual_bs = min(batch_size, num_samples - r * batch_size)
    [x0, x1], info = batched_sample_tau_leaping(
        rounds=1, batch_size=actual_bs, steps=sampling_steps, args=args,
        controller=controller, noise=noise, cond=None,
    )
    all_samples.append(x1.cpu())

all_samples = torch.cat(all_samples, dim=0)[:num_samples]
print(f"Generated {all_samples.shape[0]} samples")

# Convert to {-1, 1}
model_samples_spin = all_samples * 2 - 1

# Compute model metrics
model_mag = ising2d_mag(model_samples_spin)
model_abs_mag = model_mag.abs().mean().item()
print(f"Model ⟨|m|⟩: {model_abs_mag:.6f}")

# ∆Mag
delta_mag = abs(model_abs_mag - gt_abs_mag)
print(f"\n*** ∆Mag = {delta_mag:.6f} ***")

# ∆Corr
model_corrs = {}
delta_corr_vals = []
for r in range(1, L//2 + 1):  # Only positive distances (symmetric)
    mc = ising2d_2pt_corr(model_samples_spin, r)
    model_corrs[r] = mc
    delta = abs(mc - gt_corrs[r])
    delta_corr_vals.append(delta)
    print(f"  r={r}: model={mc:.6f}, gt={gt_corrs[r]:.6f}, delta={delta:.6f}")

delta_corr = np.mean(delta_corr_vals)
print(f"\n*** ∆Corr = {delta_corr:.6f} ***")

# EW2 - Energy Wasserstein-2 distance
ew2 = compute_energy_w2_distance(
    model_samples_spin, gt_samples_spin,
    energy_fn=lambda samp: ising2d_ham(samp, J=args.J, h=args.h)
)
print(f"\n*** EW2 = {ew2:.6f} ***")

# Summary
print(f"\n{=*50}")
print(f"FINAL METRICS:")
print(f"  ∆Mag  = {delta_mag:.6f}")
print(f"  ∆Corr = {delta_corr:.6f}")
print(f"  EW2   = {ew2:.6f}")
print(f"  Paper targets: ∆Mag=0.015, ∆Corr=0.0023, EW2=5.4")
print(f"{=*50}")
