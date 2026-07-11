#!/usr/bin/env python3
"""Evaluation script for CycMit-MRI reproduction.

Runs the full attack + mitigation pipeline on the sample data and prints
final PSNR and SSIM metrics to stdout.

Usage:
    cd /repo && python3 eval.py
"""

import os
import sys
import yaml
import torch
import numpy as np
import scipy.io as sio

from src.Utils import (
    seed_everything, IFFT, fft_loss,
    getpsnr, getssim, attack_generation,
    CG_EEH, Cyclic_Mitigation, noise_jiterring
)
from src.Unrolled_Network import UnrolledNet


def main():
    seed_everything(42)

    with open("Config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create output directories (required by Cyclic_Mitigation)
    save_path = config["Saving"]["path"]
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/iterations", exist_ok=True)
    os.makedirs(f"{save_path}/plots", exist_ok=True)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    ksp = sio.loadmat("./data/kneePD.mat")["kspace"].transpose([0, 2, 1])
    coil = sio.loadmat("./data/kneePD.mat")["coils"].transpose([0, 2, 1])

    Omega_Mask = torch.tensor(
        sio.loadmat("./data/omega_mask.mat")["mask"].transpose([1, 0])
    ).unsqueeze(0).to(device)

    D1 = torch.tensor(
        sio.loadmat("./data/delta1.mat")["mask"].transpose([1, 0]),
        dtype=torch.complex64
    ).unsqueeze(0).to(device)
    D2 = torch.tensor(
        sio.loadmat("./data/delta2.mat")["mask"].transpose([1, 0]),
        dtype=torch.complex64
    ).unsqueeze(0).to(device)
    D3 = torch.tensor(
        sio.loadmat("./data/delta3.mat")["mask"].transpose([1, 0]),
        dtype=torch.complex64
    ).unsqueeze(0).to(device)
    mask_list = [D1, D2, D3]

    # Load model
    network = UnrolledNet(mu=0.1813, Unrolls=10)
    network.load_state_dict(torch.load("./BestModel/checkpoint.pth"))
    network.to(device)
    network.eval()

    cg_EEH = CG_EEH().to(device)

    # Process sample
    ksp_t = torch.tensor(ksp).unsqueeze(0).to(device)
    coil_t = torch.tensor(coil).unsqueeze(0).to(device)

    zero_filled = IFFT(ksp_t * Omega_Mask)
    label = IFFT(ksp_t)
    zero_filled = torch.sum(zero_filled * torch.conj(coil_t), axis=1, keepdims=True)
    label = torch.sum(label * torch.conj(coil_t), axis=1, keepdims=True)
    scale = torch.max(torch.abs(zero_filled))
    zero_filled = zero_filled / scale
    label = label / scale
    zero_filled.requires_grad = True

    # Noise jittering
    with torch.no_grad():
        us_ksp = cg_EEH(
            fft_loss(zero_filled, coil_t, config["axis"]) * Omega_Mask,
            coil_t, Omega_Mask
        )
        recon_0 = network(zero_filled, coil_t, Omega_Mask)
        Ex1 = fft_loss(recon_0, coil_t, config["axis"]) * Omega_Mask
        jitt = noise_jiterring(
            us_ksp, Ex1, Omega_Mask,
            std_scale=config["Mitigation"]["noise_jittering_std"]
        ).to(device)

    # Attack generation
    zf_p = attack_generation(zero_filled, network, device, Omega_Mask, coil_t, config)

    with torch.no_grad():
        recon_p = network(zf_p, coil_t, Omega_Mask)

    # Attacked metrics
    thresh = config["Saving"]["Threshold"]
    label_np = np.abs(label.squeeze().cpu().numpy().transpose([1, 0]))
    attacked_np = np.abs(recon_p.squeeze().cpu().numpy().transpose([1, 0]))
    mask = label_np >= thresh
    attacked_psnr = getpsnr(attacked_np[mask], label_np[mask]) if mask.sum() > 0 else getpsnr(attacked_np, label_np)
    attacked_ssim = getssim(attacked_np, label_np)

    # Cyclic Mitigation
    PSNR, SSIM, LOSS = Cyclic_Mitigation(
        config, network, zf_p, label.squeeze(), coil_t,
        Omega_Mask, mask_list, device, jitt
    )

    best_psnr = max(PSNR)
    best_ssim = max(SSIM)

    # Final output
    print(f"\n{'='*60}")
    print(f"CycMit-MRI Reproduction Results")
    print(f"{'='*60}")
    print(f"Model:     MoDL (10 unrolls, ResNet, 15 coils)")
    print(f"Data:      Cor-PD knee (R=4, equispaced, 24 ACS)")
    print(f"Attack:    PGD, eps=0.01, 10 iters, L-inf, image")
    print(f"Mitigate:  100 iters, Linear alpha schedule")
    print(f"{'='*60}")
    print(f"Attacked PSNR:  {attacked_psnr:.3f} dB")
    print(f"Attacked SSIM:  {attacked_ssim:.4f}")
    print(f"Mitigated PSNR: {best_psnr:.3f} dB")
    print(f"Mitigated SSIM: {best_ssim:.4f}")
    print(f"{'='*60}")
    print(f"Paper: Proposed+MoDL PSNR=35.14, SSIM=0.92")
    print(f"Paper: AT baseline PSNR=33.99")

    # Machine-parseable output
    print(f"\nMETRICS:PSNR={best_psnr:.3f},SSIM={best_ssim:.4f}")
    return best_psnr, best_ssim


if __name__ == "__main__":
    main()
