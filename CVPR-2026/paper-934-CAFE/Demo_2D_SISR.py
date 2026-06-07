"""
2D Single Image Super-Resolution (SISR) with CAFE_Net
Adapted from the WIRE framework: https://github.com/vishwa91/wire
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from matplotlib.image import imsave
import matplotlib.pyplot as plt
from model import CAFE_Net
from utils import normalize, psnr

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_and_prepare_data(filepath, scale):
    """
    Load image, crop to be divisible by scale, and create LR/Bicubic versions.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found at: {filepath}")
        
    # Read and normalize original image
    im = plt.imread(filepath).astype(np.float32)
    # Handle RGBA images by keeping only RGB
    if im.shape[2] == 4:
        im = im[:, :, :3]
    im = normalize(im, True)
    
    H, W, _ = im.shape

    # Crop image dimensions to be perfectly divisible by the scale factor
    im = im[:scale * (H // scale), :scale * (W // scale), :]
    H, W, _ = im.shape


    im_lr = cv2.resize(im, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_AREA)
    H2, W2, _ = im_lr.shape

    # Generate Bicubic upsampled image (Baseline for comparison)
    im_bi = cv2.resize(im_lr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return im, im_lr, im_bi, (H, W), (H2, W2)

def train_sisr(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Prepare Data
    im_hr, im_lr, im_bi, (H, W), (H2, W2) = load_and_prepare_data(args.data_path, args.scale)
    print(f"Original HR shape: {H}x{W} | Downsampled LR shape: {H2}x{W2}")

    # 2. Build Model
    model = CAFE_Net(
        rff_mapping_size=args.rff_mapping_size,
        rff_scale=args.rff_scale,
        cheb_order=args.cheb_order,
        hidden_features=args.hidden_features,
        num_branches=args.num_branches,
        hidden_layers=args.hidden_layers,
    ).to(device)

    # 3. Setup Optimizer & Scheduler
    optim = torch.optim.Adam(lr=args.lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.niters, eta_min=1e-5)
    
    # Pooler to simulate downsampling process during training
    downsampler = nn.AvgPool2d(args.scale)

    # 4. Prepare Coordinate Tensors
    # LR Coordinates
    x_lr = torch.linspace(-1, 1, W2).to(device)
    y_lr = torch.linspace(-1, 1, H2).to(device)
    X_lr, Y_lr = torch.meshgrid(x_lr, y_lr, indexing='xy')
    coords_lr = torch.hstack((X_lr.reshape(-1, 1), Y_lr.reshape(-1, 1)))[None, ...]

    # HR Coordinates
    x_hr = torch.linspace(-1, 1, W).to(device)
    y_hr = torch.linspace(-1, 1, H).to(device)
    X_hr, Y_hr = torch.meshgrid(x_hr, y_hr, indexing='xy')
    coords_hr = torch.hstack((X_hr.reshape(-1, 1), Y_hr.reshape(-1, 1)))[None, ...]

    # Ground Truth Tensors (Fixed Reshape Logic)
    gt_hr_tensor = torch.tensor(im_hr).to(device).reshape(H * W, 3)[None, ...]
    gt_lr_tensor = torch.tensor(im_lr).to(device).reshape(1, H2 * W2, 3) 
    
    # 5. Training Loop
    best_mse = float('inf')
    best_img = None
    
    tbar = tqdm(range(args.niters), desc="Training SISR")
    
    for epoch in tbar:
        # Forward pass on HR coordinates
        rec_hr = model(coords_hr)
        
        # Reshape to (1, C, H, W) for downsampling
        rec_hr_img = rec_hr.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
        
        # Downsample prediction to match LR resolution
        rec_lr = downsampler(rec_hr_img)

        rec_lr_flat = rec_lr.reshape(1, 3, -1).permute(0, 2, 1)

        loss = ((gt_lr_tensor - rec_lr_flat) ** 2).mean()

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()

        # Evaluate against HR ground truth
        with torch.no_grad():
            mse_vs_hr = ((gt_hr_tensor - rec_hr) ** 2).mean().item()
            current_psnr = -10 * np.log10(mse_vs_hr)
            
            tbar.set_postfix({"PSNR (dB)": f"{current_psnr:.2f}"})

            if mse_vs_hr < best_mse:
                best_mse = mse_vs_hr
                # Save best reconstructed image format (H, W, C)
                best_img = rec_hr_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    # 6. Save Results
    os.makedirs(args.save_dir, exist_ok=True)
    final_psnr = -10 * np.log10(best_mse)
    print(f"\nTraining completed. Best PSNR: {final_psnr:.2f} dB")
    print(f"Bicubic Baseline PSNR: {psnr(im_hr, im_bi):.2f} dB")

    # Save images for comparison
    imsave(os.path.join(args.save_dir, "01_original_hr.png"), im_hr.clip(0, 1))
    imsave(os.path.join(args.save_dir, "02_input_lr.png"), im_lr.clip(0, 1))
    imsave(os.path.join(args.save_dir, "03_baseline_bicubic.png"), im_bi.clip(0, 1))
    imsave(os.path.join(args.save_dir, f"04_cafe_sr_psnr_{final_psnr:.2f}.png"), best_img.clip(0, 1))
    
    print(f"All comparison images saved to {args.save_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="2D Single Image Super-Resolution with CAFE_Net")
    
    # Data & IO arguments
    parser.add_argument('--data_path', type=str, default='data/pepper.tiff', help='Path to input HR image')
    parser.add_argument('--save_dir', type=str, default='results_2d_SISR', help='Directory to save outputs')
    parser.add_argument('--scale', type=int, default=4, help='Downsampling/Upsampling scale factor')
    
    # Training arguments
    parser.add_argument('--niters', type=int, default=2000, help='Number of SGD iterations')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model arguments
    parser.add_argument('--rff_mapping_size', type=int, default=96, help='RFF mapping dimension')
    parser.add_argument('--rff_scale', type=float, default=10.0, help='RFF scale')
    parser.add_argument('--cheb_order', type=int, default=32, help='Chebyshev polynomial order')
    parser.add_argument('--hidden_features', type=int, default=256, help='Number of hidden features')
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num_branches', type=int, default=2, help='Number of network branches')

    args = parser.parse_args()
    
    set_seed(args.seed)
    train_sisr(args)