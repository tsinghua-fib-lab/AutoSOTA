"""
2D Image Denoising with CAFE_Net
Adapted from the WIRE framework: https://github.com/vishwa91/wire
"""

import os
import time
import argparse
import numpy as np
import cv2
import torch
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

def get_noise(x, noise_std=0.2):
    """Add Gaussian noise to the input image."""
    x_meas = np.copy(x)
    noise = np.random.randn(*x_meas.shape) * noise_std
    return x_meas + noise

def load_and_prepare_image(filepath, target_size=(512, 512)):
    """Load, normalize, and resize the image."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found at: {filepath}")
        

    img = plt.imread(filepath)[:, :, :3].astype(np.float32)
    img = normalize(img, True)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img

def train_denoising(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Prepare Data
    im_gt = load_and_prepare_image(args.data_path)
    H, W, C = im_gt.shape
    
    im_noisy = get_noise(im_gt, noise_std=args.noise_std)
    initial_psnr = psnr(im_gt, im_noisy)
    print(f'Input Noisy PSNR: {initial_psnr:.2f} dB')

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

    # 4. Prepare Tensors
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...].to(device)

    gt_tensor = torch.tensor(im_gt).to(device).reshape(H * W, 3)[None, ...]
    gt_noisy_tensor = torch.tensor(im_noisy).to(device).reshape(H * W, 3)[None, ...]
    rec_tensor = torch.zeros_like(gt_tensor)

    # 5. Training Loop
    best_mse = float('inf')
    best_img = None
    batch_size = min(H * W, args.batch_size)
    
    tbar = tqdm(range(args.niters), desc="Training")
    
    for epoch in tbar:
        indices = torch.randperm(H * W)

        for b_idx in range(0, H * W, batch_size):
            b_indices = indices[b_idx:min(H * W, b_idx + batch_size)]
            b_coords = coords[:, b_indices, ...]
            
            # Forward pass
            pixelvalues = model(b_coords)

            with torch.no_grad():
                rec_tensor[:, b_indices, :] = pixelvalues

            # Loss computed against the *noisy* image
            loss = ((pixelvalues - gt_noisy_tensor[:, b_indices, :]) ** 2).mean()

            # Backward pass
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            
        scheduler.step()

        # Evaluate and log every 50 epochs
        if epoch % 50 == 0 or epoch == args.niters - 1:
            with torch.no_grad():

                mse_vs_gt = ((gt_tensor - rec_tensor) ** 2).mean().item()
                current_psnr = -10 * np.log10(mse_vs_gt)
                
                tbar.set_postfix({"PSNR (dB)": f"{current_psnr:.2f}"})

                imrec = rec_tensor[0, ...].reshape(H, W, 3).detach().cpu().numpy()

                if mse_vs_gt < best_mse:
                    best_mse = mse_vs_gt
                    best_img = imrec.copy()

    # 6. Save Results
    os.makedirs(args.save_dir, exist_ok=True)
    final_best_psnr = psnr(im_gt, best_img)
    print(f'\nTraining completed. Best PSNR: {final_best_psnr:.2f} dB')

    # Save Noisy input for reference
    noisy_save_path = os.path.join(args.save_dir, f"input_noisy_std{args.noise_std}.png")
    imsave(noisy_save_path, im_noisy.clip(0, 1))
    
    # Save best reconstruction
    pred_save_path = os.path.join(args.save_dir, f"denoised_psnr_{final_best_psnr:.2f}.png")
    imsave(pred_save_path, best_img.clip(0, 1))
    print(f"Results saved to {args.save_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="2D Image Denoising with CAFE_Net")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/pepper.tiff', help='Path to input image')
    parser.add_argument('--save_dir', type=str, default='results_2d_denoising', help='Directory to save outputs')
    parser.add_argument('--noise_std', type=float, default=0.2, help='Standard deviation of Gaussian noise')
    
    # Training arguments
    parser.add_argument('--niters', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512*512, help='Points per batch. Default is full image (512x512).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model arguments
    parser.add_argument('--rff_mapping_size', type=int, default=48, help='RFF mapping dimension')
    parser.add_argument('--rff_scale', type=float, default=2.0, help='RFF scale')
    parser.add_argument('--cheb_order', type=int, default=16, help='Chebyshev polynomial order')
    parser.add_argument('--hidden_features', type=int, default=256, help='Number of hidden features')
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num_branches', type=int, default=2, help='Number of network branches')

    args = parser.parse_args()
    
    set_seed(args.seed)
    train_denoising(args)