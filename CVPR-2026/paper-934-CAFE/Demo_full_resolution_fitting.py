"""
Large Image (eg. Full Resolution DIV2K Dataset) Fitting with CAFE_Net

Model Parameters: 0.47M  -  PSNR:35.71 - 0891_half.png

"""

import os
import time
import argparse
import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.image import imsave

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Custom modules (Ensure these are in your repository)
from model import CAFE_Net, PE_CAFE_Net


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def mse_fn(pred, gt):
    return ((pred - gt) ** 2).mean()


def psnr_fn(pred, gt):
    mse = mse_fn(pred, gt)
    psnr = -10 * torch.log10(mse + 1e-8)
    return psnr


def get_mgrid(height, width, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    y = torch.linspace(-1, 1, steps=height)
    x = torch.linspace(-1, 1, steps=width)
    if dim == 2:
        mgrid = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)  # (H, W, 2)
    else:
        raise NotImplementedError("Currently only dim=2 is supported")
    return mgrid.reshape(-1, dim)  # (H*W, 2)


def img2tensor(path, height, width, normalize_data=True):
    """Loads an image (or .mat file) and transforms it to a PyTorch tensor."""
    ext = os.path.splitext(path)[1].lower()

    if ext == '.mat':
        mat_data = sio.loadmat(path)
        img = mat_data.get('img', None)
        if img is None:
            raise ValueError(f"Cannot find 'img' key in MAT file: {path}")

        if img.ndim == 2:  # Grayscale
            img = img[..., None]
        elif img.ndim == 4:  # (1, H, W, C)
            img = img[0]

        if img.dtype != 'uint8':
            img = (img * 255).astype('uint8')

        if img.shape[2] == 4:
            img = Image.fromarray(img, mode='RGBA').convert('RGB')
        else:
            img = Image.fromarray(img, mode='RGB')
    else:
        if not os.path.exists(path):
             raise FileNotFoundError(f"Image file not found: {path}")
        img = Image.open(path).convert('RGB')

    transforms_list = [Resize((height, width)), ToTensor()]
    if normalize_data:
        # Normalize to [-1, 1]
        transforms_list.append(Normalize(torch.Tensor([0.5, 0.5, 0.5]), torch.Tensor([0.5, 0.5, 0.5])))

    transform = Compose(transforms_list)
    return transform(img)


class ImageFittingDataset(Dataset):
    """Dataset for INR image fitting. Samples random pixel coordinates and their RGB values."""
    def __init__(self, path, height, width, sample_points=512*512):
        img = img2tensor(path, height, width)  # (C, H, W)
        C, H, W = img.shape
        self.pixels = img.permute(1, 2, 0).reshape(-1, C)  # (H*W, C)
        self.coords = get_mgrid(H, W)  # (H*W, 2)
        self.sample_points = sample_points
        self.num_points = self.coords.shape[0]

    def __len__(self):
        # Return a large number to simulate an infinite iterator
        return 1000000 

    def __getitem__(self, idx):
        idxs = np.random.choice(self.num_points, self.sample_points, replace=False)
        coords_batch = self.coords[idxs]
        pixels_batch = self.pixels[idxs]
        return coords_batch, pixels_batch


def batched_inference(model, coords, batch_size=131072):
    """Performs inference in batches to prevent Out-Of-Memory (OOM) errors."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, coords.shape[0], batch_size):
            coords_batch_2d = coords[i:i+batch_size] 
            
            # Key: Add batch dimension to match expected shape [B, N, D]
            coords_batch_3d = coords_batch_2d.unsqueeze(0)
            
            out = model(coords_batch_3d) # Expected output shape: [1, batch_size, C]
            
            # Remove batch dimension: [batch_size, C]
            outs.append(out.squeeze(0)) 
            
    return torch.cat(outs, dim=0) # Final shape: [N, C]


def train_fitting(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    filename_no_ext = os.path.splitext(os.path.basename(args.data_path))[0]

    # 1. Prepare Data
    print(f"Loading image: {args.data_path} and resizing to {args.height}x{args.width}")
    try:
        dataset = ImageFittingDataset(args.data_path, args.height, args.width, args.sample_points)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # 2. Build Model
    model = CAFE_Net(
        rff_mapping_size=args.rff_mapping_size,
        rff_scale=args.rff_scale,
        cheb_order=args.cheb_order,
        hidden_features=args.hidden_features,
        num_branches=args.num_branches,
        hidden_layers=args.hidden_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params / 1e6:.2f}M")

    # 3. Setup Optimizer & Scheduler
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.total_steps, eta_min=1e-5)

    # 4. Training Loop
    best_psnr = -1e9
    best_output_img = None
    best_step = 0

    tbar = tqdm(range(args.total_steps), desc="Training Image Fitting")
    start_time = time.time()
    
    data_iter = iter(dataloader)

    for step in tbar:
        coords_batch, pixels_batch = next(data_iter)
        coords_batch, pixels_batch = coords_batch.to(device), pixels_batch.to(device)
        
        # coords_batch from dataloader (batch_size=1) is already [1, N, 2]
        model.train()
        output = model(coords_batch) 
        loss = mse_fn(output, pixels_batch)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()
        
        # Evaluate full image periodically
        if step % args.steps_til_summary == 0 or step == args.total_steps - 1:
            full_coords = dataset.coords.to(device)
            full_output = batched_inference(model, full_coords, batch_size=131072)
            
            full_output_2d = full_output.view(args.height, args.width, 3)
            gt_full = dataset.pixels.view(args.height, args.width, 3).to(device)
            
            # De-normalize from [-1, 1] to [0, 1] for PSNR calculation
            with torch.no_grad():
                current_psnr = psnr_fn(full_output_2d / 2 + 0.5, gt_full / 2 + 0.5)
            
            img_to_show = (full_output_2d / 2 + 0.5).clamp(0, 1).cpu().numpy()
            
            if current_psnr.item() > best_psnr:
                best_psnr = current_psnr.item()
                best_step = step
                best_output_img = img_to_show
            
            tbar.set_postfix({ "Best PSNR": f"{best_psnr:.2f} dB"})

    # 5. Save Results
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")
    print(f"Best PSNR: {best_psnr:.4f} dB at step {best_step}")

    if best_output_img is not None:
        save_path = os.path.join(args.save_dir, f"{filename_no_ext}_best_step{best_step}_psnr{best_psnr:.2f}.png")
        imsave(save_path, best_output_img)
        print(f"Saved best reconstructed image to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Image Fitting with Implicit Neural Representation")
    
    # Data & IO arguments
    parser.add_argument('--data_path', type=str, default='data/0891_half.png', help='Path to target image')
    parser.add_argument('--save_dir', type=str, default='results_fitting', help='Directory to save outputs')
    
    # Image resolution & sampling
    parser.add_argument('--width', type=int, default=834, help='Target image width. ')
    parser.add_argument('--height', type=int, default=1020, help='Target image height.')
    parser.add_argument('--sample_points', type=int, default=512*512, help='Points sampled per training iteration')
    
    # Training arguments
    parser.add_argument('--total_steps', type=int, default=6001, help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--steps_til_summary', type=int, default=200, help='Iterations between evaluations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model arguments
    parser.add_argument('--rff_mapping_size', type=int, default=144, help='RFF mapping dimension')
    parser.add_argument('--rff_scale', type=float, default=30.0, help='RFF scale factor (sigma)')
    parser.add_argument('--cheb_order', type=int, default=32, help='Chebyshev polynomial order')
    parser.add_argument('--hidden_features', type=int, default=288, help='Number of hidden features')
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num_branches', type=int, default=3, help='Number of network branches')

    
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    train_fitting(args)