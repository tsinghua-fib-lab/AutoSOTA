"""
3D Occupancy Fitting with CAFE_Net
Adapted from the WIRE framework: https://github.com/vishwa91/wire/tree/main
Reference: Saragadam Vishwanath et al. WIRE: Wavelet Implicit Neural Representations.

Performance Reference on Thai Dataset (CAFE+):
- hidden_layers=2, num_branches=2, rff_mapping_size=98, cheb_order=20, rff_scale=30
- Adjust hidden_features to match different model sizes for fair comparison
----------------------------------------------------------------------------------
| Hidden Features | Parameters | Expected IoU |
|-----------------|------------|--------------|
| 224             | 216,161    | > 0.9994     | 
| 196             | 178,165    | ≈ 0.9991     |                                  
| 160             | 133,921    | ≈ 0.9986     | 
| 128             |  98,945    | ≈ 0.9970     | 
----------------------------------------------------------------------------------
Note: Replacing RFF with PE (PE_CAFE_Net) may yield better performance in complex 
scenes (e.g., Replica).
"""

import os
import copy
import argparse
import numpy as np
import torch
import tqdm
from scipy import io
from scipy import ndimage

# Custom modules (Ensure these are provided in the repository)
from utils_3d import get_coords, get_IoU, march_and_save
from model import CAFE_Net, PE_CAFE_Net

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_and_preprocess_data(filepath, scale=1.0):
    """Loads .mat 3D data, applies zoom, and clips to tightest bounding box."""
    print(f"Loading data from {filepath}...")
    data = io.loadmat(filepath)['hypercube'].astype(np.float32)
    
    # Normalize and scale
    data = ndimage.zoom(data / data.max(), [scale, scale, scale], order=0)

    # Clip to tightest bounding box
    hidx, widx, tidx = np.where(data > 0.99)
    data = data[
        hidx.min():hidx.max(),
        widx.min():widx.max(),
        tidx.min():tidx.max()
    ]
    
    print(f"Data shape after preprocessing: {data.shape}")
    print(f"Data range: [{np.min(data):.4f}, {np.max(data):.4f}]")
    return data

def train(args):
    # 1. Prepare Data
    data = load_and_preprocess_data(args.data_path, args.scale)
    H, W, T = data.shape
    
    maxpoints = min(H * W * T, args.maxpoints)
    dataten = torch.tensor(data).cuda().reshape(H * W * T, 1)
    coords = get_coords(H, W, T)

    # 2. Build Model
    if args.use_pe:
        model = PE_CAFE_Net(
            in_features=3,
            out_features=1,
            N_freqs=10,
            cheb_order=args.cheb_order,
            hidden_features=args.hidden_features,
            num_branches=args.num_branches,
            hidden_layers=args.hidden_layers
        ).cuda()
    else:
        model = CAFE_Net(
            in_features=3,
            out_features=1,
            rff_mapping_size=args.rff_mapping_size,
            cheb_order=args.cheb_order,
            rff_scale=args.rff_scale,
            hidden_features=args.hidden_features,
            num_branches=args.num_branches,
            hidden_layers=args.hidden_layers
        ).cuda()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: {model.__class__.__name__}")
    print(f"Number of trainable parameters: {num_params}")

    # 3. Setup Optimizer and Loss
    optim = torch.optim.Adam(lr=args.lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.niters, eta_min=1e-4)
    criterion = torch.nn.MSELoss()

    # 4. Training Loop
    best_mse = float('inf')
    best_results = None
    im_estim = torch.zeros((H * W * T, 1), device='cuda')
    
    tbar = tqdm.tqdm(range(args.niters), desc="Training")
    
    for idx in tbar:
        indices = torch.randperm(H * W * T)
        train_loss = 0
        nchunks = 0
        
        for b_idx in range(0, H * W * T, maxpoints):
            b_indices = indices[b_idx:min(H * W * T, b_idx + maxpoints)]
            b_coords = coords[b_indices, ...].cuda()
            b_indices_cuda = b_indices.cuda()
            
            # Forward pass
            pixelvalues = model(b_coords[None, ...]).squeeze()[:, None]

            with torch.no_grad():
                im_estim[b_indices_cuda, :] = pixelvalues

            loss = criterion(pixelvalues, dataten[b_indices_cuda, :])

            # Backward pass
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            
            train_loss += loss.item()
            nchunks += 1

        avg_loss = train_loss / nchunks
        iou = get_IoU(im_estim, dataten, args.mcubes_thres)
        scheduler.step()

        # Track best results
        if avg_loss < best_mse:
            best_mse = avg_loss
            best_results = copy.deepcopy(im_estim)

        # Update progress bar
        tbar.set_postfix({"Loss": f"{avg_loss:.4e}", "IoU": f"{iou:.4f}"})

    # 5. Save Results
    print(f"\nTraining completed. Best MSE Loss: {best_mse:.4e}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    best_results_np = best_results.reshape(H, W, T).detach().cpu().numpy()
    final_iou = get_IoU(best_results_np, data, args.mcubes_thres)
    print(f'Final Best IoU: {final_iou:.4f}')

    # Save .mat
    mat_path = os.path.join(args.save_dir, 'output.mat')
    io.savemat(mat_path, {'best_results': best_results_np})
    print(f"Saved raw volume to {mat_path}")

    # Extract mesh using Marching Cubes and save .dae
    dae_path = os.path.join(args.save_dir, 'output.dae')
    march_and_save(best_results_np, args.mcubes_thres, dae_path, True)
    
    print(f"Saved 3D mesh to {dae_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CAFE_Net for 3D Occupancy")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data/thai_statue.mat', help='Path to input .mat data')
    parser.add_argument('--save_dir', type=str, default='results_3d_occupancy', help='Directory to save outputs')
    parser.add_argument('--scale', type=float, default=1.0, help='Scaling factor for data zoom')
    parser.add_argument('--mcubes_thres', type=float, default=0.5, help='Threshold for marching cubes')
    
    # Training arguments
    parser.add_argument('--niters', type=int, default=200, help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate. Try 2e-2, 5e-3, or 3e-3')
    parser.add_argument('--maxpoints', type=int, default=int(2e5), help='Maximum points per batch')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model arguments
    parser.add_argument('--use_pe', action='store_true', help='Use PE_CAFE_Net instead of CAFE_Net')
    parser.add_argument('--hidden_features', type=int, default=160, help='Number of hidden features')
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num_branches', type=int, default=2, help='Number of network branches')
    parser.add_argument('--rff_mapping_size', type=int, default=98, help='Mapping size for Random Fourier Features')
    parser.add_argument('--cheb_order', type=int, default=20, help='Chebyshev polynomial order')
    parser.add_argument('--rff_scale', type=float, default=30.0, help='RFF scale')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    train(args)