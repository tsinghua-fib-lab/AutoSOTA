import os
import time
import random
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from utils import ImageFitting
from model import CAFE_Net


def mse_fn(pred, gt):
    return ((pred - gt) ** 2).mean()


def psnr_fn(pred, gt):
    return -10. * torch.log10(mse_fn(pred, gt))


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args, model, dataloader, img_h, img_w):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    model.cuda()
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.total_steps, eta_min=1e-5)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    best_psnr = -1e9
    best_step = 0
    best_output_img = None

    start_time = time.time()

    for step in range(args.total_steps):
        model_output = model(model_input)  # [1, H*W, 3]

        loss = mse_fn(model_output, ground_truth)

        if step % args.steps_til_summary == 0:
            psnr = psnr_fn(model_output / 2 + 0.5, ground_truth / 2 + 0.5)
            print(f"Step {step:04d} | Loss: {loss.item():.6f} | PSNR: {psnr.item():.4f}")

            if psnr.item() > best_psnr:
                best_psnr = psnr.item()
                best_step = step
                best_output_img = (model_output / 2 + 0.5).view(img_h, img_w, 3).detach().cpu().numpy().clip(0, 1)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        scheduler.step()

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    if best_output_img is not None:
        save_path = os.path.join(save_dir, f"best_psnr_{best_step:04d}_{best_psnr:.2f}.png")
        Image.fromarray((best_output_img * 255).astype(np.uint8)).save(save_path)
        print(f"Saved best PSNR image at: {save_path}\n")


if __name__ == "__main__":
    # Data arguments
    parser = argparse.ArgumentParser(description="Train CAFE_Net for Image Fitting")
    parser.add_argument("--data_dir", type=str, default="data/04.png", help="Path of images")
    parser.add_argument("--save_dir", type=str, default="results_cafe", help="Directory to save results")
    parser.add_argument("--img_size", type=int, default=512, help="Image resolution for training")
    
    # Training arguments
    parser.add_argument("--total_steps", type=int, default=6001, help="Total training steps")
    parser.add_argument("--steps_til_summary", type=int, default=200, help="Steps between logging")
    parser.add_argument("--lr", type=float, default=2e-2, help="Learning rate. Try 2e-2, 5e-3, or 3e-3")

    # Model Hyperparameters
    # Default configuration corresponds to the 0.22M model in the paper.
    parser.add_argument("--num_branches", type=int, default=3,
                        help="Number of parallel linear branches for multiplication")
    parser.add_argument("--hidden_features", type=int, default=0, 
                        help="Hidden dimension of the network. If set to 0, defaults to the same dimension as encoded_dim")
    parser.add_argument("--hidden_layers", type=int, default=1, help="Number of hidden layers in Backbone MLP")
    parser.add_argument("--cheb_order", type=int, default=30, help="Order of Chebyshev polynomials")
    parser.add_argument("--rff_mapping_size", type=int, default=88, help="Mapping size for Random Fourier Features")
    parser.add_argument('--rff_scale', type=float, default=30.0, help='RFF scale')
    
    # For the 0.33M model, use:
    #   --lr 5e-3
    #   --num_branches 3
    #   --hidden_layers 2
    #   --cheb_order 32
    #   --rff_mapping_size 96
    # args = parser.parse_args()
    
    args, _ = parser.parse_known_args()
    filename = args.data_dir
    if not os.path.exists(filename):
        print(f"File {filename} not found")
    print(f"=== Processing file: {filename} ===")

    dataset = ImageFitting(path=filename, sidelength=args.img_size)
    dataloader = DataLoader(dataset, batch_size=1)

    model = CAFE_Net(
        rff_mapping_size=args.rff_mapping_size,
        rff_scale=args.rff_scale,
        cheb_order=args.cheb_order,
        hidden_features=args.hidden_features,
        num_branches=args.num_branches,
        hidden_layers=args.hidden_layers
    )

    train(args, model, dataloader, img_h=args.img_size, img_w=args.img_size)