import sys
import time
import torch
from torch import optim
from utils import *
from model import RepTRFD


def compute_tv_sstv_loss(recon, gamma1, gamma2):
    tv_h = torch.norm(recon[1:, :, :] - recon[:-1, :, :], 1)
    tv_w = torch.norm(recon[:, 1:, :] - recon[:, :-1, :], 1)
    dz = recon[:, :, 1:] - recon[:, :, :-1]
    sstv_h = torch.norm(dz[1:, :, 1:] - dz[:-1, :, 1:], 1)
    sstv_w = torch.norm(dz[:, 1:, 1:] - dz[:, :-1, 1:], 1)
    return gamma1 * (tv_h + tv_w) + gamma2 * (sstv_h + sstv_w)


def train(file_path, noise_std, ranks, depths, expansion, omega_0,
          lr, weight_decay, gamma1, gamma2, max_iter=4001, shared_depth=1):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    key = get_mat_key(file_path)
    gt, obs = preprocess_denoising(
        file_path=file_path, noise_std=noise_std, device=device, key=key)

    n1, n2, n3 = gt.shape
    U_coord = torch.linspace(-1, 1, n1).view(n1, 1).to(device)
    V_coord = torch.linspace(-1, 1, n2).view(n2, 1).to(device)
    W_coord = torch.linspace(-1, 1, n3).view(n3, 1).to(device)

    model = RepTRFD(
        ranks=ranks, hidden_dims=256,
        expansion=expansion, omega_0=omega_0, depths=depths,
        shared_depth=shared_depth
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_time = time.time()

    checkpoint_iters = [3000, 4000, 5000, max_iter - 1]
    checkpoint_outputs = []
    for iter_idx in range(max_iter):
        recon = model([U_coord, V_coord, W_coord])
        data_loss = torch.norm(recon - obs, p='fro')
        reg_loss = compute_tv_sstv_loss(recon, gamma1, gamma2)
        loss = data_loss + reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_idx in checkpoint_iters:
            with torch.no_grad():
                checkpoint_outputs.append(recon.clone())

    elapsed_time = time.time() - start_time

    with torch.no_grad():
        avg_recon = sum(checkpoint_outputs) / len(checkpoint_outputs)
        PSNR = calculate_psnr(avg_recon, gt)
        SSIM = calculate_ssim(avg_recon, gt)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return PSNR, SSIM, elapsed_time


if __name__ == '__main__':
    set_seed(1)

    expansion = 5
    ranks = [16, 16, 16]
    omega_0 = 120
    depths = [1, 1, 2]
    lr, weight_decay = 3e-4, 1.0
    gamma1, gamma2 = 1e-4, 1e-4  # TV and SSTV

    data_dir = 'data/'
    datasets = []
    for fname in ['Toy.mat', 'Face.mat']:
        import os
        if os.path.exists(data_dir + fname):
            datasets.append(data_dir + fname)

    if not datasets:
        print("ERROR: No dataset files found in data/")
        sys.exit(1)

    noise_levels = [0.1, 0.2, 0.3]
    shared_depth_map = {0.1: 2, 0.2: 1, 0.3: 1}
    max_iter_map = {0.1: 4001, 0.2: 6001, 0.3: 4001}

    for file_path in datasets:
        dataset_name = file_path.replace(data_dir, '').replace('.mat', '')
        for noise_std in noise_levels:
            psnr, ssim, elapsed = train(
                file_path=file_path, noise_std=noise_std,
                ranks=ranks, depths=depths, expansion=expansion, omega_0=omega_0,
                lr=lr, weight_decay=weight_decay,
                gamma1=gamma1, gamma2=gamma2,
                max_iter=max_iter_map[noise_std],
                shared_depth=shared_depth_map[noise_std]
            )
            print(f"RESULT dataset={dataset_name} noise_std={noise_std} "
                  f"PSNR={psnr:.4f} SSIM={ssim:.4f} Time={elapsed:.2f}s")
