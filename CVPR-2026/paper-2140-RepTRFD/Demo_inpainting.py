import sys
import time
import torch
from torch import optim
from tqdm import tqdm
from utils import *
from model import RepTRFD

def compute_tv_sstv_loss(recon, gamma1, gamma2):
    tv_h = torch.norm(recon[1:, :, :] - recon[:-1, :, :], 1)
    tv_w = torch.norm(recon[:, 1:, :] - recon[:, :-1, :], 1)
    dz = recon[:, :, 1:] - recon[:, :, :-1]
    sstv_h = torch.norm(dz[1:, :, 1:] - dz[:-1, :, 1:], 1)
    sstv_w = torch.norm(dz[:, 1:, 1:] - dz[:, :-1, 1:], 1)

    return gamma1 * (tv_h + tv_w) + gamma2 * (sstv_h + sstv_w)


def train(file_path, is_mat, obs_ratio, ranks, depths, expansion, omega_0,
          lr, weight_decay, gamma1, gamma2, max_iter=6001, log_interval=500):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"File: {file_path} | Device: {device} | Sampling Ratio: {obs_ratio}")

    key = get_mat_key(file_path) if is_mat else None
    gt, obs, mask = preprocess_inpainting(
        file_path=file_path,
        obs_ratio=obs_ratio,
        device=device,
        key=key
    )
    print(f"Ground Truth Shape: {gt.shape}")

    n1, n2, n3 = gt.shape
    U_coord = torch.linspace(-1, 1, n1).view(n1, 1).to(device)
    V_coord = torch.linspace(-1, 1, n2).view(n2, 1).to(device)
    W_coord = torch.linspace(-1, 1, n3).view(n3, 1).to(device)

    model = RepTRFD(
        ranks=ranks,
        hidden_dims=256,
        expansion=expansion,
        omega_0=omega_0,
        depths=depths
    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    start_time = time.time()

    pbar = tqdm(range(max_iter), desc="Training", file=sys.stdout)
    for iter_idx in pbar:
        recon = model([U_coord, V_coord, W_coord])
        data_loss = torch.norm(recon * mask - obs * mask, p='fro')
        recon[mask == 1] = obs[mask == 1]
        reg_loss = compute_tv_sstv_loss(recon, gamma1, gamma2)

        loss = data_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_idx % log_interval == 0 or iter_idx == max_iter - 1:
            with torch.no_grad():
                eval_recon = model([U_coord, V_coord, W_coord])
                eval_recon[mask == 1] = obs[mask == 1]
                psnr = calculate_psnr(eval_recon, gt)

                pbar.set_postfix({
                    'Loss': f"{loss.item():.2f}",
                    'PSNR': f"{psnr:.2f}",
                })

    elapsed_time = time.time() - start_time

    with torch.no_grad():
        final_recon = model([U_coord, V_coord, W_coord])
        final_recon[mask == 1] = obs[mask == 1]
        PSNR = calculate_psnr(final_recon, gt)
        SSIM = calculate_ssim(final_recon, gt)
        NRMSE = calculate_nrmse(final_recon, gt)

    print(f"\nTraining Complete in {elapsed_time:.2f}s")
    print(f"Final Metrics: PSNR: {PSNR:.2f}, SSIM: {SSIM:.3f}, NRMSE: {NRMSE:.3f}")


if __name__ == '__main__':
    set_seed(1)

    # --- Common Default Settings ---
    expansion = 10; ranks = [20, 20, 20]; lr = 3e-4

    # --- 1. RGB Inpainting ---
    omega_0 = 90; depths = [1, 1, 2]; weight_decay = 0.5
    gamma1, gamma2 = 5e-5, 5e-5  # TV and SSTV
    file_path = 'data/airplane.tiff'; obs_ratio = 0.3; is_mat = False

    # --- 2. Video Inpainting ---
    # omega_0 = 90; depths = [1, 1, 2]; weight_decay = 0.5
    # gamma1, gamma2 = 5e-5, 5e-4  # TV and SSTV
    # file_path = 'data/news.mat'; obs_ratio = 0.2; is_mat = True

    # --- 3. MSI Inpainting ---
    # omega_0 = 120; depths = [1, 1, 1]; weight_decay = 0.2
    # gamma1, gamma2 = 5e-5, 5e-4  # TV and SSTV
    # file_path = 'data/Toy.mat'; obs_ratio = 0.1; is_mat = True

    # --- 4. HSI Inpainting ---
    # omega_0 = 120; depths = [1, 1, 1]; weight_decay = 0.2
    # gamma1 = 5e-6; gamma2 = 5e-5  # TV and SSTV
    # file_path = 'data/Washington_DC.mat'; obs_ratio = 0.05; is_mat = True

    train(
        file_path=file_path, is_mat=is_mat, obs_ratio=obs_ratio,
        ranks=ranks, depths=depths, expansion=expansion, omega_0=omega_0,
        lr=lr, weight_decay=weight_decay, gamma1=gamma1, gamma2=gamma2,
        max_iter=6001, log_interval=500
    )