import sys
import time
import torch
from torch import optim
from tqdm import tqdm
from utils import *
from model import RepTRFD

def compute_tv_loss(recon, phi):
    tv_h = torch.norm(recon[1:, :, :] - recon[:-1, :, :], 1)
    tv_w = torch.norm(recon[:, 1:, :] - recon[:, :-1, :], 1)
    return phi * (tv_h + tv_w)


def train(file_path, scale, crop_size, ranks, depths, expansion, omega_0,
          lr, weight_decay, gamma, max_iter=6001, log_interval=500):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"File: {file_path} | Device: {device} | Scale: x{scale}")

    gt, obs = preprocess_super_resolution(
        image_path=file_path,
        device=device,
        scale=scale,
        crop_size=crop_size
    )

    print(f"HR Shape: {gt.shape}, LR Shape: {obs.shape}")

    n1, n2, n3 = gt.shape
    U_coord = torch.linspace(-1, 1, n1).view(n1, 1).to(device)
    V_coord = torch.linspace(-1, 1, n2).view(n2, 1).to(device)
    W_coord = torch.linspace(-1, 1, n3).view(n3, 1).to(device)

    downsampler = torch.nn.AvgPool2d(scale)

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
        rec_hr_img = model([U_coord, V_coord, W_coord])  # [H, W, C]
        rec_hr = rec_hr_img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        rec_lr = downsampler(rec_hr).squeeze(0).permute(1, 2, 0)  # [H/scale, W/scale, C]

        data_loss = torch.norm(rec_lr - obs, p='fro')
        reg_loss = compute_tv_loss(rec_hr_img, gamma)
        loss = data_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_idx % log_interval == 0 or iter_idx == max_iter - 1:
            with torch.no_grad():
                eval_hr = model([U_coord, V_coord, W_coord])
                psnr = calculate_psnr(eval_hr, gt)

                pbar.set_postfix({
                    'Loss': f"{loss.item():.2f}",
                    'PSNR': f"{psnr:.2f}"
                })

    elapsed_time = time.time() - start_time

    with torch.no_grad():
        final_hr = model([U_coord, V_coord, W_coord])
        PSNR = calculate_psnr(final_hr, gt)
        SSIM = calculate_ssim(final_hr, gt)
        NRMSE = calculate_nrmse(final_hr, gt)

    print(f"\nTraining Complete in {elapsed_time:.2f}s")
    print(f"Final Metrics: PSNR: {PSNR:.2f}, SSIM: {SSIM:.3f}, NRMSE: {NRMSE:.3f}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    set_seed(1)

    expansion = 10; ranks = [20, 20, 20]
    omega_0 = 90; depths = [1, 1, 2]
    lr, weight_decay = 3e-4, 0.5; gamma = 5e-5
    file_path = 'data/0809.png'; scale = 4; crop_size = 768

    train(
        file_path=file_path, scale=scale, crop_size=crop_size,
        ranks=ranks, depths=depths, expansion=expansion, omega_0=omega_0,
        lr=lr, weight_decay=weight_decay, gamma=gamma,
        max_iter=6001, log_interval=500
    )