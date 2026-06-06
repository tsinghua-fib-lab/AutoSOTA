import sys
import time
import torch
import random
import numpy as np
from plyfile import PlyData
from torch import optim
from tqdm import tqdm
from utils import *
from model import RepTRFD_point_cloud

def load_and_preprocess_ply(file_path, obs_ratio, device):
    """
    load and preprocess the ply file.
    Returns:
        gt_coords, gt_values: Full coords [3N, 4] and RGB values [3N, 1].
        obs_coords, obs_values: Sampled subset based on obs_ratio.
        xyz_coords: Normalized spatial coords [N, 3].
    """
    plydata = PlyData.read(file_path)
    data = plydata.elements[0].data

    xyz = np.vstack([data['x'], data['y'], data['z']]).T
    rgb = np.vstack([data['red'], data['green'], data['blue']]).T

    max_abs = np.max(np.abs(xyz))
    xyz_coords = xyz / max_abs

    N = xyz.shape[0]
    coords_list = []
    values_list = []

    for i, c_val in enumerate([-1, 0, 1]):
        channel_id = np.full((N, 1), c_val, dtype=np.float32)
        coord_4d = np.concatenate([xyz_coords, channel_id], axis=1)
        coords_list.append(coord_4d)
        values_list.append(rgb[:, i:i + 1])

    gt_coords = np.concatenate(coords_list, axis=0)
    gt_values = np.concatenate(values_list, axis=0)

    gt_coords = torch.from_numpy(gt_coords).float().to(device)
    gt_values = torch.from_numpy(gt_values).float().to(device)

    total_points = gt_coords.shape[0]
    num_observed = int(total_points * obs_ratio)

    idx = torch.randperm(total_points, device=device)[:num_observed]
    obs_coords = gt_coords[idx]
    obs_values = gt_values[idx]

    return gt_coords, gt_values, obs_coords, obs_values, xyz_coords


def train(file_path, obs_ratio, ranks, expansion, omega_0, depths,
          lr, weight_decay, max_iter=3001, log_interval=500):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"File: {file_path} | Device: {device} | Sampling Ratio: {obs_ratio}")

    gt_coords, gt_values, obs_coords, obs_values, xyz_coords = load_and_preprocess_ply(
        file_path=file_path,
        obs_ratio=obs_ratio,
        device=device
    )

    model = RepTRFD_point_cloud(
        ranks=ranks,
        hidden_dims=256,
        expansion=expansion,
        omega_0=omega_0,
        depths=depths
    ).to(device)

    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters()):,}",)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    start_time = time.time()
    pbar = tqdm(range(max_iter), desc="Training", file=sys.stdout)

    for iter_idx in pbar:
        model.train()

        pred_values = model(obs_coords)
        loss = torch.norm((pred_values - obs_values), p='fro')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_idx % log_interval == 0 or iter_idx == max_iter - 1:
            model.eval()
            with torch.no_grad():
                pred_all = model(gt_coords)
                nrmse, r2 = compute_metrics(gt_values, pred_all)

                pbar.set_postfix({
                    'Loss': f"{loss.item():.2e}",
                    'NRMSE': f"{nrmse:.3f}"
                })

    elapsed_time = time.time() - start_time
    print(f"\nTraining Complete in {elapsed_time:.2f}s")
    print(f"NRMSE: {nrmse:.3f}, R2: {r2:.3f}")

    with torch.no_grad():
        final_pred = model(gt_coords).cpu().numpy()

    num = xyz_coords.shape[0]
    channel_R = final_pred[0:num, 0]
    channel_G = final_pred[num:2 * num, 0]
    channel_B = final_pred[2 * num: 3 * num, 0]

    visualize_pointcloud(xyz_coords, channel_R, channel_G, channel_B, filename="ours.png")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    expansion = 3; ranks = [20, 20, 20, 20]
    omega_0 = 240; depths = [1, 1, 1, 1]
    lr, weight_decay = 3e-4, 2e2
    file_path = 'data/mario011.ply'; obs_ratio = 0.2

    train(
        file_path=file_path, obs_ratio=obs_ratio,
        ranks=ranks, expansion=expansion,
        omega_0=omega_0, depths=depths, lr=lr, weight_decay=weight_decay,
        max_iter=3001, log_interval=500
    )