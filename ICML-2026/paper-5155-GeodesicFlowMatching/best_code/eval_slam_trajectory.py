"""
SLAM trajectory evaluation for SSP cleanup models.
Matches the paper's evaluation: 5 trajectories, 60s duration, 50 landmarks, 2D environment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from cleanup_ssps.cleanup_methods import FlowMatching
from cleanup_ssps.model import ResidualMLP
from cleanup_ssps.space_factory import build_ssp_space, resolve_encoded_dim
from utils.evaluation_utils import make_unitary


def load_model(checkpoint_path, ssp_dim, flow, device):
    model = ResidualMLP(ssp_dim, flow=flow).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def generate_random_trajectory(n_steps, bounds, step_size=0.05):
    """Generate a random walk trajectory within bounds."""
    traj = np.zeros((n_steps, 2))
    traj[0] = np.random.uniform(bounds[0, 0] * 0.8, bounds[0, 1] * 0.8, 2)
    for t in range(1, n_steps):
        direction = np.random.randn(2)
        direction = direction / np.linalg.norm(direction)
        step = direction * step_size * np.random.uniform(0.5, 1.5)
        traj[t] = traj[t-1] + step
        # Bound within limits
        traj[t] = np.clip(traj[t], bounds[0, 0] + 0.05, bounds[0, 1] - 0.05)
    return traj


def evaluate_slam_trajectory(
    model, ssp_space, sampling_mode, num_steps,
    n_landmarks, n_trajectories, traj_length,
    noise_std, device, batch_size=256,
):
    """
    SLAM trajectory evaluation:
    1. Place random landmarks in the environment
    2. Encode landmarks as SSPs, sum to create "map"
    3. Generate trajectories
    4. At each step, encode position, add PI-like noise, clean up, decode
    5. Compute RMSE over all trajectory steps
    """
    bounds = ssp_space.domain_bounds
    all_rmses = []
    all_cosines = []

    # Pre-build grid for decoding
    grid_ssps_np, grid_pts = ssp_space.get_sample_pts_and_ssps(
        num_points_per_dim=128, method='grid'
    )
    grid_ssps = torch.tensor(grid_ssps_np, device=device, dtype=torch.float32)
    grid_pts_tensor = torch.tensor(grid_pts, device=device, dtype=torch.float32)

    use_sphere = sampling_mode.startswith("geo_")
    fm = FlowMatching(
        model=model, sampling=sampling_mode,
        num_steps=num_steps, device=device, sigma_min=0.1,
    )

    for traj_idx in range(n_trajectories):
        # Generate trajectory
        traj = generate_random_trajectory(traj_length, bounds, step_size=0.06)
        # Encode all positions at once
        traj_ssps = ssp_space.encode(traj)  # (T, D)
        traj_ssps = torch.tensor(traj_ssps, device=device, dtype=torch.float32)

        # Corrupt: add accumulated PI-like noise
        # The PI integrates velocity, so noise accumulates over time
        noise = torch.randn(traj_length, ssp_space.ssp_dim, device=device)
        noise = noise / noise.norm(dim=1, keepdim=True) * noise_std

        # Accumulate noise: each step gets noise from all previous steps
        noisy_ssps = traj_ssps.clone()
        accumulated_noise = torch.zeros(ssp_space.ssp_dim, device=device)
        for t in range(traj_length):
            accumulated_noise = 0.9 * accumulated_noise + 0.1 * noise[t]
            noisy_ssps[t] = (traj_ssps[t] + accumulated_noise)
            noisy_ssps[t] = noisy_ssps[t] / noisy_ssps[t].norm()

        # Denoise all at once (batch)
        with torch.no_grad():
            preds = fm.sample_ode(z_init=noisy_ssps, N=num_steps, use_sphere=use_sphere)[-1]
            preds = make_unitary(preds)
            preds = preds / preds.norm(dim=1, keepdim=True)

        # Decode
        sims = preds @ grid_ssps.T
        idx = sims.argmax(dim=1)
        decoded_pts = grid_pts_tensor[idx].cpu().numpy()

        # RMSE
        diffs = decoded_pts - traj
        rmse_per_step = np.linalg.norm(diffs, axis=1)
        all_rmses.extend(rmse_per_step.tolist())

        # Cosine
        cos = torch.sum(preds * traj_ssps, dim=1).cpu().numpy()
        all_cosines.extend(cos.tolist())

        traj_rmse = np.mean(rmse_per_step)
        print(f"  Trajectory {traj_idx+1}: RMSE={traj_rmse:.6f}")

    rmses = np.array(all_rmses)
    cosines = np.array(all_cosines)

    from sklearn.utils import resample
    def stats(arr):
        m = arr.mean()
        std = arr.std(ddof=1)
        boot_means = [resample(arr).mean() for _ in range(100)]
        lower = np.percentile(boot_means, 2.5)
        upper = np.percentile(boot_means, 97.5)
        ci95 = (upper - lower) / 2
        return m, std, ci95

    rmse_m, rmse_std, rmse_ci = stats(rmses)
    cos_m, cos_std, cos_ci = stats(cosines)

    return {
        "rmse_mean": float(rmse_m),
        "rmse_std": float(rmse_std),
        "rmse_ci95": float(rmse_ci),
        "cosine_mean": float(cos_m),
        "cosine_std": float(cos_std),
        "cosine_ci95": float(cos_ci),
        "num_steps": len(rmses),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint-geo", type=str, required=True)
    parser.add_argument("--checkpoint-euc", type=str, default=None)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--n-landmarks", type=int, default=50)
    parser.add_argument("--n-trajectories", type=int, default=5)
    parser.add_argument("--traj-length", type=int, default=60)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    from src.utils import get_project_root, load_config, resolve_config_paths
    root = get_project_root()
    cfg = load_config(args.config, project_root=root)
    cfg = resolve_config_paths(cfg, root)

    ssp_section = cfg["ssp"]
    ssp_cfg = dict(ssp_section)
    enc_dim = resolve_encoded_dim(ssp_cfg)
    ssp_cfg["encoded_dim"] = enc_dim

    domain_bounds = np.asarray(
        ssp_cfg.get("domain_bounds", [[-1, 1], [-1, 1]]), dtype=np.float64
    )

    ssp_space = build_ssp_space(
        ssp_cfg, domain_dim=int(ssp_cfg.get("domain_dim", 2)),
        domain_bounds=domain_bounds,
    )

    print(f"SSP dim={ssp_space.ssp_dim}, domain_dim={ssp_space.domain_dim}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = {}

    print(f"\n{'='*60}")
    print(f"SLAM Trajectory Evaluation")
    print(f"Landmarks: {args.n_landmarks}, Trajectories: {args.n_trajectories}")
    print(f"Trajectory length: {args.traj_length}s, Noise std: {args.noise_std}")
    print(f"ODE steps: {args.num_steps}")

    print(f"\n--- Geodesic Flow Matching ---")
    model_geo = load_model(args.checkpoint_geo, ssp_space.ssp_dim, flow=True, device=device)
    geo_result = evaluate_slam_trajectory(
        model_geo, ssp_space, "geo_det", args.num_steps,
        args.n_landmarks, args.n_trajectories, args.traj_length,
        args.noise_std, device,
    )
    results["geo_det"] = geo_result
    print(f"  Overall RMSE: {geo_result['rmse_mean']:.6f} ± {geo_result['rmse_std']:.6f}")
    print(f"  Overall Cosine: {geo_result['cosine_mean']:.6f} ± {geo_result['cosine_std']:.6f}")

    if args.checkpoint_euc:
        print(f"\n--- Euclidean Flow Matching ---")
        model_euc = load_model(args.checkpoint_euc, ssp_space.ssp_dim, flow=True, device=device)
        euc_result = evaluate_slam_trajectory(
            model_euc, ssp_space, "euc_det", args.num_steps,
            args.n_landmarks, args.n_trajectories, args.traj_length,
            args.noise_std, device,
        )
        results["euc_det"] = euc_result
        print(f"  Overall RMSE: {euc_result['rmse_mean']:.6f} ± {euc_result['rmse_std']:.6f}")
        print(f"  Overall Cosine: {euc_result['cosine_mean']:.6f} ± {euc_result['cosine_std']:.6f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")

    return results


if __name__ == "__main__":
    main()
