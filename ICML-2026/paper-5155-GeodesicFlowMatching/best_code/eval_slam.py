"""
SLAM-style evaluation for SSP cleanup models.
Computes RMSE between decoded positions and ground truth for denoised SSPs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cleanup_ssps.cleanup_methods import FlowMatching
from cleanup_ssps.dataset import SSPDataset
from cleanup_ssps.model import ResidualMLP
from cleanup_ssps.space_factory import build_ssp_space, resolve_encoded_dim
from utils.evaluation_utils import make_unitary


def load_model(checkpoint_path: str, ssp_dim: int, flow: bool, device: str, time_embed_dim: int = 32):
    """Load a trained ResidualMLP model from checkpoint."""
    model = ResidualMLP(ssp_dim, flow=flow, time_embed_dim=time_embed_dim).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_rmse(
    model,
    ssp_space,
    test_dir: str,
    sampling_mode: str,
    num_steps: int,
    signal_strength: float,
    noise_type: str,
    target_type: str,
    batch_size: int,
    device: str,
    num_samples: int | None = None,
):
    """
    Evaluate RMSE between decoded positions and ground truth.

    For each test sample:
    1. Get the target SSP (ground truth position encoding)
    2. Generate noise
    3. Create corrupted input: mix = signal_strength * target + (1-signal_strength) * noise
    4. Denoise using the flow matching model (with N ODE steps)
    5. Make unitary and normalize
    6. Decode by finding nearest grid point
    7. Compare decoded position with ground truth
    """
    # Build test dataset
    ds = SSPDataset(
        data_dir=test_dir,
        ssp_dim=ssp_space.ssp_dim,
        target_type=target_type,
        noise_type=noise_type,
        signal_strength=signal_strength,
        mode='test',
    )

    # Limit samples if requested
    if num_samples is not None and num_samples < len(ds):
        indices = torch.randperm(len(ds))[:num_samples]
        ds = torch.utils.data.Subset(ds, indices)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Pre-build a grid for decoding (same as baseline)
    grid_resolution = 128
    grid_ssps_np, grid_pts = ssp_space.get_sample_pts_and_ssps(
        num_points_per_dim=grid_resolution, method='grid'
    )
    grid_ssps = torch.tensor(grid_ssps_np, device=device, dtype=torch.float32)
    grid_pts_tensor = torch.tensor(grid_pts, device=device, dtype=torch.float32)

    use_sphere = sampling_mode.startswith("geo_")
    is_ff = False  # We're evaluating flow models

    all_gt_pts = []
    all_decoded_pts = []
    all_cosines = []

    with torch.no_grad():
        for inputs, targets in loader:
            z_noise = inputs.squeeze(1).to(device)
            z1 = targets.squeeze(1).to(device)

            # Build initial state with signal mixing
            s = float(signal_strength)
            if s <= 0.0:
                z_init = z_noise
            elif s >= 1.0:
                z_init = z1
            else:
                z_init = s * z1 + (1.0 - s) * z_noise
                # Renormalize
                z_init = z_init / (z_init.norm(dim=1, keepdim=True) + 1e-12)

            # Denoise using flow matching
            fm = FlowMatching(
                model=model,
                sampling=sampling_mode,
                num_steps=num_steps,
                device=device,
                sigma_min=0.1,
            )
            preds = fm.sample_ode(z_init=z_init, N=num_steps, use_sphere=use_sphere)[-1]

            # Make unitary and normalize
            preds = make_unitary(preds)
            preds = preds / preds.norm(dim=1, keepdim=True)

            # Decode: find nearest grid point
            sims = preds @ grid_ssps.T  # (B, G)
            idx = sims.argmax(dim=1)
            decoded_pts = grid_pts_tensor[idx]  # (B, 2)

            # Also get ground truth points by decoding z1
            sims_gt = z1 @ grid_ssps.T
            idx_gt = sims_gt.argmax(dim=1)
            gt_pts = grid_pts_tensor[idx_gt]  # (B, 2)

            all_gt_pts.append(gt_pts.cpu())
            all_decoded_pts.append(decoded_pts.cpu())

            # Cosine similarity
            cos = torch.sum(preds * z1, dim=1)
            all_cosines.append(cos.cpu())

    gt_all = torch.cat(all_gt_pts).numpy()
    decoded_all = torch.cat(all_decoded_pts).numpy()
    cos_all = torch.cat(all_cosines).numpy()

    # Compute RMSE
    diffs = decoded_all - gt_all
    rmses = np.linalg.norm(diffs, axis=1)

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
    cos_m, cos_std, cos_ci = stats(cos_all)

    return {
        "rmse_mean": float(rmse_m),
        "rmse_std": float(rmse_std),
        "rmse_ci95": float(rmse_ci),
        "cosine_mean": float(cos_m),
        "cosine_std": float(cos_std),
        "cosine_ci95": float(cos_ci),
        "num_samples": len(rmses),
    }


def main():
    parser = argparse.ArgumentParser(description="SLAM evaluation for SSP cleanup")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--checkpoint-geo", type=str,
                        default="/autosota_cache/checkpoints/hex_dim1015_ls0p2_bounds_m1_1__m1_1/drift_geo_det.pt",
                        help="Path to Geodesic model checkpoint")
    parser.add_argument("--checkpoint-euc", type=str,
                        default=None,
                        help="Path to Euclidean model checkpoint")
    parser.add_argument("--num-steps", type=int, default=10,
                        help="Number of ODE integration steps")
    parser.add_argument("--signal-strength", type=float, default=0.0,
                        help="Signal strength (0=pure noise, 1=clean)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit evaluation samples (None=all test samples)")
    parser.add_argument("--time-embed-dim", type=int, default=32,
                        help="Time embedding dimension (must match training config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    # Load config
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
        ssp_cfg,
        domain_dim=int(ssp_cfg.get("domain_dim", 2)),
        domain_bounds=domain_bounds,
    )

    print(f"SSP space: dim={ssp_space.ssp_dim}, domain_dim={ssp_space.domain_dim}")
    print(f"Test dir: {cfg['paths']['data_root']}")

    # Find test directory
    from cleanup_ssps.dataset_registry import dataset_group_dirname
    group_dir = dataset_group_dirname(
        enc_dim, float(ssp_cfg["length_scale"]), domain_bounds,
        bundle_type=str(ssp_cfg.get("bundle_type", "hexagonal")),
    )
    test_dir = str(Path(cfg["paths"]["data_root"]) / group_dir / "test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}

    # Evaluate Geodesic model
    print(f"\n{'='*60}")
    print(f"Evaluating Geodesic model: {args.checkpoint_geo}")
    print(f"Steps: {args.num_steps}, Signal strength: {args.signal_strength}")
    model_geo = load_model(args.checkpoint_geo, ssp_space.ssp_dim, flow=True, device=device, time_embed_dim=args.time_embed_dim)
    geo_result = evaluate_rmse(
        model_geo, ssp_space, test_dir,
        sampling_mode="geo_det",
        num_steps=args.num_steps,
        signal_strength=args.signal_strength,
        noise_type=cfg["trainer"]["noise_type"],
        target_type=cfg["trainer"]["target_type"],
        batch_size=args.batch_size,
        device=device,
        num_samples=args.num_samples,
    )
    results["geo_det"] = geo_result
    print(f"  RMSE: {geo_result['rmse_mean']:.6f} ± {geo_result['rmse_std']:.6f} (CI95: {geo_result['rmse_ci95']:.6f})")
    print(f"  Cosine: {geo_result['cosine_mean']:.6f} ± {geo_result['cosine_std']:.6f}")
    print(f"  Samples: {geo_result['num_samples']}")

    # Evaluate Euclidean model if provided
    if args.checkpoint_euc:
        print(f"\n{'='*60}")
        print(f"Evaluating Euclidean model: {args.checkpoint_euc}")
        model_euc = load_model(args.checkpoint_euc, ssp_space.ssp_dim, flow=True, device=device, time_embed_dim=args.time_embed_dim)
        euc_result = evaluate_rmse(
            model_euc, ssp_space, test_dir,
            sampling_mode="euc_det",
            num_steps=args.num_steps,
            signal_strength=args.signal_strength,
            noise_type=cfg["trainer"]["noise_type"],
            target_type=cfg["trainer"]["target_type"],
            batch_size=args.batch_size,
            device=device,
            num_samples=args.num_samples,
        )
        results["euc_det"] = euc_result
        print(f"  RMSE: {euc_result['rmse_mean']:.6f} ± {euc_result['rmse_std']:.6f} (CI95: {euc_result['rmse_ci95']:.6f})")
        print(f"  Cosine: {euc_result['cosine_mean']:.6f} ± {euc_result['cosine_std']:.6f}")

    # Also compute baseline
    print(f"\n{'='*60}")
    print("Computing grid-based cleanup baseline...")
    from utils.evaluation_utils import compute_cleanup_baseline
    bl = compute_cleanup_baseline(
        ssp_space,
        ssp_dim=ssp_space.ssp_dim,
        snr=args.signal_strength,
        grid_resolution=128,
        method='grid',
        num_trials=2000,
        device=device,
    )
    results["baseline"] = bl
    print(f"  Baseline RMSE: {bl['mean_rmse']:.6f} ± {bl['std_rmse']:.6f}")
    print(f"  Baseline Cosine: {bl['mean_cosine']:.6f} ± {bl['std_cosine']:.6f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    main()
