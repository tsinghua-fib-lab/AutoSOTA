"""
Evaluate Grad MAE for a trained SDF model.

Grad MAE = mean(|grad u_pred - grad u_gt|) / 3 on { x : |sdf_gt(x)| < band }.

The ground-truth field comes from `data/meshes/bunny_sdf_gt.pt` (precomputed
256^3 grid: sdf, grad_x/y/z, grid_x/y/z). The model is sampled at the same
coordinates, its analytic gradients are computed in one forward pass, then
compared to GT on the surface band.

Usage:
    python scripts/eval_sdf_grad_mae.py \\
        --ckpt results/sdf3d_bunny/model.pth \\
        --gt data/meshes/bunny_sdf_gt.pt \\
        --band 0.01
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# Ensure repo root is importable for hermite_ngp package + CUDA extensions
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

# Import torch BEFORE the CUDA extensions so libc10 is loaded
import hermite_encoding_cuda_3d  # noqa: F401
import hermite_mlp_cuda_3d_v2  # noqa: F401

# Re-use the model defined in the training script
sys.path.insert(0, os.path.join(REPO_ROOT, "examples"))
from sdf3d_bunny import HermiteNGP_SDF_CUDA_3D  # type: ignore


def build_default_config(ckpt_path, hidden=128, layers=2, omega=30.0,
                         n_levels=8, log2_hashmap_size=16,
                         per_level_scale=2.0, base_resolution=4):
    """Construct the same config dict the training script uses."""
    return {
        "hidden_dim": hidden,
        "n_layers": layers,
        "omega": omega,
        "n_levels": n_levels,
        "log2_hashmap_size": log2_hashmap_size,
        "per_level_scale": per_level_scale,
        "base_resolution": base_resolution,
    }


@torch.no_grad()
def evaluate_grad_mae(model, gt, band=0.01, chunk=131072, device="cuda"):
    """Sweep the full 256^3 GT grid through `model.forward_with_gradient`."""
    R = int(gt["resolution"])
    gx = gt["grid_x"].to(device)  # (R,)
    gy = gt["grid_y"].to(device)
    gz = gt["grid_z"].to(device)
    sdf_gt = gt["sdf"].to(device)
    grad_gt = torch.stack(
        [gt["grad_x"].to(device), gt["grad_y"].to(device), gt["grad_z"].to(device)],
        dim=-1,
    )  # (R, R, R, 3)

    # Build all coords once (256^3 = 16.7M points).
    X, Y, Z = torch.meshgrid(gx, gy, gz, indexing="ij")
    coords = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # (N, 3)
    N = coords.shape[0]

    u_pred = torch.empty(N, device=device)
    grad_pred = torch.empty((N, 3), device=device)

    for i in range(0, N, chunk):
        x = coords[i : i + chunk]
        # Model forward_with_gradient returns (u, du_dx, du_dy, du_dz)
        u, gxp, gyp, gzp = model.forward_with_gradient(x)
        u_pred[i : i + chunk] = u.squeeze(-1) if u.dim() > 1 else u
        grad_pred[i : i + chunk, 0] = gxp.squeeze(-1) if gxp.dim() > 1 else gxp
        grad_pred[i : i + chunk, 1] = gyp.squeeze(-1) if gyp.dim() > 1 else gyp
        grad_pred[i : i + chunk, 2] = gzp.squeeze(-1) if gzp.dim() > 1 else gzp

    sdf_pred = u_pred.reshape(R, R, R)
    grad_pred = grad_pred.reshape(R, R, R, 3)

    sdf_mae = (sdf_pred - sdf_gt).abs().mean().item()

    # Surface band mask
    mask = sdf_gt.abs() < band

    if mask.sum() == 0:
        raise RuntimeError(f"No points within band {band}")

    diff = (grad_pred - grad_gt).abs()  # (R, R, R, 3)
    l1_band = diff[mask].sum() / (mask.sum() * 3)
    l1_band = l1_band.item()
    l1_full = diff.sum() / (diff.numel())
    l1_full = l1_full.item()

    # Direction error 1 - cos sim
    g_pred_band = grad_pred[mask]  # (M, 3)
    g_gt_band = grad_gt[mask]
    pn = g_pred_band.norm(dim=-1) + 1e-8
    gn = g_gt_band.norm(dim=-1) + 1e-8
    cos = (g_pred_band * g_gt_band).sum(dim=-1) / (pn * gn)
    dir_err = (1 - cos).mean().item()

    return {
        "sdf_mae": sdf_mae,
        "grad_mae_band": l1_band,
        "grad_mae_full": l1_full,
        "dir_err_band": dir_err,
        "n_band_points": int(mask.sum().item()),
        "band": band,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to model.pth")
    p.add_argument("--gt", default="data/meshes/bunny_sdf_gt.pt",
                   help="Ground truth SDF .pt file")
    p.add_argument("--band", type=float, default=0.01,
                   help="Surface band |sdf_gt| < band for Grad MAE")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--omega", type=float, default=30.0)
    p.add_argument("--n-levels", type=int, default=8)
    p.add_argument("--log2-hashmap-size", type=int, default=16)
    args = p.parse_args()

    device = "cuda"
    torch.set_grad_enabled(False)

    # Build model with matching arch
    cfg = build_default_config(
        args.ckpt,
        hidden=args.hidden,
        layers=args.layers,
        omega=args.omega,
        n_levels=args.n_levels,
        log2_hashmap_size=args.log2_hashmap_size,
    )
    model = HermiteNGP_SDF_CUDA_3D(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    # Handle EMA/best-state vs raw state_dict
    if all(k.startswith("encoding") or k.startswith("mlp") or k.startswith("layers") for k in state):
        model.load_state_dict(state)
    else:
        # assume state is itself a state dict
        model.load_state_dict(state)

    model.eval()

    print(f"Loaded {args.ckpt}")
    print(f"GT  {args.gt}")
    gt = torch.load(args.gt, map_location="cpu", weights_only=False)
    print(f"GT grid {gt['resolution']}^3")

    t0 = time.time()
    out = evaluate_grad_mae(model, gt, band=args.band, device=device)
    dt = time.time() - t0
    print(f"Eval time: {dt:.1f} s ({out['n_band_points']:,} band points)")
    print()
    print(f"  SDF MAE:                {out['sdf_mae']:.6f}")
    print(f"  Grad MAE  (band {args.band}):  {out['grad_mae_band']:.6f}")
    print(f"  Grad MAE  (full grid):  {out['grad_mae_full']:.6f}")
    print(f"  Direction err (band):   {out['dir_err_band']:.6f}")

    # Save a JSON next to the ckpt
    out["ckpt"] = args.ckpt
    out["gt"] = args.gt
    import json
    out_json = os.path.splitext(args.ckpt)[0] + "_grad_mae.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
