"""
Evaluate MAE for a trained Poisson 3D model against the FD ground-truth volume.

MAE = mean(|u_pred(x) - u_gt(x)|) on { x : is_solve(x) }, where is_solve marks
the interior PDE voxels (excludes the inside-mesh region and the BC strip).

The ground-truth comes from `data/meshes/bunny_gt_volume_256.npy`, a 256^3
NaN-masked volume produced by an FD Laplace solver. The model is sampled at
the same voxel grid, then the per-voxel absolute error is averaged on the
is_solve mask.

Usage:
    python scripts/eval_poisson_mae.py \\
        --ckpt examples/poisson_bunny_model.pt \\
        --gt   data/meshes/bunny_gt_volume_256.npy \\
        --mesh data/meshes/bunny.ply
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "examples"))  # for poisson3d_bunny import

import hermite_encoding_cuda_3d  # noqa: F401
import hermite_mlp_cuda_3d_v2  # noqa: F401

from poisson3d_bunny import (  # type: ignore
    HermiteNGP_PINN_DomainBC,
    KaolinMeshSamplerWithDomainBC,
    compute_l2_error,
)


def evaluate(ckpt_path, gt_path, mesh_path, device="cuda"):
    sampler = KaolinMeshSamplerWithDomainBC(
        mesh_path, mesh_bc_value=1.0, domain_bc_value=0.0, device=device,
    )
    cfg = {
        "n_levels": 8, "log2_hashmap_size": 16, "hidden_dim": 128,
        "n_layers": 2, "omega": 0.2,
        "phases": [(0, float("inf"), list(range(8)))],
        "mesh_bc_weight_scale": 1.0,
        "n_bc_mesh_samples": 5000, "n_bc_domain_samples": 5000,
    }
    model = HermiteNGP_PINN_DomainBC(sampler, config=cfg).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    gt = np.load(gt_path)
    valid = ~np.isnan(gt)
    rel_l2, u_pred = compute_l2_error(model, gt, device=device)
    mae = float(np.abs(u_pred[valid] - gt[valid]).mean())

    return {
        "mae": mae,
        "rel_l2": float(rel_l2),
        "n_valid_points": int(valid.sum()),
        "grid_resolution": int(gt.shape[0]),
        "ckpt": ckpt_path,
        "gt": gt_path,
        "mesh": mesh_path,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Path to trained Poisson model.pt")
    p.add_argument("--gt", default="data/meshes/bunny_gt_volume_256.npy",
                   help="Path to FD GT volume .npy (256^3 NaN-masked)")
    p.add_argument("--mesh", default="data/meshes/bunny.ply",
                   help="Mesh file used at training time")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    for path, label in [(args.ckpt, "ckpt"), (args.gt, "gt"), (args.mesh, "mesh")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found at {path}", file=sys.stderr)
            sys.exit(2)

    print(f"Loading {args.ckpt}")
    print(f"GT      {args.gt}")
    print(f"Mesh    {args.mesh}")

    out = evaluate(args.ckpt, args.gt, args.mesh, device=args.device)

    print(f"\nResults (on is_solve mask, {out['n_valid_points']:,} points):")
    print(f"  MAE:     {out['mae']:.6e}")
    print(f"  rel L2:  {out['rel_l2']:.6e}")

    out_json = os.path.splitext(args.ckpt)[0] + "_poisson_mae.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
