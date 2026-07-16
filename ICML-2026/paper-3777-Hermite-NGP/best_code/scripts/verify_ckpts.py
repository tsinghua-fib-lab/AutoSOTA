"""Verify pretrained checkpoints against their expected metrics.

For each paper experiment, load `<tag>_model.pth`, read the headline
metric from the companion `<tag>_result.npz` / `<tag>_metrics.json` /
field-dump npz, and compare to the expected value within tolerance.

Usage:
    python scripts/verify_ckpts.py
    python scripts/verify_ckpts.py --ckpts /path/to/ckpts_release
    python scripts/verify_ckpts.py --headers-only
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Expected README headline metric per experiment. Tolerance is *relative*
# (i.e., |obs - exp| / exp <= TOL).
EXPECTED = {
    "01_helm2d_a10":     {"metric": "L2",   "value": 1.58e-5},
    "02_helm2d_a20":     {"metric": "L2",   "value": 7.12e-5},
    "03_helm2d_a100":    {"metric": "L2",   "value": 3.81e-2},
    "04_helm3d_a3":      {"metric": "L2",   "value": 6.10e-5},
    "05_helm3d_a10":     {"metric": "L2",   "value": 6.51e-3},
    "06_conv1d_c30":     {"metric": "L2",   "value": 8.84e-5},
    "07_taylor_green":   {"metric": "L2",   "value": 7.15e-5},
    "08_flow_mixing":    {"metric": "L2",   "value": 1.49e-4},
    "09_poisson3d_bunny":{"metric": "MAE",  "value": 4.69e-3},
    "10_sdf3d_bunny":    {"metric": "GradMAE","value": 0.0393},
    "11_image_recon_256":{"metric": "PSNR", "value": 32.58},
    "12_image_recon_512":{"metric": "PSNR", "value": 32.39},
}
TOL_REL = 0.05  # 5% relative tolerance on bundled-metric vs expected


def _read_npz_metric(npz_path: Path, metric: str, tag: str = ""):
    """Pull the headline metric value out of a result.npz file."""
    d = np.load(npz_path, allow_pickle=True)
    if metric == "L2":
        # Helm2D / Conv save final_l2; Helm3D / Taylor-Green / Flow-Mixing
        # only have history-tracked best_l2.
        if tag.startswith(("01_", "02_", "03_", "06_")) and "final_l2" in d.files:
            return float(d["final_l2"]), "final_l2"
        if "best_l2" in d.files:
            return float(d["best_l2"]), "best_l2"
        if "final_l2" in d.files:
            return float(d["final_l2"]), "final_l2"
    elif metric == "MAE":
        # Poisson: MAE is not in result.npz. Verify via post-hoc eval against
        # bunny_gt_volume_256.npy.
        if "best_mae" in d.files:
            return float(d["best_mae"]), "best_mae"
    elif metric == "GradMAE":
        # SDF: read from metrics.json (grad_mae_band at the requested band).
        return None, None
    return None, None


def _ckpt_summary(pth_path: Path):
    state = torch.load(pth_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        return None, f"unexpected ckpt type: {type(state).__name__}"
    n_params = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
    return n_params, next(iter(state.keys()), "(empty)")


def _recompute_l2_from_npz(npz_path: Path):
    """Helm2D/Conv etc. store u_pred + u_exact in result.npz; recompute L2."""
    d = np.load(npz_path, allow_pickle=True)
    if "u_pred" in d.files and "u_exact" in d.files:
        u_p = np.asarray(d["u_pred"]).flatten()
        u_e = np.asarray(d["u_exact"]).flatten()
        if u_p.size and u_e.size and u_p.shape == u_e.shape:
            return float(np.linalg.norm(u_p - u_e) / (np.linalg.norm(u_e) + 1e-30))
    return None


def _recompute_poisson_mae(ckpt_path: Path, gt_volume_path: Path | None):
    """Load the Poisson model + GT volume and return MAE on the is_solve mask.

    Requires `bunny_gt_volume_256.npy` (FD ground truth) at
    `data/meshes/bunny_gt_volume_256.npy`.
    """
    if gt_volume_path is None or not gt_volume_path.exists():
        return None
    try:
        import sys
        # Need the script's model class + CUDA extensions
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import hermite_encoding_cuda_3d  # noqa: F401
        import hermite_mlp_cuda_3d_v2  # noqa: F401
        from poisson3d_bunny import (
            HermiteNGP_PINN_DomainBC, KaolinMeshSamplerWithDomainBC,
            compute_l2_error, load_gt_volume,
        )
    except Exception as e:
        return None  # CUDA / module not available

    mesh_path = Path(__file__).resolve().parent.parent / "data" / "meshes" / "bunny.ply"
    if not mesh_path.exists():
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampler = KaolinMeshSamplerWithDomainBC(
        str(mesh_path), mesh_bc_value=1.0, domain_bc_value=0.0, device=device,
    )
    cfg = {"n_levels": 8, "log2_hashmap_size": 16, "hidden_dim": 128,
           "n_layers": 2, "omega": 0.2,
           "phases": [(0, float("inf"), list(range(8)))],
           "mesh_bc_weight_scale": 1.0,
           "n_bc_mesh_samples": 5000, "n_bc_domain_samples": 5000}
    model = HermiteNGP_PINN_DomainBC(sampler, config=cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    gt = load_gt_volume(str(mesh_path), str(gt_volume_path.parent))
    if gt is None:
        return None
    valid = ~np.isnan(gt)
    _, u_pred = compute_l2_error(model, gt, device=device)
    mae = float(np.abs(u_pred[valid] - gt[valid]).mean())
    return mae


def _recompute_psnr_field(npz_path: Path, image_res: int):
    """Image-recon stores u (predicted) in field-dump npz. Compare to camera GT."""
    import skimage.data
    from PIL import Image

    d = np.load(npz_path, allow_pickle=True)
    if "u" not in d.files:
        return None
    u = np.clip(np.asarray(d["u"], dtype=np.float32), 0.0, 1.0)
    # GT exactly the way image_recon.py loads it
    if image_res == 512:
        img = np.array(skimage.data.camera()).astype(np.float32) / 255.0
    else:
        img = np.array(Image.fromarray(skimage.data.camera()).resize((image_res, image_res)))
        img = img.astype(np.float32) / 255.0
    if u.shape != img.shape:
        return None
    mse = float(((u - img) ** 2).mean())
    return -10.0 * np.log10(mse + 1e-30) if mse > 0 else float("inf")


def verify(ckpts_dir: Path, headers_only: bool = False) -> int:
    rows = []
    n_ok = 0
    n_warn = 0
    n_err = 0

    for tag, exp in EXPECTED.items():
        pth = ckpts_dir / f"{tag}_model.pth"
        npz = ckpts_dir / f"{tag}_result.npz"
        fld = ckpts_dir / f"{tag}_final.npz"
        json_path = ckpts_dir / f"{tag}_metrics.json"

        if not pth.exists():
            rows.append((tag, "-", "-", "-", "❌ no .pth"))
            n_err += 1
            continue

        # 1) ckpt loads + params
        n_params, first_key = _ckpt_summary(pth)
        if n_params is None:
            rows.append((tag, "-", "-", "-", f"❌ {first_key}"))
            n_err += 1
            continue

        # 2) bundled metric from npz/json
        bundled = None
        bundled_src = None
        if npz.exists():
            bundled, bundled_src = _read_npz_metric(npz, exp["metric"], tag)
        if bundled is None and json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)
            # SDF has multiple grad MAEs; the paper's number is grad_mae_band
            for k in ("grad_mae_band", "eikonal_mae_offsurf", "grad_mae",
                      "best_grad_mae", "best_mae"):
                if k in meta:
                    bundled = float(meta[k])
                    bundled_src = k
                    break

        # 3) optional recompute. For Helm3D, Taylor-Green and Flow-Mixing the
        # saved u_pred/u_exact is at a coarser grid than the training-time
        # eval, so trust the npz's best_l2 directly.
        skip_recompute = tag.startswith(("04_", "05_", "07_", "08_", "09_", "10_"))
        recomputed = None
        if not headers_only and npz.exists() and not skip_recompute:
            recomputed = _recompute_l2_from_npz(npz)
        if not headers_only and fld.exists() and tag.startswith(("11_", "12_")):
            recomputed = _recompute_psnr_field(fld, image_res=256 if "256" in tag else 512)
        if not headers_only and tag == "09_poisson3d_bunny":
            # Post-hoc MAE eval needs GT volume; look in standard locations
            for cand in [
                Path(__file__).resolve().parent.parent / "data" / "meshes" / "bunny_gt_volume_256.npy",
                Path(__file__).resolve().parent.parent.parent / "3D" / "examples" / "gt_domainbc" / "bunny_gt_volume_256.npy",
            ]:
                if cand.exists():
                    recomputed = _recompute_poisson_mae(pth, cand)
                    break

        # 4) compare to expected
        observed = recomputed if recomputed is not None else bundled
        if observed is None:
            status = "⚠️ no metric"
            n_warn += 1
        else:
            rel = abs(observed - exp["value"]) / exp["value"]
            status = f"{'✅' if rel <= TOL_REL else '❌'} Δ={100*rel:+.1f}%"
            if rel <= TOL_REL:
                n_ok += 1
            else:
                n_err += 1

        rows.append((
            tag,
            f"{n_params:,}",
            f"{exp['value']:.3e}" if exp["metric"] != "PSNR" else f"{exp['value']:.2f}",
            f"{observed:.3e}" if observed is not None and exp["metric"] != "PSNR" else
            (f"{observed:.2f}" if observed is not None else "-"),
            status,
        ))

    # Print table
    print(f"{'experiment':<25} {'n_params':>10} {'expected':>12} {'observed':>12}   status")
    print("-" * 75)
    for r in rows:
        print(f"{r[0]:<25} {r[1]:>10} {r[2]:>12} {r[3]:>12}   {r[4]}")
    print("-" * 75)
    print(f"  {n_ok} OK,  {n_warn} no metric available,  {n_err} mismatch")
    return 0 if n_err == 0 else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent.parent
    p.add_argument("--ckpts", type=Path, default=here / "results" / "ckpts_release",
                   help="Path to the unpacked ckpts_release/ directory.")
    p.add_argument("--headers-only", action="store_true",
                   help="Skip the recompute path; only read bundled metrics.")
    args = p.parse_args()

    if not args.ckpts.is_dir():
        print(f"ERROR: {args.ckpts} does not exist. "
              f"Download from https://github.com/jinjinhe2001/hermite-ngp/releases/tag/v1.0-ckpts",
              file=sys.stderr)
        sys.exit(2)

    print(f"Verifying ckpts in {args.ckpts}")
    sys.exit(verify(args.ckpts, headers_only=args.headers_only))


if __name__ == "__main__":
    main()
