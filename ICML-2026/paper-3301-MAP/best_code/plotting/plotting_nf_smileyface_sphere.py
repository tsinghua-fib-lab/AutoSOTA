import os
import sys
import json
import math
import argparse
import numpy as np
import torch

# Force CPU-only to avoid CUDA/CPU type mismatches inside wrapped trainers
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Move to project root and ensure imports work
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from datasets import SmileyFaceDataset
from trainers import GlowTrainer, RealNVPTrainer, DDPMTrainer
from utils.constraints import SimpleConstraintProjector
from utils.plotting import (
    to_intrinsic_2d,
    _orthonormal_basis_from_pole,
    compute_shared_norm,
    save_standalone_colorbar,
    plot_2d_density_no_cbar,
)
from utils.metrics import coverage, jsd_histogram_2d, tvd_histogram_2d
from utils.timing import compute_avg_stats, _total_model_time, _total_proj_time


def load_unified_checkpoint(problem_dir: str, trainer_tag: str, epochs: int, noise_level: float, time_cond: str, seed: int, iso_only: bool = False):
    """Return (checkpoint_dict, path) or (None, attempted_path).

    If `iso_only` is True, only attempt filenames that include an ISO marker.
    Otherwise prefer non-ISO candidates but fall back to ISO variants if present.
    """
    effective_epochs = epochs
    # build candidate patterns (non-ISO first)
    candidates = []
    base = os.path.join(problem_dir, f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}_time_{time_cond}_seed_{seed}.pth")
    alt1 = os.path.join(problem_dir, f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}_time_{time_cond}.pth")
    alt2 = os.path.join(problem_dir, f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}.pth")
    candidates.extend([base, alt1, alt2])

    # ISO variants (try with _ISO_ and _iso_ placement before time/seed)
    iso_candidates = []
    iso_candidates.append(os.path.join(problem_dir, f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}_ISO_time_{time_cond}_seed_{seed}.pth"))
    iso_candidates.append(os.path.join(problem_dir, f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}_iso_time_{time_cond}_seed_{seed}.pth"))
    iso_candidates.append(os.path.join(problem_dir, f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}_ISO_time_{time_cond}.pth"))
    iso_candidates.append(os.path.join(problem_dir, f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}_iso_time_{time_cond}.pth"))
    iso_candidates.append(os.path.join(problem_dir, f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}_ISO.pth"))
    iso_candidates.append(os.path.join(problem_dir, f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}_iso.pth"))

    if iso_only:
        for c in iso_candidates:
            if os.path.exists(c):
                return torch.load(c, map_location="cpu"), c
        # not found -> return first iso candidate as attempted path
        return None, iso_candidates[0]

    # try non-ISO candidates first
    for c in candidates:
        if os.path.exists(c):
            return torch.load(c, map_location="cpu"), c
    # none found; return primary base as attempted path
    return None, base


def ensure_tensor_2d(x, D2=2):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    x = x.view(-1, D2)
    return x


def filter_valid_samples(x):
    # Drop rows with NaN/Inf
    mask = torch.isfinite(x).all(dim=1)
    return x[mask]


def lift_intrinsic_2d_to_sphere(uv: torch.Tensor, center: torch.Tensor, radius: float | torch.Tensor, pole: torch.Tensor | None = None) -> torch.Tensor:
    """Lift intrinsic 2D coordinates (Lambert azimuthal) back to 3D points on the sphere.

    Assumes the forward mapping used Lambert equal-area centered at `pole`.
    Given uv (N,2), compute (p_e1, p_e2, p_n) on the unit sphere frame and map back to 3D.
    """
    if not torch.is_tensor(uv):
        uv = torch.tensor(uv, dtype=torch.float32)
    if not torch.is_tensor(center):
        center = torch.tensor(center, dtype=torch.float32)
    if not torch.is_tensor(radius):
        radius = torch.tensor(radius, dtype=torch.float32)
    center = center.view(3)
    radius = radius.view(1)
    device = center.device
    uv = uv.to(device).view(-1, 2)
    pole = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device) if pole is None else pole.to(device)
    e1, e2, n_hat = _orthonormal_basis_from_pole(pole)
    a = uv[:, 0] / radius
    b = uv[:, 1] / radius
    R = a * a + b * b
    pn = (4.0 - 2.0 * R) / 4.0
    pn = pn.clamp(-1.0, 1.0)
    s = torch.clamp((1.0 + pn) / 2.0, min=0.0)
    scale = torch.sqrt(s)
    pe1 = a * scale
    pe2 = b * scale
    Xs = center + radius * (pe1.unsqueeze(1) * e1.unsqueeze(0) + pe2.unsqueeze(1) * e2.unsqueeze(0) + pn.unsqueeze(1) * n_hat.unsqueeze(0))
    return Xs


def main(seed: int = 42):
    # Configuration
    # Use CPU for projection and torch-based post-processing to avoid device mismatches
    # (Glow uses TF1 under-the-hood; keeping torch ops on CPU prevents CUDA/CPU type errors.)
    device = torch.device("cpu")

    noise_level = 0.05
    torch.manual_seed(seed)
    np.random.seed(seed)
    epochs = 40  # align with training runs to avoid mismatch
    time_cond = "default"
    sphere_center = [0.0, 0.0, 0.0]
    sphere_radius = 1.0

    # Dataset for dimensionality/reference (smiley face on sphere, lifted=False for ground truth intrinsic mapping)
    dataset = SmileyFaceDataset(
        num_samples=10000,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        projection_type="sphere",
        lifted=False,
        noise_level=0.0,
        device=device,
        seed=seed,
    )
    data_points = torch.stack([dataset[i] for i in range(len(dataset))])

    # Projector to enforce spherical constraint when doing "proj." variants
    projector = SimpleConstraintProjector(device)
    projector.add_constraints_from_dict({"sphere_equality": (sphere_center, sphere_radius)})

    models_dir = os.path.join("models", "smileyface_sphere")
    results_dir = os.path.join("results", "smileyface_sphere", "nf_noise_0.05")
    os.makedirs(results_dir, exist_ok=True)

    # Helper: load training args JSON saved by driver.py to match architectures
    def load_training_args(trainer_tag, epochs):
        args_path = os.path.join(models_dir, f"args_{trainer_tag}_epoch_{epochs}.json")
        if os.path.exists(args_path):
            try:
                with open(args_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}


    # Helper: build unfitted trainer instance with matching data shape
    def _instantiate_trainer(tag: str, data_np: np.ndarray, save_dir: str, epochs_inst: int, batch_size: int, hidden_dim_default: int, device: torch.device):
        if tag.upper() == "GLOW":
            # Use (N,1,1,D) reshape for vector data
            if data_np.ndim == 2:
                N, D = data_np.shape
                data_img = data_np.reshape(N, 1, 1, D)
            else:
                data_img = data_np
            return GlowTrainer(data_img, image_size=1, batch_size=batch_size, epochs=epochs_inst, save_dir=save_dir, device=device)
        if tag.upper() == "REALNVP":
            args_json = load_training_args("REALNVP", epochs)
            hidden_dim = int(args_json.get("hidden_dim", hidden_dim_default))
            n_layers = int(args_json.get("n_coupling_layers", 6))
            try:
                return RealNVPTrainer(data_np, batch_size=batch_size, epochs=epochs_inst, save_dir=save_dir, hidden_dim=hidden_dim, n_coupling_layers=n_layers, device=device)
            except TypeError:
                return RealNVPTrainer(data_np, batch_size=batch_size, epochs=epochs_inst, save_dir=save_dir, hidden_dim=hidden_dim, device=device)
        # Only GLOW and REALNVP are constructed
        raise ValueError(f"Unknown trainer tag {tag}")

    def _extract_state_dict(ckpt):
        if ckpt is None:
            return None
        if isinstance(ckpt, dict):
            for k in ("state_dict", "model_state_dict", "model_state", "state"):
                if k in ckpt:
                    return ckpt[k]
        if hasattr(ckpt, "keys") and all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt
        return None

    # Utility: verify checkpoint architecture compatibility
    def verify_state_dict_compat(model, state_dict):
        try:
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            missing = sorted(list(model_keys - ckpt_keys))
            unexpected = sorted(list(ckpt_keys - model_keys))
            shape_mismatch = []
            for k in (model_keys & ckpt_keys):
                mshape = tuple(model.state_dict()[k].shape)
                cshape = tuple(state_dict[k].shape)
                if mshape != cshape:
                    shape_mismatch.append((k, mshape, cshape))
            return {
                "missing": missing,
                "unexpected": unexpected,
                "shape_mismatch": shape_mismatch,
            }
        except Exception:
            return None

    # Samples/processing omitted (no additional baseline outputs)

    # --- Glow ---
    glow_samples = None
    try:
        base_np = data_points.cpu().numpy()
        glow_dir = os.path.join(models_dir, "glow")
        glow = _instantiate_trainer("GLOW", base_np, glow_dir, epochs_inst=1, batch_size=256, hidden_dim_default=64, device=torch.device("cpu"))
        try:
            unified_ckpt, unified_path = load_unified_checkpoint(models_dir, "GLOW", epochs, noise_level, time_cond, seed)
            sd = _extract_state_dict(unified_ckpt)
            if isinstance(sd, dict):
                report = verify_state_dict_compat(glow.model, sd)
                if report and (report["missing"] or report["unexpected"] or report["shape_mismatch"]):
                    print("[GLOW] Architecture mismatch detected:")
                    if report["missing"]:
                        print(f"  Missing keys: {report['missing'][:10]}{' ...' if len(report['missing'])>10 else ''}")
                    if report["unexpected"]:
                        print(f"  Unexpected keys: {report['unexpected'][:10]}{' ...' if len(report['unexpected'])>10 else ''}")
                    if report["shape_mismatch"]:
                        print(f"  Shape mismatches (first 5): {report['shape_mismatch'][:5]}")
                    raise RuntimeError("Glow checkpoint does not match constructed model architecture.")
                glow.model.load_state_dict(sd, strict=True)
                print(f"Loaded Glow PyTorch state_dict from {unified_path}")
        except Exception as e:
            print(f"Glow checkpoint load/verify failed: {e}")
        glow_samples, _ = glow.sample(num_samples=10000)
        # Save base glow samples; if checkpoint path indicates ISO, include suffix
        glow_iso_flag = True if (isinstance(unified_path, str) and ('iso' in unified_path.lower() or 'iso' in (unified_path or '').split('_'))) else False
        glow_samples_name = "glow_samples_iso.npy" if glow_iso_flag else "glow_samples.npy"
        np.save(os.path.join(results_dir, glow_samples_name), glow_samples)
        # Projected variant (apply projector to samples): ensure 3D
        try:
            xs2 = torch.tensor(glow_samples, dtype=torch.float32)
            xs2 = xs2.view(xs2.shape[0], -1)
            if xs2.shape[1] == 3:
                X3 = xs2
            elif xs2.shape[1] == 2:
                X3 = lift_intrinsic_2d_to_sphere(xs2, torch.tensor(sphere_center), torch.tensor(sphere_radius))
            elif xs2.shape[1] > 3:
                # If more dims, take first 3 as a fallback (rare)
                X3 = xs2[:, :3]
            else:
                # If dims <2, skip projection
                raise RuntimeError(f"Unexpected Glow sample shape {list(xs2.shape)} for projection")
            xs_proj, _, _ = projector.project(X3.to(device))
            glow_samples_proj = xs_proj.cpu().numpy()
            glow_samples_proj_name = "glow_samples_projected_iso.npy" if glow_iso_flag else "glow_samples_projected.npy"
            np.save(os.path.join(results_dir, glow_samples_proj_name), glow_samples_proj)
        except Exception as e:
            print(f"Glow projection failed: {e}")
        # Note: do not attempt to project 2D directly; lifting step above handles 2D -> 3D first.
        # Additionally try to load an ISO-only checkpoint variant (if present) and save its samples separately
        try:
            iso_ckpt, iso_path = load_unified_checkpoint(models_dir, "GLOW", epochs, noise_level, time_cond, seed, iso_only=True)
            if iso_ckpt is not None and iso_path != unified_path and os.path.exists(iso_path):
                try:
                    glow_iso_tr = _instantiate_trainer("GLOW", base_np, glow_dir, epochs_inst=1, batch_size=256, hidden_dim_default=64, device=torch.device("cpu"))
                    sd_iso = _extract_state_dict(iso_ckpt)
                    if isinstance(sd_iso, dict):
                        try:
                            glow_iso_tr.model.load_state_dict(sd_iso, strict=True)
                        except Exception:
                            pass
                    glow_samples_iso, _ = glow_iso_tr.sample(num_samples=10000)
                    np.save(os.path.join(results_dir, "glow_samples_iso.npy"), glow_samples_iso)
                    # Project iso samples similarly
                    xs2_iso = torch.tensor(glow_samples_iso, dtype=torch.float32).view(glow_samples_iso.shape[0], -1)
                    if xs2_iso.shape[1] == 3:
                        X3_iso = xs2_iso
                    elif xs2_iso.shape[1] == 2:
                        X3_iso = lift_intrinsic_2d_to_sphere(xs2_iso, torch.tensor(sphere_center), torch.tensor(sphere_radius))
                    elif xs2_iso.shape[1] > 3:
                        X3_iso = xs2_iso[:, :3]
                    else:
                        raise RuntimeError(f"Unexpected Glow ISO sample shape {list(xs2_iso.shape)} for projection")
                    xs_proj_iso, _, _ = projector.project(X3_iso.to(device))
                    glow_samples_proj_iso = xs_proj_iso.cpu().numpy()
                    np.save(os.path.join(results_dir, "glow_samples_projected_iso.npy"), glow_samples_proj_iso)
                except Exception as e:
                    print(f"Glow ISO sampling/processing failed: {e}")
        except Exception:
            pass
    except Exception as e:
        print(f"Glow load/sample failed: {e}")

    # --- RealNVP ---
    rnv_samples = None
    rnv_iso_flag = False
    try:
        # RealNVPTrainer was trained on original 3D vectors per driver.py
        X = data_points.cpu().numpy()
        rnv_dir = os.path.join(models_dir, "realnvp")
        os.makedirs(rnv_dir, exist_ok=True)
        rnv = _instantiate_trainer("REALNVP", X, rnv_dir, epochs_inst=1, batch_size=256, hidden_dim_default=64, device=torch.device("cpu"))
        unified, unified_path = load_unified_checkpoint(models_dir, "REALNVP", epochs, noise_level, time_cond, seed)
        if unified is not None and unified_path and os.path.exists(unified_path):
            try:
                sd = _extract_state_dict(unified)
                if isinstance(sd, dict):
                    report = verify_state_dict_compat(rnv.model, sd)
                    if report and (report["missing"] or report["unexpected"] or report["shape_mismatch"]):
                        print("[REALNVP] Architecture mismatch detected:")
                        if report["missing"]:
                            print(f"  Missing keys: {report['missing'][:10]}{' ...' if len(report['missing'])>10 else ''}")
                        if report["unexpected"]:
                            print(f"  Unexpected keys: {report['unexpected'][:10]}{' ...' if len(report['unexpected'])>10 else ''}")
                        if report["shape_mismatch"]:
                            print(f"  Shape mismatches (first 5): {report['shape_mismatch'][:5]}")
                        raise RuntimeError("RealNVP checkpoint does not match constructed model architecture.")
                    rnv.model.load_state_dict(sd, strict=True)
                    print(f"Loaded RealNVP PyTorch state_dict from {unified_path}")
                    rnv_iso_flag = True if (isinstance(unified_path, str) and ('iso' in unified_path.lower() or 'iso' in (unified_path or '').split('_'))) else False
            except Exception as e:
                print(f"RealNVP checkpoint load/verify failed: {e}")
        # Always attempt sampling after load (whether or not we hit the exception above)
        try:
            rnv_samples, _ = rnv.sample(num_samples=10000)
            rnv_samples_name = "realnvp_samples_iso.npy" if rnv_iso_flag else "realnvp_samples.npy"
            np.save(os.path.join(results_dir, rnv_samples_name), rnv_samples)
        except Exception as e:
            print(f"RealNVP sampling failed: {e}")
        # Projected variant: lift 2D to 3D before projecting
        try:
            if rnv_samples is None:
                raise RuntimeError("RealNVP samples missing; cannot project")
            xs2 = torch.tensor(rnv_samples, dtype=torch.float32)
            xs2 = xs2.view(xs2.shape[0], -1)
            X3 = lift_intrinsic_2d_to_sphere(xs2, torch.tensor(sphere_center), torch.tensor(sphere_radius))
            xs_proj, _, _ = projector.project(X3.to(device))
            rnv_samples_proj = xs_proj.cpu().numpy()
            rnv_samples_proj_name = "realnvp_samples_projected_iso.npy" if rnv_iso_flag else "realnvp_samples_projected.npy"
            np.save(os.path.join(results_dir, rnv_samples_proj_name), rnv_samples_proj)
        except Exception as e:
            print(f"RealNVP projection failed: {e}")
        # Try ISO-only RealNVP checkpoint sampling as a separate variant
        try:
            iso_ckpt_r, iso_path_r = load_unified_checkpoint(models_dir, "REALNVP", epochs, noise_level, time_cond, seed, iso_only=True)
            if iso_ckpt_r is not None and iso_path_r != unified_path and os.path.exists(iso_path_r):
                try:
                    rnv_iso_tr = _instantiate_trainer("REALNVP", X, rnv_dir, epochs_inst=1, batch_size=256, hidden_dim_default=64, device=torch.device("cpu"))
                    sd_iso_r = _extract_state_dict(iso_ckpt_r)
                    if isinstance(sd_iso_r, dict):
                        try:
                            rnv_iso_tr.model.load_state_dict(sd_iso_r, strict=True)
                        except Exception:
                            pass
                    rnv_samples_iso, _ = rnv_iso_tr.sample(num_samples=10000)
                    np.save(os.path.join(results_dir, "realnvp_samples_iso.npy"), rnv_samples_iso)
                    xs2 = torch.tensor(rnv_samples_iso, dtype=torch.float32).view(rnv_samples_iso.shape[0], -1)
                    X3_iso = lift_intrinsic_2d_to_sphere(xs2, torch.tensor(sphere_center), torch.tensor(sphere_radius))
                    xs_proj_iso, _, _ = projector.project(X3_iso.to(device))
                    rnv_samples_proj_iso = xs_proj_iso.cpu().numpy()
                    np.save(os.path.join(results_dir, "realnvp_samples_projected_iso.npy"), rnv_samples_proj_iso)
                except Exception as e:
                    print(f"RealNVP ISO sampling/processing failed: {e}")
        except Exception:
            pass
    except Exception as e:
        print(f"RealNVP load/sample failed: {e}")

    # --- Lifted variants for Glow and RealNVP ---
    # For lifted, we sample with a DDPMTrainer using lifted noise and then fit projection/on-manifold mapping for comparison.
    try:
        # Build lifted dataset on sphere with noise 0.05
        lifted_dataset = SmileyFaceDataset(
            num_samples=10000,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
            projection_type="sphere",
            lifted=True,
            noise_level=noise_level,
            device=device,
            seed=seed,
        )
        lifted_points = torch.stack([lifted_dataset[i] for i in range(len(lifted_dataset))])
        # Keep 3D domain for trainer instantiation to match training-time architectures
        lifted_X3d = lifted_points.view(lifted_points.shape[0], -1).cpu().numpy()

        # Glow lifted: train wrapper just to construct model, then try to restore from unified checkpoint
        glow_lift_dir = os.path.join(models_dir, "glow_lifted")
        os.makedirs(glow_lift_dir, exist_ok=True)
        glow_lift = _instantiate_trainer("GLOW", lifted_X3d, glow_lift_dir, epochs_inst=1, batch_size=256, hidden_dim_default=64, device=torch.device("cpu"))
        # Lifted checkpoints are saved with the base tag GLOW (no _LIFTED suffix)
        unified, unified_path = load_unified_checkpoint(models_dir, "GLOW", epochs, noise_level, time_cond, seed)
        # Also try ISO variant for lifted if present
        unified_iso, unified_iso_path = load_unified_checkpoint(models_dir, "GLOW", epochs, noise_level, time_cond, seed, iso_only=True)
        if unified is None or not (unified_path and os.path.exists(unified_path)):
            raise RuntimeError(
                f"Missing lifted Glow checkpoint: expected model_GLOW_epoch_{epochs}_noise_level_{noise_level}_time_{time_cond}_seed_{seed}.pth"
            )
        glow_lift.load_checkpoint(unified_path)
        glow_lift_samples, _ = glow_lift.sample(num_samples=10000)
        # Lifted samples are already lifted (on the sphere domain); do NOT re-lift or add noise.
        try:
            xs = torch.tensor(glow_lift_samples, dtype=torch.float32).view(glow_lift_samples.shape[0], -1)
            # If samples are 3D, project directly; if accidentally 2D, lift once.
            if xs.shape[1] >= 3:
                X3 = xs[:, :3]
            else:
                X3 = lift_intrinsic_2d_to_sphere(xs, torch.tensor(sphere_center), torch.tensor(sphere_radius))
            xs_proj, _, _ = projector.project(X3.to(device))
            glow_lift_samples_proj = xs_proj.cpu().numpy()
            np.save(os.path.join(results_dir, "glow_lifted_samples.npy"), glow_lift_samples_proj)
        except Exception as e:
            print(f"Glow lifted projection failed: {e}")

        # If there's an ISO-specific lifted checkpoint, sample and save its projected results too
        try:
            if unified_iso is not None and unified_iso_path and os.path.exists(unified_iso_path) and unified_iso_path != unified_path:
                try:
                    glow_lift_iso = _instantiate_trainer("GLOW", lifted_X3d, glow_lift_dir, epochs_inst=1, batch_size=256, hidden_dim_default=64, device=torch.device("cpu"))
                    glow_lift_iso.load_checkpoint(unified_iso_path)
                    glow_lift_samples_iso, _ = glow_lift_iso.sample(num_samples=10000)
                    xs_iso = torch.tensor(glow_lift_samples_iso, dtype=torch.float32).view(glow_lift_samples_iso.shape[0], -1)
                    if xs_iso.shape[1] >= 3:
                        X3_iso = xs_iso[:, :3]
                    else:
                        X3_iso = lift_intrinsic_2d_to_sphere(xs_iso, torch.tensor(sphere_center), torch.tensor(sphere_radius))
                    xs_proj_iso, _, _ = projector.project(X3_iso.to(device))
                    glow_lift_samples_proj_iso = xs_proj_iso.cpu().numpy()
                    np.save(os.path.join(results_dir, "glow_lifted_samples_iso.npy"), glow_lift_samples_proj_iso)
                except Exception as e:
                    print(f"Glow lifted ISO sampling failed: {e}")
        except Exception:
            pass

        # RealNVP lifted
        rnv_lift_dir = os.path.join(models_dir, "realnvp_lifted")
        os.makedirs(rnv_lift_dir, exist_ok=True)
        rnv_lift = _instantiate_trainer("REALNVP", lifted_X3d, rnv_lift_dir, epochs_inst=1, batch_size=256, hidden_dim_default=64, device=torch.device("cpu"))
        # Lifted checkpoints are saved with the base tag REALNVP (no _LIFTED suffix)
        unified, unified_path = load_unified_checkpoint(models_dir, "REALNVP", epochs, noise_level, time_cond, seed)
        unified_iso, unified_iso_path = load_unified_checkpoint(models_dir, "REALNVP", epochs, noise_level, time_cond, seed, iso_only=True)
        if unified is None or not (unified_path and os.path.exists(unified_path)):
            raise RuntimeError(
                f"Missing lifted RealNVP checkpoint: expected model_REALNVP_epoch_{epochs}_noise_level_{noise_level}_time_{time_cond}_seed_{seed}.pth"
            )
        rnv_lift.load_checkpoint(unified_path)
        rnv_lift_samples, _ = rnv_lift.sample(num_samples=10000)
        # Lifted samples are already lifted (on the sphere domain); do NOT re-lift or add noise.
        try:
            xs = torch.tensor(rnv_lift_samples, dtype=torch.float32).view(rnv_lift_samples.shape[0], -1)
            if xs.shape[1] >= 3:
                X3 = xs[:, :3]
            else:
                X3 = lift_intrinsic_2d_to_sphere(xs, torch.tensor(sphere_center), torch.tensor(sphere_radius))
            xs_proj, _, _ = projector.project(X3.to(device))
            rnv_lift_samples_proj = xs_proj.cpu().numpy()
            np.save(os.path.join(results_dir, "realnvp_lifted_samples.npy"), rnv_lift_samples_proj)
        except Exception as e:
            print(f"RealNVP lifted projection failed: {e}")

        # iso lifted variant
        try:
            if unified_iso is not None and unified_iso_path and os.path.exists(unified_iso_path) and unified_iso_path != unified_path:
                try:
                    rnv_lift_iso = _instantiate_trainer("REALNVP", lifted_X3d, rnv_lift_dir, epochs_inst=1, batch_size=256, hidden_dim_default=64, device=torch.device("cpu"))
                    rnv_lift_iso.load_checkpoint(unified_iso_path)
                    rnv_lift_samples_iso, _ = rnv_lift_iso.sample(num_samples=10000)
                    xs_iso = torch.tensor(rnv_lift_samples_iso, dtype=torch.float32).view(rnv_lift_samples_iso.shape[0], -1)
                    if xs_iso.shape[1] >= 3:
                        X3_iso = xs_iso[:, :3]
                    else:
                        X3_iso = lift_intrinsic_2d_to_sphere(xs_iso, torch.tensor(sphere_center), torch.tensor(sphere_radius))
                    xs_proj_iso, _, _ = projector.project(X3_iso.to(device))
                    rnv_lift_samples_proj_iso = xs_proj_iso.cpu().numpy()
                    np.save(os.path.join(results_dir, "realnvp_lifted_samples_iso.npy"), rnv_lift_samples_proj_iso)
                except Exception as e:
                    print(f"RealNVP lifted ISO sampling failed: {e}")
        except Exception:
            pass
    except Exception as e:
        print(f"Lifted variants failed: {e}")

    # Save a small manifest summarizing which arrays were written
    outputs = sorted([f for f in os.listdir(results_dir) if f.endswith('.npy')])
    manifest = {"noise_level": noise_level, "seed": seed, "outputs": outputs}
    with open(os.path.join(results_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- Metrics and plotting ----
    # Prepare ground-truth intrinsic 2D tensor
    with torch.no_grad():
        true_2d = to_intrinsic_2d(data_points.cpu(), torch.tensor(sphere_center), torch.tensor(sphere_radius))[0]
    true_2d = ensure_tensor_2d(true_2d, D2=2).cpu()
    true_2d = filter_valid_samples(true_2d)

    def load_np(name):
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            try:
                arr = np.load(path)
                return arr
            except Exception:
                return None
        return None

    def load_np_first(names):
        """Return the first successfully loaded numpy array from a list of filenames, else None."""
        for n in names:
            arr = load_np(n)
            if arr is not None:
                return arr
        return None

    # Explicitly include iso and non-iso projected variants; do NOT include raw ISO samples
    variants = {
        "Glow": load_np("glow_samples.npy"),
        "Glow_proj": load_np("glow_samples_projected.npy"),
        "Glow_proj_iso": load_np("glow_samples_projected_iso.npy"),
        "RealNVP": load_np("realnvp_samples.npy"),
        "RealNVP_proj": load_np("realnvp_samples_projected.npy"),
        "RealNVP_proj_iso": load_np("realnvp_samples_projected_iso.npy"),
        "Glow_lifted_proj": load_np("glow_lifted_samples.npy"),
        "RealNVP_lifted_proj": load_np("realnvp_lifted_samples.npy"),
    }

    metrics = {}
    # Build a projector to enforce spherical constraint prior to intrinsic mapping
    projector_metrics = SimpleConstraintProjector(torch.device("cpu"))
    projector_metrics.add_constraints_from_dict({"sphere_equality": (sphere_center, sphere_radius)})
    for name, arr in variants.items():
        if arr is None:
            metrics[name] = {"coverage": None, "jsd": None, "tvd": None}
            continue
        # Convert to intrinsic 2D if needed (Glow arrays might be image-like)
        try:
            x = torch.tensor(arr, dtype=torch.float32).view(arr.shape[0], -1)
            # Ensure 3D, then project to sphere before intrinsic mapping
            if x.shape[1] == 2:
                X3 = lift_intrinsic_2d_to_sphere(x, torch.tensor(sphere_center), torch.tensor(sphere_radius))
            elif x.shape[1] >= 3:
                X3 = x[:, :3]
            else:
                # Fallback: pad to 2D then lift
                pad = torch.zeros((x.shape[0], max(0, 2 - x.shape[1])), dtype=x.dtype)
                x2 = torch.cat([x, pad], dim=1)[:, :2]
                X3 = lift_intrinsic_2d_to_sphere(x2, torch.tensor(sphere_center), torch.tensor(sphere_radius))
            X3p, _, _ = projector_metrics.project(X3.cpu())
            with torch.no_grad():
                x2d = to_intrinsic_2d(X3p.cpu(), torch.tensor(sphere_center), torch.tensor(sphere_radius))[0]
            x2d = ensure_tensor_2d(x2d, D2=2).cpu()
            x2d = filter_valid_samples(x2d)
        except Exception:
            metrics[name] = {"coverage": None, "jsd": None, "tvd": None}
            continue

        try:
            cov = float(coverage(true_2d, x2d))
        except Exception:
            cov = None
        try:
            jsd = float(jsd_histogram_2d(true_2d, x2d, bins=25))
        except Exception:
            jsd = None
        try:
            tvd = float(tvd_histogram_2d(true_2d, x2d, bins=25))
        except Exception:
            tvd = None
        metrics[name] = {"coverage": cov, "jsd": jsd, "tvd": tvd}

    with open(os.path.join(results_dir, "metrics_nf_noise_0.05.json"), "w") as f:
        json.dump({"noise_level": noise_level, "metrics": metrics}, f, indent=2)
    print(f"Metrics written to {os.path.join(results_dir, 'metrics_nf_noise_0.05.json')}")

    # ------------------------------------------------------------------
    # Extrinsic (ambient 3D) + Intrinsic (2D) separate metrics tables
    # ------------------------------------------------------------------
    # Build true ambient (on-sphere) tensor
    true_ambient = data_points.view(-1, 3)
    true_ambient = filter_valid_samples(true_ambient).cpu()

    # Prepare ambient sample tensors for each variant (lift/project as needed)
    ambient_samples = {}
    for name, arr in variants.items():
        if arr is None:
            ambient_samples[name] = None
            continue
        try:
            x = torch.tensor(arr, dtype=torch.float32).view(arr.shape[0], -1)
            # Ensure we have 3D points: if 2D, lift to sphere via Lambert inverse
            if x.shape[1] == 2:
                X3 = lift_intrinsic_2d_to_sphere(x, torch.tensor(sphere_center), torch.tensor(sphere_radius))
            elif x.shape[1] >= 3:
                X3 = x[:, :3]
            else:
                # Fallback: pad to 2 dims then lift
                pad = torch.zeros((x.shape[0], max(0, 2 - x.shape[1])), dtype=x.dtype)
                x2 = torch.cat([x, pad], dim=1)[:, :2]
                X3 = lift_intrinsic_2d_to_sphere(x2, torch.tensor(sphere_center), torch.tensor(sphere_radius))
            # Filter invalid rows
            ambient_samples[name] = filter_valid_samples(X3).cpu()
        except Exception:
            ambient_samples[name] = None

    # Generic N-D histogram metrics (ambient 3D here)
    def _nd_hist_metrics(x: torch.Tensor, y: torch.Tensor, bins: int = 50):
        try:
            if x is None or y is None or x.numel() == 0 or y.numel() == 0:
                return float('nan'), float('nan'), float('nan')
            X = x.cpu().numpy()
            Y = y.cpu().numpy()
            d = X.shape[1]
            edges = [np.linspace(Y[:, i].min(), Y[:, i].max(), bins + 1) for i in range(d)]
            Hx, _ = np.histogramdd(X, bins=edges)
            Hy, _ = np.histogramdd(Y, bins=edges)
            Hx = Hx.astype(np.float64)
            Hy = Hy.astype(np.float64)
            sum_x, sum_y = Hx.sum(), Hy.sum()
            if sum_x <= 0 or sum_y <= 0:
                return float('nan'), float('nan'), float('nan')
            Px = Hx / sum_x
            Py = Hy / sum_y
            true_mask = Py > 0
            model_hit = (Px > 0) & true_mask
            cov = float(model_hit.sum() / max(1, true_mask.sum()))
            M = 0.5 * (Px + Py)
            with np.errstate(divide='ignore', invalid='ignore'):
                def _kl(P, Q):
                    mask = (P > 0) & (Q > 0)
                    return np.sum(P[mask] * (np.log(P[mask] + 1e-12) - np.log(Q[mask] + 1e-12)))
                jsd_val = 0.5 * _kl(Px, M) + 0.5 * _kl(Py, M)
            tvd_val = 0.5 * np.abs(Px - Py).sum()
            return cov, float(jsd_val), float(tvd_val)
        except Exception:
            return float('nan'), float('nan'), float('nan')

    extrinsic_metrics = {}
    for name, arr in ambient_samples.items():
        c, j, t = _nd_hist_metrics(arr, true_ambient, bins=25)
        extrinsic_metrics[name] = {"coverage": c, "JSD": j, "TVD": t}

    intrinsic_metrics = {}
    # Normalize key names and copy intrinsic (2D) results
    for name, vals in metrics.items():
        intrinsic_metrics[name] = {"coverage": vals.get("coverage"), "JSD": vals.get("jsd"), "TVD": vals.get("tvd")}
    # Filter intrinsic table to only show projected variants
    intrinsic_metrics_filtered = {k: v for k, v in intrinsic_metrics.items() if ("proj" in k)}

    # Formatting helpers and table writer
    def _sci(val, sig=3):
        if not isinstance(val, (int, float)) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return "--"
        if val == 0:
            return "0"
        exp = int(math.floor(math.log10(abs(val))))
        mant = val / (10 ** exp)
        return f"{mant:.{sig}f}e{exp:+d}"

    def _tex_sci(val, sig=3):
        if not isinstance(val, (int, float)) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return "--"
        if val == 0:
            return "0"
        exp = int(math.floor(math.log10(abs(val))))
        mant = val / (10 ** exp)
        return f"{mant:.{sig}f} \\times 10^{{{exp}}}"  # LaTeX sci notation

    def _write_table(data: dict, out_prefix: str, timing_training_map: dict | None = None, timing_sampling_map: dict | None = None, include_timing: bool = True):
        SCI_THRESHOLD = 1e-3  # only use scientific notation for very small magnitudes
        def _pretty_method(name: str) -> str:
            low = name.lower()
            is_glow = low.startswith("glow")
            is_rnv = low.startswith("realnvp")
            is_iso = "iso" in low
            is_lifted = "lifted" in low
            is_proj = "proj" in low or "projected" in low
            # ISO projected variants use simple "METHOD (iso.)" labels
            if is_iso and is_proj and is_glow:
                return "Glow (iso.)"
            if is_iso and is_proj and is_rnv:
                return "RealNVP (iso.)"
            # Lifted (non-ISO) -> p_sigma
            if is_lifted and is_proj and is_glow and not is_iso:
                return r"Glow ($p_{\sigma}$, ours)"
            if is_lifted and is_proj and is_rnv and not is_iso:
                return r"RealNVP ($p_{\sigma}$, ours)"
            if is_proj and is_glow:
                return "Glow (proj.)"
            if is_proj and is_rnv:
                return "RealNVP (proj.)"
            return name
        # Row schema: either timing-first or metrics-only depending on include_timing
        rows = []
        for k, v in sorted(data.items()):
            if include_timing:
                train_t = timing_training_map.get(k) if timing_training_map else None
                sample_t = timing_sampling_map.get(k) if timing_sampling_map else None
                rows.append([
                    _pretty_method(k),
                    train_t,
                    sample_t,
                    v.get('coverage', float('nan')),
                    v.get('JSD', float('nan')),
                    v.get('TVD', float('nan')),
                ])
            else:
                rows.append([
                    _pretty_method(k),
                    v.get('coverage', float('nan')),
                    v.get('JSD', float('nan')),
                    v.get('TVD', float('nan')),
                ])
        import csv
        with open(f"{out_prefix}.csv", 'w', newline='') as cf:
            w = csv.writer(cf)
            if include_timing:
                w.writerow(["Method", "Train/Epoch (s)", "Sample/Total (s)", "Coverage", "JSD", "TVD"])
            else:
                w.writerow(["Method", "Coverage", "JSD", "TVD"])
            for r in rows:
                if include_timing:
                    train_v, sample_v, cov, jsd_v, tvd_v = r[1], r[2], r[3], r[4], r[5]
                else:
                    cov, jsd_v, tvd_v = r[1], r[2], r[3]
                def _fmt_csv(val):
                    if not isinstance(val, (int, float)) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                        return "--"
                    if val == 0:
                        return "0"
                    if abs(val) < SCI_THRESHOLD:
                        return _sci(val)
                    return f"{val:.4f}"
                cov_csv = _fmt_csv(cov)
                jsd_csv = _fmt_csv(jsd_v)
                tvd_csv = _fmt_csv(tvd_v)
                if include_timing:
                    train_csv = _fmt_csv(train_v) if train_v is not None else "--"
                    sample_csv = _fmt_csv(sample_v) if sample_v is not None else "--"
                    w.writerow([r[0], train_csv, sample_csv, cov_csv, jsd_csv, tvd_csv])
                else:
                    w.writerow([r[0], cov_csv, jsd_csv, tvd_csv])
        with open(f"{out_prefix}.tex", 'w') as tf:
            if include_timing:
                tf.write("\\begin{tabular}{lcccccc}\n")
                tf.write("Method & Train/Epoch (s) & Sample/Total (s) & Coverage & JSD & TVD \\ \\hline\n")
            else:
                tf.write("\\begin{tabular}{lccc}\n")
                tf.write("Method & Coverage & JSD & TVD \\ \\hline\n")
            for r in rows:
                if include_timing:
                    train_v, sample_v, cov, jsd_v, tvd_v = r[1], r[2], r[3], r[4], r[5]
                else:
                    cov, jsd_v, tvd_v = r[1], r[2], r[3]
                def _fmt_tex(val):
                    if not isinstance(val, (int, float)) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                        return "--"
                    if val == 0:
                        return "0"
                    if abs(val) < SCI_THRESHOLD:
                        exp = int(math.floor(math.log10(abs(val))))
                        mant = val / (10 ** exp)
                        return f"${mant:.3f} \\cdot 10^{{{exp}}}$"
                    return f"{val:.4f}"
                cov_tex = _fmt_tex(cov)
                jsd_tex = _fmt_tex(jsd_v)
                tvd_tex = _fmt_tex(tvd_v)
                if include_timing:
                    train_tex = _fmt_tex(train_v) if train_v is not None else "--"
                    sample_tex = _fmt_tex(sample_v) if sample_v is not None else "--"
                    tf.write(f"{r[0]} & {train_tex} & {sample_tex} & {cov_tex} & {jsd_tex} & {tvd_tex} \\\n")
                else:
                    tf.write(f"{r[0]} & {cov_tex} & {jsd_tex} & {tvd_tex} \\\n")
            tf.write("\\end{tabular}\n")

    # Timing: build trainers and compute sampling totals via compute_avg_stats; training per-epoch via checkpoint metadata
    # Trainers for baseline methods (Glow, RealNVP)
    timing_trainers = {}
    try:
        base_np = data_points.cpu().numpy()
    except Exception:
        base_np = data_points.numpy()
    # Instantiate and load unified checkpoints (noise_level=0.05 baseline used for this script)
    # Glow
    try:
        gl = _instantiate_trainer("GLOW", base_np, os.path.join(models_dir, "glow_timing"), epochs_inst=1, batch_size=256, hidden_dim_default=64, device=torch.device("cpu"))
        ck_gl, _ = load_unified_checkpoint(models_dir, "GLOW", epochs, noise_level, time_cond, seed)
        sd_gl = _extract_state_dict(ck_gl)
        if isinstance(sd_gl, dict):
            try:
                gl.model.load_state_dict(sd_gl, strict=False)
            except Exception:
                pass
        timing_trainers["Glow"] = gl
    except Exception:
        pass

    # (sampling_totals defaults for ISO-projected variants will be set after
    # sampling_totals is computed and projection overheads are applied.)
    # RealNVP
    try:
        rn = _instantiate_trainer("REALNVP", base_np.reshape(-1, 3), os.path.join(models_dir, "realnvp_timing"), epochs_inst=1, batch_size=256, hidden_dim_default=64, device=torch.device("cpu"))
        ck_rn, _ = load_unified_checkpoint(models_dir, "REALNVP", epochs, noise_level, time_cond, seed)
        sd_rn = _extract_state_dict(ck_rn)
        if isinstance(sd_rn, dict):
            try:
                rn.model.load_state_dict(sd_rn, strict=False)
            except Exception:
                pass
        timing_trainers["RealNVP"] = rn
    except Exception:
        pass

    # Compute sampling averages
    try:
        method_names = list(timing_trainers.keys())
        avg_stats = compute_avg_stats(method_names, timing_trainers, n_trials=3, num_samples=10000)
    except Exception as e:
        avg_stats = {}
        print(f"Timing avg_stats failed: {e}")

    # Sampling totals map
    sampling_totals = {k: (avg_stats.get(k, {}).get("s") if isinstance(avg_stats.get(k, {}).get("s", None), (int, float)) else None) for k in method_names}

    # Augment sampling totals with projection overhead for *_proj and *_lifted_proj variants.
    # We approximate projection overhead by timing a projector.project call on produced samples.
    import time as _time
    def _proj_overhead(arr):
        try:
            if arr is None:
                return None
            x = torch.tensor(arr, dtype=torch.float32).view(arr.shape[0], -1)
            if x.shape[1] == 2:
                X3 = lift_intrinsic_2d_to_sphere(x, torch.tensor(sphere_center), torch.tensor(sphere_radius))
            elif x.shape[1] >= 3:
                X3 = x[:, :3]
            else:
                pad = torch.zeros((x.shape[0], max(0, 2 - x.shape[1])), dtype=x.dtype)
                x2 = torch.cat([x, pad], dim=1)[:, :2]
                X3 = lift_intrinsic_2d_to_sphere(x2, torch.tensor(sphere_center), torch.tensor(sphere_radius))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = _time.perf_counter()
            _ = projector.project(X3.to(torch.device("cpu")))  # returns (proj, norms, counts)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = _time.perf_counter()
            return float(t1 - t0)
        except Exception:
            return None

    glow_proj_overhead = _proj_overhead(glow_samples)
    rnv_proj_overhead = _proj_overhead(rnv_samples)
    glow_lifted_proj_overhead = _proj_overhead(locals().get("glow_lift_samples", locals().get("glow_lift_samples_proj", None)))  # lifted already 3D
    rnv_lifted_proj_overhead = _proj_overhead(locals().get("rnv_lift_samples", locals().get("rnv_lift_samples_proj", None)))

    # Use existing sampling times for baseline + overhead for projected variants; lifted variants reuse their own sampling_time if available.
    try:
        if "Glow" in sampling_totals and glow_proj_overhead is not None:
            sampling_totals["Glow_proj"] = (sampling_totals.get("Glow") if sampling_totals.get("Glow") is not None else 0.0) + glow_proj_overhead
        if "RealNVP" in sampling_totals and rnv_proj_overhead is not None:
            sampling_totals["RealNVP_proj"] = (sampling_totals.get("RealNVP") if sampling_totals.get("RealNVP") is not None else 0.0) + rnv_proj_overhead
        # Lifted trainers may have been instantiated; pull their sampling_time attributes if present.
        if "glow_lift" in locals():
            base_lift_time = getattr(glow_lift, "sampling_time", None)
            if base_lift_time is not None and glow_lifted_proj_overhead is not None:
                sampling_totals["Glow_lifted_proj"] = base_lift_time + glow_lifted_proj_overhead
        if "rnv_lift" in locals():
            base_lift_time = getattr(rnv_lift, "sampling_time", None)
            if base_lift_time is not None and rnv_lifted_proj_overhead is not None:
                sampling_totals["RealNVP_lifted_proj"] = base_lift_time + rnv_lifted_proj_overhead
    except Exception:
        pass

    # Ensure ISO-projected sampling times inherit from base / projected totals
    try:
        sampling_totals.setdefault("Glow_proj_iso", sampling_totals.get("Glow_proj", sampling_totals.get("Glow")))
        sampling_totals.setdefault("RealNVP_proj_iso", sampling_totals.get("RealNVP_proj", sampling_totals.get("RealNVP")))
        sampling_totals.setdefault("Glow_lifted_proj_iso", sampling_totals.get("Glow_lifted_proj", sampling_totals.get("Glow_lifted_proj")))
        sampling_totals.setdefault("RealNVP_lifted_proj_iso", sampling_totals.get("RealNVP_lifted_proj", sampling_totals.get("RealNVP_lifted_proj")))
    except Exception:
        pass

    # Training per-epoch time map
    def _per_epoch_total(tr):
        etb = getattr(tr, "epoch_timing_breakdowns", None)
        if not etb:
            return None
        vals = []
        for d in etb:
            if not isinstance(d, dict):
                continue
            m = d.get("avg_model_forward_time", d.get("model_forward", None))
            b = d.get("avg_backprop_time", d.get("backprop", None))
            o = d.get("avg_other_time", d.get("other", None))
            parts = [x for x in [m, b, o] if isinstance(x, (int, float)) and np.isfinite(x)]
            if parts:
                vals.append(sum(parts))
        return float(np.mean(vals)) if vals else None
    training_epoch_map = {k: _per_epoch_total(timing_trainers.get(k)) for k in method_names}

    # Expand training_epoch_map so keys match extrinsic_metrics kinds (include projected/lifted variants)
    def _per_epoch_total_from_ckpt(tag: str, noise: float):
        try:
            ckpt, _ = load_unified_checkpoint(models_dir, tag, epochs, noise, time_cond, seed)
            if not isinstance(ckpt, dict):
                return None
            etb = ckpt.get("epoch_timing_breakdowns") or []
            if etb:
                m_list = [d.get("avg_model_forward_time", d.get("model_forward", np.nan)) for d in etb]
                b_list = [d.get("avg_backprop_time", d.get("backprop", np.nan)) for d in etb]
                o_list = [d.get("avg_other_time", d.get("other", np.nan)) for d in etb]
                vals = []
                for a, b, c in zip(m_list, b_list, o_list):
                    parts = [x for x in (a, b, c) if isinstance(x, (int, float)) and np.isfinite(x)]
                    if parts:
                        vals.append(sum(parts))
                return float(np.mean(vals)) if vals else None
            # Fallback to checkpoint-level aggregates
            def _asf(x):
                try:
                    return float(x)
                except Exception:
                    return float('nan')
            m_cand = _asf(ckpt.get('total_model_forward_sample_time') or ckpt.get('avg_model_forward_sample_time'))
            p_cand = _asf(ckpt.get('total_projection_sample_time') or ckpt.get('avg_projection_sample_time'))
            if np.isfinite(m_cand) or np.isfinite(p_cand):
                m = m_cand if np.isfinite(m_cand) else np.nan
                p = p_cand if np.isfinite(p_cand) else np.nan
                # try to get total from training_losses length or dedicated key
                total = ckpt.get('total_epoch_time') or ckpt.get('avg_epoch_time') or None
                if total is not None:
                    try:
                        rem = float(total) - (m if np.isfinite(m) else 0.0) - (p if np.isfinite(p) else 0.0)
                        o = rem if np.isfinite(rem) and rem > 0 else 0.0
                        return float((m if np.isfinite(m) else 0.0) + (p if np.isfinite(p) else 0.0) + o)
                    except Exception:
                        return None
            return None
        except Exception:
            return None

    expanded_training_map = {}
    for kind in extrinsic_metrics.keys():
        low = kind.lower()
        base_key = None
        if low.startswith("glow"):
            for tk in training_epoch_map.keys():
                if tk.upper().startswith("GLOW"):
                    base_key = tk
                    break
        elif low.startswith("realnvp"):
            for tk in training_epoch_map.keys():
                if tk.upper().startswith("REALNVP"):
                    base_key = tk
                    break
        val = training_epoch_map.get(base_key)
        if val is None:
            # try to recover from checkpoint metadata
            try:
                if low.startswith("glow"):
                    val = _per_epoch_total_from_ckpt("GLOW", noise_level)
                elif low.startswith("realnvp"):
                    val = _per_epoch_total_from_ckpt("REALNVP", noise_level)
            except Exception:
                val = None
        expanded_training_map[kind] = val

    # Ensure ISO-projected variants inherit per-epoch training times from their non-ISO counterparts
    try:
        expanded_training_map.setdefault("Glow_proj_iso", expanded_training_map.get("Glow_proj", expanded_training_map.get("Glow")))
        expanded_training_map.setdefault("RealNVP_proj_iso", expanded_training_map.get("RealNVP_proj", expanded_training_map.get("RealNVP")))
        expanded_training_map.setdefault("Glow_lifted_proj_iso", expanded_training_map.get("Glow_lifted_proj", expanded_training_map.get("Glow_lifted_proj")))
        expanded_training_map.setdefault("RealNVP_lifted_proj_iso", expanded_training_map.get("RealNVP_lifted_proj", expanded_training_map.get("RealNVP_lifted_proj")))
    except Exception:
        pass

    # Write tables with timing columns
    try:
        print("[debug] expanded_training_map for table:", expanded_training_map)
        _write_table(extrinsic_metrics, os.path.join(results_dir, "nf_extrinsic_metrics_table"), timing_training_map=expanded_training_map, timing_sampling_map=sampling_totals, include_timing=True)
        _write_table(intrinsic_metrics_filtered, os.path.join(results_dir, "nf_intrinsic_metrics_table"), include_timing=False)
        print("Saved extrinsic and intrinsic NF metrics tables (sphere).")
    except Exception as e:
        print(f"Failed to write sphere NF extrinsic/intrinsic tables: {e}")

    # -------------------------
    # Additional table: NaN/Inf counts and Avg. Dist. to M (for NF outputs)
    # Write a robust appended table using the in-memory variant mappings (variants, ambient_samples)
    try:
        def _compute_stats_from_np(orig_np, proj_np=None, pr=None):
            import numpy as _np
            if orig_np is None:
                return "n/a", "n/a"
            # Work with numpy raw samples: count NaN/Inf rows only
            try:
                if torch.is_tensor(orig_np):
                    orig_arr = orig_np.cpu().numpy()
                else:
                    orig_arr = _np.array(orig_np)
            except Exception:
                orig_arr = _np.array(orig_np)
            if orig_arr.size == 0:
                return 0, "n/a"
            if orig_arr.ndim == 1:
                orig_arr = orig_arr.reshape(-1, orig_arr.shape[0])
            finite_mask = _np.isfinite(orig_arr).all(axis=1)
            n_bad = int(orig_arr.shape[0] - int(finite_mask.sum()))
            valid_np = orig_arr[finite_mask]
            if valid_np.shape[0] == 0:
                return n_bad, "n/a"
            # If projector provided, prefer asking the projector for per-sample distances
            if pr is None:
                pr = globals().get('projector', None)
            if pr is not None:
                try:
                    X = torch.tensor(valid_np, device=getattr(pr, 'device', torch.device('cpu')), dtype=getattr(pr, 'dtype', torch.float32))
                    res = pr.project(X, return_details=True)
                except Exception:
                    try:
                        res = pr.project(torch.tensor(valid_np))
                    except Exception:
                        res = None
                if isinstance(res, tuple) and len(res) >= 2:
                    dist = res[1]
                    try:
                        if torch.is_tensor(dist):
                            dist_np = dist.cpu().numpy()
                        else:
                            dist_np = _np.array(dist)
                        if dist_np.size == 0:
                            return n_bad, "n/a"
                        return n_bad, float(_np.mean(dist_np))
                    except Exception:
                        return n_bad, "n/a"
                # If projector didn't return distances, fall through to proj_np alignment
            # If proj_np provided, compute mean distance between raw finite rows and proj rows (align by index)
            try:
                if torch.is_tensor(proj_np):
                    proj_arr = proj_np.cpu().numpy()
                else:
                    proj_arr = _np.array(proj_np)
            except Exception:
                proj_arr = _np.array(proj_np)
            if proj_arr.ndim == 1:
                proj_arr = proj_arr.reshape(-1, proj_arr.shape[0])
            min_rows = min(valid_np.shape[0], proj_arr.shape[0])
            if min_rows == 0:
                return n_bad, "n/a"
            diffs = valid_np[:min_rows] - proj_arr[:min_rows]
            dists = _np.linalg.norm(diffs, axis=1)
            return n_bad, float(_np.mean(dists))

        out_path = os.path.join(results_dir, "nf_extrinsic_metrics_table.tex")
        print("[debug] appending additional table to:", out_path, "exists?", os.path.exists(out_path))
        methods = list(extrinsic_metrics.keys())
        print("[debug] additional-table methods:", methods)
        rows = []
        for name in methods:
            try:
                arr = None
                proj = None
                # Prefer exact in-memory ambient_samples used for extrinsic metrics when available
                if 'ambient_samples' in locals() and ambient_samples.get(name) is not None:
                    arr = ambient_samples.get(name)
                # Special handling for projected variants: use base ambient as arr and projected ambient as proj
                if name.endswith('_proj'):
                    base = name[:-5]
                    if arr is None and 'ambient_samples' in locals() and ambient_samples.get(base) is not None:
                        arr = ambient_samples.get(base)
                    if 'ambient_samples' in locals() and ambient_samples.get(name) is not None:
                        proj = ambient_samples.get(name)
                    # fall back to variants if ambient not present
                    if arr is None and 'variants' in locals():
                        arr = variants.get(base)
                    if proj is None and 'variants' in locals():
                        proj = variants.get(name)
                else:
                    # Non-projected: if ambient_samples not present, prefer variant arrays
                    if arr is None and 'variants' in locals() and name in variants:
                        arr = variants.get(name)
                    # Try ambient_samples for proj counterpart if not already set
                    if proj is None and 'ambient_samples' in locals():
                        pt = ambient_samples.get(f"{name}_proj")
                        if pt is None:
                            pt = ambient_samples.get(f"{name}_projected")
                        if pt is not None:
                            proj = pt
                    if proj is None and 'variants' in locals():
                        temp = variants.get(f"{name}_proj")
                        proj = temp if temp is not None else variants.get(f"{name}_projected")
                # Fallback to common filenames if still missing
                if arr is None:
                    for c in (f"{name.lower()}_samples.npy", f"{name.lower()}_samples_projected.npy", f"{name.lower()}.npy"):
                        pth = os.path.join(results_dir, c)
                        if os.path.exists(pth):
                            try:
                                arr = np.load(pth)
                                break
                            except Exception:
                                arr = None
                if proj is None:
                    for c in (f"{name.lower()}_samples_projected.npy", f"{name.lower()}_projected.npy", f"{name.lower()}_proj.npy"):
                        pth = os.path.join(results_dir, c)
                        if os.path.exists(pth):
                            try:
                                proj = np.load(pth)
                                break
                            except Exception:
                                proj = None
                # compute stats for this row (helper accepts torch tensors or numpy arrays)
                n_bad, avg_dist = _compute_stats_from_np(arr, proj, locals().get('projector', None))
                rows.append((name, n_bad, avg_dist))
            except Exception as e:
                print(f"[debug] additional-table row failed for {name}: {e}")
                rows.append((name, "n/a", "n/a"))
        print("[debug] additional-table rows:", rows)

        # Append table to extrinsic metrics file
        try:
            with open(out_path, 'a') as f:
                f.write('\n% --- Additional table: NaN/Inf counts and Avg. Dist. to $\\mathcal{M}$ ---\n')
                f.write('\\begin{table}[ht]\\centering\\small\\begin{tabular}{lrr}\\toprule\n')
                f.write('Method & Num NaN/Inf & Avg. Dist. to $\\mathcal{M}$ \\\\ \\midrule\n')
                for name, n_bad, avg_dist in rows:
                    if isinstance(avg_dist, float):
                        s = f"{avg_dist:.3e}"
                        try:
                            mant, exp = s.split('e')
                            exp_int = int(exp)
                            avg_str = f"${mant}\\times10^{{{exp_int}}}$"
                        except Exception:
                            avg_str = s
                    else:
                        avg_str = str(avg_dist)
                    f.write("{} & {} & {} \\\\ \n".format(name, n_bad, avg_str))
                f.write('\\bottomrule\\end{tabular}\\end{table}\n')
        except Exception:
            pass
    except Exception:
        pass

    # Sampling + Training time breakdown plots (robust version adapted from plane script)
    try:
        import matplotlib.pyplot as plt

        # Helpers
        def _safe_float(x):
            try:
                if x is None:
                    return float('nan')
                return float(x)
            except Exception:
                return float('nan')

        def _is_finite(x):
            try:
                return np.isfinite(x)
            except Exception:
                try:
                    return math.isfinite(float(x))
                except Exception:
                    return False

        # Desired ordering for sphere variants (match tables/plots)
        order_keys = [
            "Glow",
            "Glow_proj",
            "Glow_lifted_proj",
            "RealNVP",
            "RealNVP_proj",
            "RealNVP_lifted_proj",
        ]

        def _display_label(k: str) -> str:
            low = k.lower()
            is_glow = low.startswith("glow")
            is_rnv = low.startswith("realnvp")
            lifted = "lifted" in low
            projected = "proj" in low
            is_iso = "iso" in low
            # ISO projected variants use simple "METHOD (iso.)" labels
            if is_iso and projected and is_glow:
                return "Glow (iso.)"
            if is_iso and projected and is_rnv:
                return "RealNVP (iso.)"
            if lifted and projected and is_glow:
                return "Glow ($p_{\\sigma}$, proj.)"
            if lifted and projected and is_rnv:
                return "RealNVP ($p_{\\sigma}$, proj.)"
            if projected and is_glow:
                return "Glow (proj.)"
            if projected and is_rnv:
                return "RealNVP (proj.)"
            if is_glow:
                return "Glow"
            if is_rnv:
                return "RealNVP"
            return k

        labels = []
        m_vals = []
        p_vals = []
        other_vals = []
        for kind in order_keys:
            if kind not in extrinsic_metrics:
                continue
            labels.append(_display_label(kind))
            # base trainer key (Glow/RealNVP)
            base = kind.split("_")[0]
            # Prefer measured model-forward time from the timing trainer if present
            measured_m = None
            tr_meas = None
            found_key = None
            # timing_trainers keys may be capitalized differently; find match
            for tk in list(timing_trainers.keys()):
                if tk.upper().startswith(base.upper()):
                    tr_meas = timing_trainers.get(tk)
                    break
            try:
                if tr_meas is not None:
                    measured_m = _total_model_time(tr_meas)
            except Exception:
                measured_m = None
            if measured_m is not None and np.isfinite(measured_m):
                base_m = _safe_float(measured_m)
            else:
                # avg_stats keys correspond to timing trainer keys; try to find a key match
                for k in list(avg_stats.keys()):
                    if k.upper().startswith(base.upper()):
                        found_key = k
                        break
                base_m = _safe_float(avg_stats.get(found_key, {}).get("m") if avg_stats else float('nan'))
            base_p = _safe_float(avg_stats.get(found_key, {}).get("p") if avg_stats and found_key else 0.0)
            total_sample = _safe_float(sampling_totals.get(kind))
            base_total = _safe_float(sampling_totals.get(base))
            # compute projection overhead beyond base trainer's projection time
            overhead = float('nan')
            if _is_finite(total_sample) and _is_finite(base_total):
                overhead = total_sample - base_total
            proj_comp = base_p if _is_finite(base_p) else 0.0
            if _is_finite(overhead) and overhead > 0:
                proj_comp = proj_comp + overhead
            if (not _is_finite(total_sample)) or (not _is_finite(base_m)):
                other = float('nan')
            else:
                other = total_sample - base_m - proj_comp
                if _is_finite(other):
                    other = max(0.0, other)
                else:
                    other = float('nan')
            m_vals.append(base_m)
            p_vals.append(proj_comp)
            other_vals.append(other)

        # Debug numeric arrays
        try:
            with open(os.path.join(results_dir, 'proj_debug.log'), 'a') as df:
                df.write("SAMPLING_BREAKDOWN:\n")
                df.write(f"labels={labels}\n")
                df.write(f"m_vals={m_vals}\n")
                df.write(f"p_vals={p_vals}\n")
                df.write(f"other_vals={other_vals}\n")
                totals = []
                for mv, pv, ov in zip(m_vals, p_vals, other_vals):
                    mvf = float(mv) if _is_finite(mv) else float('nan')
                    pvf = float(pv) if _is_finite(pv) else float('nan')
                    ovf = float(ov) if _is_finite(ov) else float('nan')
                    totals.append(mvf + pvf + ovf)
                df.write(f"totals={totals}\n")
        except Exception:
            pass

        print("Skipping bar plot output: nf_sampling_time_breakdown")
    except Exception as e:
        print(f"Sampling time breakdown plot failed: {e}")

    # Training time breakdown plot (skip if unavailable)
    try:
        if any(v is not None for v in training_epoch_map.values()):
            import matplotlib.pyplot as plt
            order_keys = [
                "Glow",
                "Glow_proj",
                "Glow_lifted_proj",
                "RealNVP",
                "RealNVP_proj",
                "RealNVP_lifted_proj",
            ]

            labels = []
            m_vals = []
            p_vals = []
            b_vals = []
            o_vals = []
            for kind in order_keys:
                if kind not in training_epoch_map:
                    continue
                labels.append(kind)
                tr_total = training_epoch_map.get(kind)
                m = p = b = o = np.nan
                source_used = 'none'
                try:
                    # Determine tag and noise (noise not encoded in these short keys; try tag-based loader)
                    tag = kind.split("_")[0]
                    # Attempt to load checkpoint tolerant to naming
                    ckpt, tried_path = load_unified_checkpoint(models_dir, tag, epochs, noise_level, time_cond, seed)
                    try:
                        with open(os.path.join(results_dir, "training_source.log"), "a") as tf:
                            tf.write(f"{kind}: attempted_ckpt_path={tried_path}, exists={os.path.exists(tried_path)}\n")
                    except Exception:
                        pass
                    if isinstance(ckpt, dict):
                        etb = ckpt.get("epoch_timing_breakdowns") or []
                        if etb:
                            m_list = [d.get("avg_model_forward_time", d.get("model_forward", np.nan)) for d in etb]
                            p_list = [d.get("project", d.get("proj", d.get("projection", np.nan))) for d in etb]
                            b_list = [d.get("avg_backprop_time", d.get("backprop", np.nan)) for d in etb]
                            o_list = [d.get("avg_other_time", d.get("other", np.nan)) for d in etb]
                            m = float(np.nanmean(m_list))
                            p = float(np.nanmean(p_list))
                            b = float(np.nanmean(b_list))
                            o = float(np.nanmean(o_list))
                            source_used = 'epoch_timing_breakdowns'
                        else:
                            def _asf(x):
                                try:
                                    return float(x)
                                except Exception:
                                    return float('nan')
                            m_cand = _asf(ckpt.get('total_model_forward_sample_time') or ckpt.get('avg_model_forward_sample_time'))
                            p_cand = _asf(ckpt.get('total_projection_sample_time') or ckpt.get('avg_projection_sample_time'))
                            if np.isfinite(m_cand) or np.isfinite(p_cand):
                                m = m_cand if np.isfinite(m_cand) else np.nan
                                p = p_cand if np.isfinite(p_cand) else np.nan
                                try:
                                    if tr_total is not None:
                                        rem = float(tr_total) - (m if np.isfinite(m) else 0.0) - (p if np.isfinite(p) else 0.0)
                                        o = rem if np.isfinite(rem) and rem > 0 else 0.0
                                except Exception:
                                    o = float('nan')
                                source_used = 'checkpoint_aggregates'
                except Exception:
                    pass
                def _safe_isfinite_local(val):
                    try:
                        return np.isfinite(float(val))
                    except Exception:
                        return False

                if source_used == 'none' and _safe_isfinite_local(tr_total):
                    m = 0.0
                    b = 0.0
                    o = float(tr_total)
                    source_used = 'total_only'
                try:
                    with open(os.path.join(results_dir, 'training_source.log'), 'a') as tf:
                        tf.write(f"{kind}: source={source_used}, m={m}, p={p}, b={b}, o={o}\n")
                except Exception:
                    pass
                m_vals.append(m)
                p_vals.append(p)
                b_vals.append(b)
                o_vals.append(o)

            # Convert labels to display form
            display_labels = [_display_label(k) for k in labels]
            print("Skipping bar plot output: nf training_time_breakdown")
    except Exception as e:
        print(f"Training time breakdown plot failed: {e}")

    # ---- Sanity diagnostics: projection and intrinsic mapping health ----
    try:
        diag = {}
        # Helper to compute sphere norm deviation stats
        def _sphere_dev_stats(arr):
            try:
                x = torch.tensor(arr, dtype=torch.float32).view(arr.shape[0], -1)
                if x.shape[1] == 2:
                    X3 = lift_intrinsic_2d_to_sphere(x, torch.tensor(sphere_center), torch.tensor(sphere_radius))
                else:
                    X3 = x[:, :3]
                r = torch.linalg.norm(X3, dim=1)
                dev = r - float(sphere_radius)
                return {
                    "count": int(X3.shape[0]),
                    "radius_mean": float(r.mean().item()),
                    "radius_std": float(r.std().item()),
                    "dev_mean": float(dev.mean().item()),
                    "dev_max_abs": float(torch.max(torch.abs(dev)).item()),
                }
            except Exception:
                return None

        # Helper to compute intrinsic 2D range stats (after projection)
        projector_diag = SimpleConstraintProjector(torch.device("cpu"))
        projector_diag.add_constraints_from_dict({"sphere_equality": (sphere_center, sphere_radius)})
        def _intrinsic_stats(arr):
            try:
                x = torch.tensor(arr, dtype=torch.float32).view(arr.shape[0], -1)
                if x.shape[1] == 2:
                    X3 = lift_intrinsic_2d_to_sphere(x, torch.tensor(sphere_center), torch.tensor(sphere_radius))
                else:
                    X3 = x[:, :3]
                X3p, _, _ = projector_diag.project(X3.cpu())
                with torch.no_grad():
                    uv = to_intrinsic_2d(X3p.cpu(), torch.tensor(sphere_center), torch.tensor(sphere_radius))[0]
                uv = ensure_tensor_2d(uv, D2=2).cpu()
                uv = filter_valid_samples(uv)
                if uv.shape[0] == 0:
                    return {"count": 0}
                return {
                    "count": int(uv.shape[0]),
                    "u_min": float(uv[:,0].min().item()),
                    "u_max": float(uv[:,0].max().item()),
                    "v_min": float(uv[:,1].min().item()),
                    "v_max": float(uv[:,1].max().item()),
                }
            except Exception:
                return None

        for name, arr in variants.items():
            if arr is None:
                diag[name] = {"present": False}
            else:
                diag[name] = {
                    "present": True,
                    "sphere_dev": _sphere_dev_stats(arr),
                    "intrinsic_range": _intrinsic_stats(arr),
                }
        with open(os.path.join(results_dir, "sanity_report_nf.json"), "w") as f:
            json.dump(diag, f, indent=2)
        print(f"Wrote sanity diagnostics to {os.path.join(results_dir, 'sanity_report_nf.json')}")
    except Exception as e:
        print(f"Sanity diagnostics failed: {e}")

    print("Skipping bar plot outputs: coverage_nf_bar.pdf, jsd_nf_bar.pdf, tvd_nf_bar.pdf")

    # ---- Density plots in intrinsic 2D (shared colorbar) ----
    try:
        import matplotlib.pyplot as plt
        # Build tensors for intrinsic 2D for all variants, ensuring (N,2)
        # Project to sphere before mapping to intrinsic 2D to ensure consistency.
        projector_for_plot = SimpleConstraintProjector(torch.device("cpu"))
        projector_for_plot.add_constraints_from_dict({"sphere_equality": (sphere_center, sphere_radius)})

        def _to_2d_tensor(arr):
            if arr is None:
                return torch.empty((0, 2))
            x = torch.tensor(arr, dtype=torch.float32)
            x = x.view(x.shape[0], -1)
            # Lift 2D to 3D when necessary
            if x.shape[1] == 2:
                X3 = lift_intrinsic_2d_to_sphere(x, torch.tensor(sphere_center), torch.tensor(sphere_radius))
            elif x.shape[1] >= 3:
                X3 = x[:, :3]
            else:
                # Fallback: duplicate dims to reach 2, then lift
                pad = torch.zeros((x.shape[0], max(0, 2 - x.shape[1])), dtype=x.dtype)
                x2 = torch.cat([x, pad], dim=1)[:, :2]
                X3 = lift_intrinsic_2d_to_sphere(x2, torch.tensor(sphere_center), torch.tensor(sphere_radius))
            # Project onto sphere manifold for consistency
            X3_proj, _, _ = projector_for_plot.project(X3.cpu())
            with torch.no_grad():
                x2d = to_intrinsic_2d(X3_proj.cpu(), torch.tensor(sphere_center), torch.tensor(sphere_radius))[0]
            x2d = ensure_tensor_2d(x2d, D2=2).cpu()
            x2d = filter_valid_samples(x2d)
            return x2d

        true_tensor_2d = true_2d
        glow_2d = _to_2d_tensor(variants.get("Glow"))
        glow_proj_2d = _to_2d_tensor(variants.get("Glow_proj"))
        glow_proj_iso_2d = _to_2d_tensor(variants.get("Glow_proj_iso"))
        rnv_2d = _to_2d_tensor(variants.get("RealNVP"))
        rnv_proj_2d = _to_2d_tensor(variants.get("RealNVP_proj"))
        rnv_proj_iso_2d = _to_2d_tensor(variants.get("RealNVP_proj_iso"))
        glow_lifted_proj_2d = _to_2d_tensor(variants.get("Glow_lifted_proj"))
        glow_lifted_iso_2d = _to_2d_tensor(variants.get("Glow_lifted_proj_iso"))
        rnv_lifted_proj_2d = _to_2d_tensor(variants.get("RealNVP_lifted_proj"))
        rnv_lifted_iso_2d = _to_2d_tensor(variants.get("RealNVP_lifted_proj_iso"))

        all_points = [
            true_tensor_2d,
            glow_2d,
            glow_proj_2d,
            glow_proj_iso_2d,
            rnv_2d,
            rnv_proj_2d,
            rnv_proj_iso_2d,
            glow_lifted_proj_2d,
            glow_lifted_iso_2d,
            rnv_lifted_proj_2d,
            rnv_lifted_iso_2d,
        ]
        valid_points = [p for p in all_points if isinstance(p, torch.Tensor) and p.numel() > 0 and p.shape[0] >= 2]
        if len(valid_points) >= 2:
            # Compute shared normalization and save standalone colorbar
            shared_norm = compute_shared_norm(valid_points, gridsize=200, margin_frac=0.05, vmin=0.0)
            colorbar_path = os.path.join(results_dir, "density_colorbar_nf.pdf")
            save_standalone_colorbar(
                norm=shared_norm,
                cmap="viridis",
                filename=colorbar_path,
                label="Density",
                dpi=300,
                height_in=3.5,
                width_in=0.5,
                orientation="horizontal",
            )
            print(f"Saved standalone colorbar to {colorbar_path}")
        else:
            shared_norm = None
            print("Shared normalization skipped: need at least two non-empty datasets with >=2 samples.")

        # Save density plots (without colorbar) using shared norm
        plot_2d_density_no_cbar(
            true_tensor_2d,
            os.path.join(results_dir, "data_2d_density_nf.pdf"),
            "Data Density (Intrinsic 2D)",
            gridsize=200,
            cmap="viridis",
            point_alpha=0.4,
            dpi=300,
            norm=shared_norm,
        )
        if glow_2d.numel() >= 2:
            plot_2d_density_no_cbar(
                glow_2d,
                os.path.join(results_dir, "glow_2d_density.pdf"),
                "Glow Density (Intrinsic 2D)",
                gridsize=200,
                cmap="viridis",
                point_alpha=0.4,
                dpi=300,
                norm=shared_norm,
            )
        if glow_proj_2d.numel() >= 2:
            plot_2d_density_no_cbar(
                glow_proj_2d,
                os.path.join(results_dir, "glow_proj_2d_density.pdf"),
                "Glow (proj.) Density (Intrinsic 2D)",
                gridsize=200,
                cmap="viridis",
                point_alpha=0.4,
                dpi=300,
                norm=shared_norm,
            )
        if glow_proj_iso_2d.numel() >= 2:
            plot_2d_density_no_cbar(
                glow_proj_iso_2d,
                os.path.join(results_dir, "glow_proj_iso_2d_density.pdf"),
                "Glow (proj., iso.) Density (Intrinsic 2D)",
                gridsize=200,
                cmap="viridis",
                point_alpha=0.4,
                dpi=300,
                norm=shared_norm,
            )
        if rnv_2d.numel() >= 2:
            plot_2d_density_no_cbar(
                rnv_2d,
                os.path.join(results_dir, "realnvp_2d_density.pdf"),
                "RealNVP Density (Intrinsic 2D)",
                gridsize=200,
                cmap="viridis",
                point_alpha=0.4,
                dpi=300,
                norm=shared_norm,
            )
        if rnv_proj_2d.numel() >= 2:
            plot_2d_density_no_cbar(
                rnv_proj_2d,
                os.path.join(results_dir, "realnvp_proj_2d_density.pdf"),
                "RealNVP (proj.) Density (Intrinsic 2D)",
                gridsize=200,
                cmap="viridis",
                point_alpha=0.4,
                dpi=300,
                norm=shared_norm,
            )
        if rnv_proj_iso_2d.numel() >= 2:
            plot_2d_density_no_cbar(
                rnv_proj_iso_2d,
                os.path.join(results_dir, "realnvp_proj_iso_2d_density.pdf"),
                "RealNVP (proj., iso.) Density (Intrinsic 2D)",
                gridsize=200,
                cmap="viridis",
                point_alpha=0.4,
                dpi=300,
                norm=shared_norm,
            )
        if glow_lifted_proj_2d.numel() >= 2:
            plot_2d_density_no_cbar(
                glow_lifted_proj_2d,
                os.path.join(results_dir, "glow_lifted_proj_2d_density.pdf"),
                "Glow (lifted proj.) Density (Intrinsic 2D)",
                gridsize=200,
                cmap="viridis",
                point_alpha=0.4,
                dpi=300,
                norm=shared_norm,
            )
        if glow_lifted_iso_2d.numel() >= 2:
            plot_2d_density_no_cbar(
                glow_lifted_iso_2d,
                os.path.join(results_dir, "glow_lifted_iso_2d_density.pdf"),
                "Glow (lifted, iso.) Density (Intrinsic 2D)",
                gridsize=200,
                cmap="viridis",
                point_alpha=0.4,
                dpi=300,
                norm=shared_norm,
            )
        if rnv_lifted_proj_2d.numel() >= 2:
            plot_2d_density_no_cbar(
                rnv_lifted_proj_2d,
                os.path.join(results_dir, "realnvp_lifted_proj_2d_density.pdf"),
                "RealNVP (lifted proj.) Density (Intrinsic 2D)",
                gridsize=200,
                cmap="viridis",
                point_alpha=0.4,
                dpi=300,
                norm=shared_norm,
            )
        if rnv_lifted_iso_2d.numel() >= 2:
            plot_2d_density_no_cbar(
                rnv_lifted_iso_2d,
                os.path.join(results_dir, "realnvp_lifted_iso_2d_density.pdf"),
                "RealNVP (lifted, iso.) Density (Intrinsic 2D)",
                gridsize=200,
                cmap="viridis",
                point_alpha=0.4,
                dpi=300,
                norm=shared_norm,
            )
        print(f"Saved NF density plots and shared colorbar to {results_dir}")
    except Exception as e:
        print(f"Density plotting failed: {e}")

    print(f"Wrote outputs to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args, _ = parser.parse_known_args()
    main(seed=args.seed)
