#!/usr/bin/env python3
"""Normalizing Flow (NF) comparison script for the smileyface plane problem.

Generates samples for:
  - Glow (baseline)
  - Glow (projected post hoc)
  - Glow (lifted, projected mandatory)
  - RealNVP (baseline)
  - RealNVP (projected post hoc)
    - RealNVP (lifted, projected mandatory)

Computes intrinsic 2D metrics (Coverage, JSD, TVD) against the true data
distribution on the plane, saves scatter + density plots, and a timing
breakdown (where available) using compute_avg_stats. Assumes unified
checkpoints saved by driver.py under:
  models/smileyface_plane/model_<TRAINER>_epoch_<E>_noise_level_<NL>_time_<TIME>_seed_<SEED>.pth

Lifted variants are distinguished solely by noise_level (>0.0); driver.py
does not encode --lifted in filename, so we load based on noise level.

Output directory: results/smileyface_plane/nf_plane
"""
import os
import sys
import json
import math
import argparse
import numpy as np
import torch
import time as _time

# Move to project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from datasets import SmileyFaceDataset
from trainers import GlowTrainer, RealNVPTrainer
from utils.constraints import SimpleConstraintProjector
from utils.plotting import (
    to_intrinsic_2d_plane,
    plot_2d,
    plot_2d_density_no_cbar,
    compute_shared_norm,
    save_standalone_colorbar,
    save_metrics_table_paper,
    count_trainable_params
)
from utils.metrics import (
    coverage,
    jsd_histogram_2d,
    tvd_histogram_2d,
    ensure_tensor_2d,
    filter_valid_samples,
)
from utils.timing import compute_avg_stats, _total_model_time, _total_proj_time


def load_unified_checkpoint(problem_dir: str, trainer_tag: str, epochs: int, noise_level: float, time_cond: str, seed: int, iso: bool = False):
    """Return (checkpoint_dict, path) or (None, attempted_path).

    This loader is tolerant of slight filename-format differences. It first
    tries exact expected filenames then falls back to a glob search that
    parses numeric `noise_level` from candidate filenames and matches it
    numerically (avoids issues with underscore/dot formatting).
    """
    import glob

    effective_epochs = epochs

    # Exact candidates (fast path)
    base = os.path.join(
        problem_dir,
        f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}_time_{time_cond}_seed_{seed}.pth",
    )
    if os.path.exists(base):
        print(f"loader: found exact base {base}")
        return torch.load(base, map_location="cpu"), base
    alt1 = os.path.join(
        problem_dir, f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}_time_{time_cond}.pth"
    )
    if os.path.exists(alt1):
        print(f"loader: found alt1 {alt1}")
        return torch.load(alt1, map_location="cpu"), alt1
    alt2 = os.path.join(problem_dir, f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}.pth")
    if os.path.exists(alt2):
        print(f"loader: found alt2 {alt2}")
        return torch.load(alt2, map_location="cpu"), alt2

    # Also try explicit ISO-tagged exact forms when requested
    if iso:
        base_iso = os.path.join(
            problem_dir,
            f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}_ISO_time_{time_cond}_seed_{seed}.pth",
        )
        if os.path.exists(base_iso):
            print(f"loader: found exact ISO {base_iso}")
            return torch.load(base_iso, map_location="cpu"), base_iso
        alt1_iso = os.path.join(
            problem_dir,
            f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}_ISO_time_{time_cond}.pth",
        )
        if os.path.exists(alt1_iso):
            print(f"loader: found alt1_iso {alt1_iso}")
            return torch.load(alt1_iso, map_location="cpu"), alt1_iso
        alt2_iso = os.path.join(problem_dir, f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_{noise_level}_ISO.pth")
        if os.path.exists(alt2_iso):
            print(f"loader: found alt2_iso {alt2_iso}")
            return torch.load(alt2_iso, map_location="cpu"), alt2_iso

    # Fallback: glob search for any model_{TAG}_epoch_{E}_noise_level_*.pth and
    # parse the embedded noise_level numerically to find the closest match.
    pattern = os.path.join(problem_dir, f"model_{trainer_tag}_epoch_{effective_epochs}_noise_level_*.pth")
    candidates = glob.glob(pattern)
    print(f"loader: globbing pattern {pattern}, found {len(candidates)} candidates")
    best_match = None
    for c in candidates:
        # try to extract the noise substring between 'noise_level_' and
        # the next '_' or '.pth' segment
        try:
            name = os.path.basename(c)
            lname = name.lower()
            if iso and "iso" not in lname:
                continue
            if not iso and "iso" in lname:
                continue
            parts = name.split("noise_level_")
            if len(parts) < 2:
                continue
            tail = parts[1]
            # noise token ends at first occurrence of '_time', '_seed', or '.pth'
            for sep in ("_time", "_seed", ".pth", "_"):
                if sep in tail:
                    token = tail.split(sep)[0]
                    break
            else:
                token = tail
            # extract leading numeric portion of token (handles trailing tags like _ISO)
            import re
            m = re.match(r"([0-9]+(?:[._][0-9]+)?)", token)
            if not m:
                continue
            num_tok = m.group(1).replace("_", ".")
            try:
                tok_val = float(num_tok)
            except Exception:
                continue
            if abs(tok_val - float(noise_level)) < 1e-8:
                # If caller asked for ISO, prefer candidates that include 'iso' in filename
                if iso and "iso" in lname:
                    print(f"loader: choosing ISO candidate {c}")
                    return torch.load(c, map_location="cpu"), c
                # prefer files that include the seed/time if available
                if f"_seed_{seed}.pth" in name or f"_time_{time_cond}" in name:
                    print(f"loader: choosing candidate {c} (seed/time match)")
                    return torch.load(c, map_location="cpu"), c
                if best_match is None:
                    best_match = c
                    print(f"loader: noting candidate {c} as best_match")
        except Exception:
            continue

    if best_match is not None:
        return torch.load(best_match, map_location="cpu"), best_match

    # Give up: return None and the primary attempted base path for diagnostics
    return None, base


def _instantiate_trainer(tag: str, data_np: np.ndarray, save_dir: str, epochs: int, batch_size: int, hidden_dim: int, device: torch.device):
    """Build an unfitted trainer instance with matching data dimensionality so state_dict load succeeds."""
    if tag == "GLOW":
        # Use (N,1,1,D) reshape for vector data
        if data_np.ndim == 2:
            N, D = data_np.shape
            data_img = data_np.reshape(N, 1, 1, D)
        else:
            data_img = data_np
        return GlowTrainer(data_img, image_size=1, batch_size=batch_size, epochs=epochs, save_dir=save_dir, device=device)
    if tag == "REALNVP":
        return RealNVPTrainer(data_np, batch_size=batch_size, epochs=epochs, save_dir=save_dir, hidden_dim=hidden_dim, device=device)
    
    raise ValueError(f"Unknown trainer tag {tag}")


def _extract_state_dict(ckpt):
    if ckpt is None:
        return None
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model_state", "state"):
            if k in ckpt:
                return ckpt[k]
    # If checkpoint itself looks like a state_dict
    if hasattr(ckpt, "keys") and all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt
    return None


def sample_and_project(trainer, projector, num_samples: int, device: torch.device, results_dir: str):
    # Return tensors (on CPU) and measured projection overhead (seconds)
    samples_out = None
    proj_out = None
    proj_overhead = None
    try:
        samples_np, _ = trainer.sample(num_samples=num_samples)
        samples = torch.tensor(samples_np, dtype=torch.float32)
        if samples.dim() > 2:
            samples = samples.view(samples.shape[0], -1)
        # keep samples on device for projection timing
        samples_device = samples.to(device)
        # Filter invalid
        mask = torch.isfinite(samples_device).all(dim=1)
        samples_device = samples_device[mask]
        samples_out = samples_device.cpu()
        # Time projection on the device
        try:
            # write debug pre-attempt
            try:
                with open(os.path.join(results_dir, 'proj_errors.log'), 'a') as ef:
                    ef.write(f"attempting device project: samples_device.shape={samples_device.shape}\n")
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = _time.perf_counter()
            proj, _, _ = projector.project(samples_device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = _time.perf_counter()
            proj_overhead = float(t1 - t0)
            proj_out = proj.cpu()
        except Exception as e:
            # Log projection exception for debugging
            try:
                with open(os.path.join(results_dir, 'proj_errors.log'), 'a') as ef:
                    ef.write(f"device projection failed: {repr(e)}\n")
            except Exception:
                pass
            # fallback: try projecting on CPU (some projectors expect CPU tensors)
            try:
                samples_cpu = samples.cpu()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = _time.perf_counter()
                proj, _, _ = projector.project(samples_cpu)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = _time.perf_counter()
                proj_overhead = float(t1 - t0)
                proj_out = proj.cpu()
            except Exception as e2:
                try:
                    with open(os.path.join(results_dir, 'proj_errors.log'), 'a') as ef:
                        ef.write(f"cpu projection fallback failed: {repr(e2)}\n")
                except Exception:
                    pass
                proj_out = None
                proj_overhead = None
    except Exception:
        # fall back: attempt a simpler sampling call
        try:
            samples_np = trainer.sample()
            samples = torch.tensor(samples_np, dtype=torch.float32)
            if samples.dim() > 2:
                samples = samples.view(samples.shape[0], -1)
            samples_out = samples.cpu()
        except Exception:
            samples_out = None
        proj_out = None
        proj_overhead = None
    return samples_out, proj_out, proj_overhead


def main(seed: int = 42):
    # Force CPU to avoid Glow adapter CUDA/CPU mismatches
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    device = torch.device("cpu")
    problem = "smileyface_plane"
    problem_dir = os.path.join("models", problem)
    results_dir = os.path.join("results", problem, "nf_plane")
    os.makedirs(results_dir, exist_ok=True)

    # Plane parameters (same as plotting_smileyface_plane.py)
    A = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(0)
    b = torch.tensor([1.0])
    epochs = 40  # expected epoch count for plane DDPM; flows may differ but we search this first
    time_cond = "default"
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden_dim = 64
    batch_size = 128
    num_samples_eval = 10000

    # Ground-truth dataset (non-lifted) for metric reference
    dataset_gt = SmileyFaceDataset(
        num_samples=num_samples_eval,
        A=A,
        b=b,
        lifted=False,
        noise_level=0.0,
        device=device,
        seed=seed,
    )
    gt_points = torch.stack([dataset_gt[i] for i in range(len(dataset_gt))]).cpu()

    # Projector for plane
    projector = SimpleConstraintProjector(device)
    projector.add_constraints_from_dict({"linear_equality": (A.to(device), b.to(device))})

    # Helper to prepare base numpy data for trainer construction (always 3D ambient -> vector)
    base_np = gt_points.numpy()

    configs = [
        ("GLOW", 0.0, False, False),
        ("REALNVP", 0.0, False, False),
        ("GLOW", 0.0, True, False),  # projected baseline
        ("REALNVP", 0.0, True, False),
        ("GLOW", 0.05, True, False),  # lifted (projected mandatory)
        ("REALNVP", 0.05, True, False),
        ("GLOW", 0.05, False, False),  # lifted raw (for reference if desired)
        ("REALNVP", 0.05, False, False),
        # ISO-tagged noising variants (projected comparison)
        ("GLOW", 0.05, True, True),
        ("REALNVP", 0.05, True, True),
    ]

    results_manifest = {}
    point_sets_for_norm = []
    ambient_samples = {}
    projection_overheads = {}
    sampled_trainers = {}

    for tag, noise_level, projected, iso in configs:
        ckpt, ckpt_path = load_unified_checkpoint(problem_dir, tag, epochs, noise_level, time_cond, seed, iso=iso)
        save_dir = os.path.join(problem_dir, tag.lower())
        os.makedirs(save_dir, exist_ok=True)
        trainer = _instantiate_trainer(tag, base_np, save_dir, epochs=1, batch_size=batch_size, hidden_dim=hidden_dim, device=device)
        # If trainer instantiation returned None (e.g. missing optional dependency), skip this config
        if trainer is None:
            print(f"Skipping {tag} noise={noise_level}: trainer unavailable")
            label_base = f"{tag}_noise_{noise_level}".replace(".", "_")
            kind = f"{label_base}_projected" if projected else f"{label_base}_raw"
            results_manifest[kind] = {
                "noise_level": noise_level,
                "projected": bool(projected),
                "checkpoint_found": False,
                "num_samples": 0,
                "skipped": True,
            }
            continue
        try:
            trainer.model.to(device)
        except Exception:
            pass
        state_dict = _extract_state_dict(ckpt)
        if state_dict is not None:
            try:
                trainer.model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"State dict load failed for {tag} noise={noise_level}: {e}")
        else:
            print(f"No state dict found for {tag} noise={noise_level} (path tried: {ckpt_path})")

        # Sample and optionally project (projection mandatory for lifted variants with noise>0)
        samples_tensor, proj_tensor, proj_over = sample_and_project(trainer, projector, num_samples_eval, device, results_dir)
        label_base = f"{tag}_noise_{noise_level}".replace(".", "_")
        if iso:
            label_base = label_base + "_iso"
        if projected or noise_level > 0.0:
            # Use projected samples for any lifted/noisy configuration
            used_tensor = proj_tensor if proj_tensor is not None else samples_tensor
            kind = f"{label_base}_projected"
        else:
            # Keep raw samples for lifted/raw reference variants
            used_tensor = samples_tensor
            kind = f"{label_base}_raw"

        # Debug log sampling/projection results
        try:
            with open(os.path.join(results_dir, 'proj_debug.log'), 'a') as df:
                sshape = str(samples_tensor.shape) if samples_tensor is not None else 'None'
                p_present = str(proj_tensor is not None)
                df.write(f"{kind}: samples={sshape}, proj_present={p_present}, proj_over={repr(proj_over)}\n")
        except Exception:
            pass
        # Keep samples/projections in memory (do NOT save .npy files)
        ambient_samples[kind] = used_tensor if used_tensor is not None else None
        # remember the trainer instance that produced these samples (for timing introspection)
        sampled_trainers[tag] = trainer
        # Record projection_overheads (may be None if projection failed)
        projection_overheads[kind] = (float(proj_over) if proj_over is not None else None)
        results_manifest[kind] = {
            "noise_level": noise_level,
            "projected": bool(projected or noise_level > 0.0),
            "iso": bool(iso),
            "checkpoint_found": state_dict is not None,
            "num_samples": int(used_tensor.shape[0]) if used_tensor is not None else 0,
        }

        # Intrinsic 2D conversion for later shared normalization (operate on tensors)
        try:
            if ambient_samples.get(kind) is not None:
                intrinsic = to_intrinsic_2d_plane(ambient_samples.get(kind).to(device), A.to(device), b.to(device))
                intrinsic = ensure_tensor_2d(intrinsic, 2)
                intrinsic = filter_valid_samples(intrinsic).cpu()
                point_sets_for_norm.append((kind, intrinsic))
        except Exception as e:
            print(f"Intrinsic conversion failed for {kind}: {e}")

    # Shared density normalization across all intrinsic sets
    # Defer until after computing true_intrinsic so we can include data for robust vmax.
    shared_norm = None

    # Ground-truth intrinsic
    true_intrinsic = to_intrinsic_2d_plane(gt_points, A.cpu(), b.cpu())
    true_intrinsic = ensure_tensor_2d(true_intrinsic, 2)
    true_intrinsic = filter_valid_samples(true_intrinsic).cpu()

    # Now compute shared norm including true data; save standalone colorbar
    try:
        all_intrinsic = [true_intrinsic] + [t for _, t in point_sets_for_norm if t is not None and t.numel() > 0]
        if len(all_intrinsic) >= 1:
            shared_norm = compute_shared_norm(all_intrinsic, gridsize=200, margin_frac=0.05, vmin=0.0)
            colorbar_path = os.path.join(results_dir, "density_colorbar.pdf")
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
            print("No intrinsic datasets available to compute shared normalization; skipping colorbar save.")
    except Exception as e:
        print(f"Failed to compute/save standalone colorbar: {e}")

    metrics = {}
    # Build shared histogram grid edges from combined data
    stacked = [true_intrinsic.cpu().numpy()] + [t.cpu().numpy() for t in all_intrinsic]
    if len(stacked) > 0:
        all_np = np.vstack(stacked)
        xmin, xmax = float(all_np[:, 0].min()), float(all_np[:, 0].max())
        ymin, ymax = float(all_np[:, 1].min()), float(all_np[:, 1].max())
        def _expand(lo, hi):
            if hi <= lo:
                d = 1e-6 * (abs(lo) + 1.0)
                return lo - d, hi + d
            return lo, hi
        xmin, xmax = _expand(xmin, xmax)
        ymin, ymax = _expand(ymin, ymax)
        bins = 25
        xedges = np.linspace(xmin, xmax, bins + 1)
        yedges = np.linspace(ymin, ymax, bins + 1)
        grid_edges = (xedges, yedges)
    else:
        grid_edges = None

    # Density plots + metrics per set
    for kind, intrinsic in point_sets_for_norm:
        try:
            plot_2d_density_no_cbar(
                intrinsic,
                os.path.join(results_dir, f"{kind}_2d_density.pdf"),
                f"{kind} density",
                gridsize=200,
                cmap="viridis",
                point_alpha=0.4,
                dpi=300,
                norm=shared_norm,
            )
        except Exception:
            pass
        # Metrics
        try:
            cov = float(coverage(true_intrinsic, intrinsic))
        except Exception:
            cov = float("nan")
        try:
            jsd = float(jsd_histogram_2d(intrinsic, true_intrinsic, grid_edges=grid_edges))
        except Exception:
            jsd = float("nan")
        try:
            tvd = float(tvd_histogram_2d(intrinsic, true_intrinsic, grid_edges=grid_edges))
        except Exception:
            tvd = float("nan")
        metrics[kind] = {"coverage": cov, "JSD": jsd, "TVD": tvd}

    # ------------------------------------------------------------------
    # Extrinsic (ambient 3D) + Intrinsic (2D) separate metrics tables
    # ------------------------------------------------------------------
    # Build ambient-space sample tensors (projected sets already on plane; raw sets may deviate)
    true_ambient = gt_points.view(-1, gt_points.shape[-1])
    true_ambient = filter_valid_samples(true_ambient).cpu()
    # ambient_samples was populated during sampling loop; ensure all expected
    # kinds are present (fill missing with None)
    for tag, noise_level, projected, iso in configs:
        label_base = f"{tag}_noise_{noise_level}".replace('.', '_')
        if iso:
            label_base = label_base + "_iso"
        kind = f"{label_base}_projected" if projected else f"{label_base}_raw"
        ambient_samples.setdefault(kind, None)

    # Helper: generic N-D histogram-based coverage, JSD, TVD using numpy.histogramdd
    def _nd_hist_metrics(x: torch.Tensor, y: torch.Tensor, bins: int = 40):
        try:
            if x is None or y is None or x.numel() == 0 or y.numel() == 0:
                return float('nan'), float('nan'), float('nan')
            X = x.cpu().numpy()
            Y = y.cpu().numpy()
            d = X.shape[1]
            # Bin edges from true data (Y)
            edges = [np.linspace(Y[:, i].min(), Y[:, i].max(), bins + 1) for i in range(d)]
            Hx, _ = np.histogramdd(X, bins=edges)
            Hy, _ = np.histogramdd(Y, bins=edges)
            Hx = Hx.astype(np.float64)
            Hy = Hy.astype(np.float64)
            sum_x = Hx.sum()
            sum_y = Hy.sum()
            if sum_x <= 0 or sum_y <= 0:
                return float('nan'), float('nan'), float('nan')
            Px = Hx / sum_x
            Py = Hy / sum_y
            # Coverage: fraction of occupied true bins hit by model
            true_mask = Py > 0
            model_hit = (Px > 0) & true_mask
            cov = float(model_hit.sum() / max(1, true_mask.sum()))
            M = 0.5 * (Px + Py)
            # JSD
            with np.errstate(divide='ignore', invalid='ignore'):
                def _kl(P, Q):
                    mask = (P > 0) & (Q > 0)
                    return np.sum(P[mask] * (np.log(P[mask] + 1e-12) - np.log(Q[mask] + 1e-12)))
                jsd_val = 0.5 * _kl(Px, M) + 0.5 * _kl(Py, M)
            # TVD
            tvd_val = 0.5 * np.abs(Px - Py).sum()
            return cov, float(jsd_val), float(tvd_val)
        except Exception:
            return float('nan'), float('nan'), float('nan')

    extrinsic_metrics = {}
    for kind, arr in ambient_samples.items():
        try:
            arr_valid = filter_valid_samples(arr) if arr is not None else None
        except Exception:
            arr_valid = arr
        c, j, t = _nd_hist_metrics(arr_valid, true_ambient, bins=25)
        extrinsic_metrics[kind] = {"coverage": c, "JSD": j, "TVD": t}

    intrinsic_metrics = metrics  # already computed (2D)

    # Save separate CSV + TeX tables for extrinsic and intrinsic metrics
    def _write_table(data: dict, out_prefix: str, headers: list, timing_training_map: dict | None = None, timing_sampling_map: dict | None = None, include_timing: bool = True):
        SCI_THRESHOLD = 1e-3  # only use scientific notation for very small magnitudes
        def _pretty_method(name: str) -> str:
            lowered = name.lower()
            is_glow = lowered.startswith("glow")
            is_rnv = lowered.startswith("realnvp")
            
            import re
            projected = "projected" in lowered
            raw = "raw" in lowered
            iso_flag = "iso" in lowered
            lifted = False
            if "noise_" in lowered:
                m = re.search(r"noise_([0-9_\.]+)", lowered)
                if m:
                    try:
                        noise_val = float(m.group(1).replace("_", "."))
                        lifted = (noise_val > 0.0) and (not iso_flag)
                    except Exception:
                        # fallback: if noise_ present and not exactly 0, assume lifted
                        lifted = ("noise_0_0" not in lowered) and (not iso_flag)
                else:
                    lifted = ("noise_0_0" not in lowered) and (not iso_flag)
            # deterministic fallback for known lifted token formatting
            if ("_noise_0_05_" in lowered or "noise_0_05" in lowered) and ("iso" not in lowered):
                lifted = True
            if lifted and projected and is_glow:
                return r"Glow ($p_{\sigma}$, ours)"
            if lifted and projected and is_rnv:
                return r"RealNVP ($p_{\sigma}$, ours)"
            if lifted and raw and is_glow:
                return r"Glow (lifted raw)"
            if lifted and raw and is_rnv:
                return r"RealNVP (lifted raw)"
            if projected and is_glow:
                if iso_flag:
                    return "Glow (iso.)"
                return "Glow (proj.)"
            if projected and is_rnv:
                if iso_flag:
                    return "RealNVP (iso.)"
                return "RealNVP (proj.)"
            if is_glow:
                return "Glow"
            if is_rnv:
                return "RealNVP"
            return name
        # Row schema: timing-first when include_timing=True, else metrics-only
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
        def _sci(val, sig=3):
            if not isinstance(val, (int, float)) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                return "--"
            if val == 0:
                return "0"
            exp = int(math.floor(math.log10(abs(val))))
            mant = val / (10 ** exp)
            return f"{mant:.{sig}f}e{exp:+d}"
        with open(f"{out_prefix}.csv", 'w', newline='') as cf:
            w = csv.writer(cf)
            if include_timing:
                w.writerow(["Method", "Train/Epoch (s)", "Sample/Total (s)", "Coverage", "JSD", "TVD"])
            else:
                w.writerow(["Method", "Coverage", "JSD", "TVD"])
            for r in rows:
                if include_timing:
                    train_v, sample_v, cov, jsd, tvd = r[1], r[2], r[3], r[4], r[5]
                else:
                    cov, jsd, tvd = r[1], r[2], r[3]
                def _fmt_csv(val):
                    if not isinstance(val, (int, float)) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                        return "--"
                    if val == 0:
                        return "0"
                    if abs(val) < SCI_THRESHOLD:
                        return _sci(val)
                    return f"{val:.4f}"
                cov_csv = _fmt_csv(cov)
                jsd_csv = _fmt_csv(jsd)
                tvd_csv = _fmt_csv(tvd)
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
                    train_v, sample_v, cov, jsd, tvd = r[1], r[2], r[3], r[4], r[5]
                else:
                    cov, jsd, tvd = r[1], r[2], r[3]
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
                cov_s = _fmt_tex(cov)
                jsd_s = _fmt_tex(jsd)
                tvd_s = _fmt_tex(tvd)
                if include_timing:
                    train_s = _fmt_tex(train_v) if train_v is not None else "--"
                    sample_s = _fmt_tex(sample_v) if sample_v is not None else "--"
                    tf.write(f"{r[0]} & {train_s} & {sample_s} & {cov_s} & {jsd_s} & {tvd_s} \\\n")
                else:
                    tf.write(f"{r[0]} & {cov_s} & {jsd_s} & {tvd_s} \\\n")
            tf.write("\\end{tabular}\n")

    # Build timing maps for extrinsic table
    timing_trainers = {}
    # Baseline trainers: use noise 0.0 raw variants
    for tag in ["GLOW", "REALNVP"]:
        ckpt, _ = load_unified_checkpoint(problem_dir, tag, epochs, 0.0, time_cond, seed)
        tr = _instantiate_trainer(tag, base_np, os.path.join(problem_dir, tag.lower()), epochs=1, batch_size=batch_size, hidden_dim=hidden_dim, device=device)
        if tr is None:
            # Trainer unavailable; skip timing for this tag
            print(f"Timing: skipping trainer {tag} (unavailable)")
            continue
        sd = _extract_state_dict(ckpt)
        if sd is not None:
            try:
                tr.model.load_state_dict(sd, strict=False)
            except Exception:
                pass
        # Restore timing metadata from the checkpoint if available so we can
        # show per-epoch training timings without re-running training.
        try:
            if isinstance(ckpt, dict):
                if "epoch_timing_breakdowns" in ckpt:
                    etb = ckpt.get("epoch_timing_breakdowns")
                    if isinstance(etb, list) and len(etb) > 0:
                        tr.epoch_timing_breakdowns = etb
                if "projection_times" in ckpt:
                    tr.projection_times = ckpt.get("projection_times") or getattr(tr, "projection_times", [])
                if "projection_sample_times" in ckpt:
                    tr.projection_sample_times = ckpt.get("projection_sample_times") or getattr(tr, "projection_sample_times", [])
                if "training_losses" in ckpt:
                    tr.training_losses = ckpt.get("training_losses") or getattr(tr, "training_losses", [])
        except Exception:
            pass
        timing_trainers[tag] = tr
    # Compute sampling stats
    try:
        avg_stats = compute_avg_stats(list(timing_trainers.keys()), timing_trainers, n_trials=3, num_samples=num_samples_eval)
    except Exception as e:
        avg_stats = {}
        print(f"compute_avg_stats failed: {e}")
    sampling_totals = {k: (avg_stats.get(k, {}).get("s") if isinstance(avg_stats.get(k, {}).get("s", None), (int, float)) else None) for k in timing_trainers.keys()}
    # If compute_avg_stats failed to populate sampling totals (NaN/None),
    # attempt a best-effort one-shot measurement here so projection overhead
    # can be included for projected variants.
    import time as _time
    for base in list(timing_trainers.keys()):
        val = sampling_totals.get(base)
        try:
            if val is None or (isinstance(val, float) and (not np.isfinite(val))):
                tr = timing_trainers.get(base)
                if tr is None:
                    continue
                # run a single timed sampling pass (best-effort)
                try:
                    if hasattr(torch, 'no_grad'):
                        ctx = torch.no_grad()
                    else:
                        ctx = nullcontext = None
                    if ctx is not None:
                        ctx.__enter__()
                    t0 = _time.perf_counter()
                    _res = None
                    try:
                        _res = tr.sample(num_samples=num_samples_eval)
                    except Exception:
                        # try without args
                        try:
                            _res = tr.sample()
                        except Exception:
                            _res = None
                    t1 = _time.perf_counter()
                    if ctx is not None:
                        try:
                            ctx.__exit__(None, None, None)
                        except Exception:
                            pass
                    measured = float(t1 - t0)
                    if measured > 0:
                        sampling_totals[base] = measured
                        # also try to populate avg_stats model/proj fields from trainer attrs
                        try:
                            m_val = _total_model_time(tr)
                            p_val = _total_proj_time(tr)
                            if base not in avg_stats:
                                avg_stats[base] = {'m': m_val, 'p': p_val, 's': measured}
                            else:
                                avg_stats[base]['s'] = measured
                                if np.isfinite(m_val):
                                    avg_stats[base]['m'] = m_val
                                if np.isfinite(p_val):
                                    avg_stats[base]['p'] = p_val
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
    # Expand sampling totals to metric key space and add projection overhead.
    expanded_sampling_map = {}
    for kind in extrinsic_metrics.keys():
        low = kind.lower()
        if low.startswith("glow"):
            base = "GLOW"
        elif low.startswith("realnvp"):
            base = "REALNVP"
        
        else:
            base = None
        base_time = sampling_totals.get(base) if base else None
        projected = "projected" in low
        # If base_time missing, still attempt to include projection overhead for projected variants
        if base_time is None and not projected:
            expanded_sampling_map[kind] = None
            continue
        if projected:
            raw_key = kind.replace("_projected", "_raw")
            # Prefer projection overhead measured while sampling raw set, else use measured for projected set
            overhead = None
            if raw_key in projection_overheads:
                overhead = projection_overheads.get(raw_key)
            else:
                overhead = projection_overheads.get(kind, None)
            bt = base_time if (base_time is not None and np.isfinite(base_time)) else 0.0
            val = bt + (float(overhead) if (overhead is not None and np.isfinite(overhead)) else 0.0)
            expanded_sampling_map[kind] = val
        else:
            expanded_sampling_map[kind] = base_time
    # Training per-epoch totals from checkpoint timing — compute per-kind
    def _per_epoch_total_from_ckpt(tag: str, noise: float, iso: bool = False):
        try:
            ckpt, _ = load_unified_checkpoint(problem_dir, tag, epochs, noise, time_cond, seed, iso=iso)
            tr = _instantiate_trainer(tag, base_np, os.path.join(problem_dir, tag.lower()), epochs=1, batch_size=batch_size, hidden_dim=hidden_dim, device=device)
            if tr is None:
                return None
            sd = _extract_state_dict(ckpt)
            if sd is not None:
                try:
                    tr.model.load_state_dict(sd, strict=False)
                except Exception:
                    pass
            # restore timing metadata if present
            if isinstance(ckpt, dict):
                etb = ckpt.get("epoch_timing_breakdowns")
                if isinstance(etb, list) and len(etb) > 0:
                    tr.epoch_timing_breakdowns = etb
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
        except Exception:
            return None

    training_epoch_map = {}
    # Build training map keyed by the exact kinds used in tables/plots (match configs ordering)
    def _kind_name(tag, noise, projected, iso: bool = False):
        base = f"{tag}_noise_{str(noise).replace('.','_')}"
        if iso:
            base = base + "_iso"
        return base + ("_projected" if projected else "_raw")
    for tag, noise_level, projected, iso in configs:
        kind = _kind_name(tag, noise_level, projected, iso)
        if kind in extrinsic_metrics:
            training_epoch_map[kind] = _per_epoch_total_from_ckpt(tag, noise_level, iso)
    # Some keys may not appear in configs (fallback to mapping by prefix)
    for kind in extrinsic_metrics.keys():
        if kind not in training_epoch_map:
            low = kind.lower()
            if low.startswith("glow"):
                training_epoch_map[kind] = _per_epoch_total_from_ckpt("GLOW", 0.0)
            elif low.startswith("realnvp"):
                training_epoch_map[kind] = _per_epoch_total_from_ckpt("REALNVP", 0.0)
            else:
                training_epoch_map[kind] = None

    try:
        _write_table(
            extrinsic_metrics,
            os.path.join(results_dir, "nf_extrinsic_metrics_table"),
            ["Method", "Train/Epoch (s)", "Sample/Total (s)", "Coverage", "JSD", "TVD"],
            timing_training_map=training_epoch_map,
            timing_sampling_map=expanded_sampling_map,
            include_timing=True,
        )
        intrinsic_filtered = {k: v for k, v in intrinsic_metrics.items() if "projected" in k}
        _write_table(
            intrinsic_filtered,
            os.path.join(results_dir, "nf_intrinsic_metrics_table"),
            ["Method", "Coverage", "JSD", "TVD"],
            include_timing=False,
        )
        print("Saved extrinsic and intrinsic NF metrics tables.")
    except Exception as e:
        print(f"Failed to write NF extrinsic/intrinsic tables: {e}")

    # -------------------------
    # Additional table: NaN/Inf counts and Avg. Dist. to M (for NF outputs)
    # -------------------------
    try:
        import numpy as _np

        def _compute_stats_from_np(orig_np, proj_np=None, pr=None):
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
            # If projector provided, project raw finite rows and use returned per-sample distances
            if pr is None:
                pr = globals().get('projector', None)
            if proj_np is None and pr is not None:
                try:
                    X = torch.tensor(valid_np, device=pr.device, dtype=getattr(pr, 'dtype', torch.float32))
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
                return n_bad, "n/a"
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

        # Try to append to both extrinsic and intrinsic NF table files
        for suffix in ("nf_extrinsic_metrics_table.tex", "nf_intrinsic_metrics_table.tex"):
            out_path = os.path.join(results_dir, suffix)
            try:
                # Attempt to parse which methods are in the corresponding metrics dict by reading the .tex header
                # Fallback: use keys from intrinsic_metrics/extrinsic_metrics if available in locals()
                methods = None
                try:
                    if suffix.startswith("nf_extrinsic"):
                        methods = list(extrinsic_metrics.keys())
                    else:
                        methods = list(intrinsic_filtered.keys())
                except Exception:
                    methods = []

                rows = []
                for name in methods:
                    # try to load corresponding saved numpy arrays in results_dir
                    base = name
                    arr = None
                    proj = None
                    try:
                        pth = os.path.join(results_dir, f"{base}.npy")
                        if os.path.exists(pth):
                            arr = np.load(pth)
                    except Exception:
                        arr = None
                    try:
                        pthp = os.path.join(results_dir, f"{base}_projected.npy")
                        if os.path.exists(pthp):
                            proj = np.load(pthp)
                    except Exception:
                        proj = None
                    n_bad, avg_dist = _compute_stats_from_np(arr, proj, locals().get('projector', None))
                    rows.append((name, n_bad, avg_dist))

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

    # Sampling time breakdown plot
    try:
        import matplotlib.pyplot as plt
        # Safe numeric helpers
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
        # Desired ordering and kinds to show in the NF sampling plot
        order_keys = [
            f"GLOW_noise_0_0_raw",
            f"GLOW_noise_0_0_projected",
            f"GLOW_noise_0_05_projected",
            f"REALNVP_noise_0_0_raw",
            f"REALNVP_noise_0_0_projected",
            f"REALNVP_noise_0_05_projected",
        ]

        def _display_label(k: str) -> str:
            low = k.lower()
            is_glow = low.startswith("glow")
            is_rnv = low.startswith("realnvp")
            
            import re
            projected = "projected" in low
            iso_flag = "iso" in low
            lifted = False
            if "noise_" in low:
                m = re.search(r"noise_([0-9_\.]+)", low)
                if m:
                    try:
                        noise_val = float(m.group(1).replace("_", "."))
                        lifted = (noise_val > 0.0) and (not iso_flag)
                    except Exception:
                        lifted = ("noise_0_0" not in low) and (not iso_flag)
                else:
                    lifted = ("noise_0_0" not in low) and (not iso_flag)
            if ("_noise_0_05_" in low or "noise_0_05" in low) and ("iso" not in low):
                lifted = True
            if lifted and projected and is_glow:
                return "Glow ($p_{\\sigma}$, proj.)"
            if lifted and projected and is_rnv:
                return "RealNVP ($p_{\\sigma}$, proj.)"
            if projected and is_glow:
                if iso_flag:
                    return "Glow (iso.)"
                return "Glow (proj.)"
            if projected and is_rnv:
                if iso_flag:
                    return "RealNVP (iso.)"
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
                # skip missing kinds
                continue
            labels.append(_display_label(kind))
            # base trainer (GLOW/REALNVP)
            base = kind.split("_")[0].upper()
            # Prefer measured model-forward time from the trainer used for sampling
            measured_m = None
            tr_meas = sampled_trainers.get(base)
            try:
                if tr_meas is not None:
                    measured_m = _total_model_time(tr_meas)
            except Exception:
                measured_m = None
            if measured_m is not None and np.isfinite(measured_m):
                base_m = _safe_float(measured_m)
            else:
                base_m = _safe_float(avg_stats.get(base, {}).get("m") if avg_stats else float('nan'))
            base_p = _safe_float(avg_stats.get(base, {}).get("p") if avg_stats else 0.0)
            total_sample = _safe_float(expanded_sampling_map.get(kind))
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
        # Debug: write numeric arrays to proj_debug.log for inspection
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
            # Use same ordering as sampling plot so metrics rows align visually
            order_keys = [
                f"GLOW_noise_0_0_raw",
                f"GLOW_noise_0_0_projected",
                f"GLOW_noise_0_05_projected",
                f"REALNVP_noise_0_0_raw",
                f"REALNVP_noise_0_0_projected",
                f"REALNVP_noise_0_05_projected",
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
                # Attempt to read per-epoch averaged components from the checkpoint-backed trainer
                tr_total = training_epoch_map.get(kind)
                # Default breakdown: prefer `epoch_timing_breakdowns` from the checkpoint.
                # If missing, fall back to checkpoint-level aggregate fields, then to total-only.
                m = p = b = o = np.nan
                source_used = 'none'
                try:
                    import re
                    tag = kind.split("_")[0].upper()
                    m_match = re.search(r"_noise_([0-9_]+)", kind)
                    if m_match:
                        token = m_match.group(1).strip("_")
                        noise_str = token.replace("_", ".")
                    else:
                        noise_str = "0.0"
                    # Use the tolerant loader which numerically matches noise levels
                    ckpt, tried_path = load_unified_checkpoint(
                        problem_dir, tag, epochs, float(noise_str), time_cond, seed
                    )
                    # Log the path returned by the loader and whether it exists
                    try:
                        with open(os.path.join(results_dir, "training_source.log"), "a") as tf:
                            tf.write(f"{kind}: attempted_ckpt_path={tried_path}, exists={os.path.exists(tried_path)}\n")
                    except Exception:
                        pass
                    if isinstance(ckpt, dict):
                        etb = ckpt.get("epoch_timing_breakdowns") or []
                        if etb:
                            # Try several common keys for compatibility
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
                            # Check for checkpoint-level aggregates as a fallback
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
                                # distribute remaining total into 'other' when possible
                                try:
                                    if tr_total is not None:
                                        rem = float(tr_total) - (m if np.isfinite(m) else 0.0) - (p if np.isfinite(p) else 0.0)
                                        o = rem if np.isfinite(rem) and rem > 0 else 0.0
                                except Exception:
                                    o = float('nan')
                                source_used = 'checkpoint_aggregates'
                except Exception:
                    pass
                # If still missing components but have a total, put it in 'other'
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
                # Log which source was used and computed components
                try:
                    with open(os.path.join(results_dir, 'training_source.log'), 'a') as tf:
                        tf.write(f"{kind}: source={source_used}, m={m}, p={p}, b={b}, o={o}\n")
                except Exception:
                    pass
                m_vals.append(m)
                p_vals.append(p)
                b_vals.append(b)
                o_vals.append(o)

            # Convert labels to nicer display using same display rules used elsewhere
            def _display_label(k: str) -> str:
                low = k.lower()
                is_glow = low.startswith("glow")
                is_rnv = low.startswith("realnvp")
                
                import re
                projected = "projected" in low
                iso_flag = "iso" in low
                lifted = False
                if "noise_" in low:
                    m = re.search(r"noise_([0-9_\.]+)", low)
                    if m:
                        try:
                            noise_val = float(m.group(1).replace("_", "."))
                            lifted = (noise_val > 0.0) and (not iso_flag)
                        except Exception:
                            lifted = ("noise_0_0" not in low) and (not iso_flag)
                    else:
                        lifted = ("noise_0_0" not in low) and (not iso_flag)
                if ("_noise_0_05_" in low or "noise_0_05" in low) and ("iso" not in low):
                    lifted = True
                if lifted and projected and is_glow:
                    return "Glow ($p_{\\sigma}$, proj.)"
                if lifted and projected and is_rnv:
                    return "RealNVP ($p_{\\sigma}$, proj.)"
                if projected and is_glow:
                    if iso_flag:
                        return "Glow (iso.)"
                    return "Glow (proj.)"
                if projected and is_rnv:
                    if iso_flag:
                        return "RealNVP (iso.)"
                    return "RealNVP (proj.)"
                if is_glow:
                    return "Glow"
                if is_rnv:
                    return "RealNVP"
                return k

            display_labels = [_display_label(k) for k in labels]
            print("Skipping bar plot output: nf training_time_breakdown")
    except Exception:
        import traceback
        traceback.print_exc()

    # Save manifest + metrics
    # Persist manifest and measured projection overheads
    with open(os.path.join(results_dir, "manifest.json"), "w") as f:
        json.dump({"sets": results_manifest, "metrics": metrics, "projection_overheads": projection_overheads}, f, indent=2)
    # Also save a simple timing breakdown JSON for quick inspection
    try:
        with open(os.path.join(results_dir, "timing_breakdown.json"), "w") as tf:
            json.dump({"sampling_totals": sampling_totals, "expanded_sampling": expanded_sampling_map, "projection_overheads": projection_overheads}, tf, indent=2)
    except Exception:
        pass

    # (Legacy single-table build removed; replaced by separate extrinsic/intrinsic tables above.)

    print("\n--- NF Model Trainable Parameter Counts ---")
    for tag in ["GLOW", "REALNVP"]:
        tr = timing_trainers.get(tag)
        if tr is None:
            print(f"{tag}: Trainer unavailable or not instantiated.")
            continue
        try:
            param_count = count_trainable_params(tr.model)
            print(f"{tag} model: {param_count:,}")
        except Exception as e:
            print(f"{tag}: Could not compute parameter count: {e}")

    print(f"Saved NF plane results to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args, _ = parser.parse_known_args()
    main(seed=args.seed)
