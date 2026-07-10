#!/usr/bin/env python3
"""Varied-sigma NF evaluation for smileyface_sphere.

For each sigma, load unified checkpoints for Glow and RealNVP, sample, lift to 3D,
project onto the sphere, convert to intrinsic 2D, and compute Coverage/JSD/TVD
against the true intrinsic 2D data.

Produces:
  results/smileyface_sphere/nf_varied_sigmas/
    metrics_varied_sigmas.json
    combined_metrics.pdf

Strict checkpoint pattern required:
  models/smileyface_sphere/model_<TRAINER>_epoch_<E>_noise_level_<SIGMA>_time_<TIME>_seed_<SEED>.pth
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)


def _save_pdf_png(fig, output_path, **kwargs):
    fig.savefig(output_path, **kwargs)
    if output_path.lower().endswith(".pdf"):
        fig.savefig(output_path[:-4] + ".png", **kwargs)

from datasets import SmileyFaceDataset
from trainers import RealNVPTrainer, GlowTrainer
from utils.constraints import SimpleConstraintProjector
from utils.plotting import to_intrinsic_2d, _orthonormal_basis_from_pole
from utils.metrics import (
    coverage,
    jsd_histogram_2d,
    tvd_histogram_2d,
    ensure_tensor_2d,
    filter_valid_samples,
)


# Paper aesthetics
def set_paper_style():
    mpl.rcParams.update({
        "figure.figsize": (3.2, 2.4),
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 2.6,
        "lines.markersize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def checkpoint_path(problem_dir, tag, epochs, sigma, time_cond, seed):
    return os.path.join(
        problem_dir,
        f"model_{tag}_epoch_{epochs}_noise_level_{sigma}_time_{time_cond}_seed_{seed}.pth",
    )


def load_unified_checkpoint_strict(problem_dir, tag, epochs, sigma, time_cond, seed):
    path = checkpoint_path(problem_dir, tag, epochs, sigma, time_cond, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing checkpoint for tag={tag} sigma={sigma} expected file: {path}"
        )
    return torch.load(path, map_location="cpu"), path


# Helper: load training args saved by driver.py to match model hyperparameters
def load_training_args(problem_dir, tag, epochs):
    args_path = os.path.join(problem_dir, f"args_{tag}_epoch_{epochs}.json")
    if os.path.exists(args_path):
        try:
            with open(args_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


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


def verify_state_dict_compat(model, state_dict):
    try:
        mkeys = set(model.state_dict().keys())
        ckeys = set(state_dict.keys())
        missing = sorted(list(mkeys - ckeys))
        unexpected = sorted(list(ckeys - mkeys))
        mism = []
        for k in (mkeys & ckeys):
            ms = tuple(model.state_dict()[k].shape)
            cs = tuple(state_dict[k].shape)
            if ms != cs:
                mism.append((k, ms, cs))
        return {"missing": missing, "unexpected": unexpected, "shape_mismatch": mism}
    except Exception:
        return None


def build_combined_figure_from_json(results_dir):
    set_paper_style()
    metrics_path = os.path.join(results_dir, "metrics_varied_sigmas.json")
    if not os.path.exists(metrics_path):
        print(f"No metrics JSON found at {metrics_path}; skipping combined figure.")
        return
    with open(metrics_path, "r") as f:
        metrics_out = json.load(f)

    sigma_list = metrics_out.get("sigma_list", [])
    trainers = list(metrics_out.get("coverage", {}).keys())
    print(f"Building combined metrics figure in {results_dir} from JSON...")
    color_map = {
        "GLOW": "#1f77b4",
        "REALNVP": "#ff7f0e",
    }
    line_width = 3.0
    marker_size = 7
    # Larger figure to accommodate wider legends and improved readability
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharex=True)
    metric_order = [("coverage", "Coverage"), ("JSD", "JSD"), ("TVD", "TVD")]
    handles = []
    labels = []
    for ax, (metric_key, ylabel) in zip(axes, metric_order):
        for tag in trainers:
            vals = metrics_out.get(metric_key, {}).get(tag, [])
            stds = metrics_out.get(f"{metric_key}_std", {}).get(tag, [])
            color = color_map.get(tag, None)
            if len(vals) == 0:
                continue
            # Curve label: use mathtext for p_{\sigma}; escape braces/backslash for f-string
            h, = ax.plot(
                sigma_list,
                vals,
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
                color=color,
                label=f"{tag} ($p_{{\\sigma}}$, ours)",
                zorder=2,
            )
            if ax is axes[0]:
                handles.append(h)
                labels.append(f"{tag} ($p_{{\\sigma}}$, ours)")
            try:
                arr = np.array(vals, dtype=float)
                std_arr = np.array(stds, dtype=float)
                mask = np.isfinite(arr) & np.isfinite(std_arr)
                if mask.any():
                    sig = np.array(sigma_list)[mask]
                    # enforce minimal visual thickness
                    min_band = 0.005 if metric_key == "coverage" else 0.002
                    delta = np.maximum(std_arr[mask], min_band)
                    lower = arr[mask] - delta
                    upper = arr[mask] + delta
                    if metric_key in ("JSD", "TVD"):
                        eps = 1e-12
                        lower = np.maximum(lower, eps)
                        upper = np.maximum(upper, eps)
                    ax.fill_between(sig, lower, upper, color=color, alpha=0.25, linewidth=0, zorder=1)
            except Exception:
                pass
        key_map = {"Coverage": "coverage", "JSD": "JSD", "TVD": "TVD"}
        for tag in trainers:
            k = key_map.get(ylabel)
            base_mean = metrics_out.get("baseline", {}).get(tag, {}).get(f"{k}_mean") if k else None
            base_std = metrics_out.get("baseline", {}).get(tag, {}).get(f"{k}_std") if k else None
            color = color_map.get(tag, None)
            if base_mean is not None:
                hb = ax.axhline(
                    y=base_mean,
                    color=color,
                    linestyle="--",
                    linewidth=line_width,
                    alpha=0.9,
                    label=f"{tag} baseline",
                )
                if ax is axes[0]:
                    handles.append(hb)
                    labels.append(f"{tag} baseline")
                if base_std is not None and np.isfinite(base_std):
                    eps = 1e-12 if metric_key in ("JSD", "TVD") else 0.0
                    low = max(base_mean - base_std, eps)
                    high = max(base_mean + base_std, eps)
                    ax.fill_between(
                        sigma_list,
                        [low] * len(sigma_list),
                        [high] * len(sigma_list),
                        color=color,
                        alpha=0.10,
                    )
        ax.set_xscale("log")
        ax.set_xlabel("σ")
        ax.set_ylabel(ylabel)
        try:
            sigs = np.array(sigma_list, dtype=float)
            tick_idx = np.linspace(0, len(sigs) - 1, num=min(5, len(sigs)), dtype=int)
            xticks = sigs[tick_idx]
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{x:g}" for x in xticks])
        except Exception:
            pass
        ax.grid(True)
    try:
        ncols = max(1, len(labels))
        # Place legend lower and increase bottom margin so it doesn't overlap x-axis labels
        fig.legend(handles, labels, loc="lower center", ncol=ncols, frameon=False, bbox_to_anchor=(0.5, -0.14))
    except Exception:
        pass
    fig.tight_layout()
    # Make a bit more room at the bottom for the legend
    try:
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.92, wspace=0.35)
    except Exception:
        pass
    out_combined = os.path.join(results_dir, "combined_metrics.pdf")
    try:
        _save_pdf_png(fig, out_combined, bbox_inches="tight")
    except Exception:
        out_combined = os.path.join(results_dir, "combined_metrics_fallback.pdf")
        _save_pdf_png(fig, out_combined, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"Saved combined metrics figure to {out_combined}")

    # Build log-det vs sigma figure if available
    try:
        set_paper_style()
        with open(metrics_path, "r") as f:
            metrics_out = json.load(f)
        sigma_list = metrics_out.get("sigma_list", [])
        trainers = list(metrics_out.get("coverage", {}).keys())
        color_map = {
            "GLOW": "#1f77b4",
            "REALNVP": "#ff7f0e",
        }
        logdet_map = metrics_out.get("logdet_norm", {})
        logdet_std_map = metrics_out.get("logdet_norm_std", {})
        baseline_logdet = metrics_out.get("baseline_logdet", {})
        if logdet_map:
            # Make log-det plot wider so it matches legend width and leave room for legend below
            fig, ax = plt.subplots(figsize=(4.5, 2.8))
            for tag in trainers:
                vals = logdet_map.get(tag, [])
                stds = logdet_std_map.get(tag, [])
                color = color_map.get(tag, None)
                if len(vals) == 0:
                    continue
                arr = np.array(vals, dtype=float)
                std_arr = np.array(stds, dtype=float)
                ax.plot(sigma_list, arr, marker="o", linewidth=2.6, color=color, label=f"{tag} ($p_{{\\sigma}}$, ours)", markersize=7)
                try:
                    sig = np.array(sigma_list, dtype=float)
                    lower = np.maximum(arr - std_arr, 0.0)
                    upper = np.maximum(arr + std_arr, 0.0)
                    ax.fill_between(sig, lower, upper, color=color, alpha=0.22)
                except Exception:
                    pass
                # Baseline line and band
                bmean = baseline_logdet.get(tag, {}).get("logdet_norm_mean")
                bstd = baseline_logdet.get(tag, {}).get("logdet_norm_std")
                if bmean is not None:
                    ax.axhline(y=bmean, color=color, linestyle="--", linewidth=2.6, alpha=0.8, label=f"{tag} baseline")
                    if bstd is not None and np.isfinite(bstd):
                        low = max(bmean - bstd, 0.0)
                        high = max(bmean + bstd, 0.0)
                        ax.fill_between(sigma_list, [low] * len(sigma_list), [high] * len(sigma_list), color=color, alpha=0.10)
            ax.set_xscale("log")
            ax.set_xlabel("σ")
            ax.set_ylabel("mean |log det J|")
            ax.grid(True)
            try:
                # Use a figure-level legend centered under the plot so its width matches the axes
                fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.18))
            except Exception:
                pass
            # Tighten layout and increase bottom margin to prevent legend/labels overlap
            fig.tight_layout()
            fig.subplots_adjust(left=0.08, right=0.96, bottom=0.28, top=0.92)
            out_logdet = os.path.join(results_dir, "logdet_vs_sigma.pdf")
            _save_pdf_png(fig, out_logdet, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved log-det vs sigma figure to {out_logdet}")
    except Exception:
        pass


def lift_intrinsic_2d_to_sphere(uv: torch.Tensor, center: torch.Tensor, radius: float | torch.Tensor, pole: torch.Tensor | None = None) -> torch.Tensor:
    """Lift intrinsic 2D (Lambert equal-area) back to 3D points on the sphere."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NF varied-sigmas intrinsic metrics on smileyface sphere with combined figure")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    parser.add_argument("--trials", type=int, default=10, help="Trials per sigma")
    parser.add_argument("--samples", type=int, default=10000, help="Samples per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results_dir = os.path.join(ROOT_DIR, "results", "smileyface_sphere", "nf_varied_sigmas")
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cpu")
    sigma_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]  # ~[0.01 .. 0.5]
    trials = args.trials
    num_samples_eval = args.samples
    epochs = 40
    time_cond = "default"
    seed = args.seed
    trainers = ["REALNVP", "GLOW"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Sphere config
    sphere_center = [0.0, 0.0, 0.0]
    sphere_radius = 1.0
    projector = SimpleConstraintProjector(device)
    projector.add_constraints_from_dict({"sphere_equality": (sphere_center, sphere_radius)})

    # Dataset for ground-truth intrinsic mapping
    ds = SmileyFaceDataset(
        num_samples=10000,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        projection_type="sphere",
        lifted=False,
        noise_level=0.0,
        device=device,
        seed=seed,
    )
    data_points = torch.stack([ds[i] for i in range(len(ds))])
    with torch.no_grad():
        true_2d = to_intrinsic_2d(data_points.cpu(), torch.tensor(sphere_center), torch.tensor(sphere_radius))[0]
    true_intrinsic = ensure_tensor_2d(true_2d, 2).cpu()
    true_intrinsic = filter_valid_samples(true_intrinsic).cpu()

    # Instantiate trainers (match training-time domain: 3D sphere points for Glow/RealNVP)
    def instantiate_trainer(tag: str, save_dir: str, batch_size: int = 256, hidden_dim: int = 64):
        # Load saved args once to mirror original hyperparameters (hidden sizes, coupling layers, etc.)
        argsj = load_training_args(problem_dir, tag, epochs)
        hd = int(argsj.get("hidden_dim", hidden_dim))
        bs = int(argsj.get("batch_size", batch_size))
        X3d = data_points.cpu().numpy().reshape(-1, 3)
        if tag == "REALNVP":
            n_coupling = int(argsj.get("n_coupling_layers", 6))
            return RealNVPTrainer(dataset=X3d, batch_size=bs, hidden_dim=hd, n_coupling_layers=n_coupling, epochs=1, save_dir=save_dir, device=device)
        if tag == "GLOW":
            # Glow trained on 3D vectors reshaped to (N,1,1,3)
            N, D = X3d.shape
            Ximg = X3d.reshape(N, 1, 1, D)
            return GlowTrainer(dataset=Ximg, image_size=1, batch_size=bs, epochs=1, save_dir=save_dir, device=device)
        # Only RealNVP and Glow are supported.
        raise ValueError(f"Unknown trainer tag: {tag}")

    # Helper: sample, (lift to 3D if needed), project onto sphere
    def sample_and_project_sphere(trainer, projector, n_samples: int, device):
        with torch.no_grad():
            out = trainer.sample(n_samples)
            samples_np = out[0] if isinstance(out, (list, tuple)) else out
            x = torch.tensor(samples_np, dtype=torch.float32)
            x = x.view(x.shape[0], -1)
            if x.shape[1] == 2:
                X3 = lift_intrinsic_2d_to_sphere(x, torch.tensor(sphere_center), torch.tensor(sphere_radius))
            elif x.shape[1] >= 3:
                X3 = x[:, :3]
            else:
                raise RuntimeError(f"Unexpected sample shape {list(x.shape)}")
            proj, _, _ = projector.project(X3.to(device))
        return proj.cpu().numpy()

    bins = 25

    coverage_curves = {tag: [] for tag in trainers}
    jsd_curves = {tag: [] for tag in trainers}
    tvd_curves = {tag: [] for tag in trainers}
    coverage_std = {tag: [] for tag in trainers}
    jsd_std = {tag: [] for tag in trainers}
    tvd_std = {tag: [] for tag in trainers}
    baseline = {tag: {} for tag in trainers}
    # New: ambient-space log-det Jacobian norm tracking
    logdet_norm_curves = {tag: [] for tag in trainers}
    logdet_norm_std = {tag: [] for tag in trainers}
    baseline_logdet = {tag: {} for tag in trainers}

    problem_dir = os.path.join(ROOT_DIR, "models", "smileyface_sphere")

    metrics_path = os.path.join(results_dir, "metrics_varied_sigmas.json")
    if not args.quiet:
        print(f"Computing intrinsic metrics for trainers {trainers} over {len(sigma_list)} sigmas with {trials} trials and {num_samples_eval} samples per trial...")
        print("Strict checkpoint mode: expecting EXACT filenames; will fail if any are missing.")

    for tag in trainers:
        if not args.quiet:
            print(f"[Trainer {tag}] Initializing trainer and computing baseline (sigma=0.0)...")
        save_dir = os.path.join(problem_dir, tag.lower())
        trainer = instantiate_trainer(tag, save_dir)
        try:
            trainer.model.to(device)
        except Exception:
            pass

        # Baseline sigma=0.0 (must exist)
        try:
            ckpt0, path0 = load_unified_checkpoint_strict(problem_dir, tag, epochs, 0.0, time_cond, seed)
            if not args.quiet:
                print(f"Loaded baseline checkpoint: {path0}")
        except FileNotFoundError as e:
            print(str(e))
            raise SystemExit(1)
        sd0 = _extract_state_dict(ckpt0)
        if isinstance(sd0, dict):
            report0 = verify_state_dict_compat(trainer.model, sd0)
            if report0 and (report0["missing"] or report0["unexpected"] or report0["shape_mismatch"]):
                print(f"[{tag}] Baseline architecture mismatch:")
                if report0["missing"]:
                    print(f"  Missing keys: {report0['missing'][:10]}{' ...' if len(report0['missing'])>10 else ''}")
                if report0["unexpected"]:
                    print(f"  Unexpected keys: {report0['unexpected'][:10]}{' ...' if len(report0['unexpected'])>10 else ''}")
                if report0["shape_mismatch"]:
                    print(f"  Shape mismatches (first 5): {report0['shape_mismatch'][:5]}")
            try:
                trainer.model.load_state_dict(sd0, strict=True)
            except Exception:
                try:
                    trainer.model.load_state_dict(sd0, strict=False)
                except Exception:
                    pass
        cov_vals0, jsd_vals0, tvd_vals0 = [], [], []
        logdet_vals0 = []
        for trial_idx in range(trials):
            # Sample ambient points
            with torch.no_grad():
                out = trainer.sample(num_samples_eval)
                samples_np = out[0] if isinstance(out, (list, tuple)) else out
            x = torch.tensor(samples_np, dtype=torch.float32, device=device)
            x = x.view(x.shape[0], -1)
            # Ambient-space log-det via forward
            try:
                fwd_out = trainer.model(x)
                if isinstance(fwd_out, (list, tuple)) and len(fwd_out) >= 2:
                    _, logdet = fwd_out[0], fwd_out[1]
                else:
                    logdet = torch.zeros(x.shape[0], device=device)
                if logdet.dim() > 1:
                    logdet = logdet.view(logdet.shape[0], -1).sum(dim=1)
                logdet_vals0.append(float(torch.abs(logdet).mean().detach().cpu()))
            except Exception:
                logdet_vals0.append(float('nan'))
            # Project to sphere and compute intrinsic metrics
            proj_np = projector.project(x[:, :3])[0].cpu().numpy()
            with torch.no_grad():
                intrinsic2d = to_intrinsic_2d(torch.tensor(proj_np), torch.tensor(sphere_center), torch.tensor(sphere_radius))[0]
            intrinsic2d = ensure_tensor_2d(intrinsic2d, 2)
            intrinsic2d = filter_valid_samples(intrinsic2d).cpu()
            cov_vals0.append(float(coverage(true_intrinsic, intrinsic2d)))
            jsd_vals0.append(float(jsd_histogram_2d(intrinsic2d, true_intrinsic, bins=bins)))
            tvd_vals0.append(float(tvd_histogram_2d(intrinsic2d, true_intrinsic, bins=bins)))
            if not args.quiet:
                print(f"  Baseline trial {trial_idx+1}/{trials} done.")
        baseline_logdet[tag]["logdet_norm_mean"] = float(np.nanmean(np.array(logdet_vals0)))
        baseline_logdet[tag]["logdet_norm_std"] = float(np.nanstd(np.array(logdet_vals0)))
        baseline[tag]["coverage_mean"] = float(np.nanmean(np.array(cov_vals0)))
        baseline[tag]["coverage_std"] = float(np.nanstd(np.array(cov_vals0)))
        baseline[tag]["JSD_mean"] = float(np.nanmean(np.array(jsd_vals0)))
        baseline[tag]["JSD_std"] = float(np.nanstd(np.array(jsd_vals0)))
        baseline[tag]["TVD_mean"] = float(np.nanmean(np.array(tvd_vals0)))
        baseline[tag]["TVD_std"] = float(np.nanstd(np.array(tvd_vals0)))
        if not args.quiet:
            print(f"[Trainer {tag}] Baseline metrics: coverage={baseline[tag]['coverage_mean']:.4f}±{baseline[tag]['coverage_std']:.4f}, JSD={baseline[tag]['JSD_mean']:.4e}±{baseline[tag]['JSD_std']:.4e}, TVD={baseline[tag]['TVD_mean']:.4e}±{baseline[tag]['TVD_std']:.4e}, |logdetJ|={baseline_logdet[tag]['logdet_norm_mean']:.4e}±{baseline_logdet[tag]['logdet_norm_std']:.4e}")

        # (no special-casing required)

        # Sigma sweep
        for sigma in sigma_list:
            if not args.quiet:
                print(f"[Trainer {tag}] Sigma {sigma:g}: sampling trials...")
            try:
                ckpt, ckpt_path = load_unified_checkpoint_strict(problem_dir, tag, epochs, sigma, time_cond, seed)
                if not args.quiet:
                    print(f"  Loaded checkpoint {ckpt_path}")
            except FileNotFoundError as e:
                print(str(e))
                raise SystemExit(1)
            sd = _extract_state_dict(ckpt)
            if isinstance(sd, dict):
                report = verify_state_dict_compat(trainer.model, sd)
                if report and (report["missing"] or report["unexpected"] or report["shape_mismatch"]):
                    print(f"[{tag}] Sigma {sigma:g} architecture mismatch:")
                    if report["missing"]:
                        print(f"  Missing keys: {report['missing'][:10]}{' ...' if len(report['missing'])>10 else ''}")
                    if report["unexpected"]:
                        print(f"  Unexpected keys: {report['unexpected'][:10]}{' ...' if len(report['unexpected'])>10 else ''}")
                    if report["shape_mismatch"]:
                        print(f"  Shape mismatches (first 5): {report['shape_mismatch'][:5]}")
                try:
                    trainer.model.load_state_dict(sd, strict=True)
                except Exception:
                    try:
                        trainer.model.load_state_dict(sd, strict=False)
                    except Exception:
                        pass
            cov_vals, jsd_vals, tvd_vals = [], [], []
            logdet_vals = []
            for trial_idx in range(trials):
                # Sample ambient points
                with torch.no_grad():
                    out = trainer.sample(num_samples_eval)
                    samples_np = out[0] if isinstance(out, (list, tuple)) else out
                x = torch.tensor(samples_np, dtype=torch.float32, device=device)
                x = x.view(x.shape[0], -1)
                # Ambient-space log-det via forward
                try:
                    fwd_out = trainer.model(x)
                    if isinstance(fwd_out, (list, tuple)) and len(fwd_out) >= 2:
                        _, logdet = fwd_out[0], fwd_out[1]
                    else:
                        logdet = torch.zeros(x.shape[0], device=device)
                    if logdet.dim() > 1:
                        logdet = logdet.view(logdet.shape[0], -1).sum(dim=1)
                    logdet_vals.append(float(torch.abs(logdet).mean().detach().cpu()))
                except Exception:
                    logdet_vals.append(float('nan'))
                # Project to sphere for intrinsic metrics
                proj_np = projector.project(x[:, :3])[0].cpu().numpy()
                with torch.no_grad():
                    intrinsic2d = to_intrinsic_2d(torch.tensor(proj_np), torch.tensor(sphere_center), torch.tensor(sphere_radius))[0]
                intrinsic2d = ensure_tensor_2d(intrinsic2d, 2)
                intrinsic2d = filter_valid_samples(intrinsic2d).cpu()
                cov_vals.append(float(coverage(true_intrinsic, intrinsic2d)))
                jsd_vals.append(float(jsd_histogram_2d(intrinsic2d, true_intrinsic, bins=bins)))
                tvd_vals.append(float(tvd_histogram_2d(intrinsic2d, true_intrinsic, bins=bins)))
                if not args.quiet:
                    print(f"    Trial {trial_idx+1}/{trials} done.")
            coverage_curves[tag].append(float(np.nanmean(np.array(cov_vals))) if np.isfinite(np.array(cov_vals)).any() else float('nan'))
            jsd_curves[tag].append(float(np.nanmean(np.array(jsd_vals))) if np.isfinite(np.array(jsd_vals)).any() else float('nan'))
            tvd_curves[tag].append(float(np.nanmean(np.array(tvd_vals))) if np.isfinite(np.array(tvd_vals)).any() else float('nan'))
            coverage_std[tag].append(float(np.nanstd(np.array(cov_vals))) if np.isfinite(np.array(cov_vals)).sum() >= 2 else 0.0)
            jsd_std[tag].append(float(np.nanstd(np.array(jsd_vals))) if np.isfinite(np.array(jsd_vals)).sum() >= 2 else 0.0)
            tvd_std[tag].append(float(np.nanstd(np.array(tvd_vals))) if np.isfinite(np.array(tvd_vals)).sum() >= 2 else 0.0)
            logdet_norm_curves[tag].append(float(np.nanmean(np.array(logdet_vals))) if np.isfinite(np.array(logdet_vals)).any() else float('nan'))
            logdet_norm_std[tag].append(float(np.nanstd(np.array(logdet_vals))) if np.isfinite(np.array(logdet_vals)).sum() >= 2 else 0.0)
            if not args.quiet:
                print(f"[Trainer {tag}] Sigma {sigma:g} metrics: coverage={coverage_curves[tag][-1]:.4f}±{coverage_std[tag][-1]:.4f}, JSD={jsd_curves[tag][-1]:.4e}±{jsd_std[tag][-1]:.4e}, TVD={tvd_curves[tag][-1]:.4e}±{tvd_std[tag][-1]:.4e}, |logdetJ|={logdet_norm_curves[tag][-1]:.4e}±{logdet_norm_std[tag][-1]:.4e}")

    # Save JSON and build figure
    metrics_out = {
        "sigma_list": sigma_list,
        "coverage": coverage_curves,
        "coverage_std": coverage_std,
        "JSD": jsd_curves,
        "JSD_std": jsd_std,
        "TVD": tvd_curves,
        "TVD_std": tvd_std,
        "logdet_norm": logdet_norm_curves,
        "logdet_norm_std": logdet_norm_std,
        "baseline": baseline,
        "baseline_logdet": baseline_logdet,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    if not args.quiet:
        print(f"Saved metrics JSON to {metrics_path}")

    if not args.quiet:
        print("Building combined figure from freshly computed JSON...")
    build_combined_figure_from_json(results_dir)
    if not args.quiet:
        print("Done.")
