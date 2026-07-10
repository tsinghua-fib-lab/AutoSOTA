#!/usr/bin/env python3
"""Varied-sigma NF evaluation for smileyface_plane.

Sweeps a list of noise levels (sigmas) and, for each sigma, loads lifted
Glow and RealNVP unified checkpoints:
  models/smileyface_plane/model_<TRAINER>_epoch_<E>_noise_level_<SIGMA>_time_<TIME>_seed_<SEED>.pth

For each (trainer, sigma):
  - Instantiate trainer skeleton matching data dimensionality.
  - Load state_dict if available.
  - Sample num_samples points.
  - Project samples onto the plane (mandatory for lifted flows).
  - Convert to intrinsic 2D coordinates.
  - Compute Coverage, JSD, TVD (histogram-based) against true intrinsic data.

Produces:
  results/smileyface_plane/nf_varied_sigmas/
    metrics_varied_sigmas.json  (arrays per sigma per trainer)
    combined_metrics.pdf
    (Optional per-sigma scatter/density images)

Glow and RealNVP curves share the same sigma_list and histogram grid.
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
from utils.plotting import to_intrinsic_2d_plane
from utils.metrics import (
    coverage,
    jsd_histogram_2d,
    tvd_histogram_2d,
    ensure_tensor_2d,
    filter_valid_samples,
)


# Consistent paper aesthetics
def set_paper_style():
    mpl.rcParams.update({
        "figure.figsize": (3.2, 2.4),  # ~1/3 single column
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
    """Construct the ONLY allowed checkpoint path pattern.

    Required pattern:
        model_<TAG>_epoch_<E>_noise_level_<SIGMA>_time_<TIME>_seed_<SEED>.pth
    """
    return os.path.join(
        problem_dir,
        f"model_{tag}_epoch_{epochs}_noise_level_{sigma}_time_{time_cond}_seed_{seed}.pth",
    )

def load_unified_checkpoint_strict(problem_dir, tag, epochs, sigma, time_cond, seed):
    """Strictly load a checkpoint using the single supported filename pattern.

    Raises FileNotFoundError if the file does not exist.
    """
    path = checkpoint_path(problem_dir, tag, epochs, sigma, time_cond, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing checkpoint for tag={tag} sigma={sigma} expected file: {path}"
        )
    return torch.load(path, map_location="cpu"), path


def extract_state_dict(ckpt):
    if ckpt is None:
        return None
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model_state", "state"):
            if k in ckpt:
                return ckpt[k]
    return None


def load_training_args(problem_dir, tag, epochs):
    """Load saved training args JSON to mirror hyperparameters (hidden_dim, etc.)."""
    args_path = os.path.join(problem_dir, f"args_{tag}_epoch_{epochs}.json")
    if os.path.exists(args_path):
        try:
            with open(args_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


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
    """Build and save the combined figure using the already-saved metrics JSON.

    This avoids recomputing metrics and guarantees a single combined image.
    """
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
        "GLOW": "#1f77b4",      # blue
        "REALNVP": "#ff7f0e",   # orange
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
            # Curve label uses mathtext for p_{\sigma}
            h, = ax.plot(
                sigma_list,
                vals,
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
                color=color,
                label=f"{tag} ($p_{{\\sigma}}$, ours)",
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
                    lower = (arr - std_arr)[mask]
                    upper = (arr + std_arr)[mask]
                    if metric_key in ("JSD", "TVD"):
                        eps = 1e-12
                        lower = np.maximum(lower, eps)
                        upper = np.maximum(upper, eps)
                    ax.fill_between(sig, lower, upper, color=color, alpha=0.22)
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
                    alpha=0.8,
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
    try:
        if not os.path.exists(out_combined):
            fig3 = plt.figure(figsize=(3.2, 2.4))
            plt.text(0.5, 0.5, "Combined metrics (placeholder)", ha="center", va="center")
            _save_pdf_png(fig3, out_combined, bbox_inches="tight")
            plt.close(fig3)
            print(f"Wrote placeholder combined figure to {out_combined}")
    except Exception:
        pass

    # Additionally build a dedicated log-det vs sigma figure for NF trainers
    try:
        set_paper_style()
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
                # Baseline line
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NF varied-sigmas intrinsic metric computation + combined figure builder (smileyface plane)")
    # Always recompute; retained flag removed per user request.
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity (only final status messages)")
    parser.add_argument("--trials", type=int, default=10, help="Number of sampling trials per sigma for mean/std")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples per trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    # Compute intrinsic metrics (Coverage/JSD/TVD) over sigmas, save JSON, then build combined figure.
    results_dir = os.path.join(ROOT_DIR, "results", "smileyface_plane", "nf_varied_sigmas")
    os.makedirs(results_dir, exist_ok=True)

    # Config (reasonable defaults; adjust if needed)
    device = torch.device("cpu")
    # Use strictly positive, log-spaced sigma values for plotting on a log x-axis
    sigma_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]  # ~[0.01 .. 0.5]
    trials = args.trials
    num_samples_eval = args.samples
    epochs = 40
    # unified checkpoints often use string tags like 'best'
    time_cond = "default"
    seed = args.seed
    trainers = ["REALNVP", "GLOW"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Dataset and projector setup
    # Define plane constraint and set up projector
    A = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).unsqueeze(0).to(device)
    b = torch.tensor([1.0], dtype=torch.float32).to(device)
    projector = SimpleConstraintProjector(device=device)
    projector.add_linear_equality(A, b)

    ds = SmileyFaceDataset(
        device=device,
        num_samples=10000,
        A=A.cpu().numpy().reshape(-1),
        b=float(b.cpu().numpy().reshape(())),
        lifted=False,
        noise_level=0.0,
        projection_type="plane",
        embed_mode="basis",
        seed=seed,
    )
    base_np = ds.data.cpu().numpy().astype(np.float32)
    # Convert true dataset points on the plane to intrinsic 2D coordinates
    true_intrinsic = to_intrinsic_2d_plane(torch.tensor(base_np, dtype=torch.float32), A.cpu(), b.cpu())
    true_intrinsic = ensure_tensor_2d(true_intrinsic, 2)
    true_intrinsic = filter_valid_samples(true_intrinsic).cpu()

    # Helper to instantiate trainers without training
    def instantiate_trainer(tag: str, save_dir: str, batch_size: int = 256, hidden_dim: int = 64):
        argsj = load_training_args(problem_dir, tag, epochs)
        bs = int(argsj.get("batch_size", batch_size))
        hd = int(argsj.get("hidden_dim", hidden_dim))
        if tag == "REALNVP":
            n_coupling = int(argsj.get("n_coupling_layers", 6))
            return RealNVPTrainer(dataset=torch.tensor(base_np), batch_size=bs, hidden_dim=hd, n_coupling_layers=n_coupling, save_dir=save_dir, device=device)
        if tag == "GLOW":
            # Glow reshape to (N,1,1,D) for vector data to match training convention
            np_data = np.array(base_np, dtype=np.float32)
            if np_data.ndim == 2:
                N, D = np_data.shape
                np_img = np_data.reshape(N, 1, 1, D)
            else:
                np_img = np_data
            return GlowTrainer(dataset=np_img, image_size=1, batch_size=bs, epochs=1, save_dir=save_dir, device=device)
        # Only RealNVP and Glow are supported.
        raise ValueError(f"Unknown trainer tag: {tag}")

    # Helper: sample and project then convert to intrinsic 2D
    def sample_and_project(trainer, projector, n_samples: int, device):
        with torch.no_grad():
            out = trainer.sample(n_samples)
            # RealNVPTrainer returns (np_array, logpx) or (tensor, None)
            if isinstance(out, (list, tuple)):
                samples_np = out[0]
            else:
                samples_np = out
            if not torch.is_tensor(samples_np):
                samples_t = torch.tensor(samples_np, dtype=torch.float32, device=device)
            else:
                samples_t = samples_np.to(device)
            if samples_t.ndim > 2:
                samples_t = samples_t.reshape(samples_t.shape[0], -1)
            proj, _, _ = projector.project(samples_t)
        return proj.cpu().numpy()

    # Metric bin count (do not precompute edges; use automatic binning per call)
    bins = 25

    coverage_curves = {tag: [] for tag in trainers}
    jsd_curves = {tag: [] for tag in trainers}
    tvd_curves = {tag: [] for tag in trainers}
    coverage_std = {tag: [] for tag in trainers}
    jsd_std = {tag: [] for tag in trainers}
    tvd_std = {tag: [] for tag in trainers}
    baseline = {tag: {} for tag in trainers}
    # New: log-det Jacobian norm tracking (ambient space, before projection)
    logdet_norm_curves = {tag: [] for tag in trainers}
    logdet_norm_std = {tag: [] for tag in trainers}
    baseline_logdet = {tag: {} for tag in trainers}

    problem_dir = os.path.join(ROOT_DIR, "models", "smileyface_plane")

    metrics_path = os.path.join(results_dir, "metrics_varied_sigmas.json")
    # Always recompute metrics regardless of existing JSON.

    if not args.quiet:
        print(f"Computing intrinsic metrics for trainers {trainers} over {len(sigma_list)} sigmas with {trials} trials and {num_samples_eval} samples per trial...")
    if not args.quiet:
        print("Strict checkpoint mode: expecting EXACT filenames; will fail if any are missing.")

    # For each sigma and trainer, load checkpoint, sample, project, intrinsic, metrics
    for tag in trainers:
        if not args.quiet:
            print(f"[Trainer {tag}] Initializing trainer and computing baseline (sigma=0.0)...")
        save_dir = os.path.join(problem_dir, tag.lower())
        trainer = instantiate_trainer(tag, save_dir)
        try:
            trainer.model.to(device)
        except Exception:
            pass

        # Baseline at sigma=0.0 (must exist)
        try:
            ckpt0, path0 = load_unified_checkpoint_strict(problem_dir, tag, epochs, 0.0, time_cond, seed)
            if not args.quiet:
                print(f"Loaded baseline checkpoint: {path0}")
        except FileNotFoundError as e:
            print(str(e))
            raise SystemExit(1)
        state_dict0 = extract_state_dict(ckpt0)
        if state_dict0 is not None:
            report0 = verify_state_dict_compat(trainer.model, state_dict0)
            if report0 and (report0["missing"] or report0["unexpected"] or report0["shape_mismatch"]):
                print(f"[{tag}] Baseline architecture mismatch:")
                if report0["missing"]:
                    print(f"  Missing keys: {report0['missing'][:10]}{' ...' if len(report0['missing'])>10 else ''}")
                if report0["unexpected"]:
                    print(f"  Unexpected keys: {report0['unexpected'][:10]}{' ...' if len(report0['unexpected'])>10 else ''}")
                if report0["shape_mismatch"]:
                    print(f"  Shape mismatches (first 5): {report0['shape_mismatch'][:5]}")
            try:
                trainer.model.load_state_dict(state_dict0, strict=True)
            except Exception:
                try:
                    trainer.model.load_state_dict(state_dict0, strict=False)
                except Exception:
                    pass
        cov_vals0, jsd_vals0, tvd_vals0 = [], [], []
        # collect log-det norms at baseline (ambient, unprojected)
        logdet_vals0 = []
        for trial_idx in range(trials):
            # Sample from the model
            with torch.no_grad():
                out = trainer.sample(num_samples_eval)
                samples_np = out[0] if isinstance(out, (list, tuple)) else out
            # Convert to tensor on device
            samples_t = torch.tensor(samples_np, dtype=torch.float32, device=device) if not torch.is_tensor(samples_np) else samples_np.to(device)
            # Flatten if necessary for model forward
            if samples_t.ndim > 2:
                samples_flat = samples_t.reshape(samples_t.shape[0], -1)
            else:
                samples_flat = samples_t
            # Ambient-space log-det via forward (x -> z transformation)
            try:
                fwd_out = trainer.model(samples_flat)
                if isinstance(fwd_out, (list, tuple)) and len(fwd_out) >= 2:
                    _, logdet = fwd_out[0], fwd_out[1]
                else:
                    # If no logdet provided, treat as zeros
                    B = samples_flat.shape[0]
                    logdet = torch.zeros(B, device=device)
                # Reduce any extra dims and take absolute value as "norm"
                if logdet.dim() > 1:
                    logdet = logdet.view(logdet.shape[0], -1).sum(dim=1)
                logdet_vals0.append(float(torch.abs(logdet).mean().detach().cpu()))
            except Exception:
                # If model forward fails (e.g., adapter without forward logdet), record NaN
                logdet_vals0.append(float('nan'))
            # Constrained pathway for intrinsic metrics
            proj_np = projector.project(samples_flat)[0].cpu().numpy()
            intrinsic = to_intrinsic_2d_plane(torch.tensor(proj_np), A.cpu(), b.cpu())
            intrinsic = ensure_tensor_2d(intrinsic, 2)
            intrinsic = filter_valid_samples(intrinsic).cpu()
            cov_vals0.append(float(coverage(true_intrinsic, intrinsic)))
            jsd_vals0.append(float(jsd_histogram_2d(intrinsic, true_intrinsic, bins=bins)))
            tvd_vals0.append(float(tvd_histogram_2d(intrinsic, true_intrinsic, bins=bins)))
            if not args.quiet:
                print(f"  Baseline trial {trial_idx+1}/{trials} done.")
        # Aggregate baseline logdet norm
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

        # Sweep sigmas
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
            state_dict = extract_state_dict(ckpt)
            if state_dict is not None:
                report = verify_state_dict_compat(trainer.model, state_dict)
                if report and (report["missing"] or report["unexpected"] or report["shape_mismatch"]):
                    print(f"[{tag}] Sigma {sigma:g} architecture mismatch:")
                    if report["missing"]:
                        print(f"  Missing keys: {report['missing'][:10]}{' ...' if len(report['missing'])>10 else ''}")
                    if report["unexpected"]:
                        print(f"  Unexpected keys: {report['unexpected'][:10]}{' ...' if len(report['unexpected'])>10 else ''}")
                    if report["shape_mismatch"]:
                        print(f"  Shape mismatches (first 5): {report['shape_mismatch'][:5]}")
                try:
                    trainer.model.load_state_dict(state_dict, strict=True)
                except Exception:
                    try:
                        trainer.model.load_state_dict(state_dict, strict=False)
                    except Exception:
                        pass
            cov_vals, jsd_vals, tvd_vals = [], [], []
            logdet_vals = []
            for trial_idx in range(trials):
                # Sample from the model
                with torch.no_grad():
                    out = trainer.sample(num_samples_eval)
                    samples_np = out[0] if isinstance(out, (list, tuple)) else out
                samples_t = torch.tensor(samples_np, dtype=torch.float32, device=device) if not torch.is_tensor(samples_np) else samples_np.to(device)
                if samples_t.ndim > 2:
                    samples_flat = samples_t.reshape(samples_t.shape[0], -1)
                else:
                    samples_flat = samples_t
                # Ambient-space logdet
                try:
                    fwd_out = trainer.model(samples_flat)
                    if isinstance(fwd_out, (list, tuple)) and len(fwd_out) >= 2:
                        _, logdet = fwd_out[0], fwd_out[1]
                    else:
                        B = samples_flat.shape[0]
                        logdet = torch.zeros(B, device=device)
                    if logdet.dim() > 1:
                        logdet = logdet.view(logdet.shape[0], -1).sum(dim=1)
                    logdet_vals.append(float(torch.abs(logdet).mean().detach().cpu()))
                except Exception:
                    logdet_vals.append(float('nan'))
                # Project for intrinsic metrics
                proj_np = projector.project(samples_flat)[0].cpu().numpy()
                intrinsic = to_intrinsic_2d_plane(torch.tensor(proj_np), A.cpu(), b.cpu())
                intrinsic = ensure_tensor_2d(intrinsic, 2)
                intrinsic = filter_valid_samples(intrinsic).cpu()
                cov_vals.append(float(coverage(true_intrinsic, intrinsic)))
                jsd_vals.append(float(jsd_histogram_2d(intrinsic, true_intrinsic, bins=bins)))
                tvd_vals.append(float(tvd_histogram_2d(intrinsic, true_intrinsic, bins=bins)))
                if not args.quiet:
                    print(f"    Trial {trial_idx+1}/{trials} done.")
            # Aggregate
            cov_arr = np.array(cov_vals)
            jsd_arr = np.array(jsd_vals)
            tvd_arr = np.array(tvd_vals)
            logdet_arr = np.array(logdet_vals)
            coverage_curves[tag].append(float(np.nanmean(cov_arr)) if np.isfinite(cov_arr).any() else float('nan'))
            jsd_curves[tag].append(float(np.nanmean(jsd_arr)) if np.isfinite(jsd_arr).any() else float('nan'))
            tvd_curves[tag].append(float(np.nanmean(tvd_arr)) if np.isfinite(tvd_arr).any() else float('nan'))
            coverage_std[tag].append(float(np.nanstd(cov_arr)) if np.isfinite(cov_arr).sum() >= 2 else 0.0)
            jsd_std[tag].append(float(np.nanstd(jsd_arr)) if np.isfinite(jsd_arr).sum() >= 2 else 0.0)
            tvd_std[tag].append(float(np.nanstd(tvd_arr)) if np.isfinite(tvd_arr).sum() >= 2 else 0.0)
            logdet_norm_curves[tag].append(float(np.nanmean(logdet_arr)) if np.isfinite(logdet_arr).any() else float('nan'))
            logdet_norm_std[tag].append(float(np.nanstd(logdet_arr)) if np.isfinite(logdet_arr).sum() >= 2 else 0.0)
            if not args.quiet:
                print(f"[Trainer {tag}] Sigma {sigma:g} metrics: coverage={coverage_curves[tag][-1]:.4f}±{coverage_std[tag][-1]:.4f}, JSD={jsd_curves[tag][-1]:.4e}±{jsd_std[tag][-1]:.4e}, TVD={tvd_curves[tag][-1]:.4e}±{tvd_std[tag][-1]:.4e}, |logdetJ|={logdet_norm_curves[tag][-1]:.4e}±{logdet_norm_std[tag][-1]:.4e}")

    # Save metrics JSON
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

    # Build combined figure from the saved JSON
    if not args.quiet:
        print("Building combined figure from freshly computed JSON...")
    build_combined_figure_from_json(results_dir)
    if not args.quiet:
        print("Done.")
