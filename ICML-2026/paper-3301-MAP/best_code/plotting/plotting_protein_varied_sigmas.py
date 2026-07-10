import json
import os
import sys
import time
import argparse

# Move to the project root (one level up from current file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import sidechainnet as scn

from datasets import *
from datasets import _unnormalize_backbone_fragment, _normalize_backbone_fragment
from trainers import *
from utils.constraints import SimpleConstraintProjector
from utils.metrics import *
from utils.plotting import *

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


def _save_pdf_png(fig, output_path, **kwargs):
    fig.savefig(output_path, **kwargs)
    if output_path.lower().endswith(".pdf"):
        fig.savefig(output_path[:-4] + ".png", **kwargs)

def _attach_time_embed_if_needed(denoiser, state_dict, device):
    """Placeholder function - time embedding utilities have been removed."""
    pass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Experiment parameters
sigma_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
epochs = 1000
num_samples = 1000
timesteps = 250
hidden_dim = 1024
time_embed_dim = 16
time_embed_choice = "default"
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args, _ = parser.parse_known_args()
random_seed = args.seed
trials = 10
RAMA_BINS = 72

torch.manual_seed(random_seed)
np.random.seed(random_seed)

set_paper_style()

# Load protein data once
data = scn.load(name="casp12", with_coordinates=True)
fragments = extract_backbone_fragments(
    data, fragment_length=10, max_data_length=num_samples
)
projector = ProteinConstraintProjector(device=device)

# Prepare true data (no noise)
dataset_true = BackboneFragmentDataset(
    fragments, noise_level=0.0, lifted=False, seed=random_seed
)
indices = np.arange(len(dataset_true))
data_points_true = dataset_true.get_batch(indices)
print(f"Dataset size: {data_points_true.shape}")

# Compute flattened dimension for metrics
D = int(np.prod(data_points_true.shape[1:]))
true_tensor = filter_valid_samples(data_points_true.view(-1, D)).cpu()

# Initialize trainer once (weights will be reloaded per sigma)
trainer = DDPMTrainer(
    data_points_true.squeeze(),
    project_x0_sample=False,
    timesteps=timesteps,
    projector=projector,
    hidden_dim=hidden_dim,
    time_embed_dim=time_embed_dim,
    size=data_points_true.shape[1],
    unet=True
)

# Accumulators for metrics across sigmas
coverage_list = []
coverage_std_list = []
MMD_list = []
MMD_std_list = []
diversity_RMSD_list = []
diversity_RMSD_std_list = []
MMD_dist_list = []
MMD_dist_std_list = []
KL_phi_list = []
KL_phi_std_list = []
KL_psi_list = []
KL_psi_std_list = []
mmd_list = []
mmd_std_list = []
tm_score_list = []
tm_score_std_list = []
scores_list = []  # per-sigma score curves (from trial 0)
fidelity_RMSD_list = []
fidelity_RMSD_std_list = []

# Evaluate Lifted DDPM models for each sigma
for sigma in sigma_list:
    print(f"\nEvaluating sigma={sigma}")
    
    # Load model for this sigma
    checkpoint_path = f"models/protein/model_DDPM_epoch_{epochs}_noise_level_{sigma}_time_{time_embed_choice}_seed_{random_seed}.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        # Append NaN values
        coverage_list.append(float('nan'))
        coverage_std_list.append(float('nan'))
        MMD_list.append(float('nan'))
        MMD_std_list.append(float('nan'))
        diversity_RMSD_list.append(float('nan'))
        diversity_RMSD_std_list.append(float('nan'))
        MMD_dist_list.append(float('nan'))
        fidelity_RMSD_list.append(float('nan'))
        fidelity_RMSD_std_list.append(float('nan'))
        KL_phi_list.append(float('nan'))
        KL_phi_std_list.append(float('nan'))
        KL_psi_list.append(float('nan'))
        KL_psi_std_list.append(float('nan'))
        tm_score_list.append(float('nan'))
        tm_score_std_list.append(float('nan'))
        scores_list.append([])
        continue
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    _attach_time_embed_if_needed(
        trainer.denoiser, state_dict if isinstance(state_dict, dict) else {}, device
    )
    has_timeembed = (
        any(k.startswith("time_embed_module.") for k in state_dict.keys())
        if isinstance(state_dict, dict)
        else False
    )
    if (
        not has_timeembed
        and getattr(trainer.denoiser, "time_embed_module", None) is not None
    ):
        trainer.denoiser.time_embed_module = None
    trainer.denoiser.load_state_dict(state_dict if isinstance(state_dict, dict) else {})
    torch.cuda.empty_cache()
    trainer.denoiser.eval()
    trainer.denoiser.to(device)
    
    # Run multiple trials
    trial_coverages = []
    trial_mmds = []
    trial_diversity_rmsds = []
    trial_MMD_dist = []
    trial_kl_phis = []
    trial_kl_psis = []
    trial_mmd = []
    trial_tm_scores = []
    trial_fidelity_rmsds = []
    
    for t in range(trials):
        print(f"  Trial {t+1}/{trials}")
        with torch.no_grad():
            samples_lifted, _ = trainer.sample(num_samples=num_samples)
        
        samples_lifted_t = torch.tensor(samples_lifted, device=device).view(-1, 10, 3, 3)
        # Unnormalize to Ångströms before projection (projector expects ~1.46 Å bond lengths)
        samples_lifted_t_np = samples_lifted_t.cpu().numpy()
        samples_lifted_ang = np.stack([_unnormalize_backbone_fragment(frag, mean_caca=3.8) for frag in samples_lifted_t_np], axis=0)
        proj_4d_ang, _, _ = projector.project(torch.tensor(samples_lifted_ang, device=device))
        # Re-normalize for metrics (compare normalized-to-normalized)
        proj_4d_ang_np = proj_4d_ang.cpu().numpy()
        proj_4d_normalized = np.stack([_normalize_backbone_fragment(frag, target_caca=1.0) for frag in proj_4d_ang_np], axis=0)
        samples_lifted_final = proj_4d_normalized.reshape(-1, 10 * 3 * 3)

        # Collect scores from first trial
        if t == 0:
            try:
                scores_list.append(list(trainer.scores))
            except Exception:
                scores_list.append([])
        
        # Reshape for protein metrics
        samples_lifted_reshaped = proj_4d_normalized.reshape(-1, 10, 3, 3).squeeze()
        data_points_reshaped = data_points_true.reshape(-1, 10, 3, 3).squeeze()
        
        # Prepare tensors for general metrics
        samples_tensor = filter_valid_samples(torch.tensor(samples_lifted_final).view(-1, D)).cpu()
        
        # Compute metrics
        try:
            trial_coverages.append(float(coverage(true_tensor, samples_tensor)))
        except Exception:
            trial_coverages.append(float('nan'))
        
        try:
            trial_mmds.append(float(MMD(samples_tensor, true_tensor)))
        except Exception:
            trial_mmds.append(float('nan'))
        
        try:
            diversity_rmsds = pairwise_rmsd(samples_tensor)
            trial_diversity_rmsds.append(float(np.median(diversity_rmsds)))
        except Exception:
            trial_diversity_rmsds.append(float('nan'))
        
        try:
            samples_flat = samples_tensor.view(samples_tensor.shape[0], -1)
            true_flat = true_tensor.view(true_tensor.shape[0], -1)
            mmd_val = MMD(samples_flat, true_flat, kernel="rbf", bandwidths=[1.0], unbiased=True)
            trial_MMD_dist.append(float(mmd_val))
        except Exception:
            trial_MMD_dist.append(float('nan'))
        
        
        
        try:
            KL_phi, KL_psi = torsion_angle_KL(samples_lifted_reshaped, data_points_reshaped)
            trial_kl_phis.append(float(KL_phi))
            trial_kl_psis.append(float(KL_psi))
        except Exception:
            trial_kl_phis.append(float('nan'))
            trial_kl_psis.append(float('nan'))

        try:
            fidelity_rmsd = chamfer_rmsd(samples_tensor, true_tensor)
            if isinstance(fidelity_rmsd, dict):
                fidelity_rmsd = float(
                    fidelity_rmsd.get(
                        "sym",
                        0.5 * (
                            fidelity_rmsd.get("forward", 0.0) + fidelity_rmsd.get("backward", 0.0)
                        ),
                    )
                )
            trial_fidelity_rmsds.append(float(fidelity_rmsd))
        except Exception:
            trial_fidelity_rmsds.append(float('nan'))

# --- MMD with multi-scale RBF kernel ---
        try:
            # MMD with RBF kernel
            samples_flat = samples_tensor.view(samples_tensor.shape[0], -1)
            true_flat = true_tensor.view(true_tensor.shape[0], -1)
            mmd_val = MMD(samples_flat, true_flat, kernel="rbf", bandwidths=[1.0], unbiased=True)
            trial_mmd.append(float(mmd_val))
        except Exception:
            trial_mmd.append(float('nan'))
    
    # Compute mean and std across trials
    coverage_list.append(float(np.nanmean(trial_coverages)))
    coverage_std_list.append(float(np.nanstd(trial_coverages)))
    MMD_list.append(float(np.nanmean(trial_mmds)))
    MMD_std_list.append(float(np.nanstd(trial_mmds)))
    diversity_RMSD_list.append(float(np.nanmean(trial_diversity_rmsds)))
    diversity_RMSD_std_list.append(float(np.nanstd(trial_diversity_rmsds)))
    MMD_dist_list.append(float(np.nanmean(trial_MMD_dist)))
    MMD_dist_std_list.append(float(np.nanstd(trial_MMD_dist)))
    KL_phi_list.append(float(np.nanmean(trial_kl_phis)))
    KL_phi_std_list.append(float(np.nanstd(trial_kl_phis)))
    KL_psi_list.append(float(np.nanmean(trial_kl_psis)))
    KL_psi_std_list.append(float(np.nanstd(trial_kl_psis)))
    mmd_list.append(float(np.nanmean(trial_mmd)))
    mmd_std_list.append(float(np.nanstd(trial_mmd)))
    tm_score_list.append(float(np.nanmean(trial_tm_scores)))
    tm_score_std_list.append(float(np.nanstd(trial_tm_scores)))
    fidelity_RMSD_list.append(float(np.nanmean(trial_fidelity_rmsds)))
    fidelity_RMSD_std_list.append(float(np.nanstd(trial_fidelity_rmsds)))

# Evaluate baseline DDPM (NONPROJECT)
print("\n\nEvaluating baseline DDPM")
trainer_plain = DDPMTrainer(
    data_points_true.squeeze(),
    project_x0_sample=False,
    timesteps=timesteps,
    projector=projector,
    hidden_dim=hidden_dim,
    time_embed_dim=time_embed_dim,
    size=data_points_true.shape[1],
    unet=True
)
checkpoint_path = f"models/protein/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)
_attach_time_embed_if_needed(
    trainer_plain.denoiser, state_dict if isinstance(state_dict, dict) else {}, device
)
has_timeembed = (
    any(k.startswith("time_embed_module.") for k in state_dict.keys())
    if isinstance(state_dict, dict)
    else False
)
if (
    not has_timeembed
    and getattr(trainer_plain.denoiser, "time_embed_module", None) is not None
):
    trainer_plain.denoiser.time_embed_module = None
trainer_plain.denoiser.load_state_dict(state_dict if isinstance(state_dict, dict) else {})
trainer_plain.denoiser.eval()

# Run multiple trials for baseline
plain_coverages = []
plain_mmds = []
plain_diversity_rmsds = []
plain_fidelity_rmsds = []
plain_kl_phis = []
plain_kl_psis = []
plain_mmd = []
plain_tm_scores = []

for t in range(trials):
    print(f"  Trial {t+1}/{trials}")
    with torch.no_grad():
        samples_plain, norms = trainer_plain.sample(num_samples=num_samples)
    
    if t == 0:
        scores_plain = trainer_plain.scores
    
    samples_plain_t = torch.tensor(samples_plain, device=device).view(-1, 10, 3, 3)
    # Unnormalize to Ångströms before projection
    samples_plain_t_np = samples_plain_t.cpu().numpy()
    samples_plain_ang = np.stack([_unnormalize_backbone_fragment(frag, mean_caca=3.8) for frag in samples_plain_t_np], axis=0)
    proj_plain_4d_ang, _, _ = projector.project(torch.tensor(samples_plain_ang, device=device))
    # Re-normalize for metrics (compare normalized-to-normalized)
    proj_plain_4d_ang_np = proj_plain_4d_ang.cpu().numpy()
    proj_plain_4d_normalized = np.stack([_normalize_backbone_fragment(frag, target_caca=1.0) for frag in proj_plain_4d_ang_np], axis=0)
    samples_plain_projected = proj_plain_4d_normalized.reshape(-1, 10 * 3 * 3)
    
    # Reshape
    samples_plain_reshaped = samples_plain_projected.reshape(-1, 10, 3, 3).squeeze()
    data_points_reshaped = data_points_true.reshape(-1, 10, 3, 3).squeeze()
    
    # Prepare tensors
    samples_plain_tensor = filter_valid_samples(torch.tensor(samples_plain_projected).view(-1, D)).cpu()
    
    # Compute metrics
    try:
        plain_coverages.append(float(coverage(true_tensor, samples_plain_tensor)))
    except Exception:
        plain_coverages.append(float('nan'))
    
    try:
        plain_mmds.append(float(MMD(samples_plain_tensor, true_tensor)))
    except Exception:
        plain_mmds.append(float('nan'))
    
    try:
        diversity_rmsds = pairwise_rmsd(samples_plain_tensor)
        plain_diversity_rmsds.append(float(np.median(diversity_rmsds)))
    except Exception:
        plain_diversity_rmsds.append(float('nan'))
    
    try:
        fidelity_rmsd = chamfer_rmsd(samples_plain_tensor, true_tensor)
        if isinstance(fidelity_rmsd, dict):
            fidelity_rmsd = float(
                fidelity_rmsd.get(
                    "sym",
                    0.5 * (
                        fidelity_rmsd.get("forward", 0.0) +
                        fidelity_rmsd.get("backward", 0.0)
                    ),
                )
            )
        plain_fidelity_rmsds.append(float(fidelity_rmsd))
    except Exception:
        plain_fidelity_rmsds.append(float('nan'))
    
    
    
    try:
        KL_phi, KL_psi = torsion_angle_KL(samples_plain_reshaped, data_points_reshaped)
        plain_kl_phis.append(float(KL_phi))
        plain_kl_psis.append(float(KL_psi))
    except Exception:
        plain_kl_phis.append(float('nan'))
        plain_kl_psis.append(float('nan'))

    try:
        # MMD with RBF kernel
        samples_flat = samples_plain_tensor.view(samples_plain_tensor.shape[0], -1)
        true_flat = true_tensor.view(true_tensor.shape[0], -1)
        mmd_val_plain = MMD(samples_flat, true_flat, kernel="rbf", bandwidths=[1.0], unbiased=True)
        plain_mmd.append(float(mmd_val_plain))
    except Exception:
        plain_mmd.append(float('nan'))

coverage_plain = float(np.nanmean(plain_coverages))
coverage_plain_std = float(np.nanstd(plain_coverages))
MMD_plain = float(np.nanmean(plain_mmds))
MMD_plain_std = float(np.nanstd(plain_mmds))
diversity_RMSD_plain = float(np.nanmean(plain_diversity_rmsds))
diversity_RMSD_plain_std = float(np.nanstd(plain_diversity_rmsds))
fidelity_RMSD_plain = float(np.nanmean(plain_fidelity_rmsds))
fidelity_RMSD_plain_std = float(np.nanstd(plain_fidelity_rmsds))
KL_phi_plain = float(np.nanmean(plain_kl_phis))
KL_phi_plain_std = float(np.nanstd(plain_kl_phis))
KL_psi_plain = float(np.nanmean(plain_kl_psis))
KL_psi_plain_std = float(np.nanstd(plain_kl_psis))
mmd_plain = float(np.nanmean(plain_mmd))
mmd_plain_std = float(np.nanstd(plain_mmd))
tm_score_plain = float(np.nanmean(plain_tm_scores))
tm_score_plain_std = float(np.nanstd(plain_tm_scores))

# Save metrics to JSON
output_dir = "results/protein"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "metrics_varied_sigmas.json"), "w") as f:
    json.dump(
        {
            "sigma_list": sigma_list,
            "coverage_mean": coverage_list,
            "coverage_std": coverage_std_list,
            "MMD_mean": MMD_list,
            "MMD_std": MMD_std_list,
            "diversity_RMSD_mean": diversity_RMSD_list,
            "diversity_RMSD_std": diversity_RMSD_std_list,
            "fidelity_RMSD_mean": fidelity_RMSD_list,
            "fidelity_RMSD_std": fidelity_RMSD_std_list,
            "KL_phi_mean": KL_phi_list,
            "KL_phi_std": KL_phi_std_list,
            "KL_psi_mean": KL_psi_list,
            "KL_psi_std": KL_psi_std_list,
            "MMD_mean": mmd_list,
            "MMD_std": mmd_std_list,
            "tm_score_mean": tm_score_list,
            "tm_score_std": tm_score_std_list,
            "plain": {
                "coverage_mean": coverage_plain,
                "coverage_std": coverage_plain_std,
                "MMD_mean": MMD_plain,
                "MMD_std": MMD_plain_std,
                "diversity_RMSD_mean": diversity_RMSD_plain,
                "diversity_RMSD_std": diversity_RMSD_plain_std,
                "fidelity_RMSD_mean": fidelity_RMSD_plain,
                "fidelity_RMSD_std": fidelity_RMSD_plain_std,
                "KL_phi_mean": KL_phi_plain,
                "KL_phi_std": KL_phi_plain_std,
                "KL_psi_mean": KL_psi_plain,
                "KL_psi_std": KL_psi_plain_std,
                "MMD_mean": mmd_plain,
                "MMD_std": mmd_plain_std,
                "tm_score_mean": tm_score_plain,
                "tm_score_std": tm_score_plain_std,
            },
        },
        f,
    )

def build_combined_3panel_figure(
    output_dir,
    sigma_list,
    coverage_list,
    coverage_std_list,
    diversity_RMSD_list,
    diversity_RMSD_std_list,
    MMD_list,
    MMD_std_list,
    coverage_plain,
    coverage_plain_std,
    diversity_RMSD_plain,
    diversity_RMSD_plain_std,
    MMD_plain,
    MMD_plain_std,
):
    """
    Build a 3-panel figure showing Coverage, Pairwise RMSD, and MMD.
    
    Style mirrors the MNIST varied-sigma NF plots:
      - log-scale x axis (sigma)
      - ours curve with markers and shaded error band
      - baseline DDPM horizontal dashed line + band
      - legend stacked (ours vs baseline) placed below axes
    """
    set_paper_style()
    color_map = {"OURS": "#1f77b4", "DDPM": "#d62728"}
    line_width = 3.0
    marker_size = 7
    # Use consistent proportions for 3-panel figures
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharex=True)

    metric_data = [
        ("Coverage", np.array(coverage_list, dtype=float), np.array(coverage_std_list, dtype=float), coverage_plain, coverage_plain_std),
        ("Pairwise RMSD", np.array(diversity_RMSD_list, dtype=float), np.array(diversity_RMSD_std_list, dtype=float), diversity_RMSD_plain, diversity_RMSD_plain_std),
        ("MMD", np.array(MMD_list, dtype=float), np.array(MMD_std_list, dtype=float), MMD_plain, MMD_plain_std),
    ]
    handles = []
    labels = []

    for ax, (label, vals, stds, base_mean, base_std) in zip(axes, metric_data):
        # ours curve
        h, = ax.plot(
            sigma_list,
            vals,
            marker="o",
            markersize=marker_size,
            linewidth=line_width,
            color=color_map["OURS"],
            label=r"$p_{\sigma}$ (ours)",
            zorder=2,
        )
        # error band (min thickness enforcement)
        try:
            arr = vals.astype(float)
            std_arr = stds.astype(float)
            mask = np.isfinite(arr) & np.isfinite(std_arr)
            if mask.any():
                sig = np.array(sigma_list)[mask]
                min_band = 0.005 if label == "Coverage" else 0.002
                delta = np.maximum(std_arr[mask], min_band)
                lower = arr[mask] - delta
                upper = arr[mask] + delta
                if label == "Pairwise RMSD":
                    eps = 1e-12
                    lower = np.maximum(lower, eps)
                    upper = np.maximum(upper, eps)
                elif label == "MMD":
                    # MMD is non-negative
                    lower = np.maximum(lower, 0.0)
                    upper = np.maximum(upper, 0.0)
                ax.fill_between(sig, lower, upper, color=color_map["OURS"], alpha=0.25, linewidth=0, zorder=1)
        except Exception:
            pass

        # Collect legend handle once
        if ax is axes[0]:
            handles.append(h)
            labels.append(r"$p_{\sigma}$ (ours)")

        # baseline line and band
        hb = ax.axhline(
            y=base_mean,
            color=color_map["DDPM"],
            linestyle="--",
            linewidth=line_width,
            alpha=0.9,
            label="DDPM (proj.)",
        )
        if ax is axes[0]:
            handles.append(hb)
            labels.append("DDPM (proj.)")
        if base_std is not None and np.isfinite(base_std):
            if label == "Pairwise RMSD":
                eps = 1e-12
                low = max(base_mean - base_std, eps)
                high = max(base_mean + base_std, eps)
            elif label == "MMD":
                # MMD is non-negative
                low = max(base_mean - base_std, 0.0)
                high = max(base_mean + base_std, 0.0)
            else:
                low = max(base_mean - base_std, 0.0)
                high = max(base_mean + base_std, 0.0)
            ax.fill_between(
                sigma_list,
                [low] * len(sigma_list),
                [high] * len(sigma_list),
                color=color_map["DDPM"],
                alpha=0.10,
            )

        # axes cosmetics
        ax.set_xscale("log")
        ax.set_xlabel("σ")
        ax.set_ylabel(label)
        # Set x-limits and ticks to standard increments for clarity
        try:
            min_sigma = float(min(sigma_list))
            max_sigma = float(max(sigma_list))
            ax.set_xlim(min_sigma, max_sigma)
            # Major ticks at powers of 10 only
            from matplotlib.ticker import LogLocator, FuncFormatter
            ax.xaxis.set_major_locator(LogLocator(base=10.0))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}"))
            # Remove minor tick labels entirely to avoid 5s
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
        except Exception:
            pass
        # Adaptive y-limits with small padding to make lines clearer
        try:
            values_for_ylim = [vals]
            if np.isfinite(base_mean):
                values_for_ylim.append(np.array([base_mean]))
            y_all = np.concatenate([v.flatten() for v in values_for_ylim])
            y_all = y_all[np.isfinite(y_all)]
            if label == "Coverage":
                if y_all.size > 0:
                    ymin = max(0.0, float(y_all.min()) - 0.03)
                    ymax = min(1.0, float(y_all.max()) + 0.03)
                    ax.set_ylim(ymin, ymax)
                else:
                    ax.set_ylim(0, 1)
            else:
                if y_all.size > 0:
                    ymin = 0.0
                    ymax = max(1e-12, float(y_all.max()) * 1.10)
                    ax.set_ylim(ymin, ymax)
                else:
                    ax.set_ylim(bottom=0)
        except Exception:
            pass
        ax.grid(True)

    try:
        ncols = max(1, len(labels))
        fig.legend(handles, labels, loc="lower center", ncol=ncols, frameon=False, bbox_to_anchor=(0.5, -0.14))
    except Exception:
        pass

    fig.tight_layout()
    try:
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.92, wspace=0.35)
    except Exception:
        pass
    out_path = os.path.join(output_dir, "combined_metrics_3panel.pdf")
    try:
        _save_pdf_png(fig, out_path, bbox_inches="tight")
    except Exception:
        out_path = os.path.join(output_dir, "combined_metrics_3panel_fallback.pdf")
        _save_pdf_png(fig, out_path, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"Saved 3-panel combined figure to {out_path}")


# Build the combined 3-panel figure
build_combined_3panel_figure(
    output_dir,
    sigma_list,
    coverage_list,
    coverage_std_list,
    diversity_RMSD_list,
    diversity_RMSD_std_list,
    MMD_list,
    MMD_std_list,
    coverage_plain,
    coverage_plain_std,
    diversity_RMSD_plain,
    diversity_RMSD_plain_std,
    MMD_plain,
    MMD_plain_std,
)

# Scores vs time plot
if "scores_list" in globals() and "scores_plain" in globals():
    cmap = plt.cm.viridis
    try:
        norm = mpl.colors.LogNorm(
            vmin=float(min(sigma_list)), vmax=float(max(sigma_list))
        )
    except Exception:
        norm = mpl.colors.Normalize(
            vmin=float(min(sigma_list)), vmax=float(max(sigma_list))
        )

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(3.5, 4.0))
    nan_label_plotted = False
    for i, sigma in enumerate(sigma_list):
        raw = scores_list[i]
        try:
            scores_list_sigma = np.array([float(x) for x in raw])
        except Exception:
            scores_list_sigma = np.array(raw)
        invalid_mask = ~np.isfinite(scores_list_sigma)
        valid_mask = np.isfinite(scores_list_sigma)
        color = cmap(norm(sigma))
        ax.plot(scores_list_sigma, color=color, linewidth=2.0)
        if np.any(valid_mask):
            top_y = np.nanmax(scores_list_sigma[valid_mask]) * 1.1
        else:
            top_y = 1.0
        if np.any(invalid_mask):
            label = "NaN or Inf" if not nan_label_plotted else ""
            ax.plot(
                np.where(invalid_mask)[0],
                [top_y] * np.sum(invalid_mask),
                "x",
                color=color,
                markersize=5,
                label=label,
            )
            nan_label_plotted = True
    # Define invalid_mask_plain and valid_mask_plain before use
    try:
        scores_plain_plot = np.array([float(x) for x in scores_plain])
    except Exception:
        scores_plain_plot = np.array(scores_plain)
    invalid_mask_plain = ~np.isfinite(scores_plain_plot)
    valid_mask_plain = np.isfinite(scores_plain_plot)
    if np.any(valid_mask_plain):
        top_y_plain = np.nanmax(scores_plain_plot[valid_mask_plain]) * 1.1
    else:
        top_y_plain = 1.0
    ax.plot(scores_plain_plot, label="DDPM", linestyle="--", color="red", linewidth=2.6)
    if np.any(invalid_mask_plain):
        ax.plot(
            np.where(invalid_mask_plain)[0],
            [top_y_plain] * np.sum(invalid_mask_plain),
            "x",
            color="red",
            markersize=5,
            label="NaN or Inf" if not nan_label_plotted else "",
        )

    ax.set_yscale("log")

    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.20)
    cbar.set_label("σ", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    try:
        min_s = float(min(sigma_list))
        max_s = float(max(sigma_list))
        if min_s > 0 and max_s > 0:
            exp_min = int(np.floor(np.log10(min_s)))
            exp_max = int(np.ceil(np.log10(max_s)))
            exps = np.arange(exp_min, exp_max + 1)
            ticks = (10.0**exps).tolist()
            ticks = [t for t in ticks if t >= min_s and t <= max_s]
            if len(ticks) >= 1:
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(
                    [r"$10^{%d}$" % int(np.round(np.log10(t))) for t in ticks]
                )
            else:
                cbar.set_ticks([min_s, max_s])
                cbar.set_ticklabels([f"{min_s:.3g}", f"{max_s:.3g}"])
    except Exception:
        try:
            cbar.set_ticks(sigma_list)
            cbar.set_ticklabels([str(s) for s in sigma_list])
        except Exception:
            pass

    ax.legend(fontsize=9, frameon=False, loc="upper left")
    num_points = len(scores_plain_plot)
    xticks = np.linspace(0, num_points - 1, num=6, dtype=int)
    xtick_labels = [f"{num_points - 1 - x}" for x in xticks]
    xtick_labels[0] = "250"
    xtick_labels[-1] = "0"
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_xlabel(r"$t$ (reversed)", fontsize=11, labelpad=8)
    ax.set_ylabel(
        r"Median $\nabla_x(t) \, \log p_t(x(t))$", fontsize=11
    )
    ax.grid(True)
    fig.subplots_adjust(bottom=0.20)
    _save_pdf_png(fig, os.path.join(output_dir, "scores_vs_time_varied_sigmas.pdf"), bbox_inches="tight")
    plt.close(fig)

print(f"\nResults saved to {output_dir}")
print(f"Metrics JSON: {os.path.join(output_dir, 'metrics_varied_sigmas.json')}")
