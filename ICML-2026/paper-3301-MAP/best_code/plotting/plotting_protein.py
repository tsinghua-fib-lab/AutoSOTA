def _safe_scalar(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float('nan')
import os
import sys

# Move to the project root (one level up from current file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import *
from datasets import _unnormalize_backbone_fragment, _normalize_backbone_fragment
from trainers import *
from utils.constraints import SimpleConstraintProjector
from utils.metrics import *
from utils.plotting import *

epochs = 1000
noise_level = 0.001
num_samples = 1000
timesteps = 250
# keep a random_seed variable next to other hyperparameters so plotting scripts
# select matching checkpoints (checkpoints now include _seed_{seed} in filenames)
hidden_dim = 1024
time_embed_dim = 16
time_embed_choice = "default"
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

from utils.plotting import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kabsch_rmsd_pairwise(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Pairwise Kabsch RMSD between two sets of point clouds.

    P: (M,N,3)
    Q: (K,N,3)

    Returns: (M,K) RMSD (Å)
    """
    assert P.dim() == 3 and Q.dim() == 3 and P.shape[2] == 3 and Q.shape[2] == 3
    M, N, _ = P.shape
    K = Q.shape[0]
    if M == 0 or K == 0 or N == 0:
        return torch.empty((M, K), device=P.device, dtype=P.dtype)

    Pc = P - P.mean(dim=1, keepdim=True)
    Qc = Q - Q.mean(dim=1, keepdim=True)

    p2 = (Pc * Pc).sum(dim=(1, 2))  # (M,)
    q2 = (Qc * Qc).sum(dim=(1, 2))  # (K,)

    H = torch.einsum("mna,knb->mkab", Pc, Qc)  # (M,K,3,3)

    Hf = H.reshape(-1, 3, 3)
    U, S, Vh = torch.linalg.svd(Hf, full_matrices=False)

    V = Vh.transpose(-2, -1)
    Ut = U.transpose(-2, -1)
    d = torch.det(V @ Ut)  # (M*K,)

    s3 = S[:, 2]
    trace = S[:, 0] + S[:, 1] + torch.where(d < 0, -s3, s3)  # (M*K,)

    trace = trace.reshape(M, K)
    rmsd2 = (p2[:, None] + q2[None, :] - 2.0 * trace) / float(N)
    rmsd2 = rmsd2.clamp_min(0.0)
    return torch.sqrt(rmsd2 + eps)


def _attach_time_embed_if_needed(denoiser, state_dict, device):
    """Placeholder function - time embedding utilities have been removed."""
    pass
import sidechainnet as scn

data = scn.load(name="casp12", with_coordinates=True)
# Extract fragments
fragments = extract_backbone_fragments(
    data, fragment_length=10, max_data_length=num_samples
)
fragments = fragments
dataset = BackboneFragmentDataset(
    fragments, noise_level=noise_level, lifted=True, seed=random_seed
)
projector = ProteinConstraintProjector(device=device)
# Higher-accuracy projector for evaluation (more GN steps, tighter tol)
eval_projector = ProteinConstraintProjector(device=device)
try:
    eval_projector.optimizer.max_iter = 50
    eval_projector.optimizer.tol_max_res = 1e-6
    eval_projector.optimizer.step_size = 0.8
except Exception:
    pass
indices = np.arange(len(dataset))
data_points = dataset.get_batch(indices)
print(f"Dataset size: {data_points.shape}")

# Number of trials to average timings/metrics over
n_trials = 3

# Lifted Diffusion score
print("Lifted Diffusion Model")
trainer = DDPMTrainer(
    data_points.squeeze(),
    project_x0_sample=True,
    timesteps=timesteps,
    projector=projector,
    hidden_dim=hidden_dim,
    time_embed_dim=time_embed_dim,
    size=data_points.shape[1],
    unet=True
)
checkpoint_path = f"models/protein/model_DDPM_epoch_{epochs}_noise_level_{noise_level}_time_{time_embed_choice}_seed_{random_seed}.pth"
# if not os.path.exists(checkpoint_path):
#     checkpoint_path = f'models/protein/model_DDPM_epoch_{epochs}_noise_level_{noise_level}.pth'
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
trainer.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
torch.cuda.empty_cache()
trainer.denoiser.eval()
trainer.denoiser.to(device)
with torch.no_grad():
    samples_lifted, _ = trainer.sample(num_samples=num_samples)

dataset_plain = BackboneFragmentDataset(
    fragments, noise_level=0.0, lifted=False, seed=random_seed
)
indices = np.arange(len(dataset))
data_points_plain = dataset_plain.get_batch(indices)
# Traditional DDPM Score
print("Traditional Diffusion Model")
trainer_plain = DDPMTrainer(
    data_points_plain.squeeze(),
    project_x0_sample=False,
    timesteps=timesteps,
    projector=projector,
    hidden_dim=hidden_dim,
    time_embed_dim=time_embed_dim,
    size=data_points_plain.shape[1],
    unet=True
)
checkpoint_path = f"models/protein/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
# if not os.path.exists(checkpoint_path):
#     checkpoint_path = f'models/protein/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0.pth'
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
trainer_plain.load_checkpoint(
    checkpoint_path, map_location=device, load_optimizer=False
)
trainer_plain.denoiser.eval()
with torch.no_grad():
    samples_plain, norms = trainer_plain.sample(num_samples=num_samples)
    proj_time_plain_projection = float("nan")
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # Unnormalize to Ångströms, project, then re-normalize for metrics
        samples_plain_reshaped = samples_plain.reshape(-1, 10, 3, 3)
        samples_plain_ang = np.stack([_unnormalize_backbone_fragment(frag, mean_caca=3.8) for frag in samples_plain_reshaped], axis=0)
        samples_plain_projected_ang, _, _ = eval_projector.project(
            torch.tensor(samples_plain_ang).cpu()
        )
        samples_plain_projected_ang_np = samples_plain_projected_ang.cpu().numpy()
        samples_plain_projected = np.stack([_normalize_backbone_fragment(frag, target_caca=1.0) for frag in samples_plain_projected_ang_np], axis=0)
        samples_plain_projected = samples_plain_projected.reshape(samples_plain.shape[0], -1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        proj_time_plain_projection = float(t1 - t0)
        print("Average deviation of Traditional DDPM samples from the plane:", norms)
        # propagate this external projection timing into trainer_plain aggregates when present
        try:
            tr = locals().get("trainer_plain", None)
            if tr is not None:
                if not hasattr(tr, "total_projection_sample_time"):
                    tr.total_projection_sample_time = 0.0
                if not hasattr(tr, "projection_sample_times"):
                    tr.projection_sample_times = []
                if not hasattr(tr, "projection_sample_count"):
                    tr.projection_sample_count = 0
                tr.total_projection_sample_time = (
                    float(tr.total_projection_sample_time) + proj_time_plain_projection
                )
                tr.projection_sample_times.append(proj_time_plain_projection)
                tr.projection_sample_count = (
                    int(getattr(tr, "projection_sample_count", 0)) + 1
                )
                tr.avg_projection_sample_time = (
                    float(tr.total_projection_sample_time / tr.projection_sample_count)
                    if tr.projection_sample_count > 0
                    else float("nan")
                )
                try:
                    if hasattr(tr, "sampling_time") and not (
                        _np.isnan(getattr(tr, "sampling_time"))
                        if hasattr(_np, "isnan")
                        else False
                    ):
                        tr.sampling_time = (
                            float(getattr(tr, "sampling_time", 0.0))
                            + proj_time_plain_projection
                        )
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        try:
            # Fallback: unnormalize, project, re-normalize
            samples_plain_reshaped = samples_plain.reshape(-1, 10, 3, 3)
            samples_plain_ang = np.stack([_unnormalize_backbone_fragment(frag, mean_caca=3.8) for frag in samples_plain_reshaped], axis=0)
            samples_plain_projected_ang, _, _ = eval_projector.project(
                torch.tensor(samples_plain_ang).cpu()
            )
            samples_plain_projected_ang_np = samples_plain_projected_ang.cpu().numpy()
            samples_plain_projected = np.stack([_normalize_backbone_fragment(frag, target_caca=1.0) for frag in samples_plain_projected_ang_np], axis=0)
            samples_plain_projected = samples_plain_projected.reshape(samples_plain.shape[0], -1)
            print(
                "Average deviation of Traditional DDPM samples from the plane:", norms
            )
        except Exception:
            samples_plain_projected = torch.tensor([])

# PDM
print("Projected Diffusion Model")
trainer_PDM = DDPMTrainer(
    data_points_plain.squeeze(),
    project_x0_sample=True,
    timesteps=timesteps,
    hidden_dim=hidden_dim,
    time_embed_dim=time_embed_dim,
    projector=projector,
    size=data_points_plain.shape[1],
    unet=True
)
checkpoint_path = f"models/protein/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
# if not os.path.exists(checkpoint_path):
#     checkpoint_path = f'models/protein/model_DDPM_epoch_{epochs}_noise_level_0.0.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)
_attach_time_embed_if_needed(
    trainer_PDM.denoiser, state_dict if isinstance(state_dict, dict) else {}, device
)
has_timeembed = (
    any(k.startswith("time_embed_module.") for k in state_dict.keys())
    if isinstance(state_dict, dict)
    else False
)
if (
    not has_timeembed
    and getattr(trainer_PDM.denoiser, "time_embed_module", None) is not None
):
    trainer_PDM.denoiser.time_embed_module = None
trainer_PDM.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
with torch.no_grad():
    samples_PDM, norms_PDM = trainer_PDM.sample(num_samples=num_samples, PDM=True)
print("Average deviation of PDM samples from the plane:", norms_PDM)
# Project the PDM samples. Unnormalize to Ångströms, project, then re-normalize.
# samples_PDM shape: (num_samples, 90) or (num_samples, 10*3*3)
samples_PDM_reshaped = samples_PDM.reshape(-1, 10, 3, 3)  # (N, 10, 3, 3)
# Unnormalize each fragment in the batch
samples_PDM_ang = np.stack([_unnormalize_backbone_fragment(frag, mean_caca=3.8) for frag in samples_PDM_reshaped], axis=0)
samples_PDM_projected_ang, _, _ = eval_projector.project(torch.tensor(samples_PDM_ang).cpu())
# Re-normalize each fragment
samples_PDM_projected_ang_np = samples_PDM_projected_ang.cpu().numpy()
samples_PDM = np.stack([_normalize_backbone_fragment(frag, target_caca=1.0) for frag in samples_PDM_projected_ang_np], axis=0)
samples_PDM = torch.tensor(samples_PDM.reshape(samples_PDM.shape[0], -1), dtype=torch.float32)

print("Physics-Informed Diffusion Model")
trainer_PIDM = DDPMTrainer(
    data_points_plain.squeeze(),
    hidden_dim=hidden_dim,
    timesteps=timesteps,
    time_embed_dim=time_embed_dim,
    project_x0_sample=False,
    projector=projector,
    size=data_points_plain.shape[1],
    unet=True
)
checkpoint_path = f"models/protein/model_PIDM_epoch_1000_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
# if not os.path.exists(checkpoint_path):
#     checkpoint_path = f'models/protein/model_PIDM_epoch_{epochs}_noise_level_0.0.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)
_attach_time_embed_if_needed(
    trainer_PIDM.denoiser, state_dict if isinstance(state_dict, dict) else {}, device
)
has_timeembed = (
    any(k.startswith("time_embed_module.") for k in state_dict.keys())
    if isinstance(state_dict, dict)
    else False
)
if (
    not has_timeembed
    and getattr(trainer_PIDM.denoiser, "time_embed_module", None) is not None
):
    trainer_PIDM.denoiser.time_embed_module = None
trainer_PIDM.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
with torch.no_grad():
    samples_PIDM, norms_PIDM = trainer_PIDM.sample(num_samples=num_samples)
print("Average deviation of PIDM samples from the plane:", norms_PIDM)

import matplotlib.pyplot as plt

scores_conv = np.array(trainer.scores, dtype=np.float64)
scores = np.array(trainer_plain.scores, dtype=np.float64)

# Use the unified helper to plot scores vs time (handles NaN/Inf, logscale, colorbars, etc.)
os.makedirs("results/protein", exist_ok=True)
plot_scores_vs_time(
    scores_list=[scores_conv],
    scores_plain=scores,
    sigma_list=[float(noise_level)],
    output_path="results/protein/scores.pdf",
    logscale=True,
)

# Plotting
samples_lifted = samples_lifted.reshape(-1, 10, 3, 3).squeeze()
samples_plain = samples_plain.reshape(-1, 10, 3, 3).squeeze()
samples_plain_projected = samples_plain_projected.reshape(-1, 10, 3, 3).squeeze()
samples_PDM = samples_PDM.reshape(-1, 10, 3, 3).squeeze()
samples_PIDM = samples_PIDM.reshape(-1, 10, 3, 3).squeeze()
data_points = data_points.reshape(-1, 10, 3, 3).squeeze()
data_points_plain = data_points_plain.reshape(-1, 10, 3, 3).squeeze()

# Compute flattened per-sample feature dimension (e.g. residues * atoms_per_residue * coords)
# data_points_plain is shaped (num_samples, L, 3, 3) at this point, so the per-sample
# flattened dimension is the product of the trailing axes.
D = int(np.prod(data_points_plain.shape[1:]))

# Use torch.as_tensor to avoid copy-construction warnings when input is already a tensor
samples_PDM_tensor = filter_valid_samples(torch.as_tensor(samples_PDM).view(-1, D)).cpu()
samples_PIDM_tensor = filter_valid_samples(torch.as_tensor(samples_PIDM).view(-1, D)).cpu()
samples_lifted_tensor = filter_valid_samples(torch.as_tensor(samples_lifted).view(-1, D)).cpu()
samples_plain_tensor = filter_valid_samples(torch.as_tensor(samples_plain).view(-1, D)).cpu()
true_tensor = filter_valid_samples(data_points_plain.view(-1, D)).cpu()
samples_plain_projected_tensor = filter_valid_samples(
    torch.as_tensor(samples_plain_projected).view(-1, D)
).cpu()


def _project_backbone_for_coverage(samples_4d):
    samples_4d = np.asarray(samples_4d).reshape(-1, 10, 3, 3)
    samples_ang = np.stack(
        [_unnormalize_backbone_fragment(frag, mean_caca=3.8) for frag in samples_4d],
        axis=0,
    )
    projected_ang, _, _ = eval_projector.project(torch.tensor(samples_ang).cpu())
    projected_ang_np = projected_ang.cpu().numpy()
    projected_normalized = np.stack(
        [_normalize_backbone_fragment(frag, target_caca=1.0) for frag in projected_ang_np],
        axis=0,
    )
    return filter_valid_samples(
        torch.as_tensor(projected_normalized.reshape(projected_normalized.shape[0], -1))
    ).cpu()


# --- Diversity (pairwise RMSD) ---
def _safe_pairwise_median(tensor):
    try:
        arr = pairwise_rmsd(tensor)
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        a = np.asarray(arr)
        return float(np.nanmedian(a)) if a.size > 0 else float("nan")
    except Exception:
        return float("nan")

med_div_PDM = _safe_pairwise_median(samples_PDM_tensor)
print(f"Median Pairwise RMSD (diversity) of PDM Samples: {med_div_PDM:.3f} Å")
med_div_lifted = _safe_pairwise_median(samples_lifted_tensor)
print(f"Median Pairwise RMSD (diversity) of Lifted Samples: {med_div_lifted:.3f} Å")
med_div_plain = _safe_pairwise_median(samples_plain_tensor)
print(f"Median Pairwise RMSD (diversity) of DDPM Samples: {med_div_plain:.3f} Å")
med_div_plain_projected = _safe_pairwise_median(samples_plain_projected_tensor)
print(f"Median Pairwise RMSD (diversity) of Projected DDPM Samples: {med_div_plain_projected:.3f} Å")
med_div_PIDM = _safe_pairwise_median(samples_PIDM_tensor)
print(f"Median Pairwise RMSD (diversity) of PIDM Samples: {med_div_PIDM:.3f} Å")


KL_phi_PDM, KL_psi_PDM = torsion_angle_KL(samples_PDM, data_points_plain)
KL_phi_lifted, KL_psi_lifted = torsion_angle_KL(samples_lifted, data_points_plain)
KL_phi_plain, KL_psi_plain = torsion_angle_KL(samples_plain, data_points_plain)
KL_phi_projected, KL_psi_projected = torsion_angle_KL(
    samples_plain_projected, data_points_plain
)
KL_phi_PIDM, KL_psi_PIDM = torsion_angle_KL(samples_PIDM, data_points_plain)

print(f"Torsion Angle KL (phi, psi) PDM: {KL_phi_PDM:.4f}, {KL_psi_PDM:.4f}")
print(f"Torsion Angle KL (phi, psi) Lifted: {KL_phi_lifted:.4f}, {KL_psi_lifted:.4f}")
print(f"Torsion Angle KL (phi, psi) DDPM: {KL_phi_plain:.4f}, {KL_psi_plain:.4f}")
print(
    f"Torsion Angle KL (phi, psi) Projected DDPM: {KL_phi_projected:.4f}, {KL_psi_projected:.4f}"
)
print(f"Torsion Angle KL (phi, psi) PIDM: {KL_phi_PIDM:.4f}, {KL_psi_PIDM:.4f}")

# --- MMD (Maximum Mean Discrepancy) as distribution metric ---
mmd_PDM = _safe_scalar(MMD, samples_PDM_tensor, true_tensor, kernel="rbf", bandwidths=[1.0], unbiased=True)
mmd_lifted = _safe_scalar(MMD, samples_lifted_tensor, true_tensor, kernel="rbf", bandwidths=[1.0], unbiased=True)
mmd_plain = _safe_scalar(MMD, samples_plain_tensor, true_tensor, kernel="rbf", bandwidths=[1.0], unbiased=True)
mmd_projected = _safe_scalar(MMD, samples_plain_projected_tensor, true_tensor, kernel="rbf", bandwidths=[1.0], unbiased=True)
mmd_PIDM = _safe_scalar(MMD, samples_PIDM_tensor, true_tensor, kernel="rbf", bandwidths=[1.0], unbiased=True)

print(f"MMD PDM: {mmd_PDM:.4f}")
print(f"MMD Lifted: {mmd_lifted:.4f}")
print(f"MMD DDPM: {mmd_plain:.4f}")
print(f"MMD Projected DDPM: {mmd_projected:.4f}")
print(f"MMD PIDM: {mmd_PIDM:.4f}")

# Collect metrics into a dict and save as LaTeX + CSV
metrics = {
    "PDM": {
        "Coverage": _safe_scalar(coverage, true_tensor, _project_backbone_for_coverage(samples_PDM)),
        "DiversityRMSD": med_div_PDM,
        "MMD": mmd_PDM,
    },
    "PIDM": {
        "Coverage": _safe_scalar(coverage, true_tensor, _project_backbone_for_coverage(samples_PIDM)),
        "DiversityRMSD": med_div_PIDM,
        "MMD": mmd_PIDM,
    },
    "Lifted": {
        "Coverage": _safe_scalar(coverage, true_tensor, _project_backbone_for_coverage(samples_lifted)),
        "DiversityRMSD": med_div_lifted,
        "MMD": mmd_lifted,
    },
    "DDPM": {
        "Coverage": _safe_scalar(coverage, true_tensor, _project_backbone_for_coverage(samples_plain)),
        "DiversityRMSD": med_div_plain,
        "MMD": mmd_plain,
    },
    "ProjectedDDPM": {
        "Coverage": _safe_scalar(coverage, true_tensor, _project_backbone_for_coverage(samples_plain_projected)),
        "DiversityRMSD": med_div_plain_projected,
        "MMD": mmd_projected,
    },
}

# Save metrics to CSV
import pandas as pd
df = pd.DataFrame(metrics).T
df.to_csv("results/protein/metrics.csv")
print(f"Metrics saved to results/protein/metrics.csv")


# Save metrics to LaTeX table with training and sampling time columns
with open("results/protein/metrics.tex", "w") as f:
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Protein Structure Generation Metrics}\n")
    f.write("\\label{tab:protein_metrics}\n")
    f.write("\\begin{tabular}{lcccccc}\n")
    # Build general and intrinsic dicts for the paper-style table. Fill measured times when available.
    sample_time_map = {}
    try:
        sample_time_map["$p_\\sigma$ (ours)"] = avg_stats.get("Lifted", {}).get(
            "s", float("nan")
        )
        sample_time_map["DDPM"] = avg_stats.get("DDPM", {}).get("s", float("nan"))
        sample_time_map["PDM"] = avg_stats.get("PDM", {}).get("s", float("nan"))
        sample_time_map["Proj. DDPM"] = avg_stats.get("DDPM (proj.)", {}).get(
            "s", float("nan")
        )
        sample_time_map["PIDM"] = avg_stats.get("PIDM", {}).get("s", float("nan"))
    except Exception:
        sample_time_map = {
            k: float("nan")
            for k in ["$p_\\sigma$ (ours)", "DDPM", "PDM", "Proj. DDPM", "PIDM"]
        }

    # Build training-time totals from averaged epoch breakdowns
    train_time_map = {}
    try:
        for i, name in enumerate(train_method_names):
            comps = [
                model_t[i] if i < len(model_t) else float("nan"),
                proj_t[i] if i < len(proj_t) else float("nan"),
                backprop_t[i] if i < len(backprop_t) else float("nan"),
                other_t[i] if i < len(other_t) else float("nan"),
            ]
            if all([not np.isfinite(c) for c in comps]):
                total = float("nan")
            else:
                total = float(sum([float(c) for c in comps if np.isfinite(c)]))
            train_time_map[name] = total
        # Map labels used in general_metrics
        train_time_map["$p_\\sigma$ (ours)"] = train_time_map.get("Lifted", float("nan"))
        train_time_map["Proj. DDPM"] = train_time_map.get("DDPM", float("nan"))
    except Exception:
        train_time_map = {k: float("nan") for k in ["Lifted", "PDM", "DDPM", "PIDM"]}

    f.write("\\toprule\n")
    f.write(r"Method & Coverage & Pairwise RMSD & MMD \\" + "\n")
    f.write("\\midrule\n")
    method_order = ["Lifted", "PDM", "PIDM", "DDPM", "ProjectedDDPM"]
    method_labels = {
        "Lifted": "$p_{\\sigma}$ (ours)",
        "PDM": "PDM",
        "PIDM": "PIDM", 
        "DDPM": "DDPM",
        "ProjectedDDPM": "Proj. DDPM"
    }
    for method in method_order:
        m = metrics[method]
        f.write(f"{method_labels[method]} & "
                f"{m['Coverage']:.3f} & "
                f"{m['DiversityRMSD']:.3f} & "
            f"{m['MMD']:.4f} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")
print(f"Metrics saved to results/protein/metrics.tex")

# stacked breakdown of sampling time: model / projection / other
import numpy as _np


def _total_model_time(tr):
    if tr is None:
        return _np.nan
    if hasattr(tr, "total_model_forward_sample_time"):
        try:
            return float(getattr(tr, "total_model_forward_sample_time"))
        except Exception:
            pass
    if hasattr(tr, "model_forward_times"):
        lst = getattr(tr, "model_forward_times") or []
        try:
            return float(_np.sum([float(x) for x in lst])) if len(lst) > 0 else _np.nan
        except Exception:
            return _np.nan
    return _np.nan


def _total_proj_time(tr):
    if tr is None:
        return _np.nan
    if hasattr(tr, "total_projection_sample_time"):
        try:
            return float(getattr(tr, "total_projection_sample_time"))
        except Exception:
            pass
    if hasattr(tr, "projection_sample_times"):
        lst = getattr(tr, "projection_sample_times") or []
        try:
            return float(_np.sum([float(x) for x in lst])) if len(lst) > 0 else _np.nan
        except Exception:
            return _np.nan
    return _np.nan


method_names = ["Lifted", "PDM", "DDPM", "DDPM (proj.)", "PIDM"]
trainers_map = {
    "Lifted": locals().get("trainer", None),
    "PDM": locals().get("trainer_PDM", None),
    "DDPM": locals().get("trainer_plain", None),
    "DDPM (proj.)": locals().get("trainer_plain", None),
    "PIDM": locals().get("trainer_PIDM", None),
}
model_vals = []
proj_vals = []
other_vals = []


def _compute_avg_stats(method_names, trainers_map, n_trials):
    stats = {}
    for name in method_names:
        tr = trainers_map.get(name)
        m0 = _total_model_time(tr)
        p0 = _total_proj_time(tr)
        s0 = getattr(tr, "sampling_time", _np.nan) if tr is not None else _np.nan
        stats[name] = {
            "m_sum": 0.0 if _np.isnan(m0) else float(m0),
            "p_sum": 0.0 if _np.isnan(p0) else float(p0),
            "s_sum": 0.0 if _np.isnan(s0) else float(s0),
            "count": (
                1 if not (_np.isnan(m0) and _np.isnan(p0) and _np.isnan(s0)) else 0
            ),
            "external_proj_list": [],
        }

    # Capture external projection time if it was measured
    try:
        if not _np.isnan(proj_time_plain_projection):
            stats.setdefault("DDPM (proj.)", {"external_proj_list": []})["external_proj_list"].append(proj_time_plain_projection)
    except (NameError, Exception):
        pass

    avg_stats = {}
    for name in method_names:
        if name == "DDPM (proj.)":
            continue
        st = stats.get(name, None)
        if st is None or st.get("count", 0) == 0:
            avg_stats[name] = {"m": _np.nan, "p": _np.nan, "s": _np.nan}
            continue
        cnt = float(st["count"])
        m_avg = (
            float(st["m_sum"]) / cnt
            if st["m_sum"] != 0
            else (st["m_sum"] / cnt if st["m_sum"] == 0 and cnt > 0 else _np.nan)
        )
        p_avg = (
            float(st["p_sum"]) / cnt
            if st["p_sum"] != 0
            else (st["p_sum"] / cnt if st["p_sum"] == 0 and cnt > 0 else _np.nan)
        )
        s_avg = (
            float(st["s_sum"]) / cnt
            if st["s_sum"] != 0
            else (st["s_sum"] / cnt if st["s_sum"] == 0 and cnt > 0 else _np.nan)
        )
        avg_stats[name] = {"m": m_avg, "p": p_avg, "s": s_avg}

    ext_list = stats.get("DDPM (proj.)", {}).get("external_proj_list", [])
    ext_mean = float(_np.mean(ext_list)) if len(ext_list) > 0 else _np.nan
    ddpm_base = avg_stats.get("DDPM", {"m": _np.nan, "p": _np.nan, "s": _np.nan})
    avg_stats["DDPM (proj.)"] = {
        "m": ddpm_base.get("m", _np.nan),
        "p": ext_mean,
        "s": (
            (ddpm_base.get("s", _np.nan) + ext_mean)
            if _np.isfinite(ddpm_base.get("s", _np.nan)) and _np.isfinite(ext_mean)
            else _np.nan
        ),
    }

    return avg_stats


avg_stats = _compute_avg_stats(method_names, trainers_map, n_trials)

for name in method_names:
    stats = avg_stats.get(name, {"m": _np.nan, "p": _np.nan, "s": _np.nan})
    m = stats["m"]
    p = stats["p"]
    s = stats["s"]
    if name == "DDPM (proj.)":
        try:
            extra = float(proj_time_plain_projection)
        except Exception:
            extra = float("nan")
        p = (p if _np.isfinite(p) else 0.0) + (extra if _np.isfinite(extra) else 0.0)
        s = (s if _np.isfinite(s) else 0.0) + (extra if _np.isfinite(extra) else 0.0)
    if _np.isfinite(s) and _np.isfinite(m) and _np.isfinite(p):
        other = max(0.0, float(s) - float(m) - float(p))
    else:
        other = _np.nan
    model_vals.append(m)
    proj_vals.append(p)
    other_vals.append(other)

outdir = "results/protein"
os.makedirs(outdir, exist_ok=True)


# ---- Training time breakdown plot (uses checkpoint-loaded epoch timing breakdowns) ----
def _avg_training_components(tr):
    """Return (model, proj, other) averaged across epochs using epoch_timing_breakdowns."""
    import numpy as _np

    if tr is None:
        return _np.nan, _np.nan, _np.nan
    etb = getattr(tr, "epoch_timing_breakdowns", None)
    if not etb:
        return _np.nan, _np.nan, _np.nan
    model_vals = [
        _np.nan if d is None else d.get("model_forward", _np.nan) for d in etb
    ]
    proj_vals = [_np.nan if d is None else d.get("project", _np.nan) for d in etb]
    backprop_vals = [_np.nan if d is None else d.get("backprop", 0.0) for d in etb]
    other_rest_vals = [
        _np.nan if d is None else (d.get("other", 0.0) + d.get("sampling_to_t0", 0.0))
        for d in etb
    ]
    try:
        m = float(_np.nanmean(model_vals))
    except Exception:
        m = _np.nan
    try:
        p = float(_np.nanmean(proj_vals))
    except Exception:
        p = _np.nan
    try:
        bp = float(_np.nanmean(backprop_vals))
    except Exception:
        bp = _np.nan
    try:
        o = float(_np.nanmean(other_rest_vals))
    except Exception:
        o = _np.nan
    return m, p, bp, o


train_method_names = ["Lifted", "PDM", "DDPM", "PIDM"]
trainers_map_small = {
    "Lifted": locals().get("trainer", None),
    "PDM": locals().get("trainer_PDM", None),
    "DDPM": locals().get("trainer_plain", None),
    "PIDM": locals().get("trainer_PIDM", None),
}
model_t = []
proj_t = []
backprop_t = []
other_t = []
for name in train_method_names:
    tr = trainers_map_small.get(name)
    m, p, bp, o = _avg_training_components(tr)
    model_t.append(m)
    proj_t.append(p)
    backprop_t.append(bp)
    other_t.append(o)

from plotting.paper_tables import write_protein_metrics_table

# Build sampling and training time maps used in the paper table.
sample_time_map = {
    "$p_\\sigma$ (ours)": avg_stats.get("Lifted", {}).get("s", float("nan")),
    "DDPM": avg_stats.get("DDPM", {}).get("s", float("nan")),
    "PDM": avg_stats.get("PDM", {}).get("s", float("nan")),
    "DDPM (proj.)": avg_stats.get("DDPM (proj.)", {}).get("s", float("nan")),
    "PIDM": avg_stats.get("PIDM", {}).get("s", float("nan")),
}

train_time_map = {}
for i, name in enumerate(train_method_names):
    comps = [
        model_t[i] if i < len(model_t) else float("nan"),
        proj_t[i] if i < len(proj_t) else float("nan"),
        backprop_t[i] if i < len(backprop_t) else float("nan"),
        other_t[i] if i < len(other_t) else float("nan"),
    ]
    if all([not np.isfinite(c) for c in comps]):
        total = float("nan")
    else:
        total = float(sum([float(c) for c in comps if np.isfinite(c)]))
    train_time_map[name] = total

rows = []
for key, label in [
    ("Lifted", "$p_\\sigma$ (ours)"),
    ("DDPM", "DDPM"),
    ("PDM", "PDM"),
    ("ProjectedDDPM", "DDPM (proj.)"),
    ("PIDM", "PIDM"),
]:
    vals = metrics[key]
    rows.append(
        {
            "method": label,
            "Train time (s/epoch)": train_time_map.get(key, float("nan")),
            "Sampling time (s)": sample_time_map.get(label, sample_time_map.get(key, float("nan"))),
            "COV": float(vals["Coverage"]),
            "Pairwise RMSD": float(vals["DiversityRMSD"]),
            "MMD": float(vals["MMD"]),
        }
    )

write_protein_metrics_table(
    rows,
    out_tex_path="results/protein/metrics_table.tex",
    caption="Protein backbone fragment metrics at $\\sigma = 0.001$.",
    label="tab:protein_metrics",
)

print("\n--- Model Trainable Parameter Counts ---")
try:
    print(f"Lifted Model (trainer.denoiser): {count_trainable_params(trainer.denoiser):,}")
except Exception as e:
    print(f"Could not count Lifted model params: {e}")
try:
    print(f"Traditional DDPM Model (trainer_plain.denoiser): {count_trainable_params(trainer_plain.denoiser):,}")
except Exception as e:
    print(f"Could not count Traditional DDPM params: {e}")
try:
    print(f"Projected Diffusion Model (trainer_PDM.denoiser): {count_trainable_params(trainer_PDM.denoiser):,}")
except Exception as e:
    print(f"Could not count PDM params: {e}")
try:
    print(f"Physics-Informed Diffusion Model (trainer_PIDM.denoiser): {count_trainable_params(trainer_PIDM.denoiser):,}")
except Exception as e:
    print(f"Could not count PIDM params: {e}")
