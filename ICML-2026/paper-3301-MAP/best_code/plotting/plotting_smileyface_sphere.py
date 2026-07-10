import argparse
import json
import os
import sys
import time

import torch
import trimesh
import numpy as np

# Move to the project root (one level up from current file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from datasets import *
from trainers import *
from utils.constraints import *
from utils.metrics import *
from utils.plotting import *
from utils.timing import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_pdf_png(fig, output_path, **kwargs):
    fig.savefig(output_path, **kwargs)
    if output_path.lower().endswith(".pdf"):
        fig.savefig(output_path[:-4] + ".png", **kwargs)


def _attach_time_embed_if_needed(denoiser, state_dict, device):
    """Placeholder function - time embedding utilities have been removed."""
    pass
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args, _ = parser.parse_known_args()

    sphere_center = [0.0, 0.0, 0.0]
    sphere_radius = 1.0
    num_samples = 10000
    noise_level = 0.05
    epochs = 200
    hidden_dim = 64
    time_concat = True
    timesteps = 250
    time_embed_dim = 32
    # random seed used to select matching checkpoints
    random_seed = args.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # Number of trials to average timings/metrics over
    n_trials = 3
    # Allow choosing time embedding behavior via CLI so filenames and trainer
    # construction can reflect the choice (default/sinusoidal/fourier).
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--time-embed', choices=['default', 'sinusoidal', 'fourier'], default='default', help='Time embedding module to use')
    # args = parser.parse_args()
    time_embed_choice = "default"
    dataset = SmileyFaceDataset(
        device,
        num_samples=num_samples,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        projection_type="sphere",
        lifted=True,
        noise_level=noise_level,
        seed=random_seed,
    )
    data_points = torch.stack([dataset[i] for i in range(len(dataset))])
    constraints_dict = {"sphere_equality": (sphere_center, sphere_radius)}
    # Lifted Diffusion score
    print("Lifted Diffusion Model")
    trainer = DDPMTrainer(
        data_points.squeeze(),
        timesteps=timesteps,
        project_x0_sample=True,
        constraints_dict={"sphere_equality": (sphere_center, sphere_radius)},
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        time_conditioning=time_embed_choice,
        time_concat=time_concat,
    )
    # Prefer a checkpoint that encodes the time embed choice and random seed in the filename when available
    checkpoint_path = f"models/smileyface_sphere/model_DDPM_epoch_{epochs}_noise_level_{noise_level}_time_{time_embed_choice}_seed_{random_seed}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"models/smileyface_sphere/model_DDPM_epoch_{epochs}_noise_level_{noise_level}_time_{time_embed_choice}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"models/smileyface_sphere/model_DDPM_epoch_{epochs}_noise_level_{noise_level}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    _attach_time_embed_if_needed(
        trainer.denoiser, state if isinstance(state, dict) else {}, device
    )
    if isinstance(state, dict):
        has_timeembed = any(k.startswith("time_embed_module.") for k in state.keys())
    else:
        has_timeembed = False
    if (
        not has_timeembed
        and getattr(trainer.denoiser, "time_embed_module", None) is not None
    ):
        print(
            f"Checkpoint {checkpoint_path} has no time_embed_module keys — removing trainer.denoiser.time_embed_module to match checkpoint"
        )
        trainer.denoiser.time_embed_module = None
    # Use trainer.load_checkpoint so timing metadata (epoch_timing_breakdowns, projection_times, etc.) is restored
    trainer.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
    torch.cuda.empty_cache()
    with torch.no_grad():
        samples_lifted, _ = trainer.sample(num_samples=num_samples)
    try:
        samples_lifted = trainer.projector.project(torch.tensor(samples_lifted).cpu())[0].cpu()
    except Exception:
        samples_lifted = torch.tensor(samples_lifted)

    os.makedirs("results/smileyface_sphere", exist_ok=True)

    
    # PDM
    dataset_plain = SmileyFaceDataset(
        device,
        num_samples=num_samples,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        projection_type="sphere",
        lifted=False,
        seed=random_seed,
    )
    data_points_plain = torch.stack(
        [dataset_plain[i] for i in range(len(dataset_plain))]
    )

    # Traditional DDPM Score
    print("Traditional DDPM Model")
    trainer_plain = DDPMTrainer(
        data_points_plain.squeeze(),
        timesteps=timesteps,
        project_x0_sample=False,
        constraints_dict=constraints_dict,
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        time_conditioning=time_embed_choice,
        time_concat=time_concat,
    )
    checkpoint_path = f"models/smileyface_sphere/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"models/smileyface_sphere/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"models/smileyface_sphere/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_plain = checkpoint.get("model_state_dict", checkpoint)
    _attach_time_embed_if_needed(
        trainer_plain.denoiser,
        state_plain if isinstance(state_plain, dict) else {},
        device,
    )
    if isinstance(state_plain, dict):
        has_timeembed_plain = any(
            k.startswith("time_embed_module.") for k in state_plain.keys()
        )
    else:
        has_timeembed_plain = False
    if (
        not has_timeembed_plain
        and getattr(trainer_plain.denoiser, "time_embed_module", None) is not None
    ):
        print(
            f"Checkpoint {checkpoint_path} has no time_embed_module keys — removing trainer_plain.denoiser.time_embed_module to match checkpoint"
        )
        trainer_plain.denoiser.time_embed_module = None
    trainer_plain.load_checkpoint(
        checkpoint_path, map_location=device, load_optimizer=False
    )
    trainer_plain.denoiser.eval()
    with torch.no_grad():
        samples_plain, norms = trainer_plain.sample(num_samples=num_samples)
    mask = ~np.isnan(samples_plain).any(axis=1) & ~np.isinf(samples_plain).any(axis=1)
    samples_plain = samples_plain[mask]
    proj_time_plain_projection = float("nan")
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        samples_plain_projected, _, _ = trainer_plain.projector.project(
            torch.tensor(samples_plain).cpu()
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        proj_time_plain_projection = float(t1 - t0)
        samples_plain_projected = samples_plain_projected.cpu()
    except Exception:
        try:
            samples_plain_projected, _, _ = trainer_plain.projector.project(
                torch.tensor(samples_plain).cpu()
            )
            samples_plain_projected = samples_plain_projected.cpu()
        except Exception:
            samples_plain_projected = torch.tensor([])
    print("Average deviation of Traditional DDPM samples from the plane:", norms)

    # --- ISO DDPM (projected) variant ---
    # Load an isotropic-noising DDPM checkpoint and produce projected samples.
    samples_plain_iso = None
    samples_plain_iso_projected = None
    try:
        iso_ckpt_path = f"models/smileyface_sphere/model_DDPM_epoch_{epochs}_noise_level_{noise_level}_ISO_time_{time_embed_choice}_seed_{random_seed}.pth"
        if not os.path.exists(iso_ckpt_path):
            iso_ckpt_path = f"models/smileyface_sphere/model_DDPM_epoch_{epochs}_noise_level_{noise_level}_ISO_time_{time_embed_choice}.pth"
        if not os.path.exists(iso_ckpt_path):
            iso_ckpt_path = f"models/smileyface_sphere/model_DDPM_epoch_{epochs}_noise_level_{noise_level}_ISO.pth"
        if os.path.exists(iso_ckpt_path):
            trainer_iso_plain = DDPMTrainer(
                data_points_plain.squeeze(),
                timesteps=timesteps,
                project_x0_sample=False,
                constraints_dict=constraints_dict,
                hidden_dim=hidden_dim,
                time_embed_dim=time_embed_dim,
                time_conditioning=time_embed_choice,
                time_concat=time_concat,
            )
            iso_ckpt = torch.load(iso_ckpt_path, map_location=device)
            iso_state = iso_ckpt.get("model_state_dict", iso_ckpt)
            _attach_time_embed_if_needed(
                trainer_iso_plain.denoiser,
                iso_state if isinstance(iso_state, dict) else {},
                device,
            )
            # Align time_embed presence to checkpoint
            try:
                has_timeembed_iso = any(k.startswith("time_embed_module.") for k in (iso_state.keys() if isinstance(iso_state, dict) else []))
            except Exception:
                has_timeembed_iso = False
            if (
                not has_timeembed_iso
                and getattr(trainer_iso_plain.denoiser, "time_embed_module", None) is not None
            ):
                trainer_iso_plain.denoiser.time_embed_module = None
            trainer_iso_plain.load_checkpoint(iso_ckpt_path, map_location=device, load_optimizer=False)
            trainer_iso_plain.denoiser.eval()
            with torch.no_grad():
                samples_plain_iso, _ = trainer_iso_plain.sample(num_samples=num_samples)
            # Filter and project
            try:
                mask_iso = ~np.isnan(samples_plain_iso).any(axis=1) & ~np.isinf(samples_plain_iso).any(axis=1)
                samples_plain_iso = samples_plain_iso[mask_iso]
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                proj_iso, _, _ = trainer_iso_plain.projector.project(torch.tensor(samples_plain_iso).cpu())
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                # Use same external projection timing as non-ISO when composing breakdowns
                samples_plain_iso_projected = proj_iso.cpu()
            except Exception:
                try:
                    proj_iso, _, _ = trainer_iso_plain.projector.project(torch.tensor(samples_plain_iso).cpu())
                    samples_plain_iso_projected = proj_iso.cpu()
                except Exception:
                    samples_plain_iso_projected = torch.tensor([])
        else:
            print("ISO DDPM checkpoint not found; skipping ISO projected variant.")
    except Exception as e:
        print(f"ISO DDPM sampling/projection failed: {e}")

    # PIDM
    print("Physics-Informed Diffusion Model")
    trainer_PIDM = DDPMTrainer(
        data_points_plain.squeeze(),
        timesteps=timesteps,
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        project_x0_sample=False,
        constraints_dict=constraints_dict,
        time_concat=time_concat,
    )
    checkpoint_path = f"models/smileyface_sphere/model_PIDM_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = (
            f"models/smileyface_sphere/model_PIDM_epoch_{epochs}_noise_level_0.0.pth"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_pidm = checkpoint.get("model_state_dict", checkpoint)
    _attach_time_embed_if_needed(
        trainer_PIDM.denoiser,
        state_pidm if isinstance(state_pidm, dict) else {},
        device,
    )
    if isinstance(state_pidm, dict):
        has_timeembed_pidm = any(
            k.startswith("time_embed_module.") for k in state_pidm.keys()
        )
    else:
        has_timeembed_pidm = False
    if (
        not has_timeembed_pidm
        and getattr(trainer_PIDM.denoiser, "time_embed_module", None) is not None
    ):
        print(
            f"Checkpoint {checkpoint_path} has no time_embed_module keys — removing trainer_PIDM.denoiser.time_embed_module to match checkpoint"
        )
        trainer_PIDM.denoiser.time_embed_module = None
    trainer_PIDM.load_checkpoint(
        checkpoint_path, map_location=device, load_optimizer=False
    )
    with torch.no_grad():
        samples_PIDM, _ = trainer_PIDM.sample(num_samples=num_samples)

    # PDM
    print("Projected Diffusion Model")
    trainer_PDM = DDPMTrainer(
        data_points_plain.squeeze(),
        timesteps=timesteps,
        project_x0_sample=True,
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        constraints_dict=constraints_dict,
        time_concat=time_concat,
    )
    checkpoint_path = f"models/smileyface_sphere/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = (
            f"models/smileyface_sphere/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0.pth"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_pdm = checkpoint.get("model_state_dict", checkpoint)
    _attach_time_embed_if_needed(
        trainer_PDM.denoiser, state_pdm if isinstance(state_pdm, dict) else {}, device
    )
    if isinstance(state_pdm, dict):
        has_timeembed_pdm = any(
            k.startswith("time_embed_module.") for k in state_pdm.keys()
        )
    else:
        has_timeembed_pdm = False
    if (
        not has_timeembed_pdm
        and getattr(trainer_PDM.denoiser, "time_embed_module", None) is not None
    ):
        print(
            f"Checkpoint {checkpoint_path} has no time_embed_module keys — removing trainer_PDM.denoiser.time_embed_module to match checkpoint"
        )
        trainer_PDM.denoiser.time_embed_module = None
    trainer_PDM.load_checkpoint(
        checkpoint_path, map_location=device, load_optimizer=False
    )
    with torch.no_grad():
        samples_PDM, _ = trainer_PDM.sample(num_samples=10000, PDM=True)

    import matplotlib.pyplot as plt

    scores_conv = np.array(trainer.scores, dtype=np.float64)
    scores = np.array(trainer_plain.scores, dtype=np.float64)

    # Find indices where values are invalid (NaN or Inf)
    invalid_mask = ~np.isfinite(scores)
    valid_mask = np.isfinite(scores)

    # Plot the valid scores
    plt.plot(scores, label="p_0")
    plt.plot(scores_conv, label="p_sigma")
    # plt.plot(scores_conv, label="p_convolved")
    # Set y-axis to log scale
    plt.yscale("log")
    # Determine a Y position for the X markers
    if np.any(valid_mask):
        top_y = np.nanmax(scores[valid_mask]) * 1.1
    else:
        top_y = 1.0  # fallback value
    # Plot red Xs at invalid indices
    plt.plot(
        np.where(invalid_mask)[0],
        [top_y] * np.sum(invalid_mask),
        "x",
        color="red",
        markersize=1,
        label="NaN or Inf",
    )
    plt.xlabel("T")

    
    # Ensure sphere center/radius are torch tensors before intrinsic conversions
    if not torch.is_tensor(sphere_center):
        sphere_center = torch.tensor(sphere_center, dtype=torch.float32)
    if not torch.is_tensor(sphere_radius):
        sphere_radius = torch.tensor(sphere_radius, dtype=torch.float32)

    # Convert other sample sets to intrinsic 2D coordinates (safe coercion)
    def _to_tensor_safe(x):
        try:
            if x is None:
                return torch.empty((0, 3))
            if torch.is_tensor(x):
                return x
            arr = np.asarray(x)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return torch.tensor(arr, dtype=torch.float32)
        except Exception:
            return torch.empty((0, 3))

    try:
        samples_PDM_2d = to_intrinsic_2d(_to_tensor_safe(samples_PDM), sphere_center, sphere_radius)[0]
    except Exception:
        samples_PDM_2d = torch.empty((0, 2))
    try:
        samples_PIDM_2d = to_intrinsic_2d(_to_tensor_safe(samples_PIDM), sphere_center, sphere_radius)[0]
    except Exception:
        samples_PIDM_2d = torch.empty((0, 2))
    try:
        samples_lifted_2d = to_intrinsic_2d(_to_tensor_safe(samples_lifted), sphere_center, sphere_radius)[0]
    except Exception:
        samples_lifted_2d = torch.empty((0, 2))
    try:
        samples_plain_2d = to_intrinsic_2d(_to_tensor_safe(samples_plain), sphere_center, sphere_radius)[0]
    except Exception:
        samples_plain_2d = torch.empty((0, 2))
    try:
        data_points_plain_2d = to_intrinsic_2d(_to_tensor_safe(data_points_plain), sphere_center, sphere_radius)[0]
    except Exception:
        data_points_plain_2d = torch.empty((0, 2))
    try:
        samples_plain_projected_2d = to_intrinsic_2d(_to_tensor_safe(samples_plain_projected), sphere_center, sphere_radius)[0]
    except Exception:
        samples_plain_projected_2d = torch.empty((0, 2))

    D2 = 2

    # Build full-space (original dimensionality) tensors for general metrics
    try:
        D = int(data_points_plain.shape[1])
    except Exception:
        D = 3

    def _to_tensor_or_empty(x):
        try:
            if x is None:
                return filter_valid_samples(torch.tensor([]).view(-1, D)).cpu()
            if torch.is_tensor(x):
                t = x
            else:
                arr = np.asarray(x)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                t = torch.tensor(arr)
            return filter_valid_samples(t.view(-1, D)).cpu()
        except Exception:
            return filter_valid_samples(torch.tensor([]).view(-1, D)).cpu()

    samples_PDM_tensor = _to_tensor_or_empty(samples_PDM)
    samples_PIDM_tensor = _to_tensor_or_empty(samples_PIDM)
    samples_lifted_tensor = _to_tensor_or_empty(samples_lifted)
    samples_plain_tensor = _to_tensor_or_empty(samples_plain)
    samples_plain_projected_tensor = _to_tensor_or_empty(samples_plain_projected)
    true_tensor = _to_tensor_or_empty(data_points_plain)
    samples_PDM_tensor_2d = ensure_tensor_2d(samples_PDM_2d, D2).cpu()
    samples_PIDM_tensor_2d = ensure_tensor_2d(samples_PIDM_2d, D2).cpu()
    samples_lifted_tensor_2d = ensure_tensor_2d(samples_lifted_2d, D2).cpu()
    samples_plain_tensor_2d = ensure_tensor_2d(samples_plain_2d, D2).cpu()
    true_tensor_2d = ensure_tensor_2d(data_points_plain_2d, D2).cpu()
    samples_plain_projected_tensor_2d = ensure_tensor_2d(
        samples_plain_projected_2d, D2
    ).cpu()
    
    # Compute 2D intrinsic coords for ISO projected samples
    if locals().get('samples_plain_iso_projected', None) is not None:
        try:
            samples_plain_iso_projected_2d = to_intrinsic_2d(
                _to_tensor_safe(samples_plain_iso_projected), 
                sphere_center, 
                sphere_radius
            )[0]
            samples_plain_iso_projected_tensor_2d = ensure_tensor_2d(samples_plain_iso_projected_2d, D2).cpu()
        except Exception:
            samples_plain_iso_projected_tensor_2d = torch.empty((0, D2))
    else:
        samples_plain_iso_projected_tensor_2d = torch.empty((0, D2))

    samples_PDM_tensor_2d = filter_valid_samples(samples_PDM_tensor_2d).cpu()
    samples_PIDM_tensor_2d = filter_valid_samples(samples_PIDM_tensor_2d).cpu()
    samples_lifted_tensor_2d = filter_valid_samples(samples_lifted_tensor_2d).cpu()
    samples_plain_tensor_2d = filter_valid_samples(samples_plain_tensor_2d).cpu()
    true_tensor_2d = filter_valid_samples(true_tensor_2d).cpu()
    samples_plain_projected_tensor_2d = filter_valid_samples(
        samples_plain_projected_tensor_2d
    ).cpu()
    samples_plain_iso_projected_tensor_2d = filter_valid_samples(
        samples_plain_iso_projected_tensor_2d
    ).cpu()

    print("\n--- METRICS IN INTRINSIC 2D COORDINATES (ON PLANE) ---")
    print(f"Coverage (PDM):    {coverage(true_tensor_2d, samples_PDM_tensor_2d)}")
    print(f"Coverage (PIDM):   {coverage(true_tensor_2d, samples_PIDM_tensor_2d)}")
    print(f"Coverage (Lifted): {coverage(true_tensor_2d, samples_lifted_tensor_2d)}")
    print(
        f"Coverage (Proj. DDPM):   {coverage(true_tensor_2d, samples_plain_projected_tensor_2d)}"
    )

    # Precompute histogram bin edges from the TRUE data so all histogram
    # comparisons use the same binning. For intrinsic 2D metrics we compute
    # explicit x/y edges and pass them to the 2D histogram helpers. For the
    # general (original-space) ND metrics we compute per-dimension bin edges
    # from the true 3D data and pass those arrays as the `bins` argument to
    # the ND histogram helpers (they accept a list of bin-edge arrays).
    try:
        true_np_2d = (
            true_tensor_2d.cpu().numpy()
            if hasattr(true_tensor_2d, "cpu")
            else np.asarray(true_tensor_2d)
        )
        if true_np_2d.size > 0:
            # histogram2d returns (H, xedges, yedges)
            _, xedges_true, yedges_true = np.histogram2d(
                true_np_2d[:, 0], true_np_2d[:, 1], bins=25
            )
            grid_edges_2d = (xedges_true, yedges_true)
        else:
            grid_edges_2d = None
    except Exception:
        grid_edges_2d = None

    try:
        true_np = (
            true_tensor.cpu().numpy() if hasattr(true_tensor, "cpu") else np.asarray(true_tensor)
        )
        if true_np.size > 0:
            D_orig = int(true_np.shape[1])
            nd_bins_from_true = [
                np.histogram_bin_edges(true_np[:, d], bins=25) for d in range(D_orig)
            ]
        else:
            nd_bins_from_true = None
    except Exception:
        nd_bins_from_true = None

    # Use the precomputed TRUE-data-derived edges for intrinsic 2D JSD/TVD prints
    print(
        f"Histogram Approx. JSD (PDM):    {jsd_histogram_2d(samples_PDM_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)}"
    )
    print(
        f"Histogram Approx. JSD (Lifted): {jsd_histogram_2d(samples_lifted_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)}"
    )
    print(
        f"Histogram Approx. JSD (Proj. DDPM):   {jsd_histogram_2d(samples_plain_projected_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)}"
    )

    # Also print TVD (Total Variation Distance) for the same histogram approximation
    try:
        print(
            f"Histogram Approx. TVD (PDM):    {tvd_histogram_2d(samples_PDM_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)}"
        )
    except Exception:
        print("Error computing TVD for PDM 2D")
    try:
        print(
            f"Histogram Approx. TVD (Lifted): {tvd_histogram_2d(samples_lifted_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)}"
        )
    except Exception:
        print("Error computing TVD for Lifted 2D")
    try:
        print(
            f"Histogram Approx. TVD (Proj. DDPM):   {tvd_histogram_2d(samples_plain_projected_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)}"
        )
    except Exception:
        print("Error computing TVD for Proj. DDPM 2D")

    # print(f"MMD (PDM):    {MMD(samples_PDM_tensor_2d, true_tensor_2d)}")
    # print(f"MMD (PIDM):   {MMD(samples_PIDM_tensor_2d, true_tensor_2d)}")
    # print(f"MMD (Lifted): {MMD(samples_lifted_tensor_2d, true_tensor_2d)}")
    # print(f"MMD (DDPM):   {MMD(samples_plain_projected_tensor_2d, true_tensor_2d)}")

    from scipy.stats import gaussian_kde
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    from scipy.stats import gaussian_kde

    # -------------------------------
    # Example usage with your tensors
    # -------------------------------

    all_points = [
        true_tensor_2d,
        samples_PDM_tensor_2d,
        samples_PIDM_tensor_2d,
        samples_lifted_tensor_2d,
        samples_plain_tensor_2d,
    ]

    outdir = "results/smileyface_sphere"
    os.makedirs(outdir, exist_ok=True)

    # 1) Compute shared color normalization (uniform colorbar)
    shared_norm = compute_shared_norm(
        all_points, gridsize=200, margin_frac=0.05, vmin=0.0
    )

    # # 2) Save a standalone colorbar image (matches the shared_norm + cmap)
    colorbar_path = os.path.join(outdir, "density_colorbar.pdf")
    save_standalone_colorbar(
        norm=shared_norm,
        cmap="viridis",
        filename=colorbar_path,
        label="Density",
        dpi=300,
        height_in=3.5,
        width_in=0.5,
        orientation="vertical",
    )

    # 3) Save each density plot WITHOUT colorbar, using the shared norm
    plot_2d_density_no_cbar(
        true_tensor_2d,
        os.path.join(outdir, "data_2d_density.pdf"),
        "Data Density (Intrinsic 2D)",
        gridsize=200,
        cmap="viridis",
        point_alpha=0.4,
        dpi=300,
        norm=shared_norm,
    )
    plot_2d_density_no_cbar(
        samples_PDM_tensor_2d,
        os.path.join(outdir, "PDM_2d_density.pdf"),
        "PDM Density (Intrinsic 2D)",
        gridsize=200,
        cmap="viridis",
        point_alpha=0.4,
        dpi=300,
        norm=shared_norm,
    )
    plot_2d_density_no_cbar(
        samples_PIDM_tensor_2d,
        os.path.join(outdir, "PIDM_2d_density.pdf"),
        "PIDM Density (Intrinsic 2D)",
        gridsize=200,
        cmap="viridis",
        point_alpha=0.4,
        dpi=300,
        norm=shared_norm,
    )
    plot_2d_density_no_cbar(
        samples_lifted_tensor_2d,
        os.path.join(outdir, "lifted_2d_density.pdf"),
        "Lifted Density (Intrinsic 2D)",
        gridsize=200,
        cmap="viridis",
        point_alpha=0.4,
        dpi=300,
        norm=shared_norm,
    )
    plot_2d_density_no_cbar(
        samples_plain_tensor_2d,
        os.path.join(outdir, "DDPM_2d_density.pdf"),
        "DDPM Density (Intrinsic 2D)",
        gridsize=200,
        cmap="viridis",
        point_alpha=0.4,
        dpi=300,
        norm=shared_norm,
    )
    plot_2d_density_no_cbar(
        samples_plain_projected_tensor_2d,
        os.path.join(outdir, "DDPM_projected_2d_density.pdf"),
        "DDPM (proj.) Density (Intrinsic 2D)",
        gridsize=200,
        cmap="viridis",
        point_alpha=0.4,
        dpi=300,
        norm=shared_norm,
    )
    if samples_plain_iso_projected_tensor_2d.numel() > 0:
        plot_2d_density_no_cbar(
            samples_plain_iso_projected_tensor_2d,
            os.path.join(outdir, "DDPM_iso_projected_2d_density.pdf"),
            "DDPM (proj., iso.) Density (Intrinsic 2D)",
            gridsize=200,
            cmap="viridis",
            point_alpha=0.4,
            dpi=300,
            norm=shared_norm,
        )

    print(f"Saved standalone colorbar to: {colorbar_path}")

    # Collect metrics on original-space data (general) and intrinsic 2D coords (intrinsic)
    def _compute_jsd_tvd(a, b, grid_edges=None, bins=25):
        # a,b are tensors or arrays; choose 2D vs ND implementation based on dimensionality
        try:
            d = a.shape[1]
        except Exception:
            d = None
        if d == 2:
            # For 2D, prefer using the provided shared grid_edges for consistency
            if grid_edges is not None:
                jsd = jsd_histogram_2d(a, b, grid_edges=grid_edges)
                tvd = tvd_histogram_2d(a, b, grid_edges=grid_edges)
            else:
                jsd = jsd_histogram_2d(a, b, bins=bins)
                tvd = tvd_histogram_2d(a, b, bins=bins)
        else:
            # Fall back to ND (3D) histogram-based metrics
            # Use a modest number of bins for ND histograms to avoid extreme sparsity
            # If we precomputed per-dimension bin edges from the true data, use
            # those arrays as the `bins` argument; otherwise fall back to integer
            # bin counts passed in via `bins`.
            # If per-dimension bin edges were precomputed from the TRUE data,
            # prefer those arrays (one edge-array per dimension). Use a try/except
            # to access the closure variable `nd_bins_from_true` safely; if it's
            # missing, fall back to the integer `bins` argument.
            try:
                bins_arg = nd_bins_from_true if nd_bins_from_true is not None else bins
            except NameError:
                bins_arg = bins
            jsd = compute_jsd_3d(a, b, bins=bins_arg)
            tvd = compute_tvd_3d(a, b, bins=bins_arg)
        return jsd, tvd
    # For general (original-space) metrics use ND histogramming with a fixed
    # bin count per-dimension (25). Do NOT pass 2D grid_edges here — that
    # grid is intended only for intrinsic 2D comparisons and would produce
    # degenerate results when applied to 3D samples.
    def _safe_jsd_tvd(a, b, bins=25):
        try:
            jsd_val, tvd_val = _compute_jsd_tvd(a, b, grid_edges=None, bins=bins)
            # Ensure floats or NaN
            jsd_val = float(jsd_val) if (jsd_val is not None and not isinstance(jsd_val, complex)) else float('nan')
            tvd_val = float(tvd_val) if (tvd_val is not None and not isinstance(tvd_val, complex)) else float('nan')
            return jsd_val, tvd_val
        except Exception:
            return float('nan'), float('nan')

    def _safe_cov(samples, reference):
        try:
            return float(coverage(reference, samples))
        except Exception:
            return float('nan')

    metrics_general = {}
    # Ensure full-space sample tensors exist (safe conversions)
    try:
        D_orig = int(data_points_plain.shape[1])
    except Exception:
        D_orig = 3

    def _to_tensor_or_empty(x, D=D_orig):
        try:
            if x is None:
                return torch.empty((0, D))
            if torch.is_tensor(x):
                return x
            arr = np.asarray(x)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return torch.tensor(arr, dtype=torch.float32)
        except Exception:
            return torch.empty((0, D))

    samples_PDM_tensor = _to_tensor_or_empty(locals().get('samples_PDM', None))
    samples_PIDM_tensor = _to_tensor_or_empty(locals().get('samples_PIDM', None))
    samples_lifted_tensor = _to_tensor_or_empty(locals().get('samples_lifted', None))
    samples_plain_tensor = _to_tensor_or_empty(locals().get('samples_plain', None))
    samples_plain_projected_tensor = _to_tensor_or_empty(locals().get('samples_plain_projected', None))
    true_tensor = _to_tensor_or_empty(locals().get('data_points_plain', None))

    # Build per-method entries with guarding to ensure metrics_general is always created
    methods_general = [
        ("PDM", samples_PDM_tensor),
        ("PIDM", samples_PIDM_tensor),
        ("Lifted", samples_lifted_tensor),
        ("DDPM", samples_plain_tensor),
        ("ProjectedDDPM", samples_plain_projected_tensor),
        ("ProjectedDDPM (iso.)", _to_tensor_or_empty(locals().get('samples_plain_iso_projected', None))),
    ]
    for name, samp in methods_general:
        if samp is None or (hasattr(samp, 'numel') and samp.numel() == 0):
            cov = float('nan')
            jsd_val = float('nan')
            tvd_val = float('nan')
        else:
            cov = _safe_cov(samp, true_tensor)
            jsd_val, tvd_val = _safe_jsd_tvd(samp, true_tensor, bins=25)

        metrics_general[name] = {
            "Coverage": cov,
            "JSD_hist": jsd_val,
            "TVD_hist": tvd_val,
        }

    metrics_intrinsic = {
        "PDM": {
            "Coverage": float(coverage(true_tensor_2d, samples_PDM_tensor_2d)),
            "JSD_hist": float(jsd_histogram_2d(samples_PDM_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)),
            "TVD_hist": float(tvd_histogram_2d(samples_PDM_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)),
        },
        "PIDM": {
            "Coverage": float(coverage(true_tensor_2d, samples_PIDM_tensor_2d)),
            "JSD_hist": float(jsd_histogram_2d(samples_PIDM_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)),
            "TVD_hist": float(tvd_histogram_2d(samples_PIDM_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)),
        },
        "Lifted": {
            "Coverage": float(coverage(true_tensor_2d, samples_lifted_tensor_2d)),
            "JSD_hist": float(jsd_histogram_2d(samples_lifted_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)),
            "TVD_hist": float(tvd_histogram_2d(samples_lifted_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)),
        },
        "DDPM": {
            "Coverage": float(coverage(true_tensor_2d, samples_plain_tensor_2d)),
            "JSD_hist": float(jsd_histogram_2d(samples_plain_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)),
            "TVD_hist": float(tvd_histogram_2d(samples_plain_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)),
        },
        "ProjectedDDPM": {
            "Coverage": float(coverage(true_tensor_2d, samples_plain_projected_tensor_2d)),
            "JSD_hist": float(jsd_histogram_2d(samples_plain_projected_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)),
            "TVD_hist": float(tvd_histogram_2d(samples_plain_projected_tensor_2d, true_tensor_2d, grid_edges=grid_edges_2d)),
        },
        "ProjectedDDPM (iso.)": {
            "Coverage": float(coverage(true_tensor_2d, ensure_tensor_2d(to_intrinsic_2d(_to_tensor_safe(locals().get('samples_plain_iso_projected', None)), sphere_center, sphere_radius)[0], D2).cpu())) if locals().get('samples_plain_iso_projected', None) is not None else float('nan'),
            "JSD_hist": float(jsd_histogram_2d(ensure_tensor_2d(to_intrinsic_2d(_to_tensor_safe(locals().get('samples_plain_iso_projected', None)), sphere_center, sphere_radius)[0], D2).cpu(), true_tensor_2d, grid_edges=grid_edges_2d)) if locals().get('samples_plain_iso_projected', None) is not None else float('nan'),
            "TVD_hist": float(tvd_histogram_2d(ensure_tensor_2d(to_intrinsic_2d(_to_tensor_safe(locals().get('samples_plain_iso_projected', None)), sphere_center, sphere_radius)[0], D2).cpu(), true_tensor_2d, grid_edges=grid_edges_2d)) if locals().get('samples_plain_iso_projected', None) is not None else float('nan'),
        },
    }

    from utils.plotting import save_metrics_table_paper

    # train_time_map will be computed later after we build model_t/proj_t/backprop_t/other_t
    train_time_map = {}

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

    avg_stats = compute_avg_stats(
        method_names,
        trainers_map,
        n_trials,
        num_samples=num_samples,
        external_proj_time=proj_time_plain_projection,
    )

    # Build sample_time_map from avg_stats (now that avg_stats is available)
    sample_time_map = {}
    try:
        sample_time_map["Lifted"] = avg_stats.get("Lifted", {}).get("s", float("nan"))
        sample_time_map["PDM"] = avg_stats.get("PDM", {}).get("s", float("nan"))
        sample_time_map["DDPM"] = avg_stats.get("DDPM", {}).get("s", float("nan"))
        sample_time_map["ProjectedDDPM"] = avg_stats.get("DDPM (proj.)", {}).get(
            "s", float("nan")
        )
        sample_time_map["PIDM"] = avg_stats.get("PIDM", {}).get("s", float("nan"))
        # Mirror sampling time onto ISO projected variant
        sample_time_map["ProjectedDDPM (iso.)"] = sample_time_map.get("ProjectedDDPM", float("nan"))
    except Exception:
        sample_time_map = {
            k: float("nan")
            for k in ["Lifted", "PDM", "DDPM", "ProjectedDDPM", "PIDM"]
        }

    for name in method_names:
        stats = avg_stats.get(name, {"m": np.nan, "p": np.nan, "s": np.nan})
        m = stats["m"]
        base_p = stats["p"]
        s = stats["s"]
        if name == "DDPM":
            p = 0.0
        elif name == "DDPM (proj.)":
            p = base_p
        else:
            p = base_p
        if np.isfinite(s) and np.isfinite(m) and np.isfinite(p):
            other = max(0.0, float(s) - float(m) - float(p))
        else:
            other = np.nan
        model_vals.append(m)
        proj_vals.append(p)
        other_vals.append(other)

    outdir = "results/smileyface_sphere"
    os.makedirs(outdir, exist_ok=True)
    print("Skipping bar plot output: projection_time_breakdown.pdf")

    # ---- Training time breakdown plot (uses checkpoint-loaded epoch timing breakdowns) ----
    def _avg_training_components(tr):
        import numpy as _np

        # Return averaged (model_forward, project, backprop, other_rest)
        if tr is None:
            return _np.nan, _np.nan, _np.nan, _np.nan
        etb = getattr(tr, "epoch_timing_breakdowns", None)
        if not etb:
            return _np.nan, _np.nan, _np.nan, _np.nan
        model_vals = [
            _np.nan if d is None else d.get("model_forward", _np.nan) for d in etb
        ]
        proj_vals = [_np.nan if d is None else d.get("project", _np.nan) for d in etb]
        backprop_vals = [_np.nan if d is None else d.get("backprop", 0.0) for d in etb]
        other_rest_vals = [
            (
                _np.nan
                if d is None
                else (d.get("other", 0.0) + d.get("sampling_to_t0", 0.0))
            )
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

    # Now compute train_time_map from the averaged component lists we just built
    try:
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
        train_time_map["ProjectedDDPM"] = train_time_map.get("DDPM", float("nan"))
        train_time_map["ProjectedDDPM (iso.)"] = train_time_map.get("ProjectedDDPM", float("nan"))
    except Exception:
        train_time_map = {
            k: float("nan")
            for k in ["Lifted", "PDM", "DDPM", "ProjectedDDPM", "PIDM"]
        }

    # Re-build and re-save the metrics table now that train_time_map (and avg_stats)
    # have been computed. Earlier we saved the table before timings were available,
    # which caused the time columns to be missing/zero — overwrite with updated values.
    try:
        general_metrics = {}
        intrinsic_metrics = {}
        # Build general_metrics from original-space metrics
        for k, v in metrics_general.items():
            samp_val = sample_time_map.get(k, float("nan"))
            samp_entry = float(samp_val) if np.isfinite(samp_val) else "n/a"

            train_val = train_time_map.get(k, float("nan"))
            train_entry = float(train_val) if np.isfinite(train_val) else "n/a"
            general_metrics[k] = {
                "Train time (s/epoch)": train_entry,
                "Sampling time (s)": samp_entry,
                "COV": float(v.get("Coverage", 0.0)),
                "JSD": float(v.get("JSD_hist", 0.0)),
                "TVD": float(v.get("TVD_hist", 0.0)),
            }

        # Build intrinsic_metrics from intrinsic 2D metrics
        for k, v in metrics_intrinsic.items():
            intrinsic_metrics[k] = {
                "COV": float(v.get("Coverage", 0.0)),
                "JSD": float(v.get("JSD_hist", 0.0)),
                "TVD": float(v.get("TVD_hist", 0.0)),
            }

        # --- DEBUG: print PIDM-related inputs/metrics to trace discrepancy between
        # printed intrinsic metrics and values saved to the .tex table. ---
        try:
            print("\n--- DEBUG: PIDM metric inputs before writing table ---")
            pidm_gen = metrics_general.get("PIDM")
            pidm_intr = metrics_intrinsic.get("PIDM")
            print(f"metrics_general['PIDM'] = {pidm_gen}")
            print(f"metrics_intrinsic['PIDM'] = {pidm_intr}")
            # sample tensors: shapes and simple stats
            try:
                print(f"samples_PIDM_tensor shape: {getattr(samples_PIDM_tensor, 'shape', None)}")
                if getattr(samples_PIDM_tensor, 'numel', lambda: 0)() > 0:
                    arr = samples_PIDM_tensor.cpu().numpy()
                    print(f"samples_PIDM_tensor min/max: {arr.min()}/{arr.max()}")
            except Exception as _e:
                print("Error inspecting samples_PIDM_tensor:", _e)
            try:
                print(f"true_tensor shape: {getattr(true_tensor, 'shape', None)}")
                if getattr(true_tensor, 'numel', lambda: 0)() > 0:
                    tarr = true_tensor.cpu().numpy()
                    print(f"true_tensor min/max: {tarr.min()}/{tarr.max()}")
            except Exception as _e:
                print("Error inspecting true_tensor:", _e)
            try:
                print(f"samples_PIDM_tensor_2d shape: {getattr(samples_PIDM_tensor_2d, 'shape', None)}")
                if getattr(samples_PIDM_tensor_2d, 'numel', lambda: 0)() > 0:
                    a2 = samples_PIDM_tensor_2d.cpu().numpy()
                    print(f"samples_PIDM_tensor_2d min/max: {a2.min()}/{a2.max()}")
            except Exception as _e:
                print("Error inspecting samples_PIDM_tensor_2d:", _e)
            try:
                print(f"true_tensor_2d shape: {getattr(true_tensor_2d, 'shape', None)}")
                if getattr(true_tensor_2d, 'numel', lambda: 0)() > 0:
                    t2 = true_tensor_2d.cpu().numpy()
                    print(f"true_tensor_2d min/max: {t2.min()}/{t2.max()}")
            except Exception as _e:
                print("Error inspecting true_tensor_2d:", _e)
            print("general_metrics['PIDM'] (final values to write):", general_metrics.get("PIDM"))
        except Exception as e:
            print("Error during PIDM debug prints:", e)

        # For the intrinsic (2D) table include only the constrained-manifold methods
        # requested: PDM, Lifted, and ProjectedDDPM.
        allowed_intrinsic = {
            k: v
            for k, v in intrinsic_metrics.items()
            if k in ["PDM", "Lifted", "ProjectedDDPM", "ProjectedDDPM (iso.)"]
        }
        
        # Build display name map for metrics table
        display_name_map_table = {"Lifted": "$p_{\\sigma}$", "ProjectedDDPM": "DDPM (proj.)", "ProjectedDDPM (iso.)": "DDPM (proj., iso.)"}
        
        save_metrics_table_paper(
            general_metrics,
            allowed_intrinsic,
            out_tex_path="results/smileyface_sphere/metrics_table.tex",
            caption="Metrics for smile on sphere task",
            display_name_map=display_name_map_table,
        )
    except Exception as e:
        print("Failed to re-save metrics table with timings:", e)

    # -------------------------
    # Additional table: Num NaN/Inf and Avg. Dist. to the manifold
    # -------------------------
    out_path = "results/smileyface_sphere/metrics_table.tex"
    import numpy as _np

    def _compute_stats(orig, proj=None, pr=None):
        if orig is None:
            return "n/a", "n/a"
        try:
            if torch.is_tensor(orig):
                orig_np = orig.cpu().numpy()
            else:
                orig_np = _np.array(orig)
        except Exception:
            orig_np = _np.array(orig)
        if orig_np.size == 0:
            return 0, "n/a"
        if orig_np.ndim == 1:
            orig_np = orig_np.reshape(-1, orig_np.shape[0])
        finite_mask = _np.isfinite(orig_np).all(axis=1)
        n_bad = int(orig_np.shape[0] - int(finite_mask.sum()))
        valid_np = orig_np[finite_mask]
        if valid_np.shape[0] == 0:
            return n_bad, "n/a"
        if pr is None:
            pr = globals().get('projector', None)
        # Prefer projector per-sample distances when available
        if proj is None and pr is not None:
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
            return n_bad, "n/a"
        # Fallback: compute distances to provided proj array
        try:
            if torch.is_tensor(proj):
                proj_np = proj.cpu().numpy()
            else:
                proj_np = _np.array(proj)
        except Exception:
            proj_np = _np.array(proj)
        if proj_np.ndim == 1:
            proj_np = proj_np.reshape(-1, proj_np.shape[0])
        min_rows = min(valid_np.shape[0], proj_np.shape[0])
        if min_rows == 0:
            return n_bad, "n/a"
        diffs = valid_np[:min_rows] - proj_np[:min_rows]
        dists = _np.linalg.norm(diffs, axis=1)
        return n_bad, float(_np.mean(dists))

    rows = []
    for name in list(general_metrics.keys()):
        if name == 'Lifted':
            orig = locals().get('samples_lifted_tensor', None)
            proj = None
            pr = locals().get('trainer', None).projector if locals().get('trainer', None) is not None else None
        elif name == 'ProjectedDDPM':
            orig = locals().get('samples_plain_projected_tensor', None)
            proj = None
            pr = locals().get('trainer_plain', None).projector if locals().get('trainer_plain', None) is not None else None
        elif name == 'PDM':
            orig = locals().get('samples_PDM_tensor', None)
            proj = None
            pr = locals().get('trainer_PDM', None).projector if locals().get('trainer_PDM', None) is not None else None
        elif name == 'DDPM':
            orig = locals().get('samples_plain_tensor', None)
            proj = None
            pr = locals().get('trainer_plain', None).projector if locals().get('trainer_plain', None) is not None else None
        elif name == 'PIDM':
            orig = locals().get('samples_PIDM_tensor', None)
            proj = None
            pr = locals().get('trainer_PIDM', None).projector if locals().get('trainer_PIDM', None) is not None else None
        else:
            orig = None
            proj = None
            pr = None
        n_bad, avg_dist = _compute_stats(orig, proj, pr)
        rows.append((name, n_bad, avg_dist))

    try:
        display_name_map = {"Lifted": "$p_{\\sigma}$", "ProjectedDDPM": "DDPM (proj.)"}
        display_name_map["ProjectedDDPM (iso.)"] = "DDPM (proj., iso.)"
        with open(out_path, 'a') as f:
            f.write('\n% --- Additional table: Num NaN/Inf and Avg. Dist. to $\\mathcal{M}$ ---\n')
            f.write('\\begin{table}[ht]\\centering\\small\\begin{tabular}{lrr}\\toprule\n')
            f.write('Method & Num NaN/Inf & Avg. Dist. to $\\mathcal{M}$ \\\\ \\midrule\n')
            for name, n_bad, avg_dist in rows:
                display_name = display_name_map.get(name, name)
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
                f.write("{} & {} & {} \\\\ \n".format(display_name, n_bad, avg_str))
            f.write('\\bottomrule\\end{tabular}\\end{table}\n')
    except Exception as _e:
        print('Failed to append additional metrics table:', _e)

    print("Skipping bar plot output: training_time_breakdown")

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


if __name__ == "__main__":
    main()
