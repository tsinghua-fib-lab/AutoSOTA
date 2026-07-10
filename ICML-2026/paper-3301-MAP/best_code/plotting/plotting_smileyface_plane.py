import argparse
import json
import os
import sys

# Move to the project root (one level up from current file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import time

import torch
import trimesh

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

    A = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(0)  # Normal vetor (x-axis)
    b = torch.tensor([1.0])  # Offset (good gracious!)
    num_samples = 10000
    noise_level = 0.05
    epochs = 200
    hidden_dim = 64
    time_embed_dim = 32
    timesteps = 250
    time_concat = True
    # Keep a dedicated random seed here so checkpoint filenames can be tied to it
    random_seed = args.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    # Allow choosing time embedding behavior via CLI so filenames and trainer
    # construction can reflect the choice (default/sinusoidal/fourier).
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--time-embed', choices=['default', 'sinusoidal', 'fourier'], default='default', help='Time embedding module to use')
    # args = parser.parse_args()
    time_embed_choice = "default"
    # Number of trials to average timings/metrics over
    n_trials = 3

    # Helper to add seed to filenames
    def _with_seed(fname, seed):
        """Return a filename that includes the seed before the extension."""
        d = os.path.dirname(fname)
        base = os.path.basename(fname)
        name, ext = os.path.splitext(base)
        newname = f"{name}_seed_{seed}{ext}"
        return os.path.join(d, newname)

    dataset = SmileyFaceDataset(
        num_samples=num_samples,
        A=A,
        b=b,
        lifted=False,
        noise_level=0.0,
        device=device,
        seed=random_seed,
    )
    data_points = torch.stack([dataset[i] for i in range(len(dataset))])

    # Lifted Diffusion score
    print("Lifted Diffusion Model")
    trainer = DDPMTrainer(
        data_points.squeeze(),
        project_x0_sample=True,
        timesteps=timesteps,
        # Use an ordered pair (tuple) for linear equality constraints. Using a set with
        # tensors is unsafe (tensors are unhashable and set iteration order is undefined),
        # which can lead to incorrect unpacking inside the projector and unpredictable
        # timing/behavior. Pass (A, b) so add_constraints_from_dict can unpack reliably.
        constraints_dict={"linear_equality": (A.to(device), b.to(device))},
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        time_conditioning=time_embed_choice,
        time_concat=time_concat,
    )
    # Prefer checkpoint files that encode the time-embed choice and random seed in the filename when available
    checkpoint_path = f"models/smileyface_plane/model_DDPM_epoch_{epochs}_noise_level_{noise_level}_time_{time_embed_choice}_seed_{random_seed}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"models/smileyface_plane/model_DDPM_epoch_{epochs}_noise_level_{noise_level}_time_{time_embed_choice}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"models/smileyface_plane/model_DDPM_epoch_{epochs}_noise_level_{noise_level}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    _attach_time_embed_if_needed(
        trainer.denoiser, state if isinstance(state, dict) else {}, device
    )
    # If checkpoint lacks time_embed_module keys but the constructed denoiser
    # has a time_embed_module, remove it so keys match and loading succeeds.
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
    # Use trainer.load_checkpoint to restore epoch timing breakdowns and other metadata
    trainer.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
    torch.cuda.empty_cache()
    trainer.denoiser.eval()
    with torch.no_grad():
        samples_lifted, _ = trainer.sample(num_samples=num_samples)
    try:
        samples_lifted = trainer.projector.project(torch.tensor(samples_lifted).to(device))[0].cpu()
    except Exception:
        samples_lifted = torch.tensor(samples_lifted)

    os.makedirs("results/smileyface_plane", exist_ok=True)

    # # PDM
    dataset_plain = SmileyFaceDataset(
        num_samples=num_samples,
        A=A,
        b=b,
        lifted=False,
        device=device,
        seed=random_seed,
    )
    data_points_plain = torch.stack(
        [dataset_plain[i] for i in range(len(dataset_plain))]
    ).cpu()
    # trainer_PDM =   DDPMTrainer(
    #             data_points_plain.squeeze(),
    #             project_x0_sample=True,
    #             constraints_dict={"linear_equality": {A.to(device), b.to(device)}},
    #         )
    # checkpoint_path = f'models/smileyface_plane/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0.pth'
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # trainer_PDM.denoiser.load_state_dict(checkpoint['model_state_dict'])

    # samples_PDM,_ = trainer_PDM.sample(num_samples=10000)

    # Traditional DDPM Score
    print("Traditional DDPM Model")
    trainer_plain = DDPMTrainer(
        data_points_plain.squeeze(),
        timesteps=timesteps,
        project_x0_sample=False,
        constraints_dict={"linear_equality": (A.to(device), b.to(device))},
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        time_conditioning=time_embed_choice,
        time_concat=time_concat,
    )
    checkpoint_path = f"models/smileyface_plane/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"models/smileyface_plane/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"models/smileyface_plane/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0.pth"
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
        # Best-effort: fall back to un-timed projection
        try:
            samples_plain_projected, _, _ = trainer_plain.projector.project(
                torch.tensor(samples_plain).cpu()
            )
            samples_plain_projected = samples_plain_projected.cpu()
        except Exception:
            samples_plain_projected = torch.tensor([])
    print("Average deviation of Traditional DDPM samples from the sphere:", norms)

    # --- ISO DDPM (projected) variant ---
    samples_plain_iso = None
    samples_plain_iso_projected = None
    try:
        iso_ckpt_path = f"models/smileyface_plane/model_DDPM_epoch_{epochs}_noise_level_{noise_level}_ISO_time_{time_embed_choice}_seed_{random_seed}.pth"
        if not os.path.exists(iso_ckpt_path):
            iso_ckpt_path = f"models/smileyface_plane/model_DDPM_epoch_{epochs}_noise_level_{noise_level}_ISO_time_{time_embed_choice}.pth"
        if not os.path.exists(iso_ckpt_path):
            iso_ckpt_path = f"models/smileyface_plane/model_DDPM_epoch_{epochs}_noise_level_{noise_level}_ISO.pth"
        if os.path.exists(iso_ckpt_path):
            trainer_iso_plain = DDPMTrainer(
                data_points_plain.squeeze(),
                timesteps=timesteps,
                project_x0_sample=False,
                constraints_dict={"linear_equality": (A.to(device), b.to(device))},
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
        constraints_dict={"linear_equality": (A.to(device), b.to(device))},
        time_concat=time_concat,
    )
    checkpoint_path = f"models/smileyface_plane/model_PIDM_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
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
        samples_PIDM, norms_PIDM = trainer_PIDM.sample(num_samples=num_samples)
    print("Average deviation of PIDM samples from the sphere:", norms_PIDM)

    # PDM
    print("Projected Diffusion Model")
    trainer_PDM = DDPMTrainer(
        data_points_plain.squeeze(),
        timesteps=timesteps,
        project_x0_sample=True,
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        constraints_dict={"linear_equality": (A.to(device), b.to(device))},
        time_concat=time_concat,
    )
    checkpoint_path = f"models/smileyface_plane/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"models/smileyface_plane/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0.pth"
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
        samples_PDM, norms_PDM = trainer_PDM.sample(num_samples=num_samples, PDM=True)
    print("Average deviation of PDM samples from the sphere:", norms_PDM)

    import matplotlib.pyplot as plt

    scores_conv = np.array(trainer.scores, dtype=np.float64)
    scores = np.array(trainer_plain.scores, dtype=np.float64)

    # Find indices where values are invalid (NaN or Inf)
    invalid_mask = ~np.isfinite(scores)
    valid_mask = np.isfinite(scores)

    # Plot the valid scores
    plt.plot(scores, label=r"$p_{0}$")
    plt.plot(scores_conv, label=r"$p_{\sigma}$")
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
        markersize=5,
        label="NaN or Inf",
    )
    plt.xlabel(r"$t$ (time steps in reverse order)", fontsize="x-large")
    # Flip only the tick labels (not the data)
    num_points = len(scores)
    xticks = np.linspace(0, num_points - 1, num=5, dtype=int)
    xtick_labels = [f"{num_points - 1 - x}" for x in xticks]
    xtick_labels[0] = "100"
    xtick_labels[-1] = "0"
    plt.xticks(ticks=xticks, labels=xtick_labels, fontsize="large")
    plt.ylabel(
        r"Average Score $\nabla_x(t) \log p_t(x(t))$ Across $x(t)$", fontsize="x-large"
    )
    plt.legend(fontsize="x-large")
    _save_pdf_png(plt.gcf(), _with_seed("results/smileyface_plane/scores.pdf", random_seed))

    print(data_points_plain.max())
    print(samples_PDM.max())
    print(samples_PIDM.max())
    print(samples_plain.max())
    print(samples_lifted.max())

    D = data_points_plain.shape[1]

    samples_PDM_tensor = ensure_tensor_2d(samples_PDM, D).cpu()
    samples_PIDM_tensor = ensure_tensor_2d(samples_PIDM, D).cpu()
    samples_lifted_tensor = ensure_tensor_2d(samples_lifted, D).cpu()
    samples_plain_tensor = ensure_tensor_2d(samples_plain, D).cpu()
    samples_plain_projected_tensor = ensure_tensor_2d(samples_plain_projected, D).cpu()
    true_tensor = data_points_plain.view(-1, D).cpu()

    D = data_points_plain.shape[1]

    samples_PDM_tensor = filter_valid_samples(
        torch.tensor(samples_PDM).view(-1, D)
    ).cpu()
    samples_PIDM_tensor = filter_valid_samples(
        torch.tensor(samples_PIDM).view(-1, D)
    ).cpu()
    samples_lifted_tensor = filter_valid_samples(
        torch.tensor(samples_lifted).view(-1, D)
    ).cpu()
    samples_plain_tensor = filter_valid_samples(
        torch.tensor(samples_plain).view(-1, D)
    ).cpu()
    samples_plain_projected_tensor = filter_valid_samples(
        torch.tensor(samples_plain_projected).view(-1, D)
    ).cpu()
    samples_plain_iso_projected_tensor = filter_valid_samples(
        torch.tensor(locals().get('samples_plain_iso_projected', torch.tensor([]))).view(-1, D)
    ).cpu()
    true_tensor = filter_valid_samples(data_points_plain.view(-1, D)).cpu()

    # distance_mmd = SamplesLoss("sinkhorn", blur=0.00)
    # Coverage
    # try:
    #     print(f"Coverage (PIDM):   {coverage(samples_PIDM_tensor, true_tensor)}")
    #     print(f"Coverage (Lifted): {coverage(samples_lifted_tensor, true_tensor)}")
    #     print(f"Coverage (DDPM):   {coverage(samples_plain_tensor, true_tensor)}")
    #     print(f"Coverage (Proj. DDPM):   {coverage(samples_plain_projected_tensor, true_tensor)}")
    #     print(f"Coverage (PDM):    {coverage(samples_PDM_tensor, true_tensor)}")
    # except Exception as e:
    #     print("Error computing coverage:", e)

    # MMD
    # print(f"MMD (PDM):    {MMD(samples_PDM_tensor, true_tensor)}")
    # print(f"MMD (PIDM):   {MMD(samples_PIDM_tensor, true_tensor)}")
    # print(f"MMD (Lifted): {MMD(samples_lifted_tensor, true_tensor)}")
    # print(f"MMD (DDPM):   {MMD(samples_plain_tensor, true_tensor)}")
    # print(f"MMD (Proj. DDPM):   {MMD(samples_plain_projected_tensor, true_tensor)}")

    # =========================
    # --- INTRINSIC 2D PART ---
    # =========================

    # breakpoint()
    with torch.no_grad():
        data_points_plain_2d = to_intrinsic_2d_plane(data_points_plain, A, b)
        samples_PDM_2d = to_intrinsic_2d_plane(torch.tensor(samples_PDM), A, b)
        samples_PIDM_2d = to_intrinsic_2d_plane(torch.tensor(samples_PIDM), A, b)
        samples_lifted_2d = to_intrinsic_2d_plane(torch.tensor(samples_lifted), A, b)
        samples_plain_projected_2d = to_intrinsic_2d_plane(
            torch.tensor(samples_plain_projected), A, b
        )
        samples_plain_2d = to_intrinsic_2d_plane(torch.tensor(samples_plain), A, b)
        if locals().get('samples_plain_iso_projected', None) is not None:
            samples_plain_iso_projected_2d = to_intrinsic_2d_plane(torch.tensor(samples_plain_iso_projected), A, b)

    D2 = 2
    try:
        samples_PDM_tensor_2d = ensure_tensor_2d(samples_PDM_2d, D2).cpu()
    except Exception as e:
        print("Error ensuring tensor 2D for PDM 2D:", e)
    samples_PIDM_tensor_2d = ensure_tensor_2d(samples_PIDM_2d, D2).cpu()
    samples_lifted_tensor_2d = ensure_tensor_2d(samples_lifted_2d, D2).cpu()
    samples_plain_tensor_2d = ensure_tensor_2d(samples_plain_2d, D2).cpu()
    true_tensor_2d = ensure_tensor_2d(data_points_plain_2d, D2).cpu()
    samples_plain_projected_tensor_2d = ensure_tensor_2d(
        samples_plain_projected_2d, D2
    ).cpu()
    samples_plain_iso_projected_tensor_2d = ensure_tensor_2d(
        locals().get('samples_plain_iso_projected_2d', torch.empty((0, D2))), D2
    ).cpu()
    # breakpoint()
    try:
        samples_PDM_tensor_2d = filter_valid_samples(samples_PDM_tensor_2d).cpu()
    except Exception as e:
        print("Error filtering valid samples for PDM 2D:", e)
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

    # --- Build a histogram grid derived strictly from the TRUE data only ---
    # The user requested that histogram bin edges be precomputed according to a
    # histogram of the original true data (not from combined samples). Compute
    # explicit 2D edges from the true intrinsic 2D data and per-dimension
    # bin-edge arrays from the original-space true data for ND metrics.
    fixed_bins = 25
    try:
        true_np_2d = (
            true_tensor_2d.cpu().numpy()
            if hasattr(true_tensor_2d, "cpu")
            else np.asarray(true_tensor_2d)
        )
        if true_np_2d.size > 0:
            # np.histogram2d returns (H, xedges, yedges)
            _, xedges, yedges = np.histogram2d(true_np_2d[:, 0], true_np_2d[:, 1], bins=fixed_bins)
            grid_edges = (xedges, yedges)
        else:
            grid_edges = None
    except Exception:
        grid_edges = None

    # Precompute per-dimension ND bin edges from the original-space TRUE data
    try:
        true_np = (
            true_tensor.cpu().numpy() if hasattr(true_tensor, "cpu") else np.asarray(true_tensor)
        )
        if true_np.size > 0:
            D_orig = int(true_np.shape[1])
            nd_bins_from_true = [np.histogram_bin_edges(true_np[:, d], bins=fixed_bins) for d in range(D_orig)]
        else:
            nd_bins_from_true = None
    except Exception:
        nd_bins_from_true = None

    print("\n--- METRICS IN INTRINSIC 2D COORDINATES (ON PLANE) ---")
    try:
        print(f"Coverage (PDM):    {coverage(samples_PDM_tensor_2d, true_tensor_2d)}")
    except Exception as e:
        print("Error computing coverage for PDM 2D:", e)
    print(f"Coverage (PIDM):   {coverage(samples_PIDM_tensor_2d, true_tensor_2d)}")
    print(f"Coverage (Lifted): {coverage(samples_lifted_tensor_2d, true_tensor_2d)}")
    print(
        f"Coverage (Proj. DDPM):   {coverage(samples_plain_projected_tensor_2d, true_tensor_2d)}"
    )

    try:
        print(
            f"Histogram Approx. JSD (PDM):    {jsd_histogram_2d(samples_PDM_tensor_2d, true_tensor_2d, grid_edges=grid_edges)}"
        )
    except Exception as e:
        print("Error computing JSD for PDM 2D:", e)
    try:
        print(
            f"Histogram Approx. JSD (Lifted): {jsd_histogram_2d(samples_lifted_tensor_2d, true_tensor_2d, grid_edges=grid_edges)}"
        )
    except Exception as e:
        print("Error computing JSD for Lifted 2D:", e)
    try:
        print(
            f"Histogram Approx. JSD (Proj. DDPM):   {jsd_histogram_2d(samples_plain_projected_tensor_2d, true_tensor_2d, grid_edges=grid_edges)}"
        )
    except Exception as e:
        print("Error computing JSD for Proj. DDPM 2D:", e)

    # Also show Total Variation Distance (TVD) for the same histogram approximation
    try:
        print(
            f"Histogram Approx. TVD (PDM):    {tvd_histogram_2d(samples_PDM_tensor_2d, true_tensor_2d, grid_edges=grid_edges)}"
        )
    except Exception as e:
        print("Error computing TVD for PDM 2D:", e)
    try:
        print(
            f"Histogram Approx. TVD (Lifted): {tvd_histogram_2d(samples_lifted_tensor_2d, true_tensor_2d, grid_edges=grid_edges)}"
        )
    except Exception as e:
        print("Error computing TVD for Lifted 2D:", e)
    try:
        print(
            f"Histogram Approx. TVD (Proj. DDPM):   {tvd_histogram_2d(samples_plain_projected_tensor_2d, true_tensor_2d, grid_edges=grid_edges)}"
        )
    except Exception as e:
        print("Error computing TVD for Proj. DDPM 2D:", e)

    from scipy.stats import gaussian_kde

    # -------------------------------
    # Example usage with your tensors
    # -------------------------------

    all_points = [
        true_tensor_2d,
        (
            samples_PDM_tensor_2d
            if torch.isnan(samples_PDM_tensor_2d).all() == False
            else None
        ),
        samples_PIDM_tensor_2d,
        samples_lifted_tensor_2d,
        samples_plain_tensor_2d,
    ]
    all_points = [pts for pts in all_points if pts is not None]

    outdir = "results/smileyface_plane"
    os.makedirs(outdir, exist_ok=True)

    # 1) Compute shared color normalization (uniform colorbar)
    shared_norm = compute_shared_norm(
        all_points, gridsize=200, margin_frac=0.05, vmin=0.0
    )

    # 2) Save a standalone colorbar image (matches the shared_norm + cmap)
    colorbar_path = _with_seed(
        os.path.join(outdir, "density_colorbar.pdf"), random_seed
    )
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
        _with_seed(os.path.join(outdir, "data_2d_density.pdf"), random_seed),
        "Data Density (Intrinsic 2D)",
        gridsize=200,
        cmap="viridis",
        point_alpha=0.4,
        dpi=300,
        norm=shared_norm,
    )
    try:
        plot_2d_density_no_cbar(
            samples_PDM_tensor_2d,
            _with_seed(os.path.join(outdir, "PDM_2d_density.pdf"), random_seed),
            "PDM Density (Intrinsic 2D)",
            gridsize=200,
            cmap="viridis",
            point_alpha=0.4,
            dpi=300,
            norm=shared_norm,
        )
    except Exception as e:
        print("Error plotting PDM 2D density:", e)
    plot_2d_density_no_cbar(
        samples_PIDM_tensor_2d,
        _with_seed(os.path.join(outdir, "PIDM_2d_density.pdf"), random_seed),
        "PIDM Density (Intrinsic 2D)",
        gridsize=200,
        cmap="viridis",
        point_alpha=0.4,
        dpi=300,
        norm=shared_norm,
    )
    plot_2d_density_no_cbar(
        samples_lifted_tensor_2d,
        _with_seed(os.path.join(outdir, "lifted_2d_density.pdf"), random_seed),
        "Lifted Density (Intrinsic 2D)",
        gridsize=200,
        cmap="viridis",
        point_alpha=0.4,
        dpi=300,
        norm=shared_norm,
    )
    plot_2d_density_no_cbar(
        samples_plain_tensor_2d,
        _with_seed(os.path.join(outdir, "DDPM_2d_density.pdf"), random_seed),
        "DDPM Density (Intrinsic 2D)",
        gridsize=200,
        cmap="viridis",
        point_alpha=0.4,
        dpi=300,
        norm=shared_norm,
    )
    plot_2d_density_no_cbar(
        samples_plain_projected_tensor_2d,
        _with_seed(os.path.join(outdir, "DDPM_projected_2d_density.pdf"), random_seed),
        "DDPM (proj.) Density (Intrinsic 2D)",
        gridsize=200,
        cmap="viridis",
        point_alpha=0.4,
        dpi=300,
        norm=shared_norm,
    )
    if samples_plain_iso_projected_tensor_2d is not None and samples_plain_iso_projected_tensor_2d.numel() > 0:
        plot_2d_density_no_cbar(
            samples_plain_iso_projected_tensor_2d,
            _with_seed(os.path.join(outdir, "DDPM_iso_projected_2d_density.pdf"), random_seed),
            "DDPM (proj., iso.) Density (Intrinsic 2D)",
            gridsize=200,
            cmap="viridis",
            point_alpha=0.4,
            dpi=300,
            norm=shared_norm,
        )

    print(f"Saved standalone colorbar to: {colorbar_path}")

    # Build stacked bars: model time, projection time, other (rest of sampling time)
    method_names = ["Lifted", "PDM", "DDPM", "DDPM (proj.)", "PIDM"]
    trainers_map = {
        "Lifted": locals().get("trainer", None),
        "PDM": locals().get("trainer_PDM", None),
        "DDPM": locals().get("trainer_plain", None),
        # DDPM (proj.) uses the same trainer as DDPM but adds the external projection time measured above
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

    outdir = "results/smileyface_plane"
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

    print("Skipping bar plot output: training_time_breakdown")

    # --- Build and save metrics table (LaTeX + CSV) ---
    try:
        # Build both general (original-space) and intrinsic (2D intrinsic coords) metrics
        metrics_general = {}
        metrics_intrinsic = {}

        # Collect metrics computed on the original data (general)
        def _compute_jsd_tvd(a, b, grid_edges=None, bins=25):
            try:
                d = a.shape[1]
            except Exception:
                d = None
            if d == 2:
                # prefer shared grid edges for 2D comparisons when available
                if grid_edges is not None:
                    jsd = jsd_histogram_2d(a, b, grid_edges=grid_edges)
                    tvd = tvd_histogram_2d(a, b, grid_edges=grid_edges)
                else:
                    jsd = jsd_histogram_2d(a, b, bins=bins)
                    tvd = tvd_histogram_2d(a, b, bins=bins)
            else:
                # Use per-dimension bin edges computed from the TRUE original-space
                # data when available to ensure consistent binning across methods.
                # Use per-dimension bin edges computed from the TRUE original-space
                # data when available to ensure consistent binning across methods.
                # Prefer the local `nd_bins_from_true` computed above (closure); fall
                # back to the integer `bins` value if it's not present.
                try:
                    bins_arg = nd_bins_from_true if nd_bins_from_true is not None else bins
                except NameError:
                    bins_arg = bins
                jsd = compute_jsd_3d(a, b, bins=bins_arg)
                tvd = compute_tvd_3d(a, b, bins=bins_arg)
            return jsd, tvd

        metrics_general = {
            "PDM": {
                "Coverage": (
                    float(coverage(samples_PDM_tensor, true_tensor))
                    if "samples_PDM_tensor" in locals()
                    else 0.0
                ),
                "JSD_hist": (
                    float(_compute_jsd_tvd(samples_PDM_tensor, true_tensor, bins=25)[0])
                    if "samples_PDM_tensor" in locals()
                    else 0.0
                ),
                "TVD_hist": (
                    float(_compute_jsd_tvd(samples_PDM_tensor, true_tensor, bins=25)[1])
                    if "samples_PDM_tensor" in locals()
                    else 0.0
                ),
            },
            "PIDM": {
                "Coverage": (
                    float(coverage(samples_PIDM_tensor, true_tensor))
                    if "samples_PIDM_tensor" in locals()
                    else 0.0
                ),
                "JSD_hist": (
                    float(_compute_jsd_tvd(samples_PIDM_tensor, true_tensor, bins=25)[0])
                    if "samples_PIDM_tensor" in locals()
                    else 0.0
                ),
                "TVD_hist": (
                    float(_compute_jsd_tvd(samples_PIDM_tensor, true_tensor, bins=25)[1])
                    if "samples_PIDM_tensor" in locals()
                    else 0.0
                ),
            },
            "Lifted": {
                "Coverage": (
                    float(coverage(samples_lifted_tensor, true_tensor))
                    if "samples_lifted_tensor" in locals()
                    else 0.0
                ),
                "JSD_hist": (
                    float(_compute_jsd_tvd(samples_lifted_tensor, true_tensor, bins=25)[0])
                    if "samples_lifted_tensor" in locals()
                    else 0.0
                ),
                "TVD_hist": (
                    float(_compute_jsd_tvd(samples_lifted_tensor, true_tensor, bins=25)[1])
                    if "samples_lifted_tensor" in locals()
                    else 0.0
                ),
            },
            "DDPM": {
                "Coverage": (
                    float(coverage(samples_plain_tensor, true_tensor))
                    if "samples_plain_tensor" in locals()
                    else 0.0
                ),
                "JSD_hist": (
                    float(_compute_jsd_tvd(samples_plain_tensor, true_tensor, bins=25)[0])
                    if "samples_plain_tensor" in locals()
                    else 0.0
                ),
                "TVD_hist": (
                    float(_compute_jsd_tvd(samples_plain_tensor, true_tensor, bins=25)[1])
                    if "samples_plain_tensor" in locals()
                    else 0.0
                ),
            },
            "ProjectedDDPM": {
                "Coverage": (
                    float(coverage(samples_plain_projected_tensor, true_tensor))
                    if "samples_plain_projected_tensor" in locals()
                    else 0.0
                ),
                "JSD_hist": (
                    float(_compute_jsd_tvd(samples_plain_projected_tensor, true_tensor, bins=25)[0])
                    if "samples_plain_projected_tensor" in locals()
                    else 0.0
                ),
                "TVD_hist": (
                    float(_compute_jsd_tvd(samples_plain_projected_tensor, true_tensor, bins=25)[1])
                    if "samples_plain_projected_tensor" in locals()
                    else 0.0
                ),
            },
            "ProjectedDDPM (iso.)": {
                "Coverage": float(coverage(samples_plain_iso_projected_tensor, true_tensor)) if samples_plain_iso_projected_tensor is not None else 0.0,
                "JSD_hist": float(_compute_jsd_tvd(samples_plain_iso_projected_tensor, true_tensor, bins=25)[0]) if samples_plain_iso_projected_tensor is not None else 0.0,
                "TVD_hist": float(_compute_jsd_tvd(samples_plain_iso_projected_tensor, true_tensor, bins=25)[1]) if samples_plain_iso_projected_tensor is not None else 0.0,
            },
        }

        # Collect intrinsic metrics (computed on intrinsic 2D coordinates)
        metrics_intrinsic = {
            "PDM": {
                "Coverage": (
                    float(coverage(samples_PDM_tensor_2d, true_tensor_2d))
                    if "samples_PDM_tensor_2d" in locals()
                    else 0.0
                ),
                "JSD_hist": (
                    float(
                        jsd_histogram_2d(
                            samples_PDM_tensor_2d, true_tensor_2d, grid_edges=grid_edges
                        )
                    )
                    if "samples_PDM_tensor_2d" in locals()
                    else 0.0
                ),
                "TVD_hist": (
                    float(
                        tvd_histogram_2d(
                            samples_PDM_tensor_2d, true_tensor_2d, grid_edges=grid_edges
                        )
                    )
                    if "samples_PDM_tensor_2d" in locals()
                    else 0.0
                ),
            },
            "PIDM": {
                "Coverage": (
                    float(coverage(samples_PIDM_tensor_2d, true_tensor_2d))
                    if "samples_PIDM_tensor_2d" in locals()
                    else 0.0
                ),
                "JSD_hist": (
                    float(
                        jsd_histogram_2d(
                            samples_PIDM_tensor_2d, true_tensor_2d, grid_edges=grid_edges
                        )
                    )
                    if "samples_PIDM_tensor_2d" in locals()
                    else 0.0
                ),
                "TVD_hist": (
                    float(
                        tvd_histogram_2d(
                            samples_PIDM_tensor_2d, true_tensor_2d, grid_edges=grid_edges
                        )
                    )
                    if "samples_PIDM_tensor_2d" in locals()
                    else 0.0
                ),
            },
            "Lifted": {
                "Coverage": (
                    float(coverage(samples_lifted_tensor_2d, true_tensor_2d))
                    if "samples_lifted_tensor_2d" in locals()
                    else 0.0
                ),
                "JSD_hist": (
                    float(
                        jsd_histogram_2d(
                            samples_lifted_tensor_2d, true_tensor_2d, grid_edges=grid_edges
                        )
                    )
                    if "samples_lifted_tensor_2d" in locals()
                    else 0.0
                ),
                "TVD_hist": (
                    float(
                        tvd_histogram_2d(
                            samples_lifted_tensor_2d, true_tensor_2d, grid_edges=grid_edges
                        )
                    )
                    if "samples_lifted_tensor_2d" in locals()
                    else 0.0
                ),
            },
            "DDPM": {
                "Coverage": (
                    float(coverage(samples_plain_tensor_2d, true_tensor_2d))
                    if "samples_plain_tensor_2d" in locals()
                    else 0.0
                ),
                "JSD_hist": (
                    float(
                        jsd_histogram_2d(
                            samples_plain_tensor_2d, true_tensor_2d, grid_edges=grid_edges
                        )
                    )
                    if "samples_plain_tensor_2d" in locals()
                    else 0.0
                ),
                "TVD_hist": (
                    float(
                        tvd_histogram_2d(
                            samples_plain_tensor_2d, true_tensor_2d, grid_edges=grid_edges
                        )
                    )
                    if "samples_plain_tensor_2d" in locals()
                    else 0.0
                ),
            },
            "ProjectedDDPM": {
                "Coverage": (
                    float(coverage(samples_plain_projected_tensor_2d, true_tensor_2d))
                    if "samples_plain_projected_tensor_2d" in locals()
                    else 0.0
                ),
                "JSD_hist": (
                    float(
                        jsd_histogram_2d(
                            samples_plain_projected_tensor_2d, true_tensor_2d, grid_edges=grid_edges
                        )
                    )
                    if "samples_plain_projected_tensor_2d" in locals()
                    else 0.0
                ),
                "TVD_hist": (
                    float(
                        tvd_histogram_2d(
                            samples_plain_projected_tensor_2d, true_tensor_2d, grid_edges=grid_edges
                        )
                    )
                    if "samples_plain_projected_tensor_2d" in locals()
                    else 0.0
                ),
            },
            "ProjectedDDPM (iso.)": {
                "Coverage": float(coverage(samples_plain_iso_projected_tensor_2d, true_tensor_2d)) if samples_plain_iso_projected_tensor_2d is not None else 0.0,
                "JSD_hist": float(jsd_histogram_2d(samples_plain_iso_projected_tensor_2d, true_tensor_2d, grid_edges=grid_edges)) if samples_plain_iso_projected_tensor_2d is not None else 0.0,
                "TVD_hist": float(tvd_histogram_2d(samples_plain_iso_projected_tensor_2d, true_tensor_2d, grid_edges=grid_edges)) if samples_plain_iso_projected_tensor_2d is not None else 0.0,
            },
        }

        # Build mappings for sampling and training times so they can be inserted
        # into the general metrics table. Use 'n/a' (string) for missing values
        # so the LaTeX writer prints a clear marker instead of 0.0.
        sample_time_map = {}
        try:
            # avg_stats comes from compute_avg_stats earlier and uses keys like
            # 'Lifted','PDM','DDPM','DDPM (proj.)','PIDM'
            sample_time_map["Lifted"] = avg_stats.get("Lifted", {}).get(
                "s", float("nan")
            )
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

        # Build training time map from averaged per-epoch breakdowns computed above
        train_time_map = {}
        try:
            # train_method_names = ['Lifted', 'PDM', 'DDPM', 'PIDM'] and arrays model_t, proj_t, backprop_t, other_t
            for i, name in enumerate(train_method_names):
                comps = [
                    model_t[i] if i < len(model_t) else float("nan"),
                    proj_t[i] if i < len(proj_t) else float("nan"),
                    backprop_t[i] if i < len(backprop_t) else float("nan"),
                    other_t[i] if i < len(other_t) else float("nan"),
                ]
                # If all components are NaN, mark as NaN; else sum finite parts
                if all([not np.isfinite(c) for c in comps]):
                    total = float("nan")
                else:
                    total = float(sum([float(c) for c in comps if np.isfinite(c)]))
                train_time_map[name] = total
            # ProjectedDDPM uses same trainer as DDPM for training-time purposes
            train_time_map["ProjectedDDPM"] = train_time_map.get("DDPM", float("nan"))
            train_time_map["ProjectedDDPM (iso.)"] = train_time_map.get("ProjectedDDPM", float("nan"))
        except Exception:
            train_time_map = {
                k: float("nan")
                for k in ["Lifted", "PDM", "DDPM", "ProjectedDDPM", "PIDM"]
            }

        # Build general + intrinsic metrics dicts expected by save_metrics_table_paper
        general_metrics = {}
        intrinsic_metrics = {}
        # Build general_metrics from original-space metrics (metrics_general)
        for k, v in metrics_general.items():
            # sampling time: prefer numeric avg if available, otherwise 'n/a'
            samp_val = sample_time_map.get(k, float("nan"))
            samp_entry = float(samp_val) if np.isfinite(samp_val) else "n/a"

            # training time: prefer numeric avg if available, otherwise 'n/a'
            train_val = train_time_map.get(k, float("nan"))
            train_entry = float(train_val) if np.isfinite(train_val) else "n/a"

            general_metrics[k] = {
                "Train time (s/epoch)": train_entry,
                "Sampling time (s)": samp_entry,
                "COV": float(v.get("Coverage", 0.0)),
                "JSD": float(v.get("JSD_hist", 0.0)),
                "TVD": float(v.get("TVD_hist", 0.0)),
            }
        # Build intrinsic_metrics from intrinsic 2D metrics (metrics_intrinsic)
        for k, v in metrics_intrinsic.items():
            intrinsic_metrics[k] = {
                "COV": float(v.get("Coverage", 0.0)),
                "JSD": float(v.get("JSD_hist", 0.0)),
                "TVD": float(v.get("TVD_hist", 0.0)),
            }

        from utils.plotting import save_metrics_table_paper

        out_tex = _with_seed(os.path.join(outdir, "metrics_table.tex"), random_seed)
        # For the intrinsic table, only include the constrained-manifold methods requested
        allowed_intrinsic = {k: v for k, v in intrinsic_metrics.items() if k in ["PDM", "Lifted", "ProjectedDDPM", "ProjectedDDPM (iso.)"]}
        
        # Build display name map for metrics table
        display_name_map_table = {"Lifted": "$p_{\\sigma}$", "ProjectedDDPM": "DDPM (proj.)", "ProjectedDDPM (iso.)": "DDPM (proj., iso.)"}
        
        save_metrics_table_paper(
            general_metrics,
            allowed_intrinsic,
            out_tex_path=out_tex,
            caption="Metrics for smile on plane task",
            display_name_map=display_name_map_table,
        )
        # -------------------------
        # Additional table: NaN/Inf counts and Avg. Dist. to M
        # -------------------------
        try:
            out_path = out_tex
            import numpy as _np

            def _compute_stats(orig, proj=None, pr=None):
                import torch

                if orig is None:
                    return "n/a", "n/a"
                # Convert to numpy array for simple NaN/Inf counting
                try:
                    if torch.is_tensor(orig):
                        orig_np = orig.cpu().numpy()
                    else:
                        orig_np = _np.array(orig)
                except Exception:
                    orig_np = _np.array(orig)
                if orig_np.size == 0:
                    return 0, "n/a"
                # Ensure 2D
                if orig_np.ndim == 1:
                    orig_np = orig_np.reshape(-1, orig_np.shape[0])
                # Count NaN/Inf rows only
                finite_mask = _np.isfinite(orig_np).all(axis=1)
                n_bad = int(orig_np.shape[0] - int(finite_mask.sum()))
                valid_np = orig_np[finite_mask]
                if valid_np.shape[0] == 0:
                    return n_bad, "n/a"
                # If a projector is available, use it on the raw finite 3-D points
                if pr is None:
                    pr = globals().get('projector', None)
                if proj is None and pr is not None:
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
                # If proj is provided, compute mean distance between raw finite rows and proj rows (aligned by index)
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

            methods = list(general_metrics.keys())
            rows = []
            for name in methods:
                # For Lifted and ProjectedDDPM use the exact filtered tensors
                # that were used for computing the extrinsic metrics.
                if name == 'Lifted':
                    orig = locals().get('samples_lifted_tensor', None)
                    proj = None
                    pr = locals().get('trainer', None).projector if locals().get('trainer', None) is not None else None
                elif name == 'ProjectedDDPM':
                    orig = locals().get('samples_plain_projected_tensor', None)
                    proj = None
                    pr = locals().get('trainer_plain', None).projector if locals().get('trainer_plain', None) is not None else None
                # For other methods use the same filtered tensors used for extrinsic metrics
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
                    f.write('\n% --- Additional table: NaN/Inf counts and Avg. Dist. to $\\mathcal{M}$ ---\n')
                    f.write('\\begin{table}[ht]\\centering\\small\\begin{tabular}{lrr}\\toprule\n')
                    f.write('Method & Num NaN/Inf & Avg. Dist. to $\\mathcal{M}$ \\\\ \\midrule\n')
                    for name, n_bad, avg_dist in rows:
                        display_name = display_name_map.get(name, name)
                        # Use scientific notation so very small mean distances are visible
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
        except Exception:
            pass
    except Exception as e:
        print("Failed to save metrics table for smileyface_plane:", e)

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
