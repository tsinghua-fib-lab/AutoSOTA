import argparse
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
from trainers import *
from utils.constraints import SimpleConstraintProjector
from utils.metrics import *
from utils.plotting import *
from utils.timing import *


def _attach_time_embed_if_needed(denoiser, state_dict, device):
    """Placeholder function - time embedding utilities have been removed."""
    pass
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args, _ = parser.parse_known_args()

    epochs = 200
    timesteps = 250
    noise_level = 0.0005
    num_samples = 10000
    hidden_dim = 128
    time_embed_dim = 32
    # random seed used to select matching checkpoints
    random_seed = args.seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    mode = "heat"
    time_concat = True
    time_embed_choice = "default"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    constraints_dict = {"bunny": "data/stanford-bunny.obj"}
    dataset = BunnyDataset(
        num_samples=num_samples,
        mean_idx=10500,
        bunny_path="data/stanford-bunny.obj",
        mode=mode,
        noise_level=noise_level,
        lifted=True,
    )
    data_points = torch.stack([dataset[i] for i in range(len(dataset))])

    # Number of trials to average timings/metrics over
    n_trials = 3

    mesh = True
    projector = MeshConstraintProjector("data/stanford-bunny.obj", device)
    # Lifted Diffusion score
    print("Lifted Diffusion")
    trainer = DDPMTrainer(
        data_points.squeeze(),
        project_x0_sample=True,
        timesteps=timesteps,
        constraints_dict=constraints_dict,
        projector=projector,
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        mesh=mesh,
        time_concat=time_concat,
    )
    checkpoint_path = f"models/bunny/model_DDPM_epoch_{epochs}_noise_level_{noise_level}_time_{time_embed_choice}_seed_{random_seed}.pth"
    # if not os.path.exists(checkpoint_path):
    #     checkpoint_path = f'models/bunny/model_DDPM_epoch_{epochs}_noise_level_{noise_level}.pth'
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
        trainer.denoiser.time_embed_module = None
    trainer.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
    torch.cuda.empty_cache()
    trainer.denoiser.eval()
    trainer.denoiser.to(device)
    with torch.no_grad():
        samples_lifted, _ = trainer.sample(num_samples=num_samples)
    try:
        samples_lifted = projector.project(torch.tensor(samples_lifted).cpu())[0].cpu()
    except Exception:
        samples_lifted = torch.tensor(samples_lifted)
    dataset_plain = BunnyDataset(
        num_samples=num_samples,
        mean_idx=10500,
        bunny_path="data/stanford-bunny.obj",
        mode=mode,
        noise_level=0.0,
        lifted=False,
    )
    data_points_plain = torch.stack(
        [dataset_plain[i] for i in range(len(dataset_plain))]
    )

    # Traditional DDPM Score
    print("Traditional DDPM")
    trainer_plain = DDPMTrainer(
        data_points_plain.squeeze(),
        timesteps=timesteps,
        project_x0_sample=False,
        constraints_dict={"bunny": "data/stanford-bunny.obj"},
        projector=projector,
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        mesh=mesh,
        time_concat=time_concat,
    )
    checkpoint_path = f"models/bunny/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = (
            f"models/bunny/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0.pth"
        )
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
        trainer_plain.denoiser.time_embed_module = None
    trainer_plain.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
    trainer_plain.denoiser.eval()
    with torch.no_grad():
        samples_plain, norms = trainer_plain.sample(num_samples=num_samples)
        print("Average deviation of Traditional DDPM samples from the plane:", norms)
        proj_time_plain_projection = float("nan")
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            samples_plain_projected = projector.project(
                torch.tensor(samples_plain).to(device)
            )[0].cpu()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            proj_time_plain_projection = float(t1 - t0)
            # Keep measured projection time local (do not mutate trainer objects),
            # so DDPM's averaged stats remain the baseline without projection.
        except Exception:
            try:
                samples_plain_projected = projector.project(
                    torch.tensor(samples_plain).to(device)
                )[0].cpu()
            except Exception:
                samples_plain_projected = torch.tensor([])

    # PDM
    print("Projected Diffusion Model")
    trainer_PDM = DDPMTrainer(
        data_points_plain.squeeze(),
        timesteps=timesteps,
        project_x0_sample=True,
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        constraints_dict={"bunny": "data/stanford-bunny.obj"},
        projector=projector,
        mesh=mesh,
        time_concat=time_concat,
    )
    checkpoint_path = f"models/bunny/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"models/bunny/model_DDPM_NONPROJECT_epoch_{epochs}_noise_level_0.0.pth"
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
        trainer_PDM.denoiser.time_embed_module = None
    trainer_PDM.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
    with torch.no_grad():
        samples_PDM, norms_PDM = trainer_PDM.sample(num_samples=num_samples, PDM=True)
    print("Average deviation of PDM samples from the plane:", norms_PDM)

    # PIDM
    print("Physics-Informed Diffusion Model")
    trainer_PIDM = DDPMTrainer(
        data_points_plain.squeeze(),
        timesteps=timesteps,
        hidden_dim=hidden_dim,
        time_embed_dim=time_embed_dim,
        project_x0_sample=False,
        constraints_dict={"bunny": "data/stanford-bunny.obj"},
        projector=projector,
        mesh=mesh,
        time_concat=time_concat,
    )
    checkpoint_path = f"models/bunny/model_PIDM_epoch_{epochs}_noise_level_0.0_time_{time_embed_choice}_seed_{random_seed}.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"models/bunny/model_PIDM_epoch_{epochs}_noise_level_0.0.pth"
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
        trainer_PIDM.denoiser.time_embed_module = None
    trainer_PIDM.load_checkpoint(checkpoint_path, map_location=device, load_optimizer=False)
    with torch.no_grad():
        samples_PIDM, norms_PIDM = trainer_PIDM.sample(num_samples=num_samples)
    print("Average deviation of PIDM samples from the plane:", norms_PIDM)

    import matplotlib.pyplot as plt

    # scores_conv = np.array(trainer.scores, dtype=np.float64)
    # scores = np.array(trainer_plain.scores, dtype=np.float64)
    from utils.plotting import plot_scores_vs_time

    scores_conv = np.array(
        [s.item() if torch.is_tensor(s) else s for s in trainer.scores]
    )
    scores = np.array(
        [s.item() if torch.is_tensor(s) else s for s in trainer_plain.scores]
    )
    # Use unified scores plotter (pass a dummy sigma for coloring the conv curve)
    os.makedirs("results/bunny", exist_ok=True)
    plot_scores_vs_time(
        scores_list=[scores_conv],
        scores_plain=scores,
        sigma_list=[1.0],
        output_path="results/bunny/scores.pdf",
    )

    save_mesh_point_plot(
        data_points,
        dataset.mesh,
        dataset.mesh.vertices,
        f"results/bunny/mesh_density_data_lifted_{mode}.pdf",
        view="xy",
    )
    data_points_projected = projector.project(data_points.cpu())[0].cpu()
    save_mesh_point_plot(
        data_points_projected,
        dataset.mesh,
        dataset.mesh.vertices,
        f"results/bunny/mesh_density_data_lifted_projected_{mode}.pdf",
        view="xy",
    )
    save_mesh_point_plot(
        data_points_plain,
        dataset_plain.mesh,
        dataset_plain.mesh.vertices,
        f"results/bunny/mesh_density_data_{mode}.pdf",
        view="xy",
    )
    save_mesh_point_plot(
        samples_lifted,
        dataset.mesh,
        dataset.mesh.vertices,
        f"results/bunny/mesh_density_samples_lifted_{mode}.pdf",
        view="xy",
    )
    # Diagnostic: report shape / finite counts for raw DDPM samples before plotting
    try:
        arr = samples_plain
        if torch.is_tensor(arr):
            arr_np = arr.cpu().numpy()
        else:
            arr_np = np.array(arr)
        print("[DIAG] samples_plain.shape:", getattr(arr_np, "shape", None))
        if arr_np.size == 0:
            print("[DIAG] samples_plain is empty (size==0)")
        else:
            finite_mask = np.isfinite(arr_np)
            n_rows = arr_np.shape[0] if arr_np.ndim > 0 else 0
            n_finite_rows = np.sum(finite_mask.all(axis=1)) if arr_np.ndim == 2 else (
                np.all(finite_mask).item() if arr_np.ndim == 1 else 0
            )
            print(f"[DIAG] samples_plain finite rows: {n_finite_rows} / {n_rows}")
            # compute per-dimension stats for non-NaN rows
            try:
                valid = arr_np[np.isfinite(arr_np).all(axis=1)]
                if valid.size > 0:
                    xs = valid[:, 0]
                    ys = valid[:, 1]
                    zs = valid[:, 2] if valid.shape[1] > 2 else None
                    def pct_stats(a):
                        return (float(np.nanmin(a)),
                                float(np.percentile(a, 1)),
                                float(np.percentile(a, 25)),
                                float(np.percentile(a, 50)),
                                float(np.percentile(a, 75)),
                                float(np.percentile(a, 99)),
                                float(np.nanmax(a)))
                    xs_stats = pct_stats(xs)
                    ys_stats = pct_stats(ys)
                    print("[DIAG] X stats (min,1,25,50,75,99,max):", " ".join([f"{x:.6f}" for x in xs_stats]))
                    print("[DIAG] Y stats (min,1,25,50,75,99,max):", " ".join([f"{y:.6f}" for y in ys_stats]))
                    if zs is not None:
                        zs_stats = pct_stats(zs)
                        print("[DIAG] Z stats (min,1,25,50,75,99,max):", " ".join([f"{z:.6f}" for z in zs_stats]))
            except Exception as _e:
                print("[DIAG] failed to compute percentiles:", _e)
            # inspect mesh verts extents
            try:
                if getattr(dataset_plain, 'mesh', None) is not None:
                    mv = np.asarray(dataset_plain.mesh.vertices)
                    mv_norm = np.linalg.norm(mv, axis=1) if mv.size>0 else np.array([])
                    if mv.size>0:
                        print(f"[DIAG] mesh verts range x: {mv[:,0].min():.6f}..{mv[:,0].max():.6f}, y: {mv[:,1].min():.6f}..{mv[:,1].max():.6f}")
                        print(f"[DIAG] mesh verts norm min/max: {mv_norm.min():.6f}/{mv_norm.max():.6f}")
            except Exception:
                pass
    except Exception as _e:
        print("[DIAG] failed to inspect samples_plain:", _e)
    save_mesh_point_plot(
        samples_plain,
        dataset_plain.mesh,
        dataset_plain.mesh.vertices,
        f"results/bunny/mesh_density_samples_plain_{mode}.pdf",
        view="xy",
    )
    save_mesh_point_plot(
        samples_PDM,
        dataset_plain.mesh,
        dataset_plain.mesh.vertices,
        f"results/bunny/mesh_density_samples_PDM_{mode}.pdf",
        view="xy",
    )
    save_mesh_point_plot(
        samples_PIDM,
        dataset_plain.mesh,
        dataset_plain.mesh.vertices,
        f"results/bunny/mesh_density_samples_PIDM_{mode}.pdf",
        view="xy",
    )
    save_mesh_point_plot(
        samples_plain_projected,
        dataset_plain.mesh,
        dataset_plain.mesh.vertices,
        f"results/bunny/mesh_density_samples_plain_projected_{mode}.pdf",
        view="xy",
    )

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
    true_tensor = filter_valid_samples(data_points_plain.view(-1, D)).cpu()

    # distance_mmd = SamplesLoss("sinkhorn", blur=0.00)
    # Coverage
    print(f"Coverage (PDM):    {coverage(samples_PDM_tensor, true_tensor)}")
    print(f"Coverage (PIDM):   {coverage(samples_PIDM_tensor, true_tensor)}")
    print(f"Coverage (Lifted): {coverage(samples_lifted_tensor, true_tensor)}")
    print(f"Coverage (DDPM):   {coverage(samples_plain_tensor, true_tensor)}")
    print(
        f"Coverage (Proj. DDPM):   {coverage(samples_plain_projected_tensor, true_tensor)}"
    )

    # MMD removed from table per request (kept out of printed metrics)

    # JSD (3D histograms) and TVD (3D histograms)
    try:
        # Exclude extreme-magnitude outliers from histogrammed metrics to match plotting behaviour
        def _filter_by_mag_np(arr, verts, mag_thresh=None):
            if arr is None:
                return np.array([])
            a = np.asarray(arr)
            if a.size == 0:
                return a
            # remove NaN/Inf rows
            finite_mask = np.isfinite(a).all(axis=1)
            a = a[finite_mask]
            try:
                if mag_thresh is None and verts is not None and getattr(verts, 'size', 0) > 0:
                    vnorms = np.linalg.norm(np.asarray(verts), axis=1)
                    mesh_scale = np.nanmax(vnorms) if vnorms.size > 0 else 1.0
                    mag_thresh_use = max(2.0, float(mesh_scale) * 3.0)
                elif mag_thresh is None:
                    mag_thresh_use = 2.0
                else:
                    mag_thresh_use = float(mag_thresh)
            except Exception:
                mag_thresh_use = 2.0
            norms = np.linalg.norm(a, axis=1)
            keep = np.isfinite(norms) & (norms <= mag_thresh_use)
            return a[keep]

        verts = getattr(dataset_plain, 'mesh', None)
        verts_arr = None
        try:
            if verts is not None and getattr(verts, 'vertices', None) is not None:
                verts_arr = np.asarray(verts.vertices)
            elif verts is not None and isinstance(verts, np.ndarray):
                verts_arr = verts
        except Exception:
            verts_arr = None

        jsd_pdm = compute_jsd_3d(_filter_by_mag_np(samples_PDM_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
        print(f"JSD (PDM):    {jsd_pdm}")
    except Exception:
        print("JSD (PDM):    [ERROR]")
    try:
        jsd_pidm = compute_jsd_3d(_filter_by_mag_np(samples_PIDM_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
        print(f"JSD (PIDM):   {jsd_pidm}")
    except Exception:
        print("JSD (PIDM):   [ERROR]")
    try:
        jsd_lifted = compute_jsd_3d(_filter_by_mag_np(samples_lifted_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
        print(f"JSD (Lifted): {jsd_lifted}")
    except Exception:
        print("JSD (Lifted): [ERROR]")
    try:
        jsd_ddpm = compute_jsd_3d(_filter_by_mag_np(samples_plain_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
        print(f"JSD (DDPM):   {jsd_ddpm}")
    except Exception:
        print("JSD (DDPM):   [ERROR]")
    try:
        jsd_projddpm = compute_jsd_3d(_filter_by_mag_np(samples_plain_projected_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
        print(f"JSD (Proj. DDPM):   {jsd_projddpm}")
    except Exception:
        print("JSD (Proj. DDPM):   [ERROR]")

    try:
        tvd_pdm = compute_tvd_3d(_filter_by_mag_np(samples_PDM_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
        print(f"TVD (PDM):    {tvd_pdm}")
    except Exception:
        print("TVD (PDM):    [ERROR]")
    try:
        tvd_pidm = compute_tvd_3d(_filter_by_mag_np(samples_PIDM_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
        print(f"TVD (PIDM):   {tvd_pidm}")
    except Exception:
        print("TVD (PIDM):   [ERROR]")
    try:
        tvd_lifted = compute_tvd_3d(_filter_by_mag_np(samples_lifted_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
        print(f"TVD (Lifted): {tvd_lifted}")
    except Exception:
        print("TVD (Lifted): [ERROR]")
    try:
        tvd_ddpm = compute_tvd_3d(_filter_by_mag_np(samples_plain_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
        print(f"TVD (DDPM):   {tvd_ddpm}")
    except Exception:
        print("TVD (DDPM):   [ERROR]")
    try:
        tvd_projddpm = compute_tvd_3d(_filter_by_mag_np(samples_plain_projected_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
        print(f"TVD (Proj. DDPM):   {tvd_projddpm}")
    except Exception:
        print("TVD (Proj. DDPM):   [ERROR]")

    # Collect metrics and save table
    metrics = {
        "PDM": {
            "Coverage": float(coverage(samples_PDM_tensor, true_tensor)),
            "JSD": float(
                compute_jsd_3d(_filter_by_mag_np(samples_PDM_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
            ),
            "TVD": float(
                compute_tvd_3d(_filter_by_mag_np(samples_PDM_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
            ),
        },
        "PIDM": {
            "Coverage": float(coverage(samples_PIDM_tensor, true_tensor)),
            "JSD": float(
                compute_jsd_3d(_filter_by_mag_np(samples_PIDM_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
            ),
            "TVD": float(
                compute_tvd_3d(_filter_by_mag_np(samples_PIDM_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
            ),
        },
        "Lifted": {
            "Coverage": float(coverage(samples_lifted_tensor, true_tensor)),
            "JSD": float(
                compute_jsd_3d(_filter_by_mag_np(samples_lifted_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
            ),
            "TVD": float(
                compute_tvd_3d(_filter_by_mag_np(samples_lifted_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
            ),
        },
        "DDPM": {
            "Coverage": float(coverage(samples_plain_tensor, true_tensor)),
            "JSD": float(
                compute_jsd_3d(_filter_by_mag_np(samples_plain_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
            ),
            "TVD": float(
                compute_tvd_3d(_filter_by_mag_np(samples_plain_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
            ),
        },
        "ProjectedDDPM": {
            "Coverage": float(coverage(samples_plain_projected_tensor, true_tensor)),
            "JSD": float(
                compute_jsd_3d(_filter_by_mag_np(samples_plain_projected_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
            ),
            "TVD": float(
                compute_tvd_3d(_filter_by_mag_np(samples_plain_projected_tensor.numpy(), verts_arr), _filter_by_mag_np(true_tensor.numpy(), verts_arr), bins=50)
            ),
        },
    }

    # from utils.plotting import save_metrics_table_paper

    # Build metrics table but prefer measured sampling/training times when available.
    # Use 'n/a' for missing values so LaTeX output is explicit.
    # Build sample_time_map from the same breakdowns used for plotting so
    # the annotated totals on the bars match the table values.
    sample_time_map = {}
    try:
        # method_names corresponds to model_vals/proj_vals/other_vals
        method_totals = {}
        for i, name in enumerate(method_names):
            m = model_vals[i] if i < len(model_vals) else np.nan
            p = proj_vals[i] if i < len(proj_vals) else np.nan
            o = other_vals[i] if i < len(other_vals) else np.nan
            if np.isfinite(m) and np.isfinite(p) and np.isfinite(o):
                method_totals[name] = float(m) + float(p) + float(o)
            else:
                method_totals[name] = float("nan")

        sample_time_map["Lifted"] = method_totals.get("Lifted", float("nan"))
        sample_time_map["PDM"] = method_totals.get("PDM", float("nan"))
        sample_time_map["DDPM"] = method_totals.get("DDPM", float("nan"))
        # ProjectedDDPM in the table corresponds to "DDPM (proj.)" in method_names
        sample_time_map["ProjectedDDPM"] = method_totals.get("DDPM (proj.)", float("nan"))
        sample_time_map["PIDM"] = method_totals.get("PIDM", float("nan"))

        # If any entry is NaN, fall back to avg_stats where available
        for k in ["Lifted", "PDM", "DDPM", "ProjectedDDPM", "PIDM"]:
            if not np.isfinite(sample_time_map.get(k, np.nan)):
                # map ProjectedDDPM fallback to DDPM+measured extra when possible
                if k == "ProjectedDDPM":
                    ddpm_s = avg_stats.get("DDPM", {}).get("s", float("nan"))
                    proj_extra = locals().get("proj_time_plain_projection", float("nan"))
                    try:
                        if np.isfinite(ddpm_s) and np.isfinite(proj_extra):
                            sample_time_map[k] = float(ddpm_s) + float(proj_extra)
                        elif np.isfinite(ddpm_s):
                            sample_time_map[k] = float(ddpm_s)
                        elif np.isfinite(proj_extra):
                            sample_time_map[k] = float(proj_extra)
                        else:
                            sample_time_map[k] = float("nan")
                    except Exception:
                        sample_time_map[k] = float("nan")
                else:
                    sample_time_map[k] = avg_stats.get(k, {}).get("s", float("nan"))
    except Exception:
        sample_time_map = {k: float("nan") for k in ["Lifted", "PDM", "DDPM", "ProjectedDDPM", "PIDM"]}

    # Build training time map from averaged per-epoch breakdowns computed below
    train_time_map = {}
    try:
        proj_extra = locals().get("proj_time_plain_projection", float("nan"))
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
        # For ProjectedDDPM, add the measured projection extra (if available) to DDPM's training total
        try:
            ddpm_train = train_time_map.get("DDPM", float("nan"))
            if np.isfinite(ddpm_train) and np.isfinite(proj_extra):
                train_time_map["ProjectedDDPM"] = float(ddpm_train) + float(proj_extra)
            elif np.isfinite(ddpm_train):
                train_time_map["ProjectedDDPM"] = float(ddpm_train)
            elif np.isfinite(proj_extra):
                train_time_map["ProjectedDDPM"] = float(proj_extra)
            else:
                train_time_map["ProjectedDDPM"] = float("nan")
        except Exception:
            train_time_map["ProjectedDDPM"] = train_time_map.get("DDPM", float("nan"))
    except Exception:
        train_time_map = {
            k: float("nan") for k in ["Lifted", "PDM", "DDPM", "ProjectedDDPM", "PIDM"]
        }

    # (metrics table will be written after timing averages are computed below)

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

    avg_stats = compute_avg_stats(method_names, trainers_map, n_trials)

    # Use DDPM base stats to ensure DDPM entry has no projection time,
    # and DDPM (proj.) is DDPM + measured extra projection time.
    ddpm_base = avg_stats.get("DDPM", {"m": np.nan, "p": np.nan, "s": np.nan})

    for name in method_names:
        stats = avg_stats.get(name, {"m": np.nan, "p": np.nan, "s": np.nan})
        m = stats["m"]
        p = stats["p"]
        s = stats["s"]
        # keep DDPM stats from avg_stats (do not force p=0 here)
        if name == "DDPM (proj.)":
            try:
                extra = float(proj_time_plain_projection)
            except Exception:
                extra = float("nan")
            # base model/scan stats come from ddpm_base
            if np.isfinite(ddpm_base.get("m", np.nan)):
                m = ddpm_base.get("m")
            if np.isfinite(ddpm_base.get("s", np.nan)):
                s = ddpm_base.get("s")
            # projection for the 'proj' variant is exactly the measured extra
            p = (extra if np.isfinite(extra) else 0.0)
            s = (s if np.isfinite(s) else 0.0) + (extra if np.isfinite(extra) else 0.0)
        if np.isfinite(s) and np.isfinite(m) and np.isfinite(p):
            other = max(0.0, float(s) - float(m) - float(p))
        else:
            other = np.nan
        model_vals.append(m)
        proj_vals.append(p)
        other_vals.append(other)

    outdir = "results/bunny"
    os.makedirs(outdir, exist_ok=True)

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

    train_method_names = ["Lifted", "PDM", "DDPM", "ProjectedDDPM", "PIDM"]
    trainers_map_small = {
        "Lifted": locals().get("trainer", None),
        "PDM": locals().get("trainer_PDM", None),
        "DDPM": locals().get("trainer_plain", None),
        "ProjectedDDPM": locals().get("trainer_plain", None),
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

    print("Skipping bar plot output: bunny training_time_breakdown")

    # Re-build and save the metrics table now that avg_stats and train timings are available
    try:
        # If a sample_time_map was already built from the plotted breakdowns above,
        # keep it. Otherwise, fall back to building from avg_stats.
        def _map_all_nan(m):
            try:
                return all([not np.isfinite(v) for v in m.values()])
            except Exception:
                return True

        if "sample_time_map" in locals() and (not _map_all_nan(sample_time_map)):
            # keep existing sample_time_map built earlier
            pass
        else:
            # Build sample_time_map from avg_stats as a fallback
            sample_time_map = {}
            try:
                sample_time_map["Lifted"] = avg_stats.get("Lifted", {}).get("s", float("nan"))
                sample_time_map["PDM"] = avg_stats.get("PDM", {}).get("s", float("nan"))
                sample_time_map["DDPM"] = avg_stats.get("DDPM", {}).get("s", float("nan"))
                # Ensure ProjectedDDPM is always DDPM sampling time + projection cost
                ddpm_s = sample_time_map.get("DDPM", float("nan"))
                proj_extra = locals().get("proj_time_plain_projection", float("nan"))
                try:
                    if np.isfinite(ddpm_s) and np.isfinite(proj_extra):
                        sample_time_map["ProjectedDDPM"] = float(ddpm_s) + float(proj_extra)
                    elif np.isfinite(ddpm_s):
                        sample_time_map["ProjectedDDPM"] = float(ddpm_s)
                    elif np.isfinite(proj_extra):
                        sample_time_map["ProjectedDDPM"] = float(proj_extra)
                    else:
                        sample_time_map["ProjectedDDPM"] = float("nan")
                except Exception:
                    sample_time_map["ProjectedDDPM"] = float("nan")
                sample_time_map["PIDM"] = avg_stats.get("PIDM", {}).get("s", float("nan"))
            except Exception:
                sample_time_map = {k: float("nan") for k in ["Lifted", "PDM", "DDPM", "ProjectedDDPM", "PIDM"]}

        # Build train_time_map from averaged per-epoch breakdowns computed above
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
        except Exception:
            train_time_map = {k: float("nan") for k in ["Lifted", "PDM", "DDPM", "ProjectedDDPM", "PIDM"]}

        from plotting.paper_tables import write_mesh_metrics_table

        rows = []
        for key, label in [
            ("PDM", "PDM"),
            ("PIDM", "PIDM"),
            ("Lifted", r"$p_{\\sigma}$ (ours)"),
            ("DDPM", "DDPM"),
            ("ProjectedDDPM", "DDPM (proj.)"),
        ]:
            vals = metrics[key]
            rows.append(
                {
                    "method": label,
                    "Train time (s/epoch)": train_time_map.get(key, float("nan")),
                    "Sampling time (s)": sample_time_map.get(key, float("nan")),
                    "COV": float(vals.get("Coverage", float("nan"))),
                    "JSD": float(vals.get("JSD", float("nan"))),
                    "TVD": float(vals.get("TVD", float("nan"))),
                }
            )

        write_mesh_metrics_table(
            rows,
            out_tex_path="results/bunny/metrics_table.tex",
            caption="Mesh task metrics at $\\sigma = 0.0005$. Learning $p_{\\sigma}$ is consistently competitive or an improvement upon other methods.",
            label="tab:MeshMetrics",
        )
    except Exception as e:
        print("Failed to write mesh metrics table:", e)

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
