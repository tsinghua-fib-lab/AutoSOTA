#!/usr/bin/env python3
"""
data_generator.py

Generates or loads marginals with velocities.
Downstream, we train a WLM model from the initial velocities and a subset of the generated marginals.

Outputs a single torch bundle:
  {
    "X_em_torch": (num_p0, N, T+1, d) float32 CPU tensor,
    "V_em_torch": (num_p0, N, T+1, d) float32 CPU tensor (optional),
    "time_grid":  (T+1,) float32 CPU tensor,
    "blur":       float, (esitmated from data, and to be used as a width parameter for distributional loss during training)
    "meta":       dict of run parameters (includes blur, vel_mode, has_vel)
  }
For all experiments, num_p0 = 1, since we just consider a single population dynamics (one curve of marginals)
"""
import os

# On login/CPU nodes, force JAX to CPU so it doesn't try to init CUDA.
if os.environ.get("WLF_JAX_GPU", "0") != "1":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
from plot_utils import make_compare_gif
from dataset_modules.boids import generate_boids, add_boids_parser
from dataset_modules.gf_sde import generate_potential_sde, add_potential_sde_parser
from dataset_modules.eb import generate_eb, add_eb_parser
from dataset_modules.oceans import generate_oceans, add_oceans_parser
import argparse
from pathlib import Path
from typing import Optional
import torch

from parse_save_helpers import (
    _parse_args_with_config,
    get_device,
    set_seed,
    save_bundle
)

# used downstream for WLM as the kernel blur size for training distributional loss
@torch.no_grad()
def estimate_geom_blur_from_data(
        X_em_torch: torch.Tensor,
        *,
        t_indices: Optional[list[int]] = None,
        num_times: int = 8,
        pairs_per_time: int = 4096,
        particles_subsample: Optional[int] = 4000,
        blur_scale: float = 0.5,
        blur_min: float = 1e-3,
        blur_max: float = 10.0,
) -> float:
    device = X_em_torch.device
    num_p0, N, T, d = X_em_torch.shape

    if t_indices is None:
        t_indices = torch.randint(low=0, high=T, size=(num_times,), device=device).tolist()
    else:
        t_indices = [int(t) for t in t_indices]

    medians = []
    for t in t_indices:
        p0_idx = int(torch.randint(0, num_p0, (1,), device=device).item())
        x = X_em_torch[p0_idx, :, t, :]  # (N,d)
        # filter NaNs
        mask = torch.isfinite(x).all(dim=1)
        x = x[mask]

        if particles_subsample is not None and particles_subsample < x.shape[0]:
            idx = torch.randint(0, x.shape[0], (particles_subsample,), device=device)
            x = x[idx]

        M = x.shape[0]
        if M < 2:
            continue

        i = torch.randint(0, M, (pairs_per_time,), device=device)
        j = torch.randint(0, M, (pairs_per_time,), device=device)
        j = (j + (j == i).long()) % M

        dist = torch.linalg.norm(x[i] - x[j], dim=1)
        medians.append(dist.median())

    if len(medians) == 0:
        raise RuntimeError("Could not estimate blur: no valid samples.")

    med_all = torch.stack(medians).median().item()
    blur = float(blur_scale * med_all)
    blur = max(blur_min, min(blur, blur_max))
    return blur


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate training data bundles")

    # Config
    p.add_argument("--config", type=str, default=None, help="YAML/JSON config file")
    p.add_argument("--set", action="append", default=None, help="Override config with dot-keys")

    p.add_argument("--out", type=str, required=False, default=None, help="Output .pt path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    # blur params
    p.add_argument("--blur-scale", type=float, default=0.5, help="blur = blur_scale * median(pairwise distance)")
    p.add_argument("--blur-min", type=float, default=1e-3)
    p.add_argument("--blur-max", type=float, default=10.0)
    p.add_argument("--blur-num-times", type=int, default=8)
    p.add_argument("--blur-pairs-per-time", type=int, default=4096)
    p.add_argument("--blur-particles-subsample", type=int, default=4000)

    # Subcommands for each mode
    sub = p.add_subparsers(dest="mode")
    add_boids_parser(sub)
    add_potential_sde_parser(sub)
    add_eb_parser(sub)
    add_oceans_parser(sub)

    return p


def main() -> None:
    parser = build_parser()
    args = _parse_args_with_config(parser)

    if args.out is None:
        raise SystemExit("data_generator.py: --out is required (either in config or on CLI).")
    if args.mode is None:
        raise SystemExit("data_generator.py: mode is required (e.g., 'boids') (either in config or on CLI).")

    # For simulated data, we need to specify dimensionality
    if args.mode in ("boids", "potential_sde"):
        N = getattr(args, "N", None)
        steps = getattr(args, "steps", None)
        dt = getattr(args, "dt", None)
        if N is None or steps is None or dt is None:
            raise ValueError("Missing required args: --N, --steps, --dt (check config argv and mode parser).")

    # specify drift potential and diffusivity if we are generating data from a gradient-flow SDE
    if args.mode == "potential_sde":
        if args.pot is None or args.sigma is None:
            raise SystemExit("data_generator.py: mode=potential_sde requires pot and sigma (via config or CLI).")

    device = get_device(args.device)
    set_seed(args.seed)

    out_path = Path(args.out)


    if args.mode == "boids":
        X, V, tgrid, meta = generate_boids(
            N=args.N,
            steps=args.steps,
            dt=args.dt,
            d=args.d,
            num_p0=args.num_p0,
            outer_radius=args.outer_radius,
            inner_radius=args.inner_radius,
            w_cohesion=args.w_cohesion,
            w_separation=args.w_separation,
            w_alignment=args.w_alignment,
            w_boundary=args.w_boundary,
            boundary=args.boundary,
            init_pos_std=args.init_pos_std,
            init_vel_std=args.init_vel_std,
            sigma=args.sigma,
            device=device,
            # GMM initialization parameters
            init_mode=getattr(args, 'init_mode', 'gaussian'),
            gmm_n_components_min=getattr(args, 'gmm_n_components_min', 1),
            gmm_n_components_max=getattr(args, 'gmm_n_components_max', 5),
            gmm_mean_range=getattr(args, 'gmm_mean_range', 3.0),
            gmm_std_min=getattr(args, 'gmm_std_min', 0.3),
            gmm_std_max=getattr(args, 'gmm_std_max', 1.5),
        )

    elif args.mode == "potential_sde":
        X, V, tgrid, meta = generate_potential_sde(
            N=args.N,
            steps=args.steps,
            dt=args.dt,
            d=args.d,
            num_p0=args.num_p0,
            pot=args.pot,
            sigma=args.sigma,
            p0=args.p0,
            p0_mean=args.p0_mean,
            p0_var=args.p0_var,
            center_uniform=args.center_uniform,
            center_range=args.center_range,
            kill_condition=bool(args.kill_condition),
            device=device,
            score_method=getattr(args, 'score_method', 'kernel'),
            score_hidden=getattr(args, 'score_hidden', 64),
        )

    elif args.mode == "eb":
        X, V, tgrid, meta = generate_eb(
            npz_path=Path(args.npz_path),
            num_times=int(args.num_times),
            pca_dim=int(args.pca_dim),
            num_p0=int(args.num_p0),
            label_subset=(None if args.label_subset is None else list(args.label_subset)),
            seed=int(args.seed),
            device=device,
        )
    elif args.mode == "oceans":
       X, V, tgrid, meta = generate_oceans(
           npz_path=Path(args.npz_path),
           num_p0=int(args.num_p0),
           train_ts=args.train_ts,
           seed=int(args.seed),
           device=device,
       )

    else:
        raise SystemExit(f"data_generator.py: unknown mode={args.mode!r}")

    # Estimate blur
    blur = estimate_geom_blur_from_data(
        X,
        num_times=args.blur_num_times,
        pairs_per_time=args.blur_pairs_per_time,
        particles_subsample=args.blur_particles_subsample,
        blur_scale=args.blur_scale,
        blur_min=args.blur_min,
        blur_max=args.blur_max,
    )

    # Update meta
    meta.update({"seed": args.seed, "device": str(device)})

    # Save bundle
    save_kwargs = {"V_em_torch": V} if V is not None else {}
    save_bundle(out_path, X, tgrid, meta, blur=blur, **save_kwargs)

    print(f"Saved: {out_path}")
    # --- Optional: save a GIF of the generated dynamics (one p0) ---
    try:
        p0_idx = 0  # or pick randomly / make this a CLI arg
        X0 = X[p0_idx]  # (N, T+1, d)

        gif_path = out_path.with_suffix(".gif")
        dt_for_gif = float(meta.get("dt", getattr(args, "dt", 1.0)))
        make_compare_gif(
            X_true=X0,
            X_learned=X0,                 # same data: "just generated dynamics"
            dt=dt_for_gif,
            true_label="generated",
            est_label="generated",
            save_path=str(gif_path),
            always_show=False,
            show_null=False,
            subsample=1000,
            frame_skip=1,
            fps=8,
            times=tgrid,
            projection="auto",
            render="auto",
        )
        print(f"Saved GIF: {gif_path}")
    except Exception as e:
        print(f"[warn] GIF generation failed: {e}")

    print(f"X_em_torch: {tuple(X.shape)}  time_grid: {tuple(tgrid.shape)}  blur={blur:.6g}")
    if V is not None:
        print(f"V_em_torch: {tuple(V.shape)}")


if __name__ == "__main__":
    main()