"""
Trajectory-level Hessian norm 3D visualization (autograd exact).
X: timestep, Y: action dimension, Z: spectral norm of mixed Hessian row.
All methods on same trajectory (same eval seed).
"""
import sys, os
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import TD3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.envs import (make_lunar_env, make_walker_env, make_pendulum_env,
                           make_reacher_env, make_ant_env, make_hopper_env)

seed_file_path = "./sac/tests/validation_seeds.txt"


def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def compute_trajectory_hessian(model, env, eval_seed, max_steps=200, device="cuda"):
    """Run one episode, compute per-action-dim Hessian spectral norm at each timestep."""
    obs, _ = env.reset(seed=eval_seed)

    d_a = env.action_space.shape[0]
    d_s = env.observation_space.shape[0]

    timesteps = []
    hessian_per_adim = []  # list of (d_a,) arrays — per-action-dim contribution
    hessian_spectral = []  # spectral norm at each timestep

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)

        obs_t = th.as_tensor(obs, device=device).float().unsqueeze(0).requires_grad_(True)
        act_t = th.as_tensor(action, device=device).float().unsqueeze(0).requires_grad_(True)

        q1, _ = model.critic(obs_t, act_t)
        grad_a = th.autograd.grad(q1.sum(), act_t, create_graph=True, retain_graph=True)[0]

        # Full mixed Hessian
        H_sa = th.zeros(d_a, d_s, device=device)
        for i in range(d_a):
            g = th.autograd.grad(grad_a[0, i], obs_t, retain_graph=True, create_graph=False)[0]
            H_sa[i, :] = g[0, :]

        # Spectral norm
        spec_norm = th.linalg.svdvals(H_sa)[0].item()

        # Per-action-dim: L2 norm of each row of H_sa
        row_norms = th.norm(H_sa, dim=1).detach().cpu().numpy()

        timesteps.append(t)
        hessian_spectral.append(spec_norm)
        hessian_per_adim.append(row_norms)

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    hessian_per_adim = np.array(hessian_per_adim)  # (T, d_a)
    hessian_spectral = np.array(hessian_spectral)   # (T,)

    return timesteps, hessian_per_adim, hessian_spectral


def plot_3d_trajectory(timesteps, hessian_2d, save_path, vmax=None, title=""):
    """
    3D surface: X=timestep, Y=action_dim, Z=Hessian row norm.
    hessian_2d: (T, d_a)
    """
    LABEL_SIZE = 30
    T, d_a = hessian_2d.shape

    X = np.arange(T)
    Y = np.arange(d_a)
    X, Y = np.meshgrid(X, Y)
    Z = hessian_2d.T  # (d_a, T)

    if vmax is None:
        vmax = np.percentile(Z, 99)
    Z_clipped = np.clip(Z, 0, vmax)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    surf = ax.plot_surface(X, Y, Z_clipped, cmap='magma',
                           edgecolor='none', alpha=0.9, antialiased=True,
                           vmin=0, vmax=vmax)

    ax.set_xlabel("Timestep", fontsize=LABEL_SIZE, labelpad=10)
    ax.set_ylabel("Action Dim", fontsize=LABEL_SIZE, labelpad=10)
    ax.set_zlabel(r'$\|\nabla^2_{sa} Q\|$', fontsize=LABEL_SIZE, labelpad=5)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=0)
    ax.set_zlim(0, vmax)
    ax.view_init(elev=35, azim=225)
    ax.set_box_aspect(None, zoom=0.85)

    if title:
        ax.set_title(title, fontsize=LABEL_SIZE + 5, pad=20)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_2d_timeseries(all_spectral, method_labels, save_path, title=""):
    """2D line plot: spectral norm over time for all methods."""
    LABEL_SIZE = 20

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    for i, (spec, label) in enumerate(zip(all_spectral, method_labels)):
        ax.plot(spec, color=colors[i], alpha=0.8, linewidth=1.5, label=label)

    ax.set_xlabel("Timestep", fontsize=LABEL_SIZE)
    ax.set_ylabel(r'$\|\nabla^2_{sa} Q\|_2$', fontsize=LABEL_SIZE)
    ax.legend(fontsize=LABEL_SIZE - 4, loc='upper right')
    ax.tick_params(labelsize=14)

    if title:
        ax.set_title(title, fontsize=LABEL_SIZE + 2)

    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="lunar", choices=["lunar", "walker", "pendulum", "reacher", "ant", "hopper"])
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed_idx", type=int, default=0)
    parser.add_argument("--vmax", type=float, default=None)
    args = parser.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"
    seeds = load_seeds(seed_file_path)
    eval_seed = seeds[args.seed_idx]

    env_funcs = {"lunar": make_lunar_env, "walker": make_walker_env,
                 "pendulum": make_pendulum_env, "reacher": make_reacher_env,
                 "ant": make_ant_env, "hopper": make_hopper_env}
    env = env_funcs[args.env]("rgb_array")()

    pth_root = "./Full/td3/pths/"
    methods = ["base_td3", "caps_td3", "grad_td3", "asap_td3", "pave_td3"]
    method_labels = ["Base", "CAPS", "GRAD", "ASAP", "PAVE"]

    viz_dir = f"./viz/{args.env}_trajectory/"
    os.makedirs(viz_dir, exist_ok=True)

    all_spectral = []
    all_2d = {}

    for method, label in zip(methods, method_labels):
        print(f"\n=== {label} ({args.env}) ===")
        method_dir = os.path.join(pth_root, args.env)
        model_files = sorted([
            os.path.join(method_dir, d, "final.zip")
            for d in os.listdir(method_dir)
            if method in d and os.path.exists(os.path.join(method_dir, d, "final.zip"))
        ])
        if args.seed_idx >= len(model_files):
            print(f"  Seed idx {args.seed_idx} not available")
            continue

        model = TD3.load(model_files[args.seed_idx], env=env)
        timesteps, hessian_2d, hessian_spectral = compute_trajectory_hessian(
            model, env, eval_seed, args.max_steps, device
        )

        all_spectral.append(hessian_spectral)
        all_2d[label] = hessian_2d

        print(f"  T={len(timesteps)}, spec max={hessian_spectral.max():.1f}, "
              f"mean={hessian_spectral.mean():.1f}, p99={np.percentile(hessian_spectral, 99):.1f}")

    # Shared vmax from Base
    if args.vmax is not None:
        vmax = args.vmax
    elif "Base" in all_2d:
        vmax = np.percentile(all_2d["Base"], 99)
        print(f"\nUsing Base 99th percentile as vmax: {vmax:.1f}")
    else:
        vmax = 300

    # 3D plots per method
    for label, hessian_2d in all_2d.items():
        save_path = os.path.join(viz_dir, f"traj3d_{label}_{args.env}_seed{eval_seed}.pdf")
        plot_3d_trajectory(range(len(hessian_2d)), hessian_2d, save_path, vmax=vmax, title=label)

    # 2D overlay plot (all methods)
    save_2d = os.path.join(viz_dir, f"traj2d_all_{args.env}_seed{eval_seed}.pdf")
    plot_2d_timeseries(all_spectral, method_labels, save_2d,
                       title=f"Spectral Norm along Trajectory ({args.env})")

    print(f"\nDone! All plots saved to {viz_dir}")


if __name__ == "__main__":
    main()
