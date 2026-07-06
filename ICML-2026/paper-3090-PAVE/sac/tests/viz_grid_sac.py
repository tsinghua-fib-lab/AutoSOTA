"""
SAC grid Hessian 3D visualization (autograd exact, spectral norm).
"""
import sys, os
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import SAC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../td3/tests/modules'))

from envs import (make_lunar_env, make_walker_env, make_pendulum_env,
                   make_reacher_env, make_ant_env, make_hopper_env)

seed_file_path = "./sac/tests/validation_seeds.txt"
TRAIN_SEEDS = [178132, 410580, 922852, 787576, 660993]


def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def find_dominant_axis(model, env, seed, device):
    obs, _ = env.reset(seed=seed)
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, _ = env.step(action)
        if done or trunc:
            obs, _ = env.reset(seed=seed)

    obs_t = th.as_tensor(obs, device=device).float().unsqueeze(0).requires_grad_(True)
    with th.no_grad():
        base_action = model.predict(obs, deterministic=True)[0]
    act_t = th.as_tensor(base_action, device=device).float().unsqueeze(0).requires_grad_(True)

    q1 = model.critic(obs_t, act_t)[0]
    grad_a = th.autograd.grad(q1.sum(), act_t, create_graph=True)[0]

    max_val, best_pair = -1.0, (0, 0)
    for a_idx in range(base_action.shape[0]):
        grad_sa = th.autograd.grad(grad_a[0, a_idx], obs_t, retain_graph=True)[0]
        local_max = th.abs(grad_sa[0]).max().item()
        if local_max > max_val:
            max_val = local_max
            best_pair = (th.argmax(th.abs(grad_sa[0])).item(), a_idx)

    print(f"    [Axis] State[{best_pair[0]}] <-> Action[{best_pair[1]}] (Score: {max_val:.4f})")
    return best_pair, obs


def compute_grid_hessian(model, obs, s_dim, a_dim, device, res=50):
    with th.no_grad():
        base_action = model.predict(obs, deterministic=True)[0]

    s_grid = np.linspace(obs[s_dim] - 1.0, obs[s_dim] + 1.0, res)
    a_grid = np.linspace(base_action[a_dim] - 1.5, base_action[a_dim] + 1.5, res)
    S, A = np.meshgrid(s_grid, a_grid)

    d_a = base_action.shape[0]
    d_s = obs.shape[0]
    hessian_norms = np.zeros((res, res))

    for i in range(res):
        for j in range(res):
            obs_ij = obs.copy()
            obs_ij[s_dim] = S[i, j]
            act_ij = base_action.copy()
            act_ij[a_dim] = A[i, j]

            obs_t = th.as_tensor(obs_ij, device=device).float().unsqueeze(0).requires_grad_(True)
            act_t = th.as_tensor(act_ij, device=device).float().unsqueeze(0).requires_grad_(True)

            q1 = model.critic(obs_t, act_t)[0]
            grad_a = th.autograd.grad(q1.sum(), act_t, create_graph=True, retain_graph=True)[0]

            H_sa = th.zeros(d_a, d_s, device=device)
            for k in range(d_a):
                g = th.autograd.grad(grad_a[0, k], obs_t, retain_graph=True, create_graph=False)[0]
                H_sa[k, :] = g[0, :]

            hessian_norms[i, j] = th.linalg.svdvals(H_sa)[0].item()

        if (i + 1) % 10 == 0:
            print(f"      Row {i+1}/{res} done")

    return S, A, hessian_norms


def plot_3d_surface(S, A, hessian, save_path, vmax=None, title=""):
    LABEL_SIZE = 40
    if vmax is None:
        vmax = np.percentile(hessian, 99)
    hessian_clipped = np.clip(hessian, 0, vmax)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    surf = ax.plot_surface(S, A, hessian_clipped, cmap='magma',
                           edgecolor='none', alpha=0.9, antialiased=True,
                           vmin=0, vmax=vmax)

    ax.set_xlabel("State", fontsize=LABEL_SIZE, labelpad=10)
    ax.set_ylabel("Action", fontsize=LABEL_SIZE, labelpad=10)
    ax.set_zlabel(r'$\|\nabla^2_{sa} Q\|_2$', fontsize=LABEL_SIZE, labelpad=5)
    ax.tick_params(axis='both', which='major', labelsize=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_zlim(0, vmax)
    ax.view_init(elev=45, azim=225)
    ax.set_box_aspect(None, zoom=0.9)

    if title:
        ax.set_title(title, fontsize=LABEL_SIZE + 5, pad=20)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"    Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="lunar",
                        choices=["lunar", "walker", "pendulum", "reacher", "ant", "hopper"])
    parser.add_argument("--res", type=int, default=50)
    parser.add_argument("--seed_idx", type=int, default=0)
    parser.add_argument("--vmax", type=float, default=None)
    args = parser.parse_args()

    device = "cuda" if th.cuda.is_available() else "cpu"
    eval_seeds = load_seeds(seed_file_path)
    eval_seed = eval_seeds[args.seed_idx]
    train_seed = TRAIN_SEEDS[args.seed_idx]

    env_funcs = {"lunar": make_lunar_env, "walker": make_walker_env,
                 "pendulum": make_pendulum_env, "reacher": make_reacher_env,
                 "ant": make_ant_env, "hopper": make_hopper_env}
    env = env_funcs[args.env]("rgb_array")()

    pth_root = "./Full/sac/pths/"
    methods = ["vanilla", "caps_sac", "grad_sac", "asap_sac", "pave_sac"]
    method_labels = ["Base", "CAPS", "GRAD", "ASAP", "PAVE"]

    viz_dir = f"./viz_grid_sac/{args.env}/"
    os.makedirs(viz_dir, exist_ok=True)

    # Find axis from Base
    base_path = os.path.join(pth_root, args.env, f"vanilla_{train_seed}.zip")
    model = SAC.load(base_path, env=env)
    (s_dim, a_dim), obs = find_dominant_axis(model, env, eval_seed, device)

    # Warm-up to consistent state
    obs, _ = env.reset(seed=eval_seed)
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, _ = env.step(action)
        if done or trunc:
            obs, _ = env.reset(seed=eval_seed)

    # Compute all methods
    all_results = {}
    for method, label in zip(methods, method_labels):
        print(f"\n=== {label} ({args.env}) ===")
        model_path = os.path.join(pth_root, args.env, f"{method}_{train_seed}.zip")
        if not os.path.exists(model_path):
            print(f"  Not found: {model_path}")
            continue

        model = SAC.load(model_path, env=env)
        S, A, hessian = compute_grid_hessian(model, obs, s_dim, a_dim, device, args.res)
        all_results[label] = (S, A, hessian)
        print(f"  max={hessian.max():.1f}, mean={hessian.mean():.1f}, p99={np.percentile(hessian, 99):.1f}")

    # Shared vmax
    if args.vmax is not None:
        vmax = args.vmax
    elif "Base" in all_results:
        vmax = np.percentile(all_results["Base"][2], 99)
        print(f"\nUsing Base 99th percentile as vmax: {vmax:.1f}")
    else:
        vmax = 300

    # Plot
    for label, (S, A, hessian) in all_results.items():
        save_path = os.path.join(viz_dir, f"hessian_3d_{label}_{args.env}_seed{eval_seed}.pdf")
        plot_3d_surface(S, A, hessian, save_path, vmax=vmax, title=label)

    env.close()
    print(f"\nDone! All plots saved to {viz_dir}")


if __name__ == "__main__":
    main()
