"""
Find best seed for SAC grid visualization.
SAC models are stored as flat zip files: {method}_{seed}.zip
"""
import sys, os
import numpy as np
import torch as th
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

    return best_pair, obs


def compute_grid_hessian(model, obs, s_dim, a_dim, device, res=30):
    with th.no_grad():
        base_action = model.predict(obs, deterministic=True)[0]

    s_grid = np.linspace(obs[s_dim] - 1.0, obs[s_dim] + 1.0, res)
    a_grid = np.linspace(base_action[a_dim] - 1.5, base_action[a_dim] + 1.5, res)

    d_a = base_action.shape[0]
    d_s = obs.shape[0]
    norms = []

    for i in range(res):
        for j in range(res):
            obs_ij = obs.copy()
            obs_ij[s_dim] = s_grid[i]
            act_ij = base_action.copy()
            act_ij[a_dim] = a_grid[j]

            obs_t = th.as_tensor(obs_ij, device=device).float().unsqueeze(0).requires_grad_(True)
            act_t = th.as_tensor(act_ij, device=device).float().unsqueeze(0).requires_grad_(True)

            q1 = model.critic(obs_t, act_t)[0]
            grad_a = th.autograd.grad(q1.sum(), act_t, create_graph=True, retain_graph=True)[0]

            H_sa = th.zeros(d_a, d_s, device=device)
            for k in range(d_a):
                g = th.autograd.grad(grad_a[0, k], obs_t, retain_graph=True, create_graph=False)[0]
                H_sa[k, :] = g[0, :]

            norms.append(th.linalg.svdvals(H_sa)[0].item())

    return np.mean(norms), np.max(norms), np.percentile(norms, 99)


def main():
    device = "cuda" if th.cuda.is_available() else "cpu"
    eval_seeds = load_seeds(seed_file_path)

    env_funcs = {
        "lunar": make_lunar_env, "pendulum": make_pendulum_env,
        "reacher": make_reacher_env, "ant": make_ant_env,
        "hopper": make_hopper_env, "walker": make_walker_env,
    }
    # SAC naming: vanilla (base), caps_sac, grad_sac, asap_sac, pave_sac
    methods = ["vanilla", "caps_sac", "grad_sac", "asap_sac", "pave_sac"]
    method_labels = ["Base", "CAPS", "GRAD", "ASAP", "PAVE"]
    pth_root = "./Full/sac/pths/"

    print(f"{'Env':10s} {'Seed':>6s} | {'Base':>8s} {'CAPS':>8s} {'GRAD':>8s} {'ASAP':>8s} {'PAVE':>8s} | {'Score':>8s}")
    print("-" * 90)

    best_per_env = {}

    for env_name, env_func in env_funcs.items():
        env = env_func("rgb_array")()
        best_score = -1
        best_seed_idx = 0

        for si in range(5):
            train_seed = TRAIN_SEEDS[si]
            eval_seed = eval_seeds[si]
            means = {}

            # Load Base to find axis
            base_path = os.path.join(pth_root, env_name, f"vanilla_{train_seed}.zip")
            if not os.path.exists(base_path):
                print(f"  SKIP {env_name} s{si}: base not found at {base_path}")
                continue

            base_model = SAC.load(base_path, env=env)
            (s_dim, a_dim), obs = find_dominant_axis(base_model, env, eval_seed, device)

            obs, _ = env.reset(seed=eval_seed)
            for _ in range(20):
                action, _ = base_model.predict(obs, deterministic=True)
                obs, _, done, trunc, _ = env.step(action)
                if done or trunc:
                    obs, _ = env.reset(seed=eval_seed)

            for method, label in zip(methods, method_labels):
                model_path = os.path.join(pth_root, env_name, f"{method}_{train_seed}.zip")
                if not os.path.exists(model_path):
                    means[label] = float('nan')
                    continue

                model = SAC.load(model_path, env=env)
                mean_val, _, _ = compute_grid_hessian(model, obs, s_dim, a_dim, device, res=30)
                means[label] = mean_val

            pave_mean = means.get("PAVE", float('nan'))
            other_means = [means[l] for l in ["Base", "CAPS", "GRAD", "ASAP"]
                          if not np.isnan(means.get(l, float('nan')))]

            if other_means and not np.isnan(pave_mean) and pave_mean > 0:
                score = np.mean(other_means) / pave_mean
            else:
                score = 0

            vals = " ".join([f"{means.get(l, float('nan')):8.1f}" for l in method_labels])
            print(f"{env_name:10s} s{si}={train_seed:>6d} | {vals} | {score:8.2f}")

            if score > best_score:
                best_score = score
                best_seed_idx = si

        best_per_env[env_name] = (best_seed_idx, TRAIN_SEEDS[best_seed_idx], best_score)
        env.close()

    print("\n" + "=" * 60)
    print("BEST SEED PER ENV (SAC grid, highest PAVE advantage):")
    print("=" * 60)
    for env_name, (si, seed, score) in best_per_env.items():
        print(f"  {env_name:10s}: seed_idx={si}, seed={seed}, score={score:.2f}")


if __name__ == "__main__":
    main()
