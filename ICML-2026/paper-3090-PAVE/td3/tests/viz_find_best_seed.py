"""
Find the best seed for visualization: where PAVE has the largest
spectral norm advantage over other methods along trajectories.
Scans all 5 seeds × 6 envs for TD3.
"""
import sys, os
import numpy as np
import torch as th
from stable_baselines3 import TD3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.envs import (make_lunar_env, make_walker_env, make_pendulum_env,
                           make_reacher_env, make_ant_env, make_hopper_env)

seed_file_path = "./sac/tests/validation_seeds.txt"


def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def compute_trajectory_spectral(model, env, eval_seed, max_steps=300, device="cuda"):
    """Compute spectral norm of mixed Hessian along trajectory."""
    obs, _ = env.reset(seed=eval_seed)
    d_a = env.action_space.shape[0]
    d_s = env.observation_space.shape[0]
    norms = []

    for t in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs_t = th.as_tensor(obs, device=device).float().unsqueeze(0).requires_grad_(True)
        act_t = th.as_tensor(action, device=device).float().unsqueeze(0).requires_grad_(True)

        q1, _ = model.critic(obs_t, act_t)
        grad_a = th.autograd.grad(q1.sum(), act_t, create_graph=True, retain_graph=True)[0]

        H_sa = th.zeros(d_a, d_s, device=device)
        for i in range(d_a):
            g = th.autograd.grad(grad_a[0, i], obs_t, retain_graph=True, create_graph=False)[0]
            H_sa[i, :] = g[0, :]

        norms.append(th.linalg.svdvals(H_sa)[0].item())

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    return np.array(norms)


def main():
    device = "cuda" if th.cuda.is_available() else "cpu"
    seeds = load_seeds(seed_file_path)

    env_funcs = {
        "lunar": make_lunar_env, "pendulum": make_pendulum_env,
        "reacher": make_reacher_env, "ant": make_ant_env,
        "hopper": make_hopper_env, "walker": make_walker_env,
    }
    methods = ["base_td3", "caps_td3", "grad_td3", "asap_td3", "pave_td3"]
    method_labels = ["Base", "CAPS", "GRAD", "ASAP", "PAVE"]
    pth_root = "./Full/td3/pths/"
    max_steps = 300

    print(f"{'Env':10s} {'Seed':>6s} | {'Base':>8s} {'CAPS':>8s} {'GRAD':>8s} {'ASAP':>8s} {'PAVE':>8s} | {'Ratio':>8s} {'Score':>8s}")
    print("-" * 100)

    best_per_env = {}

    for env_name, env_func in env_funcs.items():
        env = env_func("rgb_array")()
        best_score = -1
        best_seed_idx = 0

        for si in range(5):
            eval_seed = seeds[si]
            means = {}

            for method, label in zip(methods, method_labels):
                method_dir = os.path.join(pth_root, env_name)
                model_files = sorted([
                    os.path.join(method_dir, d, "final.zip")
                    for d in os.listdir(method_dir)
                    if method in d and os.path.exists(os.path.join(method_dir, d, "final.zip"))
                ])
                if si >= len(model_files):
                    means[label] = float('nan')
                    continue

                model = TD3.load(model_files[si], env=env)
                norms = compute_trajectory_spectral(model, env, eval_seed, max_steps, device)
                means[label] = norms.mean()

            # Score: how much lower PAVE mean is compared to others
            pave_mean = means.get("PAVE", float('nan'))
            other_means = [means[l] for l in ["Base", "CAPS", "GRAD", "ASAP"] if not np.isnan(means.get(l, float('nan')))]

            if other_means and not np.isnan(pave_mean) and pave_mean > 0:
                ratio = np.mean(other_means) / pave_mean  # higher = PAVE more advantageous
                # Also consider max ratio (worst competitor / PAVE)
                max_ratio = max(other_means) / pave_mean
                score = ratio  # use mean ratio as score
            else:
                ratio = float('nan')
                max_ratio = float('nan')
                score = 0

            vals = " ".join([f"{means.get(l, float('nan')):8.1f}" for l in method_labels])
            print(f"{env_name:10s} s{si}={seeds[si]:>6d} | {vals} | {ratio:8.2f} {score:8.2f}")

            if score > best_score:
                best_score = score
                best_seed_idx = si

        best_per_env[env_name] = (best_seed_idx, seeds[best_seed_idx], best_score)
        env.close()

    print("\n" + "=" * 60)
    print("BEST SEED PER ENV (highest PAVE advantage):")
    print("=" * 60)
    for env_name, (si, seed, score) in best_per_env.items():
        print(f"  {env_name:10s}: seed_idx={si}, seed={seed}, score={score:.2f}")


if __name__ == "__main__":
    main()
