"""
Q-gradient cosine similarity between consecutive timesteps.
Measures: cos(∇_a Q(s_t, a_t), ∇_a Q(s_{t+1}, a_{t+1})) at each transition.
Reports per-seed mean/std, then across-seed statistics.

Run from PAVE_Merge root: python td3/tests/eval_cosine_sim.py
"""
import sys, os, csv
import numpy as np
import torch as th

sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.envs import (make_ant_env, make_hopper_env, make_lunar_env,
                           make_pendulum_env, make_reacher_env, make_walker_env)
from stable_baselines3 import TD3

# ── Config ──
MAX_EPISODES = 10
EVAL_SEED_FILE = "./sac/tests/validation_seeds.txt"
TRAIN_SEEDS = [178132, 410580, 922852, 787576, 660993]  # seeds #1-#5
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

SAVE_DIR_ROOT = "./Full/td3/pths/"
PTH_NAMES = ["base_td3", "pave_td3"]  # Base vs PAVE comparison
OUTPUT_DIR = "./Full/td3/pths/eval_cosine_sim/"

ENVS = {
    "lunar": make_lunar_env,
    "pendulum": make_pendulum_env,
    "reacher": make_reacher_env,
    "ant": make_ant_env,
    "hopper": make_hopper_env,
    "walker": make_walker_env,
}

def load_eval_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

def find_matching_files(save_dir, al_name):
    if not os.path.isdir(save_dir):
        return []
    matched = []
    for fname in os.listdir(save_dir):
        full_path = os.path.join(save_dir, fname)
        if al_name in fname and os.path.isdir(full_path):
            final = os.path.join(full_path, "final.zip")
            if os.path.isfile(final):
                matched.append(os.path.join(fname, "final.zip"))
    return sorted(matched)

def compute_q_gradient(model, obs):
    """Compute ∇_a Q(s, a) at the deterministic action."""
    action, _ = model.predict(obs, deterministic=True)
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    action_tensor = th.as_tensor(action, device=obs_tensor.device).unsqueeze(0)
    action_tensor.requires_grad_(True)

    q1 = model.critic(obs_tensor, action_tensor)[0].sum()
    grad_a = th.autograd.grad(q1, action_tensor, retain_graph=False)[0]

    return action, grad_a.detach().cpu().squeeze(0)  # (action_dim,)

def measure_cosine_sim(model, env, eval_seed, max_episodes):
    """Compute cosine similarity between consecutive Q-gradients."""
    cosine_sims = []

    obs, info = env.reset(seed=eval_seed)
    prev_grad = None

    for ep in range(max_episodes):
        prev_grad = None  # reset at episode start
        while True:
            action, grad_a = compute_q_gradient(model, obs)

            if prev_grad is not None:
                # Cosine similarity
                cos = th.nn.functional.cosine_similarity(
                    prev_grad.unsqueeze(0), grad_a.unsqueeze(0)
                ).item()
                cosine_sims.append(cos)

            prev_grad = grad_a

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                prev_grad = None  # reset across episodes
                break

    return np.array(cosine_sims)

def main():
    eval_seeds = load_eval_seeds(EVAL_SEED_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env_names = list(ENVS.keys())
    headers = ["al_name"] + env_names

    rows_mean = [headers]
    rows_std = [headers]
    rows_median = [headers]
    rows_below0 = [headers]  # fraction of cos < 0 (direction flips)

    for al_name in PTH_NAMES:
        print(f"\n=== {al_name} ===")
        row_mean = [al_name]
        row_std = [al_name]
        row_median = [al_name]
        row_below0 = [al_name]

        for env_name in env_names:
            save_dir = os.path.join(SAVE_DIR_ROOT, env_name, "")
            files = find_matching_files(save_dir, al_name)

            # Match files to training seeds
            seed_to_file = {}
            for f in files:
                for s in TRAIN_SEEDS:
                    if str(s) in f:
                        seed_to_file[s] = f
                        break

            if not seed_to_file:
                print(f"  {env_name}: no files")
                row_mean.append(0.0); row_std.append(0.0)
                row_median.append(0.0); row_below0.append(0.0)
                continue

            seed_means = []
            seed_stds = []
            seed_medians = []
            seed_flip_rates = []

            env = ENVS[env_name]("rgb_array")()

            for si, train_seed in enumerate(TRAIN_SEEDS):
                if train_seed not in seed_to_file:
                    continue
                filename = seed_to_file[train_seed]
                eval_seed = eval_seeds[si]
                model = TD3.load(os.path.join(save_dir, filename), env=env)

                cos_sims = measure_cosine_sim(model, env, eval_seed, MAX_EPISODES)

                if len(cos_sims) > 0:
                    seed_means.append(np.mean(cos_sims))
                    seed_stds.append(np.std(cos_sims))
                    seed_medians.append(np.median(cos_sims))
                    seed_flip_rates.append(np.mean(cos_sims < 0))

            env.close()

            if seed_means:
                m = np.mean(seed_means)
                s = np.mean(seed_stds)
                med = np.mean(seed_medians)
                flip = np.mean(seed_flip_rates)
                print(f"  {env_name}: mean_cos={m:.3f}, std={s:.3f}, "
                      f"median={med:.3f}, flip_rate={flip:.3f}")
                row_mean.append(m); row_std.append(s)
                row_median.append(med); row_below0.append(flip)
            else:
                row_mean.append(0.0); row_std.append(0.0)
                row_median.append(0.0); row_below0.append(0.0)

        rows_mean.append(row_mean)
        rows_std.append(row_std)
        rows_median.append(row_median)
        rows_below0.append(row_below0)

    # Write CSVs
    for name, rows in [
        ("cosine_mean.csv", rows_mean),
        ("cosine_std.csv", rows_std),
        ("cosine_median.csv", rows_median),
        ("cosine_flip_rate.csv", rows_below0),
    ]:
        path = os.path.join(OUTPUT_DIR, name)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"Saved: {path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
