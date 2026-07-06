"""
Robustness evaluation under observation noise (SiLU-unified models).
Scale-aware noise: δ ~ U(-σ, σ) ⊙ σ_base
Outputs CSV with re/sm per method per sigma.
Uses 5 TRAIN_SEEDS only, Base obs_scale locked for all methods.
"""
import sys, os
import numpy as np
import csv
import gymnasium as gym
import torch as th
from stable_baselines3 import TD3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modules.envs import make_lunar_env, make_walker_env
from modules.action_extractor import calculate_smoothness_np

SEED_FILE_PATH = "./sac/tests/validation_seeds.txt"
TRAIN_SEEDS = [178132, 410580, 922852, 787576, 660993]
PTH_ROOT = "./Full/td3/pths/"
OUTPUT_DIR = "./Full/td3/pths/eval_robustness/"

SIGMAS = [0.01, 0.05, 0.1]
NUM_EPISODES = 20
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

METHODS = ["base_td3", "caps_td3", "grad_td3", "asap_td3", "pave_td3"]
METHOD_LABELS = ["Base", "CAPS", "GRAD", "ASAP", "PAVE"]
ENVS = {
    "lunar": make_lunar_env,
    "walker": make_walker_env,
}


def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def estimate_obs_scale(model, env, num_episodes=10):
    all_obs = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            all_obs.append(obs)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    all_obs = np.array(all_obs)
    obs_std = np.std(all_obs, axis=0)
    obs_std = np.maximum(obs_std, 1e-6)
    return obs_std


def test_robustness(model, env, noise_sigma, obs_scale, num_episodes=20):
    all_rewards = []
    all_sm = []
    low, high = env.observation_space.low, env.observation_space.high

    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        actions = []
        while True:
            base_noise = np.random.uniform(-noise_sigma, noise_sigma, size=obs.shape)
            scaled_noise = base_noise * obs_scale
            noisy_obs = np.clip(obs + scaled_noise, low, high)
            action, _ = model.predict(noisy_obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            actions.append(action)
            if terminated or truncated:
                all_rewards.append(ep_reward)
                all_sm.append(calculate_smoothness_np(np.array(actions)))
                break

    return np.mean(all_rewards), np.std(all_rewards), np.mean(all_sm), np.std(all_sm)


def find_model_files(env_name, method):
    """Find model files matching TRAIN_SEEDS."""
    env_dir = os.path.join(PTH_ROOT, env_name)
    files = []
    for ts in TRAIN_SEEDS:
        for d in os.listdir(env_dir):
            if method in d and str(ts) in d:
                full_path = os.path.join(env_dir, d, "final.zip")
                if os.path.exists(full_path):
                    files.append((ts, full_path))
                    break
    return files


def main():
    eval_seeds = load_seeds(SEED_FILE_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []

    for env_name, env_func in ENVS.items():
        env = env_func("rgb_array")()
        print(f"\n{'='*60}")
        print(f"  Environment: {env_name}")
        print(f"{'='*60}")

        # Step 1: Compute obs_scale from Base (locked for all methods)
        base_files = find_model_files(env_name, "base_td3")
        if not base_files:
            print(f"  No Base files found for {env_name}")
            continue

        base_model = TD3.load(base_files[0][1], env=env)
        obs_scale = estimate_obs_scale(base_model, env, num_episodes=20)
        print(f"  obs_scale computed from Base (locked)")

        for method, label in zip(METHODS, METHOD_LABELS):
            model_files = find_model_files(env_name, method)
            print(f"\n  {label}: {len(model_files)} seeds found")

            for sigma in SIGMAS:
                seed_re, seed_sm = [], []
                for si, (ts, model_path) in enumerate(model_files):
                    model = TD3.load(model_path, env=env)
                    re_mean, re_std, sm_mean, sm_std = test_robustness(
                        model, env, sigma, obs_scale, NUM_EPISODES
                    )
                    seed_re.append(re_mean)
                    seed_sm.append(sm_mean)
                    print(f"    σ={sigma} seed={ts}: re={re_mean:.1f}, sm={sm_mean:.3f}")

                if seed_re:
                    all_rows.append({
                        'env': env_name, 'method': label, 'sigma': sigma,
                        're_mean': np.mean(seed_re), 're_std': np.std(seed_re),
                        'sm_mean': np.mean(seed_sm), 'sm_std': np.std(seed_sm),
                        'n_seeds': len(seed_re)
                    })

        env.close()

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "robustness_silu.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved: {csv_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    for env_name in ENVS:
        for sigma in SIGMAS:
            print(f"\n  {env_name} σ={sigma}:")
            for row in all_rows:
                if row['env'] == env_name and row['sigma'] == sigma:
                    print(f"    {row['method']:6s}: re={row['re_mean']:8.1f}({row['re_std']:5.1f})  sm={row['sm_mean']:.3f}({row['sm_std']:.3f})")

    print("\nDone!")


if __name__ == "__main__":
    main()
