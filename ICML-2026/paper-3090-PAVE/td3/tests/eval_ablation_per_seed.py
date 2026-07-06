"""
Ablation eval per seed — Walker + Lunar, 10 seeds.
Includes Base from Full experiment (same SLURM server for seeds #6-#10).
Outputs sm/re for each config × each seed individually.

Run from PAVE_Merge root: python td3/tests/eval_ablation_per_seed.py
"""
import sys, os, csv
import numpy as np
import torch as th

sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.envs import make_walker_env, make_lunar_env
from modules.action_extractor import calculate_smoothness_np
from stable_baselines3 import TD3

MAX_EPISODES = 10
EVAL_SEED_FILE = "./sac/tests/validation_seeds.txt"

# Ablation configs (from pths_10seed)
ABLATION_DIR = "./Ablation/td3/pths_10seed/"
ABLATION_CONFIGS = [
    ("Curv_only",  "pave_td3_S0.0_T0.0_C0.01_sig0.01_del1.0"),
    ("VFC_only",   "pave_td3_S0.0_T0.1_C0.0_sig0.01_del1.0"),
    ("VFC+Curv",   "pave_td3_S0.0_T0.1_C0.01_sig0.01_del1.0"),
    ("MPR_only",   "pave_td3_S0.1_T0.0_C0.0_sig0.01_del1.0"),
    ("MPR+Curv",   "pave_td3_S0.1_T0.0_C0.01_sig0.01_del1.0"),
    ("MPR+VFC",    "pave_td3_S0.1_T0.1_C0.0_sig0.01_del1.0"),
    ("Full_PAVE",  "pave_td3_S0.1_T0.1_C0.01_sig0.01_del1.0"),
]

# Base from Full experiment
FULL_DIR = "./Full/td3/pths/"
BASE_CONFIG = ("Base", "base_td3")

OUTPUT_DIR = "./Ablation/td3/pths_10seed/eval_per_seed/"

def load_eval_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

def eval_one_model(model, env, eval_seed, max_episodes):
    all_rewards = []
    all_actions = []
    obs, info = env.reset(seed=eval_seed)
    for ep in range(max_episodes):
        ep_reward = 0
        ep_actions = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            ep_actions.append(action.copy())
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                obs, info = env.reset()
                break
        all_rewards.append(ep_reward)
        all_actions.extend(ep_actions)
    re = np.mean(all_rewards)
    sm = calculate_smoothness_np(np.array(all_actions))
    return re, sm

def find_zip(model_dir):
    for f in os.listdir(model_dir):
        if f == "final.zip":
            return os.path.join(model_dir, f)
        elif f.endswith(".zip"):
            return os.path.join(model_dir, f)
    return None

def main():
    eval_seeds = load_eval_seeds(EVAL_SEED_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for env_name, env_func in [("walker", make_walker_env), ("lunar", make_lunar_env)]:
        env = env_func("rgb_array")()
        print(f"\n{'='*60}")
        print(f"  {env_name}")
        print(f"{'='*60}")

        all_rows = []

        # 1. Base from Full (all 10 seeds)
        label, al_name = BASE_CONFIG
        base_dir = os.path.join(FULL_DIR, env_name)
        base_folders = sorted([d for d in os.listdir(base_dir) if d.startswith(al_name + "_")])

        for folder in base_folders:
            seed_str = folder.replace(al_name + "_", "")
            try:
                seed = int(seed_str)
            except:
                continue

            zip_path = find_zip(os.path.join(base_dir, folder))
            if zip_path is None:
                continue

            try:
                model = TD3.load(zip_path, env=env)
                seed_idx = base_folders.index(folder)
                eval_seed = eval_seeds[seed_idx % len(eval_seeds)]
                re, sm = eval_one_model(model, env, eval_seed, MAX_EPISODES)
                print(f"  {label:12s} seed={seed}: re={re:.1f}, sm={sm:.3f}")
                all_rows.append({
                    'env': env_name, 'config': label,
                    'seed': seed, 're': round(re, 1), 'sm': round(sm, 4),
                })
            except Exception as e:
                print(f"  {label} seed {seed}: ERROR {e}")

        # 2. Ablation configs from pths_10seed
        abl_env_dir = os.path.join(ABLATION_DIR, env_name)
        for label, config in ABLATION_CONFIGS:
            folders = sorted([d for d in os.listdir(abl_env_dir) if d.startswith(config + "_")])

            for folder in folders:
                seed_str = folder.replace(config + "_", "")
                try:
                    seed = int(seed_str)
                except:
                    continue

                zip_path = find_zip(os.path.join(abl_env_dir, folder))
                if zip_path is None:
                    continue

                try:
                    model = TD3.load(zip_path, env=env)
                    seed_idx = folders.index(folder)
                    eval_seed = eval_seeds[seed_idx % len(eval_seeds)]
                    re, sm = eval_one_model(model, env, eval_seed, MAX_EPISODES)
                    print(f"  {label:12s} seed={seed}: re={re:.1f}, sm={sm:.3f}")
                    all_rows.append({
                        'env': env_name, 'config': label,
                        'seed': seed, 're': round(re, 1), 'sm': round(sm, 4),
                    })
                except Exception as e:
                    print(f"  {label} seed {seed}: ERROR {e}")

        env.close()

        # Save CSV
        if all_rows:
            csv_path = os.path.join(OUTPUT_DIR, f"ablation_per_seed_{env_name}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"\nSaved: {csv_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
