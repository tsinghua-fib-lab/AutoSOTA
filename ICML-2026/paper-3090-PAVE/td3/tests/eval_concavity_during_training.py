"""
Concavity satisfaction during training — using mid checkpoints.
Base vs PAVE, lunar + walker, seed #1 (178132).
Measures tr(∇²_aa Q) < -1.0 at each checkpoint.

Run from PAVE_Merge root: python td3/tests/eval_concavity_during_training.py
"""
import sys, os
import numpy as np
import torch as th

sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.envs import make_lunar_env, make_walker_env
from stable_baselines3 import TD3

MAX_EPISODES = 10
EVAL_SEED_FILE = "./sac/tests/validation_seeds.txt"
TRAIN_SEEDS = [178132, 410580, 922852, 787576, 660993]
DELTA = 1.0
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

def load_eval_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

CONFIGS = [
    {
        "env_name": "lunar",
        "env_func": make_lunar_env,
        "base_dir": "./Full/td3/pths/lunar/",
        "methods": ["base_td3", "pave_td3"],
    },
    {
        "env_name": "walker",
        "env_func": make_walker_env,
        "base_dir": "./Full/td3/pths/walker/",
        "methods": ["base_td3", "pave_td3"],
    },
]

def measure_concavity(model, env, eval_seed, max_episodes):
    """Measure concavity satisfaction rate for one model."""
    traces = []
    obs, info = env.reset(seed=eval_seed)

    for ep in range(max_episodes):
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            action_tensor = th.as_tensor(action, device=obs_tensor.device).unsqueeze(0)
            action_tensor.requires_grad_(True)

            q1 = model.critic(obs_tensor, action_tensor)[0].sum()
            grad_a = th.autograd.grad(q1, action_tensor, create_graph=True, retain_graph=True)[0]

            trace = 0.0
            for i in range(grad_a.shape[1]):
                grad_aa_i = th.autograd.grad(grad_a[0, i], action_tensor,
                                              create_graph=True, retain_graph=True)[0]
                trace += grad_aa_i[0, i].item()
            traces.append(trace)

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                break

    traces = np.array(traces)
    satisfy_rate = float((traces < -DELTA).mean())
    return satisfy_rate, float(traces.mean()), len(traces)

def main():
    eval_seeds = load_eval_seeds(EVAL_SEED_FILE)
    print("=== Concavity During Training (5 seeds) ===\n")

    for config in CONFIGS:
        env_name = config["env_name"]
        base_dir = config["base_dir"]
        env = config["env_func"]("rgb_array")()

        print(f"--- {env_name} ---")

        for method in config["methods"]:
            # Collect checkpoints across 5 seeds
            # First, find checkpoint steps from seed #1
            seed1_dir = os.path.join(base_dir, f"{method}_{TRAIN_SEEDS[0]}")
            if not os.path.isdir(seed1_dir):
                print(f"  {method}: seed dir not found")
                continue

            step_list = []
            for f in sorted(os.listdir(seed1_dir)):
                if f.startswith("mid_") and f.endswith(".zip"):
                    step_str = f.replace("mid_", "").replace("_steps.zip", "")
                    try:
                        step = int(step_str)
                        if step > 0:
                            step_list.append(step)
                    except ValueError:
                        pass
            step_list.sort()

            print(f"  {method} (steps: {step_list}):")

            for step in step_list:
                seed_rates = []
                for si, train_seed in enumerate(TRAIN_SEEDS):
                    model_dir = os.path.join(base_dir, f"{method}_{train_seed}")
                    ckpt_path = os.path.join(model_dir, f"mid_{step:05d}_steps.zip")
                    if not os.path.exists(ckpt_path):
                        # Try without leading zeros
                        ckpt_path = os.path.join(model_dir, f"mid_{step}_steps.zip")
                    if not os.path.exists(ckpt_path):
                        continue

                    try:
                        model = TD3.load(ckpt_path, env=env)
                        eval_seed = eval_seeds[si]
                        sat_rate, _, _ = measure_concavity(model, env, eval_seed, MAX_EPISODES)
                        seed_rates.append(sat_rate)
                    except Exception as e:
                        pass

                if seed_rates:
                    mean_rate = np.mean(seed_rates)
                    std_rate = np.std(seed_rates)
                    print(f"    step={step:>8d}: concavity={mean_rate:.3f}±{std_rate:.3f} ({len(seed_rates)} seeds)")

        env.close()
        print()

if __name__ == "__main__":
    main()
