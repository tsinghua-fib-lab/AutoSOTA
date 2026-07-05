import glob
import json
import os
import uuid
from typing import Dict, List

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def generate_gif(
    model, env_name: str, seeds: List[int], idx: int, mean_reward: float, trial: int
):
    save_dir = "Rendered/PPO"
    os.makedirs(save_dir, exist_ok=True)

    for seed in seeds:
        env = gym.make(env_name, render_mode="rgb_array")
        obs, _ = env.reset(seed=seed)
        images = []
        done = False
        step_count = 0
        max_steps = 1000

        rewards = 0.0
        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            img = env.render()
            images.append(img)
            step_count += 1
            rewards += reward

        env.close()

        if images:
            filename = f"Task{idx}_{env_name}_trial{trial + 1}_seed{seed}_reward_{rewards:.2f}.gif"
            save_path = os.path.join(save_dir, filename)
            imageio.mimsave(save_path, images, fps=30, loop=0)
            print(f"Saved GIF to {save_path}")


def make_env(env_name: str):
    def _init():
        env = gym.make(env_name)
        env.reset()
        return env

    return _init


class RolloutEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
            )
            self.rewards.append(mean_reward)
            self.timesteps.append(self.num_timesteps)
            # if self.verbose:
            #     print(f"[Eval] timesteps={self.num_timesteps}, mean_reward={mean_reward:.2f}")
        return True


def train_ppo(
    env_name: str,
    iterations: int = 500,
    env_nums: int = 16,
    eval_freq: int = None,
    device: str = "auto",
) -> Dict:
    # Equivalent total interaction steps
    total_timesteps = iterations
    if eval_freq is None:
        eval_freq = max(total_timesteps // 500 // env_nums, 1)

    # Multiple parallel environments with different seeds
    env_fns = [make_env(env_name) for i in range(env_nums)]
    train_env = DummyVecEnv(env_fns)
    train_env = VecMonitor(train_env)

    # Single instance evaluation environment
    eval_env = gym.make(env_name)

    # PPO initialization
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": [64, 64],
            "activation_fn": torch.nn.Tanh,
            "ortho_init": True,
        },
        verbose=0,
        device=device,
    )

    # Callback evaluation
    eval_callback = RolloutEvalCallback(
        eval_env=eval_env, eval_freq=eval_freq, n_eval_episodes=3, verbose=1
    )

    # Training
    model.learn(
        total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True
    )

    # Close environments
    train_env.close()
    eval_env.close()

    return {
        "model": model,
        "timesteps": eval_callback.timesteps,
        "rewards": eval_callback.rewards,
    }


def get_latest_run_file(env_name: str, idx: int) -> str:
    """Find the latest run file for the given environment"""
    log_dir = "ppo_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Pattern to match files for this environment
    pattern = os.path.join(log_dir, f"T{idx}_{env_name}.json")
    files = glob.glob(pattern)

    if not files:
        return None

    # Get the most recently modified file
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def run_multiple_trials(
    env_name: str,
    idx: int = 0,
    num_trials: int = 10,
    iterations: int = 1_000_000,
    device: str = "auto",
) -> Dict:
    # 路径配置
    log_dir = "ppo_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"T{idx}_{env_name}.json")

    # 1. 尝试恢复现有数据
    existing_data = None
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                existing_data = json.load(f)
            print(f"Found existing data in {log_file}, will resume/append.")
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    all_rewards = existing_data["all_rewards"] if existing_data else []
    all_timesteps = existing_data.get("timesteps") if existing_data else None

    # 2. 循环运行直到达到目标 Trial 数
    while len(all_rewards) < num_trials:
        current_trial_idx = len(all_rewards)
        print(
            f"\n[PPO] Running trial {current_trial_idx + 1}/{num_trials} for {env_name}"
        )

        try:
            # 执行训练
            result = train_ppo(env_name=env_name, iterations=iterations, device=device)

            # 提取最后评估结果
            mean_reward = result["rewards"][-1] if result["rewards"] else 0.0

            # 3. 成功后生成 GIF 记录
            generate_gif(
                model=result["model"],
                env_name=env_name,
                seeds=[42, 123, 999],
                idx=idx,
                mean_reward=mean_reward,
                trial=current_trial_idx,
            )

            # 4. 更新内存数据并即时保存到硬盘
            if all_timesteps is None:
                all_timesteps = result["timesteps"]
            all_rewards.append(result["rewards"])

            result_data = {
                "env": env_name,
                "iterations": iterations,
                "num_trials": len(all_rewards),
                "all_rewards": all_rewards,
                "timesteps": all_timesteps,
            }
            with open(log_file, "w") as f:
                json.dump(result_data, f, indent=2)

            print(f"Successfully completed and saved trial {len(all_rewards)}.")

        except Exception as e:
            print(f"\n[CRITICAL ERROR] Trial {current_trial_idx + 1} crashed: {e}")
            print("Restarting this trial to ensure data integrity...")
            continue

    print(f"All {num_trials} trials for {env_name} (PPO) are done.")
    return result_data


if __name__ == "__main__":
    env_names = [
        "MountainCarContinuous-v0",
        "MountainCar-v0",
        "Pendulum-v1",
        "CartPole-v1",
        "Acrobot-v1",
        "LunarLander-v3",
        "BipedalWalker-v3",
        "InvertedPendulum-v5",
        "InvertedDoublePendulum-v5",
        "Reacher-v5",
        "Pusher-v5",
        "HalfCheetah-v5",
        "Hopper-v5",
        "Walker2d-v5",
        "Swimmer-v5",
        "Ant-v5",
        "Humanoid-v5",
        "HumanoidStandup-v5",
    ]

    device = "cpu"
    if device == "mps":
        print("Using Apple Silicon GPU (MPS) for training.")
    elif device == "cuda":
        print("Using NVIDIA GPU for training.")
    else:
        print("Using CPU for training.")
        torch.set_num_threads(1)

    # Create results dictionary to store all data
    all_results = {}

    idx = 0
    for env_name in env_names:
        idx += 1

        print(f"\nStarting training for {env_name}")
        env_results = run_multiple_trials(
            env_name=env_name,
            idx=idx,
            num_trials=10,  # 10
            iterations=10_000_000,  # 10_000_000
            device=device,
        )
        all_results[env_name] = env_results
        print(f"Completed training for {env_name}")

    print("\nAll training completed. Results saved to individual files in ppo_logs/")
