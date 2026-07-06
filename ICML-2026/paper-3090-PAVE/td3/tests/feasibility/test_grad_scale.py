import argparse
import numpy as np
import torch as th
from torch.autograd.functional import jacobian
import gymnasium as gym
from stable_baselines3 import TD3

def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def run_episode(env, model):
    obs, _ = env.reset(seed=None)
    done, trunc = False, False
    obs_list, act_list, rew_list = [], [], []
    while not (done or trunc):
        with th.no_grad():
            action, _ = model.predict(obs, deterministic=True)
        obs_list.append(np.asarray(obs, dtype=np.float32))
        act_list.append(np.asarray(action, dtype=np.float32))
        obs, reward, done, trunc, _ = env.step(action)
        rew_list.append(float(reward))
    return np.array(obs_list), np.array(act_list), np.array(rew_list)

def main():
    model_path = "td3/tests/feasibility/pths/hopper.zip"
    device = "cpu"
    env = gym.make("Hopper-v5")

    model = TD3.load(model_path, device=device)
    model.policy.set_training_mode(True)
    model.policy.critic.train(True)
    critic = model.policy.critic #.to(device)

    obs_np, act_np, _ = run_episode(env, model)

    T = len(obs_np)