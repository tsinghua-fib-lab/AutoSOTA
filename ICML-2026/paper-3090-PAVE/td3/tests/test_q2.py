import sys
import gymnasium as gym
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import argparse

from stable_baselines3.common.env_util import make_vec_env

from modules.q_extractor2 import test_some_path

from modules.controller import train_vanilla, train_lips_l, train_lips_g, train_caps, train_grad, train_pave, train_asap
from modules.envs import make_ant_env, make_hopper_env, make_humanoid_env, make_lunar_env, make_pendulum_env, make_reacher_env, make_walker_env
from modules.params import env_timestep, env_args, alg_args

train_envs_dict = dict({
    "ant" : make_ant_env,
    "hopper" : make_hopper_env,
    "humanoid" : make_humanoid_env,
    "lunar" : make_lunar_env,
    "pendulum" : make_pendulum_env,
    "reacher" : make_reacher_env,
    "walker" : make_walker_env
})

alg_cnts = dict({
    "vanilla": train_vanilla,
    "lips_l": train_lips_l,
    "lips_g": train_lips_g,
    "caps" : train_caps,
    "grad" : train_grad,
    "pave" : train_pave,
    "asap" : train_asap
})

save_dir_root = f"./td3/results/pths/"
logs_dir_root = f"./td3/results/tensorboard_logs/"
seed_file_path = "./td3/tests/seeds.txt"
result_dir_root = f"./td3/results/"


max_num = 5
max_concurrent_num = 5

def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--envs",
    nargs="+",
    choices=["ant", "hopper", "lunar", "pendulum", "reacher", "walker"],
    default=["ant", "hopper", "lunar", "pendulum", "reacher", "walker"],
    help="List of environments to train on."
)
parser.add_argument(
    "--algs",
    nargs="+",
    choices=["vanilla", "lips_l", "lips_g", "caps", "grad","pave","asap"],
    default=["vanilla", "caps", "grad"],
    help="List of environments to train on."
)
parser.add_argument(
    "--device",
    type=str,
    default="auto",
    help="Device to use for training: 'cpu', 'cuda', or 'cuda:<id>'."
)
args = parser.parse_args()
print(f"Selected envs: {args.envs}")
print(f"Selected algs: {args.algs}")
print(f"Selected device: {args.device}")

# 테스트할 컨트롤러 목록
train_algs = args.algs
# 테스트할 env 목록
train_envs = args.envs 

pth_names = ["base_td3", "caps_td3", "grad_td3", "asap_td3","pave_td3"]
test_some_path(save_dir_root, True, pth_names, "test_all")