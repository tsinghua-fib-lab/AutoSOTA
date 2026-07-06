import sys
import gymnasium as gym
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import argparse

from stable_baselines3.common.env_util import make_vec_env

from modules.action_extractor import test_some_path

from modules.controller import train_vanilla, train_caps, train_asap, train_l2c2, train_grad,train_pave
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
    "caps" : train_caps,
    "l2c2" : train_l2c2,
    "grad" : train_grad,
    "asap" : train_asap,
    "pave" : train_pave,
})

save_dir_root = f"./sac/results/pths/"
logs_dir_root = f"./sac/results/tensorboard_logs/"
seed_file_path = "./sac/tests/seeds.txt"
result_dir_root = f"./sac/results/"
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
    choices=["vanilla", "caps", "l2c2", "grad", "asap","pave"],
    default=["vanilla", "caps", "l2c2", "grad", "asap","pave"],
    help="List of environments to train on."
)
args = parser.parse_args()
print(f"Selected envs: {args.envs}")
print(f"Selected algs: {args.algs}")

# 테스트할 컨트롤러 목록
train_algs = args.algs
# 테스트할 env 목록
train_envs = args.envs #["ant", "hopper", "humanoid", "lunar", "pendulum", "reacher"]

# 한개 알고리즘을 1개 env에서 테스트하는데 약 2시간 소요
# -> 모든 env 테스트시 12시간, 모든 알고리즘 테스트시 약 60시간



pth_names = ["vanilla", "caps_sac", "grad_sac", "asap_sac","pave_sac"]
test_some_path(save_dir_root, True, pth_names, "test_all")