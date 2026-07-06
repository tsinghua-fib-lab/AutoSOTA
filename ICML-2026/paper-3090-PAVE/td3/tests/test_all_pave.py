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

# argparse에 추가
parser.add_argument("--grad_lamT", type=float, default=None)
parser.add_argument("--grad_lamS", type=float, default=None)
parser.add_argument("--grad_lamC", type=float, default=None)
parser.add_argument("--grad_sigma", type=float, default=None, help="MPR noise std (default: 0.01 in model)")
parser.add_argument("--grad_delta", type=float, default=None, help="Curvature margin (default: 1.0 in model)")
parser.add_argument("--start_num", type=int, default=0, help="Start index for seeds (0-based).")
parser.add_argument("--max_num", type=int, default=5, help="Number of seeds to run.")

args = parser.parse_args()
print(f"Selected envs: {args.envs}")
print(f"Selected algs: {args.algs}")
print(f"Selected device: {args.device}")

# 테스트할 컨트롤러 목록
train_algs = args.algs
# 테스트할 env 목록
train_envs = args.envs #["ant", "hopper", "humanoid", "lunar", "pendulum", "reacher"]

# 한개 알고리즘을 1개 env에서 테스트하는데 약 2시간 소요
# -> 모든 env 테스트시 12시간, 모든 알고리즘 테스트시 약 60시간

seeds = load_seeds(seed_file_path)
start_num = args.start_num
max_num = args.max_num
if len(seeds) < start_num + max_num:
    raise ValueError(f"Not enough seeds in {seed_file_path}: required {start_num + max_num}, found {len(seeds)}")


if "pave" in train_algs:
    for env_name in train_envs:
        if args.grad_lamT is not None:
            alg_args["pave"][env_name]["grad_lamT"] = args.grad_lamT
        if args.grad_lamS is not None:
            alg_args["pave"][env_name]["grad_lamS"] = args.grad_lamS
        if args.grad_lamC is not None:
            alg_args["pave"][env_name]["grad_lamC"] = args.grad_lamC
        if args.grad_sigma is not None:
            alg_args["pave"][env_name]["grad_sigma"] = args.grad_sigma
        if args.grad_delta is not None:
            alg_args["pave"][env_name]["grad_delta"] = args.grad_delta
            
jobs = []
for env_name in train_envs:

    save_dir = os.path.join(save_dir_root, env_name) + '/'
    log_dir = os.path.join(logs_dir_root, env_name) + '/'


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        time.sleep(2)

    for num in range(start_num, start_num + max_num):
        seed = seeds[num]
        for alg_name in train_algs:
            # alg_cnts[alg_name](seed, env_timestep[env_name], save_dir, log_dir, train_envs_dict[env_name], env_args[env_name], alg_args[alg_name][env_name])
            jobs.extend([
                partial(alg_cnts[alg_name], seed, env_timestep[env_name], save_dir, log_dir, train_envs_dict[env_name], env_args[env_name], alg_args[alg_name][env_name], args.device),
            ])

with ProcessPoolExecutor(max_workers=max_concurrent_num) as executor:
    future_to_job = {executor.submit(job): job for job in jobs}

    for future in as_completed(future_to_job):
        job_func = future_to_job[future]
        try:
            future.result()
        except Exception as exc:
            print(f"Job {job_func} generated an exception: {exc}")
        else:
            print(f"Job {job_func} completed successfully.")

pth_names = ["base_td3", "caps_td3", "grad_td3", "asap_td3","pave_td3","asap_td3"]
test_some_path(save_dir_root, True, pth_names, "test_all")