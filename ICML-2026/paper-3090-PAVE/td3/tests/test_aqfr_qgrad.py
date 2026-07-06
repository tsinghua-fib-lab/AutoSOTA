import sys
import gymnasium as gym
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import argparse

from stable_baselines3.common.env_util import make_vec_env
from typing import Callable, Optional
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

from modules.action_extractor import test_some_path

from modules.envs import make_ant_env, make_hopper_env, make_humanoid_env, make_lunar_env, make_pendulum_env, make_reacher_env, make_walker_env
from modules.params import env_timestep, env_args, alg_args

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.aqfr_qgrad_td3 import AQFR_QGRAD_TD3

def train_aqfr_qgrad(seed:int, total_time_steps:int, save_dir:str, log_dir:str, 
                  mkenv_func : Callable, env_args:dict, alg_args:dict, 
                  device: str = 'auto', detail_info: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pth_name = f"aqfr_qgrad_td3_{seed}"
    log_name = f"AQFR_QGRAD_TD3_{seed}"
    if detail_info :
        log_name_parts = [
            f"lambda_{alg_args['adv_lambda']}"
        ]
        combind_part = "_".join(log_name_parts)
        pth_name = f"aqfr_qgrad_td3_{combind_part}_{seed}"
        log_name = f"AQFR_QGRAD_TD3_{combind_part}_{seed}"

    # save dir 변경
    local_save_dir = os.path.join(save_dir, pth_name)
    if not os.path.exists(local_save_dir):
        os.makedirs(local_save_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=total_time_steps//5,
        save_path=local_save_dir,
        name_prefix='mid'
    )

    n_envs = 1  # 원하는 env 개수
    exploration_noise = 0.1
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    if 'exploration_noise' in env_args:
        exploration_noise = env_args['exploration_noise']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    action_dim = vec_env.action_space.shape[0]
    mean    = np.zeros(action_dim)
    sigma   = exploration_noise * np.ones(action_dim)
    action_noise = NormalActionNoise(mean=mean, sigma=sigma)
    td3_args = {k: v for k, v in env_args.items() if k != "n_envs" and k != "exploration_noise"}
    td3_args.update(alg_args)
    model = AQFR_QGRAD_TD3("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                      device=device, action_noise=action_noise, **td3_args)
    # 강제 저장
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))

    model.learn(total_timesteps=total_time_steps, tb_log_name=log_name,
                callback=checkpoint_callback)
    #save file 이름
    save_name = os.path.join(local_save_dir, f"final")
    model.save(save_name)
    vec_env.close()
    del model

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
    "aqfr": train_aqfr_qgrad
})

save_dir_root = f"./td3/results/pths/"
logs_dir_root = f"./td3/results/tensorboard_logs/"
seed_file_path = "./td3/tests/seeds.txt"
result_dir_root = f"./td3/results/"
max_num = 10
max_concurrent_num = 5

def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]
    
def mk_param_list_lam(lam_list, eps_list, steps_list, alpha_list):
    param_list = []
    pth_list=[]
    for lam in lam_list:
        param_list.append(dict(
            adv_lambda = lam,))
        log_name_parts = [
            f"lambda_{lam}"
        ]
        combind_part = "_".join(log_name_parts)
        pth_name = f"aqfr_qgrad_td3_{combind_part}"
        pth_list.append(pth_name)
    return param_list, pth_list


parser = argparse.ArgumentParser()
parser.add_argument(
    "--envs",
    nargs="+",
    choices=["ant", "hopper", "lunar", "pendulum", "reacher", "walker"],
    default=["ant", "hopper", "lunar", "pendulum", "reacher", "walker"],
    help="List of environments to train on."
)

parser.add_argument(
    "--lam",
    nargs="+",
    type=float,
    default=[0.1, 0.25, 0.5],
    help="List of lambda values (floats). 예: --lams 0.1 1.5 2.3"
)

parser.add_argument(
    "--eps",
    nargs="+",
    type=float,
    default=[0.1, 0.2],
    help="List of lambda values (floats). 예: --lams 0.1 1.5 2.3"
)

parser.add_argument(
    "--steps",
    nargs="+",
    type=int,
    default=[3, 5, 10],
    help="List of lambda values (floats). 예: --lams 0.1 1.5 2.3"
)

parser.add_argument(
    "--alpha",
    nargs="+",
    type=float,
    default=[0.01, 0.03, 0.05],
    help="List of lambda values (floats). 예: --lams 0.1 1.5 2.3"
)

parser.add_argument(
    "--num",           # 옵션 이름
    type=int,        # 타입은 float
    default=5,       # 기본값
    help="test num for each parameter (int). 예: --num 3"
)
parser.add_argument(
    "--cnum",           # 옵션 이름
    type=int,        # 타입은 float
    default=5,       # 기본값
    help="test num for each parameter (int). 예: --num 3"
)

parser.add_argument(
    "--device",
    type=str,
    default="auto",
    help="Device to use for training: 'cpu', 'cuda', or 'cuda:<id>'."
)
args = parser.parse_args()
print(f"Selected envs: {args.envs}")
print(f"Selected device: {args.device}")

# 테스트할 컨트롤러 목록
train_algs = ["aqfr"]
# 테스트할 env 목록
train_envs = args.envs #["ant", "hopper", "humanoid", "lunar", "pendulum", "reacher"]

test_params, pth_list = mk_param_list_lam(args.lam, args.eps, args.steps, args.alpha)
max_num = args.num
max_concurrent_num = args.cnum

# 한개 알고리즘을 1개 env에서 테스트하는데 약 2시간 소요
# -> 모든 env 테스트시 12시간, 모든 알고리즘 테스트시 약 60시간

seeds = load_seeds(seed_file_path)
if len(seeds) < max_num:
    raise ValueError(f"Not enough seeds in {seed_file_path}: required {max_num}, found {len(seeds)}")

jobs = []
for env_name in train_envs:

    save_dir = os.path.join(save_dir_root, env_name) + '/'
    log_dir = os.path.join(logs_dir_root, env_name) + '/'


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        time.sleep(2)

    for num in range(max_num):
        seed = seeds[num]
        for alg_name in train_algs:
            for test_param in test_params:
                # alg_cnts[alg_name](seed, env_timestep[env_name], save_dir, log_dir, train_envs_dict[env_name], env_args[env_name], test_param, args.device, True)
                jobs.extend([
                    partial(alg_cnts[alg_name], seed, env_timestep[env_name], save_dir, log_dir, train_envs_dict[env_name], env_args[env_name], test_param, args.device, True),
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

# pth_names = ["base_td3", "caps_td3", "grad_td3", "asap_td3"]
test_some_path(save_dir_root, True, pth_list, f"test_aqfr_qgrad_env_{args.envs[0]}_lam_{args.lam[0]}")