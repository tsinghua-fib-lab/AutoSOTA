import sys
import gymnasium as gym
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import argparse
from typing import Callable, Optional
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from stable_baselines3.common.env_util import make_vec_env
from models.asap_sac_feasibility import ASAPSAC_feasibility, ASAPPolicy_nonshare

from tests.modules.action_extractor import test_some
from tests.modules.envs import make_ant_env, make_hopper_env, make_humanoid_env, make_lunar_env, make_pendulum_env, make_reacher_env, make_walker_env

from modules.params import env_timestep, env_args, alg_args

def train_base(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable,
             env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    sac_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    sac_args.update(alg_args)
    model = ASAPSAC_feasibility(ASAPPolicy_nonshare, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    **sac_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"SAC_BASE_{seed}")
    model.save(f"{save_dir}sac_base_{seed}")
    vec_env.close()
    del model

def finetune_asap(seed:int, total_time_steps:int, base_dir:str, save_dir:str, log_dir:str, mkenv_func : Callable,
             env_args:dict, alg_args:dict, num_samples:int, predict_train_step:int):
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    sac_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    sac_args.update(alg_args)
    model = ASAPSAC_feasibility.load(f"{base_dir}sac_base_{seed}",
                                   env=vec_env,
                                   seed=seed,
                                   tensorboard_log=log_dir,
                                   **sac_args)
    model.reset_buffer(num_samples)
    model.train_predictor(predict_train_step)
    model.learn_with_asap(total_timesteps=total_time_steps, tb_log_name=f"ASAP_FINE_lamS{alg_args['lam_smooth']}_{seed}")
    model.save(f"{save_dir}asap_fine_lamS{alg_args['lam_smooth']}_{seed}")


train_envs_dict = dict({
    "ant" : make_ant_env,
    "hopper" : make_hopper_env,
    "humanoid" : make_humanoid_env,
    "lunar" : make_lunar_env,
    "pendulum" : make_pendulum_env,
    "reacher" : make_reacher_env,
    "walker" : make_walker_env
})

save_dir_root = f"./sac/tests/feasibility/results/pths/"
save_base_dir_root = f"./sac/tests/feasibility/base/pths/"
logs_dir_root = f"./sac/tests/feasibility/results/tensorboard_logs/"
seed_file_path = "./sac/tests/seeds.txt"
result_dir_root = f"./sac/tests/feasibility/results/"
max_num = 10
max_concurrent_num = 5

def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

def mk_param_list_lam(lam_list):
    param_list = []
    pth_list=[]
    for lam in lam_list:
        param_list.append(dict(
            lam_predict = 1.0,
            lam_smooth = lam))
        pth_list.append(f"asap_fine_lamS{lam}_")
    pth_list.append("sac_base_")
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
    "--lams",
    nargs="+",
    type=float,
    default=[30.0, 3.0, 0.3, 0.03],
    help="List of lambda values (floats). 예: --lams 0.1 1.5 2.3"
)
parser.add_argument(
    "--seedidx",           # 옵션 이름
    type=int,        # 타입은 float
    default=0,       # 기본값
    help="Single tau value (float). 예: --tau 0.01"
)
parser.add_argument(
    "--timestep",           # 옵션 이름
    type=int,        # 타입은 float
    default=300000,       # 기본값
    help="Single tau value (float). 예: --tau 0.01"
)
parser.add_argument(
    "--samples",           # 옵션 이름
    type=int,        # 타입은 float
    default=1000000,       # 기본값
    help="Single tau value (float). 예: --tau 0.01"
)
parser.add_argument(
    "--predict_train_step",           # 옵션 이름
    type=int,        # 타입은 float
    default=100000,       # 기본값
    help="Single tau value (float). 예: --tau 0.01"
)
parser.add_argument(
    "--num",           # 옵션 이름
    type=int,        # 타입은 float
    default=10,       # 기본값
    help="test num for each parameter (int). 예: --num 3"
)
args = parser.parse_args()
print(f"Selected envs: {args.envs}")
print(f"Selected lams: {args.lams}")
print(f"Selected seedidx: {args.seedidx}")
print(f"Selected timestep: {args.timestep}")
print(f"Selected samples: {args.samples}")
print(f"Selected max_num: {args.num}")
print(f"Selected predict_train_step: {args.predict_train_step}")


# 테스트할 env 목록
train_envs = args.envs #["ant", "hopper", "humanoid", "lunar", "pendulum", "reacher"]

test_params, pth_list = mk_param_list_lam(args.lams)
default_param = dict(
            lam_predict = 1.0,
            lam_smooth = 3.0)
max_num = args.num

# 한개 알고리즘을 1개 env에서 테스트하는데 약 2시간 소요
# -> 모든 env 테스트시 12시간, 모든 알고리즘 테스트시 약 60시간

seeds = load_seeds(seed_file_path)
if len(seeds) < max_num:
    raise ValueError(f"Not enough seeds in {seed_file_path}: required {max_num}, found {len(seeds)}")

jobs = []
jobs_after = []

for env_name in train_envs:

    save_dir = os.path.join(save_dir_root, env_name) + '/'
    base_save_dir = os.path.join(save_base_dir_root, env_name) + '/'
    log_dir = os.path.join(logs_dir_root, env_name) + '/'


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        time.sleep(2)

    for num in range(max_num):
        seed = seeds[args.seedidx + num]
        # train_base(seed, env_timestep[env_name], base_save_dir, log_dir, train_envs_dict[env_name], env_args[env_name], test_param)
        # train_base(seed, 50000, base_save_dir, log_dir, train_envs_dict[env_name], env_args[env_name], default_param)
        jobs.extend([
                    partial(train_base, seed, env_timestep[env_name], base_save_dir, log_dir, train_envs_dict[env_name], env_args[env_name], default_param),
                ])
    for test_param in test_params:
        for num in range(max_num):
            seed = seeds[args.seedidx + num]
            finetune_timestep = args.timestep
            # finetune_asap(seed, finetune_timestep, base_save_dir, save_dir, log_dir, train_envs_dict[env_name], env_args[env_name], test_param, args.samples, args.predict_train_step)
            jobs_after.extend([
                        partial(finetune_asap, seed, finetune_timestep, base_save_dir, save_dir, log_dir, train_envs_dict[env_name], env_args[env_name], test_param, args.samples, args.predict_train_step),
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

with ProcessPoolExecutor(max_workers=max_concurrent_num) as executor:
    future_to_job = {executor.submit(job): job for job in jobs_after}

    for future in as_completed(future_to_job):
        job_func = future_to_job[future]
        try:
            future.result()
        except Exception as exc:
            print(f"Job {job_func} generated an exception: {exc}")
        else:
            print(f"Job {job_func} completed successfully.")


test_some(save_base_dir_root, True, pth_list)
test_some(save_dir_root, True, pth_list)