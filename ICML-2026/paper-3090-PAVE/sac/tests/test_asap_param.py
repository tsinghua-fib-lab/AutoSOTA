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

from modules.controller import train_asap_lam_test
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
    "asap" : train_asap_lam_test
})

save_dir_root = f"./sac/results/pths/"
logs_dir_root = f"./sac/results/tensorboard_logs/"
seed_file_path = "./sac/tests/seeds.txt"
result_dir_root = f"./sac/results/"
max_num = 3
max_concurrent_num = 5

def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

def mk_param_list_lam(lam_list, tau, lamt_list):
    param_list = []
    pth_list=[]
    for lamt in lamt_list:
        for lam in lam_list:
            param_list.append(dict(
                lam_p = 2.0,
                lam_t = lamt,
                lam_s = lam,
                asap_tau = tau))
            pth_list.append(f"asap_sac_lamS{lam}_lamT{lamt}_tau{tau}_")
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
    default=[30.0, 3.0, 0.3],
    help="List of lambda values (floats). 예: --lams 0.1 1.5 2.3"
)
parser.add_argument(
    "--lamt",
    nargs="+",
    type=float,
    default=[1.0, 0.1],
    help="List of lambda values (floats). 예: --lams 0.1 1.5 2.3"
)
parser.add_argument(
    "--tau",          
    type=float,      
    default=0.01,     
    help="Single tau value (float). 예: --tau 0.01"
)
parser.add_argument(
    "--num",      
    type=int,   
    default=3,   
    help="test num for each parameter (int). 예: --num 3"
)
parser.add_argument(
    "--cnum",   
    type=int,  
    default=5, 
    help="test num for each parameter (int). 예: --num 3"
)
args = parser.parse_args()
print(f"Selected envs: {args.envs}")
print(f"Selected lams: {args.lams}")
print(f"Selected tau: {args.tau}")
print(f"Selected max_num: {args.num}")

# 테스트할 컨트롤러 목록
train_algs = ["asap"]
# 테스트할 env 목록
train_envs = args.envs #["ant", "hopper", "humanoid", "lunar", "pendulum", "reacher"]

test_params, pth_list = mk_param_list_lam(args.lams, args.tau, args.lamt)
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
                jobs.extend([
                    partial(alg_cnts[alg_name], seed, env_timestep[env_name], save_dir, log_dir, train_envs_dict[env_name], env_args[env_name], test_param),
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

# extract_all(result_dir_root)
subpath = f"asap_{args.envs[0]}_lamt{args.lamt[0]}/"
test_some_path(save_dir_root, True, pth_list, subpath)