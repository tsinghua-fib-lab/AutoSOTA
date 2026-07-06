"""
TD3 LipsNet experiments: lips_td3, asap_lips_td3, pave_lips_td3
Run from PAVE_Merge root:
  /home/airlab1tb/anaconda3/envs/gym2/bin/python td3/tests/test_lipsnet_td3.py \
    --envs lunar walker --algs lips asap_lips pave_lips --device cuda:0
"""
import sys
import gymnasium as gym
import os
import time
import argparse
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.envs import make_lunar_env, make_walker_env, make_ant_env, make_hopper_env, make_pendulum_env, make_reacher_env
from modules.params import env_timestep, env_args

from models.lips_td3 import LipsTD3, LipsTD3Policy
from models.asap_lips_td3 import ASAPLipsTD3, ASAPLipsTD3Policy
from models.pave_lips_td3 import PaveLipsTD3

train_envs_dict = {
    "lunar": make_lunar_env,
    "walker": make_walker_env,
    "ant": make_ant_env,
    "hopper": make_hopper_env,
    "pendulum": make_pendulum_env,
    "reacher": make_reacher_env,
}

seed_file_path = "./td3/tests/seeds.txt"

def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


# ── LipsNet hyperparameters (same across all envs, from LipsNet paper) ──
LIPS_KWARGS = dict(
    lips_lam=1e-5,
    lips_eps=1e-4,
    lips_k_init=50.0,
    lips_f_size=[64, 64],
    lips_k_size=[32],
    lips_global=False,
)

# ── ASAP TD3 hyperparameters (from params.py alg_args["asap"]) ──
ASAP_ARGS = {
    "lunar": dict(asap_lamP=2.0, asap_lamS=0.03, asap_lamT=0.005),
    "walker": dict(asap_lamP=2.0, asap_lamS=0.3, asap_lamT=0.05),
    "ant": dict(asap_lamP=2.0, asap_lamS=0.3, asap_lamT=0.05),
    "hopper": dict(asap_lamP=2.0, asap_lamS=0.3, asap_lamT=0.07),
    "pendulum": dict(asap_lamP=2.0, asap_lamS=0.03, asap_lamT=0.005),
    "reacher": dict(asap_lamP=2.0, asap_lamS=0.1, asap_lamT=0.1),
}

# ── PAVE TD3 hyperparameters (from params.py alg_args["pave"]) ──
PAVE_ARGS = {
    "lunar": dict(grad_lamS=0.1, grad_lamT=0.1, grad_lamC=0.01),
    "walker": dict(grad_lamS=0.1, grad_lamT=0.1, grad_lamC=0.01),
    "ant": dict(grad_lamS=0.1, grad_lamT=0.005, grad_lamC=0.5),
    "hopper": dict(grad_lamS=0.1, grad_lamT=0.005, grad_lamC=0.5),
    "pendulum": dict(grad_lamS=2.0, grad_lamT=0.005, grad_lamC=2.0),
    "reacher": dict(grad_lamS=0.1, grad_lamT=0.1, grad_lamC=0.01),
}


def _make_env_and_noise(mkenv_func, env_args_dict):
    n_envs = env_args_dict.get("n_envs", 1)
    exploration_noise = env_args_dict.get("exploration_noise", 0.1)
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    action_dim = vec_env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim),
        sigma=exploration_noise * np.ones(action_dim),
    )
    td3_args = {k: v for k, v in env_args_dict.items() if k not in ("n_envs", "exploration_noise")}
    return vec_env, action_noise, td3_args


def train_lips(seed, total_steps, save_dir, log_dir, mkenv_func, env_args_dict, lips_kwargs, device="auto"):
    local_save_dir = os.path.join(save_dir, f"lips_td3_lips{lips_kwargs['lips_lam']}_{seed}")
    os.makedirs(local_save_dir, exist_ok=True)
    cb = CheckpointCallback(save_freq=total_steps // 5, save_path=local_save_dir, name_prefix="mid")
    vec_env, action_noise, td3_args = _make_env_and_noise(mkenv_func, env_args_dict)
    model = LipsTD3(LipsTD3Policy, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    device=device, action_noise=action_noise, **td3_args, **lips_kwargs)
    model.save(os.path.join(local_save_dir, "mid_00000_steps"))
    model.learn(total_timesteps=total_steps, tb_log_name=f"LIPS_TD3_{seed}", callback=cb)
    model.save(os.path.join(local_save_dir, "final"))
    vec_env.close()
    del model


def train_asap_lips(seed, total_steps, save_dir, log_dir, mkenv_func, env_args_dict, asap_args, lips_kwargs, device="auto"):
    lamT = asap_args.get("asap_lamT", 0.005)
    lamS = asap_args.get("asap_lamS", 0.03)
    lamP = asap_args.get("asap_lamP", 2.0)
    local_save_dir = os.path.join(save_dir, f"asap_lips_td3_T{lamT}_S{lamS}_P{lamP}_{seed}")
    os.makedirs(local_save_dir, exist_ok=True)
    cb = CheckpointCallback(save_freq=total_steps // 5, save_path=local_save_dir, name_prefix="mid")
    vec_env, action_noise, td3_args = _make_env_and_noise(mkenv_func, env_args_dict)
    model = ASAPLipsTD3(ASAPLipsTD3Policy, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                        device=device, action_noise=action_noise, **td3_args, **asap_args, **lips_kwargs)
    model.save(os.path.join(local_save_dir, "mid_00000_steps"))
    model.learn(total_timesteps=total_steps, tb_log_name=f"ASAP_LIPS_TD3_{seed}", callback=cb)
    model.save(os.path.join(local_save_dir, "final"))
    vec_env.close()
    del model


def train_pave_lips(seed, total_steps, save_dir, log_dir, mkenv_func, env_args_dict, pave_args, lips_kwargs, device="auto"):
    lamS = pave_args.get("grad_lamS", 0.1)
    lamT = pave_args.get("grad_lamT", 0.1)
    lamC = pave_args.get("grad_lamC", 0.01)
    local_save_dir = os.path.join(save_dir, f"pave_lips_td3_S{lamS}_T{lamT}_C{lamC}_{seed}")
    os.makedirs(local_save_dir, exist_ok=True)
    cb = CheckpointCallback(save_freq=total_steps // 5, save_path=local_save_dir, name_prefix="mid")
    vec_env, action_noise, td3_args = _make_env_and_noise(mkenv_func, env_args_dict)
    model = PaveLipsTD3(LipsTD3Policy, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                        device=device, action_noise=action_noise, **td3_args, **pave_args, **lips_kwargs)
    model.save(os.path.join(local_save_dir, "mid_00000_steps"))
    model.learn(total_timesteps=total_steps, tb_log_name=f"PAVE_LIPS_TD3_{seed}", callback=cb)
    model.save(os.path.join(local_save_dir, "final"))
    vec_env.close()
    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", choices=list(train_envs_dict.keys()), default=["lunar", "walker"])
    parser.add_argument("--algs", nargs="+", choices=["lips", "asap_lips", "pave_lips"], default=["lips", "asap_lips", "pave_lips"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--start_num", type=int, default=0)
    parser.add_argument("--max_num", type=int, default=5)
    parser.add_argument("--save_root", type=str, default="./LipsNet_TD3/pths/")
    parser.add_argument("--log_root", type=str, default="./LipsNet_TD3/tensorboard_logs/")
    # PAVE hyperparameter overrides (for sweep)
    parser.add_argument("--grad_lamS", type=float, default=None, help="Override PAVE λ1 (MPR)")
    parser.add_argument("--grad_lamT", type=float, default=None, help="Override PAVE λ2 (VFC)")
    parser.add_argument("--grad_lamC", type=float, default=None, help="Override PAVE λ3 (Curv)")
    args = parser.parse_args()

    # Apply PAVE overrides if specified
    if any([args.grad_lamS, args.grad_lamT, args.grad_lamC]):
        for env_name in args.envs:
            if args.grad_lamS is not None:
                PAVE_ARGS[env_name]["grad_lamS"] = args.grad_lamS
            if args.grad_lamT is not None:
                PAVE_ARGS[env_name]["grad_lamT"] = args.grad_lamT
            if args.grad_lamC is not None:
                PAVE_ARGS[env_name]["grad_lamC"] = args.grad_lamC

    seeds = load_seeds(seed_file_path)
    print(f"Envs: {args.envs}, Algs: {args.algs}, Device: {args.device}, Seeds: {args.start_num}-{args.start_num + args.max_num - 1}")
    if "pave_lips" in args.algs:
        for e in args.envs:
            print(f"  PAVE[{e}]: S={PAVE_ARGS[e]['grad_lamS']}, T={PAVE_ARGS[e]['grad_lamT']}, C={PAVE_ARGS[e]['grad_lamC']}")

    jobs = []
    for env_name in args.envs:
        save_dir = os.path.join(args.save_root, env_name) + "/"
        log_dir = os.path.join(args.log_root, env_name) + "/"
        os.makedirs(save_dir, exist_ok=True)

        for num in range(args.start_num, args.start_num + args.max_num):
            seed = seeds[num]

            if "lips" in args.algs:
                jobs.append(partial(train_lips, seed, env_timestep[env_name], save_dir, log_dir,
                                    train_envs_dict[env_name], env_args[env_name], LIPS_KWARGS, args.device))
            if "asap_lips" in args.algs:
                jobs.append(partial(train_asap_lips, seed, env_timestep[env_name], save_dir, log_dir,
                                    train_envs_dict[env_name], env_args[env_name], ASAP_ARGS[env_name], LIPS_KWARGS, args.device))
            if "pave_lips" in args.algs:
                jobs.append(partial(train_pave_lips, seed, env_timestep[env_name], save_dir, log_dir,
                                    train_envs_dict[env_name], env_args[env_name], PAVE_ARGS[env_name], LIPS_KWARGS, args.device))

    print(f"Total jobs: {len(jobs)}")

    # Sequential execution (GPU 1개이므로)
    for i, job in enumerate(jobs):
        print(f"\n[{i+1}/{len(jobs)}] Starting...")
        try:
            job()
            print(f"[{i+1}/{len(jobs)}] Done.")
        except Exception as e:
            print(f"[{i+1}/{len(jobs)}] ERROR: {e}")
