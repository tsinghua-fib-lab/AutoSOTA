import sys
import gymnasium as gym
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Callable, Optional
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.custom_td3 import CustomTD3
from models.lips_td3 import LipsTD3, LipsTD3Policy
from models.caps_td3 import CAPSTD3
from models.grad_td3 import GRADTD3
from models.aqfr_td3 import AQFRTD3
from models.asap_td3 import ASAPTD3, ASAPPolicy
from models.sr2l import *
from models.pave_td3 import PaveTD3

def train_vanilla(seed:int, total_time_steps:int, save_dir:str, log_dir:str, 
                  mkenv_func : Callable, env_args:dict, alg_args:dict, device: str = 'auto'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save dir 변경
    local_save_dir = os.path.join(save_dir, f"base_td3_{seed}")
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
    model = CustomTD3("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                      device=device, action_noise=action_noise, **td3_args)
    # 강제 저장
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))

    model.learn(total_timesteps=total_time_steps, tb_log_name=f"BASE_TD3_{seed}",
                callback=checkpoint_callback)

    #save file 이름
    save_name = os.path.join(local_save_dir, f"final")
    model.save(save_name)
    vec_env.close()
    del model

def train_lips_l(seed:int, total_time_steps:int, save_dir:str, log_dir:str, 
                 mkenv_func : Callable, env_args:dict, alg_args:dict, device: str = 'auto'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save dir 변경
    local_save_dir = os.path.join(save_dir, f"lips_l_td3_{seed}")
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
    model = LipsTD3(LipsTD3Policy, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    device=device, action_noise=action_noise, **td3_args)
    # 강제 저장
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))

    model.learn(total_timesteps=total_time_steps, tb_log_name=f"Lips_L_TD3_{seed}",
                callback=checkpoint_callback)
    #save file 이름
    save_name = os.path.join(local_save_dir, f"final")
    model.save(save_name)
    vec_env.close()
    del model

def train_lips_g(seed:int, total_time_steps:int, save_dir:str, log_dir:str, 
                 mkenv_func : Callable, env_args:dict, alg_args:dict, device: str = 'auto'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save dir 변경
    local_save_dir = os.path.join(save_dir, f"lips_g_td3_{seed}")
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
    model = LipsTD3(LipsTD3Policy, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    device=device, action_noise=action_noise, **td3_args)
    # 강제 저장
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))

    model.learn(total_timesteps=total_time_steps, tb_log_name=f"Lips_G_TD3_{seed}",
                callback=checkpoint_callback)
    #save file 이름
    save_name = os.path.join(local_save_dir, f"final")
    model.save(save_name)
    vec_env.close()
    del model

def train_caps(seed:int, total_time_steps:int, save_dir:str, log_dir:str, 
                  mkenv_func : Callable, env_args:dict, alg_args:dict, device: str = 'auto'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # save dir 변경
    local_save_dir = os.path.join(save_dir, f"caps_td3_{seed}")
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
    model = CAPSTD3("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                      device=device, action_noise=action_noise, **td3_args)
    # 강제 저장
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))

    model.learn(total_timesteps=total_time_steps, tb_log_name=f"CAPS_TD3_{seed}",
                callback=checkpoint_callback)
    #save file 이름
    save_name = os.path.join(local_save_dir, f"final")
    model.save(save_name)
    vec_env.close()
    del model

def train_grad(seed:int, total_time_steps:int, save_dir:str, log_dir:str, 
                  mkenv_func : Callable, env_args:dict, alg_args:dict, device: str = 'auto'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save dir 변경
    local_save_dir = os.path.join(save_dir, f"grad_td3_{seed}")
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
    model = GRADTD3("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                      device=device, action_noise=action_noise, **td3_args)
    # 강제 저장
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))

    model.learn(total_timesteps=total_time_steps, tb_log_name=f"GRAD_TD3_{seed}",
                callback=checkpoint_callback)
    #save file 이름
    save_name = os.path.join(local_save_dir, f"final")
    model.save(save_name)
    vec_env.close()
    del model

def train_asap(seed:int, total_time_steps:int, save_dir:str, log_dir:str, 
                  mkenv_func : Callable, env_args:dict, alg_args:dict, device: str = 'auto'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save dir 변경
    local_save_dir = os.path.join(save_dir, f"asap_td3_{seed}")
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
    model = ASAPTD3(ASAPPolicy, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                      device=device, action_noise=action_noise, **td3_args)
    # 강제 저장
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))

    model.learn(total_timesteps=total_time_steps, tb_log_name=f"ASAP_TD3_{seed}",
                callback=checkpoint_callback)
    #save file 이름
    save_name = os.path.join(local_save_dir, f"final")
    model.save(save_name)
    vec_env.close()
    del model


def train_pave(seed:int, total_time_steps:int, save_dir:str, log_dir:str, 
                  mkenv_func : Callable, env_args:dict, alg_args:dict, device: str = 'auto'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save dir 변경 (모든 PAVE 하이퍼파라미터를 이름에 포함)
    lamS = alg_args.get('grad_lamS', 0.1)
    lamT = alg_args.get('grad_lamT', 0.1)
    lamC = alg_args.get('grad_lamC', 0.01)
    sig = alg_args.get('grad_sigma', 0.01)
    delta = alg_args.get('grad_delta', 1.0)
    suffix = f"_S{lamS}_T{lamT}_C{lamC}_sig{sig}_del{delta}"
    local_save_dir = os.path.join(save_dir, f"pave_td3{suffix}_{seed}")
    tb_log_name = f"PAVE_TD3{suffix}_{seed}"
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
    model = PaveTD3("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                      device=device, action_noise=action_noise, **td3_args)
    # 강제 저장
    model.save(os.path.join(local_save_dir, 'mid_00000_steps'))

    model.learn(total_timesteps=total_time_steps, tb_log_name=tb_log_name,
                callback=checkpoint_callback)
    #save file 이름
    save_name = os.path.join(local_save_dir, f"final")
    model.save(save_name)
    vec_env.close()
    del model

def train_s2rl_a(seed:int, total_time_steps:int, save_dir:str, log_dir:str, 
                  mkenv_func : Callable, env_args:dict, alg_args:dict, 
                  device: str = 'auto', detail_info: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pth_name = f"s2rl_a_td3_{seed}"
    log_name = f"S2RL_a_TD3_{seed}"
    if detail_info :
        log_name_parts = [
            f"lambda_{alg_args['adv_lambda']}",
            f"eps_{alg_args['adv_epsilon']}",
            f"steps_{alg_args['adv_steps']}",
            f"alpha_{alg_args['adv_alpha']}",
        ]
        combind_part = "_".join(log_name_parts)
        pth_name = f"s2rl_a_td3_{combind_part}_{seed}"
        log_name = f"S2RL_a_TD3_{combind_part}_{seed}"

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
    model = SR2L_A("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
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



def train_s2rl_c(seed:int, total_time_steps:int, save_dir:str, log_dir:str, 
                  mkenv_func : Callable, env_args:dict, alg_args:dict, 
                  device: str = 'auto', detail_info: bool = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    pth_name = f"s2rl_c_td3_{seed}"
    log_name = f"S2RL_c_TD3_{seed}"
    if detail_info :
        log_name_parts = [
            f"lambda_{alg_args['adv_lambda']}",
            f"eps_{alg_args['adv_epsilon']}",
            f"steps_{alg_args['adv_steps']}",
            f"alpha_{alg_args['adv_alpha']}",
        ]
        combind_part = "_".join(log_name_parts)
        pth_name = f"s2rl_c_td3_{combind_part}_{seed}"
        log_name = f"S2RL_c_TD3_{combind_part}_{seed}"

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
    model = SR2L_C("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
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