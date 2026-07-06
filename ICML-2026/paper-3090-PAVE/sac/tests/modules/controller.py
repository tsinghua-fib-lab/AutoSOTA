import sys
import gymnasium as gym
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Callable, Optional
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.custom_sac import CustomSAC
from models.caps_sac import CAPSSAC
from models.lips_sac import LipsSAC, LipsSACPolicy
from models.l2c2_sac import L2C2SAC
from models.grad_sac import GRADSAC
from models.asap_sac import ASAPSAC, ASAPPolicy_soft
from models.asap_lips_sac import ASAPLIPSSAC, ASAPLIPSPolicy_soft
from models.caps_lips_sac import CAPSLipsSAC
from models.pave_sac import PAVE_SAC
from models.pave_lips_sac import PAVE_LIPS_SAC

def train_vanilla(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    sac_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    sac_args.update(alg_args)
    model = CustomSAC("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                      **sac_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"Vanilla_{seed}")
    model.save(f"{save_dir}vanilla_{seed}")
    vec_env.close()
    del model

def train_caps(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    sac_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    sac_args.update(alg_args)
    model = CAPSSAC("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    **sac_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"CAPS_SAC_{seed}")
    model.save(f"{save_dir}caps_sac_{seed}")
    vec_env.close()
    del model

def train_l2c2(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    sac_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    sac_args.update(alg_args)
    model = L2C2SAC("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    **sac_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"L2C2_SAC_{seed}")
    model.save(f"{save_dir}l2c2_sac_{seed}")
    vec_env.close()
    del model

def train_lips(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    sac_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    sac_args.update(alg_args)
    model = LipsSAC(LipsSACPolicy, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    lips_lam=1e-5, lips_eps=1e-4)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"Lips_SAC_{seed}")
    model.save(f"{save_dir}lips_sac_{seed}")
    vec_env.close()
    del model

def train_grad(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    sac_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    sac_args.update(alg_args)
    model = GRADSAC("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    **sac_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"GRAD_SAC_{seed}")
    model.save(f"{save_dir}grad_sac_{seed}")
    vec_env.close()
    del model

def train_pave(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    sac_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    sac_args.update(alg_args)
    model = PAVE_SAC("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    **sac_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"PAVE_SAC_{seed}")
    model.save(f"{save_dir}pave_sac_{seed}")
    vec_env.close()
    del model

def train_asap(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable,
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
    model = ASAPSAC(ASAPPolicy_soft, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    **sac_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"ASAP_SAC_{seed}")
    model.save(f"{save_dir}asap_sac_{seed}")
    vec_env.close()
    del model

def train_asap_lam_test(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable,
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
    model = ASAPSAC(ASAPPolicy_soft, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    **sac_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"ASAP_SAC_lamS{alg_args['lam_s']}_lamT{alg_args['lam_t']}_tau{alg_args['asap_tau']}_{seed}")
    model.save(f"{save_dir}asap_sac_lamS{alg_args['lam_s']}_lamT{alg_args['lam_t']}_tau{alg_args['asap_tau']}_{seed}")
    vec_env.close()
    del model

def train_asap_lips_lam_test(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable,
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
    model = ASAPLIPSSAC(ASAPLIPSPolicy_soft, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    **sac_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"ASAP_LIPS_SAC_lamS{alg_args['lam_s']}_lamT{alg_args['lam_t']}_lips{alg_args['lips_lam']}_{seed}")
    model.save(f"{save_dir}asap_lips_sac_lamS{alg_args['lam_s']}_lamT{alg_args['lam_t']}_lips{alg_args['lips_lam']}_{seed}")
    vec_env.close()
    del model

def train_lips_lam_test(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1  # 원하는 env 개수
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    sac_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    sac_args.update(alg_args)
    model = LipsSAC(LipsSACPolicy, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    **sac_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"Lips_SAC_lips{alg_args['lips_lam']}_{seed}")
    model.save(f"{save_dir}lips_sac_lips{alg_args['lips_lam']}_{seed}")
    vec_env.close()
    del model

def train_caps_lips_lam_test(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable,
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
    model = CAPSLipsSAC(LipsSACPolicy, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    **sac_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"CAPS_LIPS_SAC_lamS{alg_args['caps_lamS']}_lamT{alg_args['caps_lamT']}_lips{alg_args['lips_lam']}_{seed}")
    model.save(f"{save_dir}caps_lips_sac_lamS{alg_args['caps_lamS']}_lamT{alg_args['caps_lamT']}_lips{alg_args['lips_lam']}_{seed}")
    vec_env.close()
    del model

def train_pave_lips(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, env_args:dict, alg_args:dict):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    n_envs = 1
    if 'n_envs' in env_args:
        n_envs = env_args['n_envs']
    vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    sac_args = {k: v for k, v in env_args.items() if k != "n_envs"}
    sac_args.update(alg_args)
    model = PAVE_LIPS_SAC(LipsSACPolicy, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    **sac_args)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"PAVE_LIPS_SAC_{seed}")
    model.save(f"{save_dir}pave_lips_sac_{seed}")
    vec_env.close()
    del model

