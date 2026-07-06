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
from models.ent_sac import EntSAC, EntPolicy
from models.caps_sac import CAPSSAC
from models.lips_sac import LipsSAC, LipsSACPolicy
from models.da_sac import DASAC, DAPolicy, DASAC_nonshare, DAPolicy_nonshare, DASAC_buffer, DASAC_buffer_async, DASAC_scale, DASAC_soft, DAPolicy_soft, DASAC_norm

def train_vanilla(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # n_envs = 4  # 원하는 env 개수
    # vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    # vec_env = VecMonitor(vec_env)
    vec_env = mkenv_func()
    model = CustomSAC("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"Vanilla_{seed}")
    model.save(f"{save_dir}vanilla_{seed}")
    vec_env.close()
    del model

def train_ent(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # n_envs = 4  # 원하는 env 개수
    # vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    # vec_env = VecMonitor(vec_env)
    vec_env = mkenv_func()
    model = EntSAC(EntPolicy, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"Ent_{seed}")
    model.save(f"{save_dir}ent_sac_{seed}")
    vec_env.close()
    del model

def train_alpha(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable, alpha : float):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # n_envs = 4  # 원하는 env 개수
    # vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    # vec_env = VecMonitor(vec_env)
    vec_env = mkenv_func()
    model = CustomSAC("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                      ent_coef=alpha)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"Alpha_{alpha}_{seed}")
    model.save(f"{save_dir}alpha_{alpha}_{seed}")
    vec_env.close()
    del model

def train_caps(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # n_envs = 4  # 원하는 env 개수
    # vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    # vec_env = VecMonitor(vec_env)
    vec_env = mkenv_func()
    model = CAPSSAC("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir, seed=seed)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"CAPS_SAC_{seed}")
    model.save(f"{save_dir}caps_sac_{seed}")
    vec_env.close()
    del model

def train_lips(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # n_envs = 4  # 원하는 env 개수
    # vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    # vec_env = VecMonitor(vec_env)
    vec_env = mkenv_func()
    model = LipsSAC(LipsSACPolicy, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    lips_lam=1e-5, lips_eps=1e-4)
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"Lips_SAC_{seed}")
    model.save(f"{save_dir}lips_sac_{seed}")
    vec_env.close()
    del model

def train_da(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable,
             pargs: Optional[dict] = None):
    if pargs is None:
        pargs = {
            "da_lamP": 1,
            "da_lam_L": 1,
        }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # n_envs = 4  # 원하는 env 개수
    # vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    # vec_env = VecMonitor(vec_env)
    vec_env = mkenv_func()
    model = DASAC(DAPolicy, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    da_lamP=pargs["da_lamP"], da_lamL=pargs["da_lam_L"])
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"DA_SAC_{seed}")
    model.save(f"{save_dir}da_sac_{seed}")
    vec_env.close()
    del model

def train_da_nonshare(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable,
                pargs: Optional[dict] = None):
    if pargs is None:
        pargs = {
            "da_lamP": 1,
            "da_lam_L": 1,
        }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # n_envs = 4  # 원하는 env 개수
    # vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    # vec_env = VecMonitor(vec_env)
    vec_env = mkenv_func()
    model = DASAC_nonshare(DAPolicy_nonshare, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    da_lamP=pargs["da_lamP"], da_lamL=pargs["da_lam_L"])
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"DANS_SAC_{seed}")
    model.save(f"{save_dir}dans_sac_{seed}")
    vec_env.close()
    del model

def train_da_buffer(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable,
                      pargs: Optional[dict] = None):
    if pargs is None:
        pargs = {
            "da_lamP": 1,
            "da_lam_L": 1,
            "da_buffer_size": 10000,
            "da_batch_size": 256,
            "da_n_epoch": 1,
        }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # n_envs = 4  # 원하는 env 개수
    # vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    # vec_env = VecMonitor(vec_env)
    vec_env = mkenv_func()
    model = DASAC_buffer(DAPolicy_nonshare, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    da_lamP=pargs["da_lamP"], da_lamL=pargs["da_lam_L"],
                    da_batch_size=pargs["da_batch_size"], da_buffer_size=pargs["da_buffer_size"],
                    da_n_epoch=pargs["da_n_epoch"])
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"DABF_SAC_{seed}")
    model.save(f"{save_dir}dabf_sac_{seed}")
    vec_env.close()
    del model


def train_da_buffer_async(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable,
                      pargs: Optional[dict] = None):
    if pargs is None:
        pargs = {
            "da_lamP": 1,
            "da_lam_L": 1,
            "da_buffer_size": 10000,
            "da_batch_size": 256,
            "da_n_epoch": 1,
        }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # n_envs = 4  # 원하는 env 개수
    # vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    # vec_env = VecMonitor(vec_env)
    vec_env = mkenv_func()
    model = DASAC_buffer_async(DAPolicy_nonshare, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    da_lamP=pargs["da_lamP"], da_lamL=pargs["da_lam_L"],
                    da_batch_size=pargs["da_batch_size"], da_buffer_size=pargs["da_buffer_size"],
                    da_n_epoch=pargs["da_n_epoch"])
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"DABFAC_SAC_{seed}")
    model.save(f"{save_dir}dabfac_sac_{seed}")
    vec_env.close()
    del model

def train_da_scale(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable,
             pargs: Optional[dict] = None):
    if pargs is None:
        pargs = {
            "da_lamLscale": 0.05,
            "da_lamP": 1,
            "da_tau": 0.01,
            "da_eps": 0.01,
        }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # n_envs = 4  # 원하는 env 개수
    # vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    # vec_env = VecMonitor(vec_env)
    vec_env = mkenv_func()
    model = DASAC_scale(DAPolicy, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    da_lamP=pargs["da_lamP"], da_lamLscale=pargs["da_lamLscale"],
                    da_eps=pargs["da_eps"], da_tau=pargs["da_tau"])
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"DA_SAC_{seed}")
    model.save(f"{save_dir}da_sac_{seed}")
    vec_env.close()
    del model

def train_da_soft(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable,
             pargs: Optional[dict] = None):
    if pargs is None:
        pargs = {
            "da_lamL": 1,
            "da_lamP": 1,
            "da_tau": 0.005,
            "da_ui": 1,
        }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # n_envs = 4  # 원하는 env 개수
    # vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    # vec_env = VecMonitor(vec_env)
    vec_env = mkenv_func()
    model = DASAC_soft(DAPolicy_soft, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    da_lamP=pargs["da_lamP"], da_lamL=pargs["da_lamL"],
                    da_tau=pargs["da_tau"], da_target_update_interval=pargs["da_ui"])
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"DA_SOFT_SAC_{seed}")
    model.save(f"{save_dir}da_soft_sac_{seed}")
    vec_env.close()
    del model


def train_da_norm(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable,
             pargs: Optional[dict] = None):
    if pargs is None:
        pargs = {
            "da_lamL": 1,
            "da_lamP": 1,
            "da_lamS" : 0.05,
            "da_tau": 1,
            "da_ui": 1,
        }
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # n_envs = 4  # 원하는 env 개수
    # vec_env = DummyVecEnv([mkenv_func() for _ in range(n_envs)])
    # vec_env = VecMonitor(vec_env)
    vec_env = mkenv_func()
    model = DASAC_norm(DAPolicy_soft, vec_env, verbose=0, tensorboard_log=log_dir, seed=seed,
                    da_lamP=pargs["da_lamP"], da_lamL=pargs["da_lamL"], da_lamS=pargs["da_lamS"],
                    da_tau=pargs["da_tau"], da_target_update_interval=pargs["da_ui"])
    model.learn(total_timesteps=total_time_steps, tb_log_name=f"DA_NORM_SAC_{seed}")
    model.save(f"{save_dir}da_norm_sac_{seed}")
    vec_env.close()
    del model

# def train_caps(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     vec_env = mkenv_func()
#     model = CAPSPPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir,
#                     caps_sigma=0.2, caps_lamT=0.01, caps_lamS=0.05, seed=seed)
#     model.learn(total_timesteps=total_time_steps, tb_log_name="CAPS_PPO")
#     model.save(f"{save_dir}caps_ppo_{seed}")
#     vec_env.close()
#     del model

# def train_l2c2(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     vec_env = mkenv_func()
#     model = L2C2PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir,
#                     l2c2_lamD=0.01, l2c2_lamU=1.0, seed=seed)
#     model.learn(total_timesteps=total_time_steps, tb_log_name="L2C2PPO")
#     model.save(f"{save_dir}l2c2_ppo_{seed}")
#     vec_env.close()
#     del model

# def train_lips(seed:int, total_time_steps:int, save_dir:str, log_dir:str, mkenv_func : Callable):
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     vec_env = mkenv_func()
#     model = LipsPPO(LipsPolicy, vec_env, verbose=0, tensorboard_log=log_dir,
#                     lips_lam=1e-1, lips_eps=1e-4, lips_k_init=1, seed=seed)
#     model.learn(total_timesteps=total_time_steps, tb_log_name="Lips_PPO")
#     model.save(f"{save_dir}lips_ppo_{seed}")
#     vec_env.close()
#     del model

# def train_pid(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable,
#               pargs: Optional[dict] = None):
#     if pargs is None:
#         pargs = {
#             "pid_kp": 0.2,
#             "pid_ki": 1,
#             "pid_kd": 1,
#             "pid_lam_s": 0.01,
#             "pid_lam_d": 0.05
#         }

#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     vec_env = mkenv_func()
#     model = PidPPO(PidPolicy, vec_env, verbose=0, tensorboard_log=log_dir, 
#                     pid_kd=pargs["pid_kd"], pid_ki=pargs["pid_ki"], pid_kp=pargs["pid_kp"],
#                       pid_lam_s=pargs["pid_lam_s"], pid_lam_d=pargs["pid_lam_d"], seed=seed)
#     model.learn(total_timesteps=total_time_steps, tb_log_name=f"Pid_PPO_{seed}")
#     model.save(f"{save_dir}pid_ppo_{seed}")
#     vec_env.close()
#     del model

# def train_pid_kp(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable,
#               pargs: Optional[dict] = None):
#     if pargs is None:
#         pargs = {
#             "pid_kp": 0.2,
#             "pid_ki": 1,
#             "pid_kd": 1,
#             "pid_lam_s": 0.01,
#             "pid_lam_d": 0.05
#         }

#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     vec_env = mkenv_func()
#     model = PidPPO(PidPolicy, vec_env, verbose=0, tensorboard_log=log_dir, 
#                     pid_kd=pargs["pid_kd"], pid_ki=pargs["pid_ki"], pid_kp=pargs["pid_kp"],
#                       pid_lam_s=pargs["pid_lam_s"], pid_lam_d=pargs["pid_lam_d"], seed=seed)
#     model.learn(total_timesteps=total_time_steps, tb_log_name=f"Pid_PPO_kp{pargs['pid_kp']}_{seed}")
#     model.save(f"{save_dir}pid_ppo_kp{pargs['pid_kp']}_{seed}")
#     vec_env.close()
#     del model

# def train_spd(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable,
#               pargs: Optional[dict] = None):
#     if pargs is None:
#         pargs = {
#             "pid_kp": 0.2,
#             "pid_ki": 1,
#             "pid_kd": 1,
#             "pid_lam_s": 0.01,
#             "pid_lam_d": 0.05
#         }

#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     vec_env = mkenv_func()
#     model = SpdPPO(SpdPolicy, vec_env, verbose=0, tensorboard_log=log_dir, 
#                     pid_kd=pargs["pid_kd"], pid_ki=pargs["pid_ki"], pid_kp=pargs["pid_kp"],
#                       pid_lam_s=pargs["pid_lam_s"], pid_lam_d=pargs["pid_lam_d"], seed=seed)
#     model.learn(total_timesteps=total_time_steps, tb_log_name=f"Spd_PPO_lam{pargs['pid_lam_d']}_{seed}")
#     model.save(f"{save_dir}spd_ppo_lam{pargs['pid_lam_d']}_{seed}")
#     vec_env.close()
#     del model

# def train_time(seed: int, total_time_steps: int, save_dir: str, log_dir: str, mkenv_func: Callable,
#               pargs: Optional[dict] = None):
#     if pargs is None:
#         pargs = {
#             "pid_kp": 0.2,
#             "pid_ki": 1,
#             "pid_kd": 1,
#             "pid_lam_s": 0.01,
#             "pid_lam_d": 0.05
#         }

#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     vec_env = mkenv_func()
#     model = TimePPO(TimePolicy, vec_env, verbose=0, tensorboard_log=log_dir, 
#                     pid_kd=pargs["pid_kd"], pid_ki=pargs["pid_ki"], pid_kp=pargs["pid_kp"],
#                       pid_lam_s=pargs["pid_lam_s"], pid_lam_d=pargs["pid_lam_d"], seed=seed)
#     model.learn(total_timesteps=total_time_steps, tb_log_name=f"Time_PPO_lam{pargs['pid_lam_d']}_{seed}")
#     model.save(f"{save_dir}Time_ppo_lam{pargs['pid_lam_d']}_{seed}")
#     vec_env.close()
#     del model