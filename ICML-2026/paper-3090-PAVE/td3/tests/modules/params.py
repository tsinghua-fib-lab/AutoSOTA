from torch import nn
from stable_baselines3.common.utils import get_schedule_fn
from typing import Any, ClassVar, Optional, TypeVar, Union
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

# same timesteps with ASAP's SAC implementation
env_timestep = dict({
    "ant" : 1000000,  # 1e6
    "hopper" : 1000000,  # 1e6
    "humanoid" : 1000000,  # 1e6
    "lunar" : 500000,  # 5e5
    "pendulum" : 100000,  # 1e5
    "reacher" : 500000,  # 5e5
    "walker" : 1000000,  # 1e6
})

# default hyperparameters from stable_baselines3/td3/td3.py
# https://github.com/DLR-RM/stable-baselines3/blob/8fccf7f1c421deff6b54bd595c430604b24724b0/stable_baselines3/td3/td3.py

'''
    def __init__(
        self,
        policy: str | type[TD3Policy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
'''

# hyperparameters based on rl-baselines3-zoo
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/d29756c456caadbbebc15c35893674abb2453e0d/hyperparams/td3.yml 

'''
# === Mujoco Envs ===
HalfCheetah-v4: &mujoco-defaults
  n_timesteps: !!float 1e6
  policy: 'MlpPolicy'
  learning_starts: 10000
  noise_type: 'normal'
  noise_std: 0.1
  train_freq: 1
  gradient_steps: 1
  learning_rate: !!float 1e-3
  batch_size: 256
  policy_kwargs: "dict(net_arch=[400, 300])"
  '''

env_args = dict({
    "ant" : dict(    # mujoco-defaults
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
    ),
    "hopper" : dict(    # mujoco-defaults
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
    ),
    "humanoid" : dict(  # mujoco-defaults & n_timesteps : 2e6, but we use 1e6
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
    ),
    "lunar" : dict( # n_timesteps : 3e5, but we use 5e5
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 200000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=(1, "step"),
        gradient_steps=1,
    ),
    "pendulum" : dict(  # n_timesteps : 20k, but we use 1e6
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 200000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=(1, "step"),
        gradient_steps=1,
    ),
    "reacher" : dict(   # there is no tuned hyperparameters, we use mujoco-defaults
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
    ),
    "walker" : dict(    # mujoco-defaults
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
    )
})

alg_args = dict({
    "vanilla" : dict(
        ant = dict(),
        hopper = dict(),
        humanoid = dict(),
        lunar = dict(),
        reacher = dict(),
        pendulum = dict(),
        walker = dict(),
    ),
    "caps" : dict(
        ant = dict(
            caps_sigma = 0.2,
            caps_lamT = 0.1,
            caps_lamS = 0.5,),
        hopper = dict(
            caps_sigma = 0.2,
            caps_lamT = 0.1,
            caps_lamS = 0.5,),
        humanoid = dict(
            caps_sigma = 0.2,
            caps_lamT = 0.1,
            caps_lamS = 0.5,),
        lunar = dict(
            caps_sigma = 0.2,
            caps_lamT = 0.1,
            caps_lamS = 0.5,),
        reacher = dict(
            caps_sigma = 0.2,
            caps_lamT = 0.1,
            caps_lamS = 0.5,),
        pendulum = dict(
            caps_sigma = 0.2,
            caps_lamT = 1.0,
            caps_lamS = 5.0,),
        walker = dict(
            caps_sigma = 0.2,
            caps_lamT = 0.1,
            caps_lamS = 0.5,),
    ),
    "asap" : dict(
        ant = dict(
            asap_lamP = 2.0,
            asap_lamS = 0.3,
            asap_lamT = 0.05),
        hopper = dict(
            asap_lamP = 2.0,
            asap_lamS = 0.3,
            asap_lamT = 0.07),
        humanoid = dict(
            asap_lamP = 2.0,
            asap_lamS = 0.3,
            asap_lamT = 0.05),
        lunar = dict(
            asap_lamP = 2.0,
            asap_lamS = 0.03,
            asap_lamT = 0.005),
        reacher = dict(
            asap_lamP = 2.0,
            asap_lamS = 0.1,
            asap_lamT = 0.1),
        pendulum = dict(
            asap_lamP = 2.0,
            asap_lamS = 0.03,
            asap_lamT = 0.005),
        walker = dict(
            asap_lamP = 2.0,
            asap_lamS = 0.3,
            asap_lamT = 0.05),
    ),
    "grad" : dict(
        ant = dict(
            grad_lamT = 1.0
        ),
        hopper = dict(
            grad_lamT = 1.0
        ),
        humanoid = dict(
            grad_lamT = 1.0
        ),
        lunar = dict(
            grad_lamT = 1.0
        ),
        reacher = dict(
            grad_lamT = 1.0
        ),
        pendulum = dict(
            grad_lamT = 1.0
        ),
        walker = dict(
            grad_lamT = 1.0
        ),
    ),
    "pave" : dict(
        ant = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.005,
            grad_lamC = 0.5,
        ),
        hopper = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.005,
            grad_lamC = 0.5,
        ),
        humanoid = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.005,
            grad_lamC = 0.5,
        ),
        lunar = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.1,
            grad_lamC = 0.01,
        ),
        reacher = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.1,
            grad_lamC = 0.01,
        ),
        pendulum = dict(
            grad_lamS = 2.0,
            grad_lamT = 0.005,
            grad_lamC = 2.0,
        ),
        walker = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.1,
            grad_lamC = 0.01,
        ),
    ),
})


# alg_args = dict({
#     "vanilla" : dict(
#         ant = dict(),
#         hopper = dict(),
#         humanoid = dict(),
#         lunar = dict(),
#         reacher = dict(),
#         pendulum = dict(),
#         walker = dict(),
#     ),
#     "caps" : dict(
#         ant = dict(
#             caps_sigma = 0.2,
#             caps_lamT = 0.1,
#             caps_lamS = 0.5,),
#         hopper = dict(
#             caps_sigma = 0.2,
#             caps_lamT = 0.1,
#             caps_lamS = 0.5,),
#         humanoid = dict(
#             caps_sigma = 0.2,
#             caps_lamT = 0.1,
#             caps_lamS = 0.5,),
#         lunar = dict(
#             caps_sigma = 0.2,
#             caps_lamT = 0.1,
#             caps_lamS = 0.5,),
#         reacher = dict(
#             caps_sigma = 0.2,
#             caps_lamT = 0.1,
#             caps_lamS = 0.5,),
#         pendulum = dict(
#             caps_sigma = 0.2,
#             caps_lamT = 1.0,
#             caps_lamS = 5.0,),
#         walker = dict(
#             caps_sigma = 0.2,
#             caps_lamT = 0.1,
#             caps_lamS = 0.5,),
#     ),
#     "l2c2" :dict(
#         ant = dict(
#             l2c2_sigma = 1.0,
#             l2c2_lamD = 0.1,
#             l2c2_lamU = 10.0,
#             l2c2_beta = 0.1),
#         hopper = dict(
#             l2c2_sigma = 1.0,
#             l2c2_lamD = 0.1,
#             l2c2_lamU = 10.0,
#             l2c2_beta = 0.1),
#         humanoid = dict(
#             l2c2_sigma = 1.0,
#             l2c2_lamD = 0.1,
#             l2c2_lamU = 10.0,
#             l2c2_beta = 0.1),
#         lunar = dict(
#             l2c2_sigma = 1.0,
#             l2c2_lamD = 0.1,
#             l2c2_lamU = 10.0,
#             l2c2_beta = 0.1),
#         reacher = dict(
#             l2c2_sigma = 1.0,
#             l2c2_lamD = 0.1,
#             l2c2_lamU = 10.0,
#             l2c2_beta = 0.1),
#         pendulum = dict(
#             l2c2_sigma = 1.0,
#             l2c2_lamD = 1.0,
#             l2c2_lamU = 100.0,
#             l2c2_beta = 0.1),
#         walker = dict(
#             l2c2_sigma = 1.0,
#             l2c2_lamD = 0.1,
#             l2c2_lamU = 10.0,
#             l2c2_beta = 0.1),
#     ),
#     "nadp" : dict(
#         ant = dict(
#             lam_predict = 1.0,
#             lam_smooth = 0.5,
#             da_tau = 0.01),
#         hopper = dict(
#             lam_predict = 1.0,
#             lam_smooth = 0.5,
#             da_tau = 0.01),
#         humanoid = dict(
#             lam_predict = 1.0,
#             lam_smooth = 0.5,
#             da_tau = 0.01),
#         lunar = dict(
#             lam_predict = 1.0,
#             lam_smooth = 0.5,
#             da_tau = 0.01),
#         reacher = dict(
#             lam_predict = 1.0,
#             lam_smooth = 0.5,
#             da_tau = 0.01),
#         pendulum = dict(
#             lam_predict = 1.0,
#             lam_smooth = 5.0,
#             da_tau = 0.01),
#         walker = dict(
#             lam_predict = 1.0,
#             lam_smooth = 0.5,
#             da_tau = 0.01),
#     ),
# })


    # "lips_l" : dict(
    #     cartpole = dict(
    #         lips_lam= 1e-3,
    #         lips_eps = 1e-4,
    #         lips_k_init = 50.0,
    #         lips_f_size = [256, 256],
    #         lips_k_size = [32],
    #         lips_global = False,
    #     ),
    #     reacher = dict(
    #         lips_lam= 1e-5,
    #         lips_eps = 1e-4,
    #         lips_k_init = 50.0,
    #         lips_f_size = [64, 64],
    #         lips_k_size = [32],
    #         lips_global = False,
    #     ),
    #     cheetah = dict(
    #         lips_lam= 1e-7,
    #         lips_eps = 1e-4,
    #         lips_k_init = 50.0,
    #         lips_f_size = [64, 64],
    #         lips_k_size = [32],
    #         lips_global = False,
    #     ),
    #     walker = dict(
    #         lips_lam= 1e-5,
    #         lips_eps = 1e-4,
    #         lips_k_init = 50.0,
    #         lips_f_size = [64, 64],
    #         lips_k_size = [32],
    #         lips_global = False,
    #     ),
    # ),
    # "lips_g" : dict(
    #     cartpole = dict(
    #         lips_lam= 1e-3,
    #         lips_eps = 1e-4,
    #         lips_k_init = 50.0,
    #         lips_f_size = [64, 64],
    #         lips_k_size = [32],
    #         lips_global = True,
    #     ),
    #     reacher = dict(
    #         lips_lam= 1e-5,
    #         lips_eps = 1e-4,
    #         lips_k_init = 50.0,
    #         lips_f_size = [64, 64],
    #         lips_k_size = [32],
    #         lips_global = True,
    #     ),
    #     cheetah = dict(
    #         lips_lam= 1e-7,
    #         lips_eps = 1e-4,
    #         lips_k_init = 50.0,
    #         lips_f_size = [64, 64],
    #         lips_k_size = [32],
    #         lips_global = True,
    #     ),
    #     walker = dict(
    #         lips_lam= 1e-5,
    #         lips_eps = 1e-4,
    #         lips_k_init = 50.0,
    #         lips_f_size = [64, 64],
    #         lips_k_size = [32],
    #         lips_global = True,
    #     ),
    # ),