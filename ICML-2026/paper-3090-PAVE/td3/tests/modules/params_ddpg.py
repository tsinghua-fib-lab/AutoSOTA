from torch import nn
from stable_baselines3.common.utils import get_schedule_fn
from typing import Any, ClassVar, Optional, TypeVar, Union
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

# env_timestep = dict({
#     "ant" : 1000000,
#     "hopper" : 1000000,
#     "humanoid" : 1000000,
#     "lunar" : 500000,
#     "pendulum" : 100000,
#     "reacher" : 100000,
#     "walker" : 1000000
# })

# env_timestep = dict({
#     "ant" : 2000,
#     "hopper" : 2000,
#     "humanoid" : 2000,
#     "lunar" : 2000,
#     "pendulum" : 2000,
#     "reacher" : 2000,
#     "walker" : 2000
# })

env_timestep = dict({
    "ant" : 500000,
    "hopper" : 500000,
    "humanoid" : 500000,
    "lunar" : 500000,
    "pendulum" : 100000,
    "reacher" : 100000,
    "walker" : 500000,
    "cheetah" : 500000
})


env_args = dict({
    "ant" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=1000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=(50, "step"),
        gradient_steps=50,
        policy_delay=1,
        target_noise_clip=0.0,
        target_policy_noise=0.1,
        policy_kwargs={"n_critics": 1},
    ),
    "hopper" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=1000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=(50, "step"),
        gradient_steps=50,
        policy_delay=1,
        target_noise_clip=0.0,
        target_policy_noise=0.1,
        policy_kwargs={"n_critics": 1},
    ),
    "humanoid" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=1000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=(50, "step"),
        gradient_steps=50,
        policy_delay=1,
        target_noise_clip=0.0,
        target_policy_noise=0.1,
        policy_kwargs={"n_critics": 1},
    ),
    "lunar" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=1000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=(50, "step"),
        gradient_steps=50,
        policy_delay=1,
        target_noise_clip=0.0,
        target_policy_noise=0.1,
        policy_kwargs={"n_critics": 1},
    ),
    "pendulum" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=1000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=(50, "step"),
        gradient_steps=50,
        policy_delay=1,
        target_noise_clip=0.0,
        target_policy_noise=0.1,
        policy_kwargs={"n_critics": 1},
    ),
    "reacher" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=1000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=(50, "step"),
        gradient_steps=50,
        policy_delay=1,
        target_noise_clip=0.0,
        target_policy_noise=0.1,
        policy_kwargs={"n_critics": 1},
    ),
    "walker" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=1000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=(50, "step"),
        gradient_steps=50,
        policy_delay=1,
        target_noise_clip=0.0,
        target_policy_noise=0.1,
        policy_kwargs={"n_critics": 1},
    ),
    "cheetah" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=1000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=(50, "step"),
        gradient_steps=50,
        policy_delay=1,
        target_noise_clip=0.0,
        target_policy_noise=0.1,
        policy_kwargs={"n_critics": 1},
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
        cheetah = dict(),
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
        cheetah = dict(
            caps_sigma = 0.2,
            caps_lamT = 0.1,
            caps_lamS = 0.5,),
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
        cheetah = dict(
            grad_lamT = 1.0
        ),
    ),
})

