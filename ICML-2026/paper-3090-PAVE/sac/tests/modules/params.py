from torch import nn
from stable_baselines3.common.utils import get_schedule_fn

env_timestep = dict({
    "ant" : 1000000,
    "hopper" : 1000000,
    "humanoid" : 1000000,
    "lunar" : 500000,
    "pendulum" : 100000,
    "reacher" : 500000,
    "walker" : 1000000
})

env_args = dict({
    "ant" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=10000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=50,
        gradient_steps=50,
        ent_coef=0.2,
    ),
    "hopper" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=10000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=50,
        gradient_steps=50,
        ent_coef=0.2,
    ),
    "humanoid" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=10000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=50,
        gradient_steps=50,
        ent_coef=0.2,
    ),
    "lunar" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=10000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=50,
        gradient_steps=50,
        ent_coef=0.2,
    ),
    "pendulum" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=10000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=50,
        gradient_steps=50,
        ent_coef=0.2,
    ),
    "reacher" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=10000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=50,
        gradient_steps=50,
        ent_coef=0.2,
    ),
    "walker" : dict(
        n_envs= 1,
        learning_rate=1e-3,
        buffer_size = 1_000_000,
        learning_starts=10000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=50,
        gradient_steps=50,
        ent_coef=0.2,
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
    "l2c2" :dict(
        ant = dict(
            l2c2_sigma = 1.0,
            l2c2_lamD = 0.1,
            l2c2_lamU = 10.0,
            l2c2_beta = 0.1),
        hopper = dict(
            l2c2_sigma = 1.0,
            l2c2_lamD = 0.1,
            l2c2_lamU = 10.0,
            l2c2_beta = 0.1),
        humanoid = dict(
            l2c2_sigma = 1.0,
            l2c2_lamD = 0.1,
            l2c2_lamU = 10.0,
            l2c2_beta = 0.1),
        lunar = dict(
            l2c2_sigma = 1.0,
            l2c2_lamD = 0.1,
            l2c2_lamU = 10.0,
            l2c2_beta = 0.1),
        reacher = dict(
            l2c2_sigma = 1.0,
            l2c2_lamD = 0.1,
            l2c2_lamU = 10.0,
            l2c2_beta = 0.1),
        pendulum = dict(
            l2c2_sigma = 1.0,
            l2c2_lamD = 1.0,
            l2c2_lamU = 100.0,
            l2c2_beta = 0.1),
        walker = dict(
            l2c2_sigma = 1.0,
            l2c2_lamD = 0.1,
            l2c2_lamU = 10.0,
            l2c2_beta = 0.1),
    ),
    "asap" : dict(
        ant = dict(
            lam_p = 2.0,
            lam_s = 30.0,
            lam_t = 0.5,
            asap_tau = 0.01),
        hopper = dict(
            lam_p = 2.0,
            lam_s = 10.0,
            lam_t = 0.5,
            asap_tau = 0.01),
        humanoid = dict(
            lam_p = 2.0,
            lam_s = 5.0,
            lam_t = 0.5,
            asap_tau = 0.01),
        lunar = dict(
            lam_p = 2.0,
            lam_s = 10.0,
            lam_t = 0.5,
            asap_tau = 0.01),
        reacher = dict(
            lam_p = 2.0,
            lam_s = 5.0,
            lam_t = 0.5,
            asap_tau = 0.01),
        pendulum = dict(
            lam_p = 2.0,
            lam_s = 5.0,
            lam_t = 0.5,
            asap_tau = 0.01),
        walker = dict(
            lam_p = 2.0,
            lam_s = 5.0,
            lam_t = 0.5,
            asap_tau = 0.01),
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
            grad_lamT = 0.0005,
            grad_lamC = 1.0,
        ),
        hopper = dict(
            grad_lamS = 2.0,
            grad_lamT = 0.0005,
            grad_lamC = 3.0,
        ),
        humanoid = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.0005,
            grad_lamC = 1.0,
        ),
        lunar = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.5,
            grad_lamC = 0.05,
        ),
        reacher = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.0005,
            grad_lamC = 1.0,
        ),
        pendulum = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.005,
            grad_lamC = 0.5,
        ),
        walker = dict(
            grad_lamS = 2.0,
            grad_lamT = 0.005,
            grad_lamC = 2.0,
        ),
    ),
    "pave_lips" : dict(
        ant = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.0005,
            grad_lamC = 1.0,
        ),
        hopper = dict(
            grad_lamS = 2.0,
            grad_lamT = 0.0005,
            grad_lamC = 3.0,
        ),
        humanoid = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.0005,
            grad_lamC = 1.0,
        ),
        lunar = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.5,
            grad_lamC = 0.05,
        ),
        reacher = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.0005,
            grad_lamC = 1.0,
        ),
        pendulum = dict(
            grad_lamS = 0.1,
            grad_lamT = 0.005,
            grad_lamC = 0.5,
        ),
        walker = dict(
            grad_lamS = 2.0,
            grad_lamT = 0.005,
            grad_lamC = 2.0,
        ),
    ),
})
