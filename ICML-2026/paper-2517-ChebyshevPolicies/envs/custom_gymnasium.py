"""custom_gymnasium.py: Modified versions of existing gymnasium environments.
"""

import numpy as np
import gymnasium as gym
from gymnasium.envs.classic_control import PendulumEnv
from gymnasium.envs.classic_control import utils
from stable_baselines3.common.vec_env import VecEnvWrapper
from typing import Optional


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class DeterministicPendulumEnv(PendulumEnv):
    '''
    Wrapper enabling deterministic initialization of Pendulum-v1 Gymnasium env.
    '''
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is None:
            self.state = np.array([DEFAULT_X, DEFAULT_Y])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            self.state = np.array([x, y])
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}