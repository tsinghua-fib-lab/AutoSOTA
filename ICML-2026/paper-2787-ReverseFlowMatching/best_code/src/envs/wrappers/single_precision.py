# Code adapted from https://github.com/ikostrikov/jaxrl

import copy

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict


class SinglePrecision(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        if isinstance(self.observation_space, Box):
            obs_space = self.observation_space
            # Cast bounds to float32 and preserve dtype
            low = obs_space.low.astype(np.float32)
            high = obs_space.high.astype(np.float32)
            self.observation_space = Box(low, high, obs_space.shape, dtype=np.float32)
        elif isinstance(self.observation_space, Dict):
            obs_spaces = copy.copy(self.observation_space.spaces)
            for k, v in obs_spaces.items():
                # Cast bounds to float32 and preserve dtype
                low = v.low.astype(np.float32)
                high = v.high.astype(np.float32)
                obs_spaces[k] = Box(low, high, v.shape, dtype=np.float32)
            self.observation_space = Dict(obs_spaces)
        else:
            raise NotImplementedError

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            return observation.astype(np.float32)
        elif isinstance(observation, dict):
            observation = copy.copy(observation)
            for k, v in observation.items():
                observation[k] = v.astype(np.float32)
            return observation
