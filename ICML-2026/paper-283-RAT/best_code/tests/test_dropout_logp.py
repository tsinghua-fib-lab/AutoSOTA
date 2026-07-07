from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union

import gym
import torch
import numpy as np
import pytest
from gymnasium import spaces

from vec_env import DummyVecEnv, VecNormalize
from utils.utils import build_vit
from utils.utils import CategoricalActorCritic

class FakeImageEnv(gym.Env):
    def __init__(
        self,
        action_dim: int = 6,
        screen_height: int = 84,
        screen_width: int = 84,
        n_channels: int = 1,
        discrete: bool = True,
        channel_first: bool = False,
    ) -> None:
        self.observation_shape = (screen_height, screen_width, n_channels)
        if channel_first:
            self.observation_shape = (n_channels, screen_height, screen_width)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8)
        if discrete:
            self.action_space = spaces.Discrete(action_dim)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.ep_length = 10
        self.current_step = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            super().reset(seed=seed)
        self.current_step = 0
        return self.observation_space.sample()

    def step(self, action: Union[np.ndarray, int]):
        reward = 0.0
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.ep_length
        return self.observation_space.sample(), reward, terminated, truncated, {}

    def render(self, mode: str = "human") -> None:
        pass

def test_dropout_logp():
    # Fake grayscale with frameskip
    # Atari after preprocessing: 84x84x1, here we are using lower resolution
    # to check that the network handle it automatically
    def make_fn(h, w, c):
        return lambda: FakeImageEnv(screen_height=h, screen_width=w, n_channels=c)
    fns = [make_fn(64, 64, 3) for _ in range(4)]
    venv= DummyVecEnv(fns)
    venv = VecNormalize(venv) # img transform and reward normalization

    obs = venv.reset()
    obs_space = venv.observation_space

    embed_dim=256
    kwargs = {'patch_size': 16, 'dim': 256, 'depth': 3, 'heads': 3, 'dropout': 0.2}
    neural_nets = build_vit(obs_space, embed_dim, **kwargs)

    act_num = venv.action_space.n
    actor_critic = CategoricalActorCritic(neural_nets, embed_dim, act_num)

    actor_critic.eval()
    obs_tensor = torch.from_numpy(obs)
    with torch.no_grad():
        _, act, _, logp_old, _ = actor_critic.step(obs_tensor)
    act_tensor = torch.from_numpy(act)

    all_preds = []
    actor_critic.train()
    for _ in range(100):
        pi = actor_critic(obs_tensor)[3]
        logp = pi.log_prob(act_tensor)
        all_preds.append(logp.detach().numpy() - logp_old)

    assert np.allclose(0.0, np.mean(all_preds))