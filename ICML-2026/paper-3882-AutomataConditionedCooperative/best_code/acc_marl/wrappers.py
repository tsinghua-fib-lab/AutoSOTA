import jax
import chex
import jax.numpy as jnp
from flax import struct
from functools import partial
from typing import Tuple, Union
from dfa_gym.env import MultiAgentEnv, State


@struct.dataclass
class LogEnvState:
    env_state: State
    episode_returns: float
    episode_disc_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_last_returns: float
    returned_episode_disc_returns: float
    returned_episode_lengths: int
    timestep: int

class LogWrapper(object):

    def __init__(self, env: MultiAgentEnv, config: dict):
        self._env = env
        self.config = config

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros((self._env.num_agents,)),
            episode_disc_returns=jnp.zeros((self._env.num_agents,)),
            episode_lengths=jnp.zeros((self._env.num_agents,)),
            returned_episode_returns=jnp.zeros((self._env.num_agents,)),
            returned_episode_last_returns=jnp.zeros((self._env.num_agents,)),
            returned_episode_disc_returns=jnp.zeros((self._env.num_agents,)),
            returned_episode_lengths=jnp.zeros((self._env.num_agents,)),
            timestep=0
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        _rew = self._batchify_floats(reward)
        new_episode_return = state.episode_returns + _rew
        new_episode_disc_return = state.episode_disc_returns + _rew * self.config["GAMMA"]**state.episode_lengths
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_disc_returns=new_episode_disc_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done) + new_episode_return * ep_done,
            returned_episode_last_returns=state.returned_episode_last_returns * (1 - ep_done) + _rew * ep_done,
            returned_episode_disc_returns=state.returned_episode_disc_returns * (1 - ep_done) + new_episode_disc_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done) + new_episode_length * ep_done,
            timestep=state.timestep + 1
        )

        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_last_returns"] = state.returned_episode_last_returns
        info["returned_episode_disc_returns"] = state.returned_episode_disc_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        info["timestep"] = jnp.full((self._env.num_agents,), state.timestep)
        return obs, state, reward, done, info

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])

