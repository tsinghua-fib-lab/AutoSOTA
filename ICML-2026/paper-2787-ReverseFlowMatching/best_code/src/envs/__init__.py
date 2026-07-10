# Code adapted from https://github.com/ikostrikov/jaxrl

from typing import Optional

import gymnasium as gym
from gymnasium.wrappers import RescaleAction

from src.envs import wrappers
from src.envs.dmc_env import DMCEnv


def make_dmc_env(
    env_name: str,
    save_folder: Optional[str] = None,
    add_episode_monitor: bool = True,
    action_repeat: int = 1,
    flatten: bool = True,
) -> gym.Env:

    assert action_repeat == 1, "We only support action_repeat=1."

    def thunk(seed):
        all_envs = gym.envs.registry.values()
        env_ids = [env_spec.id for env_spec in all_envs]

        if env_name in env_ids:
            env = gym.make(env_name, render_mode="rgb_array")
        else:
            domain_name, task_name = env_name.split("-")
            env = DMCEnv(
                domain=domain_name, task=task_name, task_kwargs={"random": seed}
            )
        if flatten and isinstance(env.observation_space, gym.spaces.Dict):
            env = gym.wrappers.FlattenObservation(env)

        if add_episode_monitor:
            env = wrappers.EpisodeMonitor(env)

        env = RescaleAction(env, -1.0, 1.0)

        if save_folder is not None:
            env = gym.wrappers.RecordVideo(env, save_folder)
        env = wrappers.SinglePrecision(env)

        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return env

    return thunk
