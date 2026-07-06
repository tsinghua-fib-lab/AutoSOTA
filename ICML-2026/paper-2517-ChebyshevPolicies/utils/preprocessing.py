"""observation_utils.py: Helper functions for preprocessing tasks.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TransformObservation, RescaleAction
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor


__author__ = "Hannes Unger"
__version__ = "1.0"
__email__ = "hannes.unger@fh-salzburg.ac.at"


def get_normalized_vec_env(env_name, normalize_actions=True, min_action=-1.0, max_action=1.0, observation_space_low=None, observation_space_high=None, new_min=-1.0, new_max=1.0, render=False):
    """
    Returns env and eval env wrapped in TransformObservation (min-max scaling to [-1, 1]) and RescaleAction (scaling to [min_action, max_action]) wrappers.
    If observation_space_low=None or observation_space_high=None, env.observation_space.low and env.observation_space.high is used for min-max scaling.  
    """
    if render:
        env1 = gym.make(env_name, render_mode='human')
        env2 = gym.make(env_name, render_mode='human')
    else:
        env1 = gym.make(env_name)
        env2 = gym.make(env_name)
    env = VecNormalize(DummyVecEnv([lambda: Monitor(get_normalized_env(env1, normalize_actions=normalize_actions, min_action=min_action, max_action=max_action, observation_space_low=observation_space_low, observation_space_high=observation_space_high, new_min=new_min, new_max=new_max))]), norm_obs=False, norm_reward=False) # min-max scaling for easier reconstruction in plots, etc.
    eval_env = VecNormalize(DummyVecEnv([lambda: Monitor(get_normalized_env(env2, normalize_actions=normalize_actions, min_action=min_action, max_action=max_action, observation_space_low=observation_space_low, observation_space_high=observation_space_high, new_min=new_min, new_max=new_max))]), norm_obs=False, norm_reward=False)
    return env, eval_env


def get_normalized_env(env, normalize_actions=True, min_action=-1.0, max_action=1.0, observation_space_low=None, observation_space_high=None, new_min=-1.0, new_max=1.0):
    """
    Wraps environment in TransformObservation (min-max scaling to [-1, 1]) and RescaleAction (scaling to [min_action, max_action]) wrappers.
    """
    all_finite = np.isfinite(env.observation_space.low).all() and np.isfinite(env.observation_space.high).all()
    if all_finite:
        # Min-max scaling for finite observation spaces using observation space bounds
        observation_space_low = env.observation_space.low
        observation_space_high = env.observation_space.high
    elif observation_space_low is None or observation_space_high is None:
        raise Exception('Min-max scaling based on observation space bounds only possible for finite observation spaces. Please provide custom bounds.')
    
    low_state = np.array(
        [new_min for e in env.observation_space.low], dtype=np.float32
    )
    high_state = np.array(
        [new_max for e in env.observation_space.high], dtype=np.float32
    )

    observation_space = spaces.Box(
        low=low_state, high=high_state, dtype=np.float32
    )

    e = TransformObservation(env, lambda obs: normalize(obs, max_value=observation_space_high, min_value=observation_space_low, new_max=new_max, new_min=new_min), observation_space)

    if normalize_actions:
        return get_normalized_action_space_env(e, min_action=min_action, max_action=max_action)
    else:
        return e


def get_normalized_action_space_env(env, min_action=-1.0, max_action=1.0):
    """
    Wraps environment in RescaleAction wrapper normalizing action space to [-1, 1].  
    This is recommended, see, e.g. https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html .
    """
    return RescaleAction(env, min_action=min_action, max_action=max_action)


def load_vecnormalize_from_disk(env_name, vec_path, vec_name, normalize_actions=True):
    if normalize_actions:
        env = DummyVecEnv([lambda: get_normalized_action_space_env(gym.make(env_name), min_action=-1.0, max_action=1.0)])
    else:
        env = DummyVecEnv([lambda: get_normalized_action_space_env(gym.make(env_name))])
    return VecNormalize.load(vec_path + vec_name + '_env', env)


def normalize(value, max_value, min_value, new_max=1.0, new_min=-1.0, flip=False):
    """
    Normalize a single value to a specified range [new_min, new_max].

    Args:
    - value: The value to be normalized.
    - min_value: The minimum value of the original range.
    - max_value: The maximum value of the original range.
    - new_min: The minimum value of the new range.
    - new_max: The maximum value of the new range.
    - flip: If True, the axis is flipped. This mainly is there for compatibility to experiments of our mountain car paper.

    Returns:
    - The normalized value.
    """
    if flip:
        return ((value - min_value) / (max_value - min_value)) * (new_min - new_max) + new_max  
    else:
        return ((value - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min 


def denormalize(value, max_value, min_value, new_max=1.0, new_min=-1.0, flip=False):
    """
    Denormalize a single value given a specified range [new_min, new_max].

    Args:
    - value: The value to be denormalized.
    - min_value: The minimum value of the original range.
    - max_value: The maximum value of the original range.
    - new_min: The minimum value of the new range, the value was normalized to.
    - new_max: The maximum value of the new range, the value was normalized to.
    - flip: If True, the axis is flipped back in case it was reverted during
    normalization. This mainly is there for compatibility to experiments
    of our mountain car paper.

    Returns:
    - The denormalized value.
    """
    if flip:
        return (value-new_max)/(new_min-new_max)*(max_value-min_value)+min_value
    else:
        return (value-new_min)/(new_max-new_min)*(max_value-min_value)+min_value