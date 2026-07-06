import gymnasium as gym
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def make_ant_env(render_mode = "rgb_array"):
    def _init():
        env = gym.make("Ant-v5", render_mode=render_mode)
        return env
    return _init

def make_lunar_env(render_mode = "rgb_array"):
    def _init():
        env = gym.make(
            "LunarLander-v3",
            continuous=True,
            render_mode=render_mode
        )
        return env
    return _init

def make_hopper_env(render_mode = "rgb_array"):
    def _init():
        env = gym.make('Hopper-v5', render_mode=render_mode)
        return env
    return _init

def make_humanoid_env(render_mode = "rgb_array"):
    def _init():
        env = gym.make("Humanoid-v5", render_mode=render_mode)
        return env
    return _init

def make_pendulum_env(render_mode = "rgb_array"):
    def _init():
        env = gym.make("Pendulum-v1", render_mode=render_mode)
        return env
    return _init

def make_reacher_env(render_mode = "rgb_array"):
    def _init():
        env = gym.make("Reacher-v5", render_mode=render_mode)
        return env
    return _init

def make_walker_env(render_mode = "rgb_array"):
    def _init():
        env = gym.make('Walker2d-v5', render_mode=render_mode)
        return env
    return _init

def make_cheetah_env(render_mode = "rgb_array"):
    def _init():
        env =gym.make('HalfCheetah-v5', render_mode=render_mode)
        return env
    return _init