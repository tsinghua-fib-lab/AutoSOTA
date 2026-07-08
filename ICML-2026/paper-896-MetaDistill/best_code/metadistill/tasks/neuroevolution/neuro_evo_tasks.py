import torch
import numpy
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import Wrapper
from tasks.neuroevolution.neuroevo_net import EvoNet
from torch_basic_settings import *


TASKS = {}


class TaskEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.tot_reward = 0

    def reset(self, init_state=None):
        obs, info = self.env.reset()
        if init_state is not None:
            self.env.unwrapped.state = init_state
            obs = numpy.copy(self.env.unwrapped.state)

        self.tot_reward = 0

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.tot_reward += reward

        info["episode_total_reward"] = self.tot_reward

        return obs, reward, terminated, truncated, info


def cart_pole(x, net, state=None, steps=256, render=None):  # fitness in [-64, 0]
    _env = gym.make("CartPole-v1", render_mode=render)
    env = TaskEnv(_env)

    b, n, d = x.shape
    x = x.view(-1, d)
    pop_fitness = []
    fitness = 0
    with torch.no_grad():
        for indv in x:
            net.set_params(indv)
            obs, info = env.reset(init_state=state)
            for step in range(steps):
                logits = net(torch.tensor(obs, dtype=DTYPE, device=DEVICE))
                logits = torch.sigmoid(logits).cpu().numpy()
                action = 1 if logits[0] > 0.5 else 0

                obs, reward, terminated, truncated, info = env.step(action)
                fitness = -info["episode_total_reward"]
                if terminated or truncated or (step == steps - 1):
                    break

            pop_fitness.append(fitness)

    env.close()
    pop_fitness = torch.tensor(pop_fitness, dtype=DTYPE, device=DEVICE).view(b, n, 1)
    x = x.view(b, n, d)
    x = torch.cat((pop_fitness, x), dim=-1)

    return x, pop_fitness


TASKS[1] = {
    "fid": "CartPole",
    "fun": cart_pole,
    "init_state": None,
    "xlb": -5,
    "xub": 5,
}


def lunar_lander(x, net, state=None, steps=256, render=None):
    """
    obs:  8, continuous
    action: {0, 1, 2, 3}
    """
    _env = gym.make("LunarLander-v3", render_mode=render)
    env = TaskEnv(_env)

    b, n, d = x.shape
    x = x.view(-1, d)
    pop_fitness = []
    fitness = 0
    with torch.no_grad():
        for indv in x:
            net.set_params(indv)
            obs, info = env.reset(init_state=state)
            for step in range(steps):
                logits = net(torch.tensor(obs, dtype=DTYPE, device=DEVICE))
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                fitness = -info["episode_total_reward"]
                if terminated or truncated or (step == steps - 1):
                    break

            pop_fitness.append(fitness)

    env.close()
    pop_fitness = torch.tensor(pop_fitness, dtype=DTYPE, device=DEVICE).view(b, n, 1)
    x = x.view(b, n, d)
    x = torch.cat((pop_fitness, x), dim=-1)

    return x, pop_fitness


TASKS[2] = {
    "fid": "LunarLander",
    "fun": lunar_lander,
    "init_state": None,
    "xlb": -5,
    "xub": 5,
}


def bipedal_walker(x, net, state=None, steps=256, render=None):
    """
    obs: 24, continuous
    action: 4, continuous
    """
    _env = gym.make("BipedalWalker-v3", render_mode=render)
    env = TaskEnv(_env)

    b, n, d = x.shape
    x = x.view(-1, d)
    pop_fitness = []
    fitness = 0
    with torch.no_grad():
        for indv in x:
            net.set_params(indv)
            obs, info = env.reset(init_state=state)
            for step in range(steps):
                logits = net(torch.tensor(obs, dtype=DTYPE, device=DEVICE))
                action = torch.tanh(logits).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                fitness = -info["episode_total_reward"]
                if terminated or truncated or (step == steps - 1):
                    break

            pop_fitness.append(fitness)

    env.close()
    pop_fitness = torch.tensor(pop_fitness, dtype=DTYPE, device=DEVICE).view(b, n, 1)
    x = x.view(b, n, d)
    x = torch.cat((pop_fitness, x), dim=-1)

    return x, pop_fitness


TASKS[3] = {
    "fid": "BipedalWalker",
    "fun": bipedal_walker,
    "init_state": None,
    "xlb": -5,
    "xub": 5,
}


def mountain_car_continuous(x, net, state=None, steps=256, render=None):
    """
    obs: 2, continuous
    action: 1, continuous
    """
    _env = gym.make("MountainCarContinuous-v0", render_mode=render)
    env = TaskEnv(_env)

    b, n, d = x.shape
    x = x.view(-1, d)
    pop_fitness = []
    fitness = 0
    with torch.no_grad():
        for indv in x:
            net.set_params(indv)
            obs, info = env.reset(init_state=state)
            for step in range(steps):
                logits = net(torch.tensor(obs, dtype=DTYPE, device=DEVICE))
                action = torch.tanh(logits).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                fitness = -info["episode_total_reward"]
                if terminated or truncated or (step == steps - 1):
                    break

            pop_fitness.append(fitness)

    env.close()
    pop_fitness = torch.tensor(pop_fitness, dtype=DTYPE, device=DEVICE).view(b, n, 1)
    x = x.view(b, n, d)
    x = torch.cat((pop_fitness, x), dim=-1)

    return x, pop_fitness


TASKS[4] = {
    "fid": "MountainCarContinuous",
    "fun": mountain_car_continuous,
    "init_state": None,
    "xlb": -5,
    "xub": 5,
}


def acrobot(x, net, state=None, steps=256, render=None):
    """
    obs: 6, continuous
    action: {0, 1, 2}
    """
    _env = gym.make("Acrobot-v1", render_mode=render)
    env = TaskEnv(_env)

    b, n, d = x.shape
    x = x.view(-1, d)
    pop_fitness = []
    fitness = 0
    with torch.no_grad():
        for indv in x:
            net.set_params(indv)
            obs, info = env.reset(init_state=state)
            for step in range(steps):
                logits = net(torch.tensor(obs, dtype=DTYPE, device=DEVICE))
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                fitness = -info["episode_total_reward"]
                if terminated or truncated or (step == steps - 1):
                    break

            pop_fitness.append(fitness)

    env.close()
    pop_fitness = torch.tensor(pop_fitness, dtype=DTYPE, device=DEVICE).view(b, n, 1)
    x = x.view(b, n, d)
    x = torch.cat((pop_fitness, x), dim=-1)

    return x, pop_fitness


TASKS[5] = {
    "fid": "Acrobot",
    "fun": acrobot,
    "init_state": None,
    "xlb": -5,
    "xub": 5,
}


def pendulum(x, net, state=None, steps=256, render=None):
    """
    obs: 3, continuous
    action: 1, continuous
    """
    _env = gym.make("Pendulum-v1", render_mode=render)
    env = TaskEnv(_env)

    b, n, d = x.shape
    x = x.view(-1, d)
    pop_fitness = []
    fitness = 0
    with torch.no_grad():
        for indv in x:
            net.set_params(indv)
            obs, info = env.reset(init_state=state)
            for step in range(steps):
                logits = net(torch.tensor(obs, dtype=DTYPE, device=DEVICE))
                action = 2 * torch.tanh(logits).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                fitness = -info["episode_total_reward"]
                if terminated or truncated or (step == steps - 1):
                    break

            pop_fitness.append(fitness)

    env.close()
    pop_fitness = torch.tensor(pop_fitness, dtype=DTYPE, device=DEVICE).view(b, n, 1)
    x = x.view(b, n, d)
    x = torch.cat((pop_fitness, x), dim=-1)

    return x, pop_fitness


TASKS[6] = {
    "fid": "Pendulum",
    "fun": pendulum,
    "init_state": None,
    "xlb": -5,
    "xub": 5,
}
