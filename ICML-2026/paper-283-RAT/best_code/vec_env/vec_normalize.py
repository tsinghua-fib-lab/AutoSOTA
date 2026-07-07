from . import VecEnvWrapper
import numpy as np
import torch
import gym

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class ObsTransform(VecEnvWrapper):
    def __init__(self, venv, device):
        from torchvision import transforms
        VecEnvWrapper.__init__(self, venv)
        self.device = device
        self.transforms =  transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # change from [h, w, c] to [c, h, w]
        obs_dim = self.observation_space.shape
        obs_dim = (obs_dim[2], obs_dim[0], obs_dim[1])
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=obs_dim, dtype=float)

    def tensorize(self, obs):
        transformed_obs = [self.transforms(o) for o in obs]
        return torch.stack(transformed_obs, dim=0).to(self.device)
    
    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        return self.tensorize(obs), rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        return self.tensorize(obs)

class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, norm_ret=True, obs_preprocess=None, cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ret_rms = RunningMeanStd(shape=()) if norm_ret else None
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.obs_preprocess = obs_preprocess

        if len(self.observation_space.shape) == 3:
            # change from [h, w, c] to [c, h, w]
            obs_dim = self.observation_space.shape
            obs_dim = (obs_dim[2], obs_dim[0], obs_dim[1])
            self.observation_space = gym.spaces.Box(0.0, 1.0, shape=obs_dim, dtype=float)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)

        for info in infos:
            if 'terminal_observation' in info:
                info['terminal_observation'] = self._obfilt(info['terminal_observation'])

        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        return obs if self.obs_preprocess is None else self.obs_preprocess(obs)

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)
