import torch
from torch import nn
import abc


class Noise(abc.ABC, nn.Module):
    @abc.abstractmethod
    def rate_noise(self, t: torch.Tensor):
        r'''
        \gamma(t), 0 <= t <= 1
        
        Args:
            t: shape (..., 1)
        Returns:
            shape (..., 1)
        '''
        pass

    @abc.abstractmethod
    def integral_noise(self, s: torch.Tensor, t: torch.Tensor):
        r'''
        \bar\gamma(s, t) = \int_s^t \gamma(u) du, 0 <= s <= t <= 1

        Args:
            s: shape (..., 1)
            t: shape (..., 1)
        Returns:
            shape (..., 1)
        '''
        pass


class ConstNoise(Noise):
    def __init__(self, gamma: float):
        super().__init__()
        assert gamma > 0, "gamma must be positive."
        self.gamma = gamma

    def rate_noise(self, t):
        return self.gamma * torch.ones_like(t)

    def integral_noise(self, s, t):
        return self.gamma * (t - s)


class LogLinearNoise(Noise):
    def __init__(self, gamma: float, alpha: float):
        super().__init__()
        assert gamma > 0 and alpha > 0, "gamma and alpha must be positive."
        self.gamma, self.alpha = gamma, alpha
    
    def rate_noise(self, t):
        return self.gamma / (self.alpha + t)
    
    def integral_noise(self, s, t):
        return self.gamma * ((self.alpha + t) / (self.alpha + s)).clamp(min=1e-10).log()


def get_noise(args) -> Noise:
    if args.noise.type == 'const':
        return ConstNoise(args.noise.gamma)
    elif args.noise.type == 'log_linear':
        return LogLinearNoise(args.noise.gamma, args.noise.alpha)
    else:
        raise NotImplementedError(f'Unknown noise type {args.noise.type}.')
