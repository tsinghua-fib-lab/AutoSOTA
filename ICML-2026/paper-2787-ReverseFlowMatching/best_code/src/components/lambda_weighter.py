from abc import ABC, abstractmethod

from src.components.noise_schedule import noise_h


class BaseLambdaWeighter(ABC):
    @abstractmethod
    def __call__(self, t):
        raise NotImplementedError


class BasicLambdaWeighter(BaseLambdaWeighter):
    def __init__(self, sigma_min, sigma_diff, epsilon):
        self.sigma_min = sigma_min
        self.sigma_diff = sigma_diff
        self.epsilon = epsilon

    def __call__(self, t):
        return noise_h(t, self.sigma_min, self.sigma_diff) + self.epsilon


class NoLambdaWeighter(BaseLambdaWeighter):
    def __call__(self, t):
        return 1.0
