from abc import ABC, abstractmethod

import jax.random as jr


def get_time_to_censoring(config):
    if config.data_obs.time_to_censoring_name == "exponential":
        return Exponential(
            config.data_obs.time_to_censoring_params,
        )


class TimeToCensoringDistribution(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, key, num_samples):
        pass


class Exponential(TimeToCensoringDistribution):
    def __init__(self, rate):
        self.rate = rate

    def sample(self, key, num_samples):
        return jr.exponential(key, (num_samples,)) / self.rate
