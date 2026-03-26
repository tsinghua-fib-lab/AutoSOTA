from abc import ABC, abstractmethod

import jax.random as jr
import jax.numpy as jnp


def get_time_to_event(config):
    if config.data_obs.time_to_event_name == "weibull":
        return Weibull(
            config.data_obs.time_to_event_params[0],
            config.data_obs.time_to_event_params[1],
        )


class TimeToEventDistribution(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, key, num_samples):
        pass


class Weibull(TimeToEventDistribution):
    def __init__(self, concentration, scale):
        self.concentration = concentration
        self.scale = scale

    def sample(self, key, num_samples, x):
        if True:
            return jr.weibull_min(
                key, self.scale, self.concentration, shape=(num_samples,)
            )
        else:
            beta = jr.normal(key, (x.shape[1],))
            print(beta)
            return jr.weibull_min(
                key,
                jnp.exp(x @ beta),
                self.concentration,
                shape=(num_samples,),
            )
