import numpy as np
from anomaly_tpp.data import Sequence
from typing import List

__all__ = [
    "self_correcting",
]


def self_correcting(
    t_max: float, alpha: float, mu: float, base_rate: float = 1.0
) -> Sequence:
    """Simulate a univariate self-correcting process.

    Args:
        t_max: Duration of the observed interval.
        alpha: Every time an event happens, intensity is divided by exp(alpha).
        mu: Intensity increases over time proportional to mu * t.
        base_rate: Average number of events in a unit interval (if alpha = mu = 0).
    """
    x = 0
    t = 0
    arrival_times = []
    while True:
        z = np.random.exponential()
        tau = np.log(z * mu * np.exp(-x) / base_rate + 1) / mu
        # Equivalent to z = (np.exp(mu * tau) - 1) * np.exp(x) / mu
        x = x + mu * tau - alpha
        t += tau
        if t > t_max:
            break
        arrival_times.append(t)
    return Sequence(arrival_times=np.array(arrival_times), t_max=t_max)