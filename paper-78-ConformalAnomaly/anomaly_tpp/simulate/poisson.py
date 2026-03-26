import numpy as np
from anomaly_tpp.data import Sequence

__all__ = [
    "homogeneous_poisson",
    "jump_poisson",
    "inhomogeneous_poisson",
]


def homogeneous_poisson(t_max: float, rate: float) -> Sequence:
    """Homogeneous Poisson process with constant intensity.

    Args:
        t_max: Length of the observed time interval.
        rate: Intensity of the Poisson process.
    """
    total_intensity = t_max * rate
    N = np.random.poisson(total_intensity)  # number of observed events
    inter_times = t_max * np.random.dirichlet(np.ones(N + 1))
    arrival_times = np.cumsum(inter_times)[:-1]
    return Sequence(arrival_times=arrival_times, t_max=t_max)


def jump_poisson(
    t_max: float, t_jump: float, rate_before: float, rate_after: float
) -> Sequence:
    """Inhomogeneous Poisson process with piecewise constant intensity function.

    Args:
        t_max: Length of the observed time interval.
        t_jump: Time when the intensity changes.
        rate_before: Intensity on interval [0, t_jump]
        rate_after: Intensity on interval (t_jump, t_max]

    """
    if t_jump > t_max:
        raise ValueError("t_jump must be <= t_max")
    if t_jump < 0 or t_max < 0:
        raise ValueError("t_jump and t_max must be nonnegative")

    if t_jump == 0:
        return homogeneous_poisson(t_max, rate_after)
    elif t_jump == t_max:
        return homogeneous_poisson(t_max, rate_before)
    else:  # 0 < t_jump < t_max
        first = homogeneous_poisson(t_jump, rate_before)
        second = homogeneous_poisson(t_max - t_jump, rate_after)
        return first + second


def inhomogeneous_poisson(
    t_max: float, amplitude: float = 0.99, period: float = 50
) -> Sequence:
    """Inhomogeneous Poisson process with intensity defined by a shifted sin wave.

    lambda(t) = amplitude * sin(2 * pi * t / period) + 1

    Args:
        t_max: Length of the observed time interval.
        amplitude, period: Parameters defining the intensity function (see above).
    """
    l_t = lambda t: np.sin(2 * np.pi * t / period) * amplitude + 1
    while 1:
        arrival_times = np.random.exponential(size=int(t_max * 3)).cumsum() * 0.5
        r = np.random.rand(int(t_max * 3))
        index = r < l_t(arrival_times) / 2.0

        arrival_times = arrival_times[index]

        if arrival_times.max() > t_max:
            arrival_times = arrival_times[arrival_times < t_max]
            break
    return Sequence(arrival_times=arrival_times, t_max=t_max)
