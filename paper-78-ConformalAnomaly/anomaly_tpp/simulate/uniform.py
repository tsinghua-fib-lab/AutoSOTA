import numpy as np
from anomaly_tpp.data import Sequence

__all__ = [
    "uniform_sequence",
]



def uniform_sequence(t_max: float, det: float) -> Sequence:
    """Generates a sequence of event arrival times with uniform intervals.

    Args:
        t_max: Length of the observed time interval.
        det: detectability
    """

    arrival_times = np.arange(0, t_max, 1+2*det)
    arrival_times = arrival_times[(arrival_times > 0) & (arrival_times < t_max)]

    return Sequence(arrival_times=np.array(arrival_times), t_max=t_max)