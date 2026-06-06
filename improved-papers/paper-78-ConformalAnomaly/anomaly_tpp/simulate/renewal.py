import numpy as np
from anomaly_tpp.data import Sequence
from typing import List, Union

from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats import gamma

from .utils import merge_arrival_times

__all__ = [
    "renewal",
]


def renewal(
    t_max: float,
    renewal_distributions: List[Union[rv_frozen, None]] = gamma(2, scale=0.5),
) -> Sequence:
    """Marked renewal process where the inter-event times for each mark are sampled iid.

    Args:
        renewal_distributions: List of renewal distributions for each mark.
    """

    def single_sample_renewal(dist: rv_frozen) -> np.ndarray:
        """Draw a single event sequence from a univariate renewal process."""
        if dist is None:
            return np.array([])
        else:
            num_samples = max(int(2 * t_max / dist.mean()), 1)
            while True:
                inter_times = dist.rvs(num_samples)
                arrival_times = np.cumsum(inter_times)
                if arrival_times[-1] >= t_max:
                    break
                else:
                    num_samples = int(1.5 * num_samples)

            return arrival_times[arrival_times < t_max]

    if isinstance(renewal_distributions, rv_frozen):
        renewal_distributions = [renewal_distributions]

    list_of_times = [single_sample_renewal(dist) for dist in renewal_distributions]
    times, marks = merge_arrival_times(list_of_times)
    if len(renewal_distributions) == 1:
        marks = None
    return Sequence(arrival_times=times, t_max=t_max, marks=marks)
