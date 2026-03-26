import numpy as np
import torch
from anomaly_tpp.data import Sequence

from typing import Optional, Union
from tick.hawkes import SimuHawkesExpKernels

from .utils import merge_arrival_times

__all__ = [
    "hawkes_exp_kernels",
]


def hawkes_exp_kernels(
    t_max: float,
    base_rates: Union[float, np.ndarray],
    influence: Union[float, np.ndarray],
    decays: Union[float, np.ndarray] = 1.0,
    adjust_radius: Optional[float] = None,
) -> Sequence:
    """Multivariate Hawkes process with exponential decay kernels.

    Args:
        t_max: Duration of the observed interval.
        base_rates: Base rate for each mark, shape [K] or []
        influence: Influence matrix, shape [K, K] or []
        decays: Decay of each mark, shape [K] or []
        adjust_radius: Branching factor for the MHP (rescales the influence matrix).
    """
    influence = np.array(influence, dtype=np.float64)
    num_marks = int(np.sqrt(influence.size))
    influence = influence.reshape([num_marks, num_marks])
    if adjust_radius is not None:
        # Make sure that the largest singular value equals adjust_radius
        lambda_max = np.abs(np.linalg.eigvals(influence)).max()
        influence *= adjust_radius / lambda_max

    if isinstance(base_rates, float):
        base_rates = base_rates * np.ones(num_marks)
    base_rates = np.array(base_rates)

    if isinstance(decays, float):
        decays = decays * np.ones([num_marks, num_marks])
    decays = decays

    generator = SimuHawkesExpKernels(
        influence, decays, base_rates, end_time=t_max, verbose=False
    )
    generator.simulate()
    times, marks = merge_arrival_times(generator.timestamps)
    if num_marks == 1:
        marks = None
    return Sequence(arrival_times=torch.as_tensor(times, dtype=torch.float32), t_max=t_max, marks=marks)
