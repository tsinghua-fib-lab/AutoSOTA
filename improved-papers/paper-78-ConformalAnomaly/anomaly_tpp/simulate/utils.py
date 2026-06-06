from typing import List, Tuple, Union

import numpy as np


def merge_arrival_times(
    list_of_times: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine the arrival times of each mark into a single event sequence.

    Args:
        list_of_times: Arrival times for each mark.

    Returns:
        arrival_times: List of arrival times of all the events.
        marks: Categorical mark corresponding to each event.
    """
    t_concat = np.concatenate(list_of_times)
    marks = np.concatenate(
        [i * np.ones_like(t) for (i, t) in enumerate(list_of_times)]
    ).astype(int)
    o = np.argsort(t_concat)
    return np.ascontiguousarray(t_concat[o]), np.ascontiguousarray(marks[o])
