"""Function for preprocessing the datasets"""

import numpy as np

def normalise_by_dimension(data: np.ndarray, mean: np.array, std: np.array) -> np.ndarray:
    """For each dimension, normalise by subtracting the mean
    and dividing by the standard deviation

    Args:
        data: Shape is (num_observations, dim)

    Returns:
        Normalised dataset
    """
    normalised_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        normalised_data[:,i] = (data[:,i] - mean[i])/std[i]
    return normalised_data

