# Angus Dempster, Francois Petitjean, Geoff Webb

# Dempster A, Petitjean F, Webb GI (2019) ROCKET: Exceptionally fast and
# accurate time series classification using random convolutional kernels.
# arXiv:1910.13051
import time
from tqdm.auto import tqdm

import numpy as np
from numba import njit, prange
from sklearn.linear_model import RidgeCV

from src.models.tsr.time_series_models import TimeSeriesRegressor
from src.utils.runtime.tsr.tools import save_test_duration, save_train_duration

name = "Rocket"


@njit(fastmath=True)
def generate_kernels(input_length, num_kernels, num_channels=1):
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    candidate_lengths = candidate_lengths[candidate_lengths < input_length]
    lengths = np.random.choice(candidate_lengths, num_kernels)

    # exponential
    num_channel_indices = (2 ** np.random.uniform(0, np.log2(num_channels + 1), num_kernels)).astype(np.int32)
    channel_indices = np.zeros(num_channel_indices.sum(), dtype=np.int32)

    weights = np.zeros((num_channels, lengths.sum()), dtype=np.float32)
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    for i in range(num_kernels):

        _weights = np.random.normal(0, 1, (num_channels, lengths[i]))

        a = lengths[:i].sum()
        b = a + lengths[i]
        for j in range(num_channels):
            _weights[j] = _weights[j] - _weights[j].mean()
        weights[:, a:b] = _weights

        a1 = num_channel_indices[:i].sum()
        b1 = a1 + num_channel_indices[i]
        channel_indices[a1:b1] = np.random.choice(np.arange(0, num_channels), num_channel_indices[i], replace=False)

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) // (lengths[i] - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((lengths[i] - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

    return weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices


@njit(fastmath=True)
def apply_kernel(X, weights, length, bias, dilation, padding, num_channel_indices, channel_indices, stride):
    # zero padding
    if padding > 0:
        _input_length, _num_channels = X.shape
        _X = np.zeros((_input_length + (2 * padding), _num_channels))
        _X[padding:(padding + _input_length), :] = X
        X = _X

    input_length, num_channels = X.shape

    output_length = input_length - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    for i in range(0, output_length, stride):
        _sum = bias

        for j in range(length):
            for k in range(num_channel_indices):
                _sum += weights[channel_indices[k], j] * X[i + (j * dilation), channel_indices[k]]

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max


#@njit(parallel=True, fastmath=True)
@njit(fastmath=True)
def apply_kernels(X, kernels, stride=1):
    weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices = kernels

    num_examples = len(X)
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 2), dtype=np.float32)  # 2 features per kernel

    for i in tqdm(prange(num_examples), desc="Applying Kernels"):
        a = 0
        a1 = 0
        for j in range(num_kernels):
            b = a + lengths[j]
            b1 = a1 + num_channel_indices[j]

            _X[i, (j * 2):((j * 2) + 2)] = \
                apply_kernel(X[i], weights[:, a:b], lengths[j], biases[j], dilations[j], paddings[j],
                             num_channel_indices[j], channel_indices[a1:b1], stride)

            a = b
            a1 = b1

    return _X


class RocketRegressor(TimeSeriesRegressor):
    """
    This is a class implementing Rocket for time series regression.
    The code is adapted by the authors from the original Rocket implementation at https://github.com/angus924/rocket
    """

    def __init__(self,
                 output_directory: str,
                 n_kernels: int = 10000):
        """
        Initialise the Rocket model

        Inputs:
            output_directory: path to store results/models
            n_kernels: number of random kernels
        """
        super().__init__(output_directory)
        print('[{}] Creating Regressor'.format(self.name))
        self.name = name
        self.n_kernels = n_kernels
        self.kernels = None
        self.regressor = RidgeCV(alphas=np.logspace(-3, 3, 10),)
                                 #normalize=True) # error, argument not known

    def fit(self,
            x_train: np.array,
            y_train: np.array,
            x_val: np.array = None,
            y_val: np.array = None):
        """
        Fit Rocket

        Inputs:
            x_train: training data (num_examples, num_timestep, num_channels)
            y_train: training target
            x_val: validation data (num_examples, num_timestep, num_channels)
            y_val: validation target
        """
        start_time = time.perf_counter()
        print('[{}] Generating kernels'.format(self.name))
        self.kernels = generate_kernels(x_train.shape[1], self.n_kernels, x_train.shape[2])
        print('[{}] Applying kernels'.format(self.name))
        x_training_transform = apply_kernels(x_train, self.kernels)

        print('[{}] Training'.format(self.name))
        self.regressor.fit(x_training_transform, y_train)
        self.train_duration = time.perf_counter() - start_time

        save_train_duration(self.output_directory + 'train_duration.csv', self.train_duration)

        print('[{}] Training done!, took {}s'.format(self.name, self.train_duration))

    def predict(self, x: np.array):
        """
        Do prediction with Rocket

        Inputs:
            x: data for prediction (num_examples, num_timestep, num_channels)
        Outputs:
            y_pred: prediction
        """
        print('[{}] Predicting'.format(self.name))
        start_time = time.perf_counter()
        print('[{}] Applying kernels'.format(self.name))
        x_test_transform = apply_kernels(x, self.kernels)
        y_pred = self.regressor.predict(x_test_transform)

        test_duration = time.perf_counter() - start_time
        save_test_duration(self.output_directory + 'test_duration.csv', test_duration)

        print("[{}] Predicting completed, took {}s".format(self.name, test_duration))

        return y_pred
