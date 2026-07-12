"""
Motion models for filtering equations
"""

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, Distribution
from torch.types import _size

from typing import Callable


class MotionModel:

    def __init__(self, state_size: int,
                 f: Callable[[Tensor, Tensor, int], Tensor],
                 x0_distribution: Distribution,
                 noise_distribution: Distribution):
        """
        Base class for SDE-based state space motion models of the form x_k+1 = f(x_k, n_k, k) where n_k is process noise
        distributed independently of x.

        Args:
            state_size: Size of state space.
            f: Dynamics step function.
            x0_distribution: Distribution for initial state.
            noise_distribution: Distribution for noise process noise.

        """
        self.state_size = state_size
        self.f = f

        self.x0_distribution = x0_distribution
        self.noise_distribution = noise_distribution

    def to(self):
        raise NotImplementedError

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        """
        Sample trajectory. Trajectory length is given by first dimension of sample_shape.

        Args:
            sample_shape: Sample shape, in form (batch dims | trajectory length).

        Returns:
            Sampled trajectory.

        """
        assert sample_shape[0] > 1

        x0_sample = self.x0_distribution.sample(sample_shape[1:])
        noise_samples = self.noise_distribution.sample(sample_shape)

        trajectory_sample = torch.empty(*sample_shape, self.state_size)
        trajectory_sample[0] = x0_sample

        for i, noise_sample in enumerate(noise_samples[1:], start=1):
            trajectory_sample[i] = self.f(trajectory_sample[i-1], noise_sample, i-1)

        return trajectory_sample


class TimeInvariantMotionModel(MotionModel):

    def __init__(self, state_size: int,
                 f: Callable[[Tensor, Tensor], Tensor],
                 x0_distribution: Distribution,
                 noise_distribution: Distribution):
        """
        Base class for time-invariant SDE-based state space motion models of the form x_k+1 = f(x_k, n_k)
        where n_k is process noise distributed according to noise_distribution.

        Args:
            state_size: Size of state space.
            f: Dynamics step function.
            x0_distribution: Distribution for initial state.
            noise_distribution: Distribution for noise process noise.

        """
        super().__init__(state_size,
                         lambda x, n, k: f(x, n),
                         x0_distribution,
                         noise_distribution)


class LTIMotionModel(TimeInvariantMotionModel):

    def __init__(self,
                 state_transition_matrix: Tensor,
                 process_noise_scale_tril: Tensor,
                 x0_distribution: MultivariateNormal,
                 constant_vector: Tensor = None):
        """
        Linear time-invariant SDE-based state space motion models of the form x_k+1 = Ax_k + sqrt(Q)n_k
        where n_k is multivariate normal distributed with zero mean and identity covariance matrix.

        Args:
            state_transition_matrix: State transition matrix, A. Must be of shape state_size X state_size.
            process_noise_scale_tril: Cholesky decomposition of process noise covariance matrix, sqrt(Q). Must be of
                shape state_size X noise_size.
            x0_distribution: Distribution for initial state.
            constant_vector: Constant vector added each dynamics step. Defaults to 0


        """
        state_size = state_transition_matrix.shape[-1]
        noise_size = process_noise_scale_tril.shape[-1]


        self.state_transition_matrix = state_transition_matrix
        self.process_noise_scale_tril = process_noise_scale_tril
        self.process_noise_covariance_matrix = process_noise_scale_tril @ process_noise_scale_tril.mT
        self.x0_distribution: MultivariateNormal = x0_distribution
        self.constant_vector = constant_vector if constant_vector is not None else torch.zeros(state_transition_matrix.shape[:-1])

        super().__init__(state_size,
                         lambda x, n: self.constant_vector
                                      + torch.einsum("...ij, ...j -> ...i", state_transition_matrix, x - self.constant_vector)
                                      + torch.einsum("...ij, ...j -> ...i", process_noise_scale_tril, n),
                         x0_distribution,
                         MultivariateNormal(torch.zeros(noise_size), torch.eye(noise_size)))

