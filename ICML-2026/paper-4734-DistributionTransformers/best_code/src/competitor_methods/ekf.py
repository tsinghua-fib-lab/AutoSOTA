"""
Extended Kalman filter competitor for filtering tasks
"""

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.func import vmap, jacrev

from functools import partial
from time import time

from distributions.distributions import (ObservationModel, DirectGaussianObservationModel,
                                         MappedGaussianObservationModel, LinearGaussianObservationModel,
                                         RangefinderObservationModel, GaussianAngleObservationModel)
from distributions.utils import batch_diag
from dynamic.motion_models import MotionModel, LTIMotionModel
from dynamic.filters import Filter


class EKF(Filter):

    def __init__(self, state_size: int,
                 motion_model: MotionModel,
                 jitter: float = 1e-6,
                 **observation_model: ObservationModel):
        """
        Extended Kalman Filter (EKF). Assumes Gaussianity and linearises dynamics and observations.

        Args:
            state_size: Size of state.
            motion_model: Motion model. Linearises mapping from state to mean of next state distribution.
            observation_model: Observation model. Linearises mapping from state to mean of observation distribution.
        """
        self.state_size = state_size
        self.motion_model = motion_model
        self.observation_model = observation_model
        self.jitter = jitter

    def filter(self, observation_series: dict[str, Tensor],
               x0_distribution: Distribution
               ) -> tuple[dict[str, Tensor], float]:
        """
        Filter the provided series (assuming sequence in first dimension) assuming initial uncertainty
        x0_distribution.

        Args:
            observation_series: Dictionary of tensors containing observations of series to filter.
            x0_distribution: Initial state distribution.

        Returns:
            Tensor of multivariate Gaussian parameters of shape (series_length | batch_shape | mu sigma).

        """
        device = list(observation_series.values())[0].device

        batched_series_shape = list(observation_series.values())[0].shape[:-1]
        batch_shape = batched_series_shape[1:]
        if batch_shape == torch.Size():
            batch_shape = (1,)

        init_loc = x0_distribution.mean.broadcast_to(*batch_shape, self.state_size).to(device)
        init_covariance_matrix = getattr(x0_distribution, "covariance_matrix", batch_diag(x0_distribution.variance)
                                         ).to(device)
        prior_params = {"loc": init_loc, "covariance_matrix": init_covariance_matrix}

        filtered_loc = torch.empty(*batched_series_shape, self.state_size)
        filtered_covariance_matrix = torch.empty(*batched_series_shape, self.state_size, self.state_size)

        inference_times = []
        for i, observations in enumerate(zip(*observation_series.values())):
            start_time = time()
            posterior_params = self._update(prior_params, **dict(zip(observation_series, observations)))
            filtered_loc[i] = posterior_params["loc"]
            filtered_covariance_matrix[i] = posterior_params["covariance_matrix"]
            prior_params = self._predict(posterior_params, i)
            inference_times.append(time() - start_time)

        return ({"loc": filtered_loc, "covariance_matrix": filtered_covariance_matrix},
                sum(inference_times) / len(inference_times))

    def _predict(self, prev_posterior_params: dict[str, Tensor],
                 k: int,
                 ) -> dict[str, Tensor]:
        """
        Prediction step of EKF.

        Args:
            prev_posterior_params: Parameters of posterior distribution for previous step.
            k: Timestep index.

        Returns:
            Parameters of prior distribution for current step.

        """
        x = prev_posterior_params["loc"]
        P = prev_posterior_params["covariance_matrix"]

        device = x.device

        posterior_loc = self.motion_model.f(x.cpu(), self.motion_model.noise_distribution.mean, k).to(device)

        if isinstance(self.motion_model, LTIMotionModel):
            F = self.motion_model.state_transition_matrix.to(device)
            L = self.motion_model.process_noise_covariance_matrix.to(device)
            Q = torch.eye(self.state_size, device=device)
        else:
            F = vmap(jacrev(partial(self.motion_model.f, n=self.motion_model.noise_distribution.mean, k=k))
                     )(x.cpu()).to(device)
            L = jacrev(lambda n: self.motion_model.f(x.cpu(), n, k)
                       )(self.motion_model.noise_distribution.mean).to(device)
            Q = getattr(self.motion_model.noise_distribution, "covariance_matrix",
                        batch_diag(self.motion_model.noise_distribution.variance)).to(device)

        posterior_covariance_matrix = (F @ P @ F.mT + L @ Q @ L.mT)

        return {"loc": posterior_loc, "covariance_matrix": posterior_covariance_matrix}

    def _update(self, prior_params: dict[str, Tensor],
                **observations: Tensor
                ) -> dict[str, Tensor]:
        """
        Update step of EKF.

        Args:
            prior_params: Parameters of prior distribution for current step.

        Returns:
            Parameters of posterior distribution for current step.

        """
        x = prior_params["loc"]
        P = prior_params["covariance_matrix"]

        device = x.device

        for key, observation in observations.items():
            observation_model = self.observation_model[key]

            y = observation - observation_model.conditional_mean(x)

            if isinstance(observation_model, LinearGaussianObservationModel):
                H = observation_model.observation_matrix
                R = observation_model.covariance_matrix
            elif isinstance(observation_model, MappedGaussianObservationModel):
                H = vmap(jacrev(observation_model.mapping))(x).nan_to_num()
                R = observation_model.covariance_matrix
            elif isinstance(observation_model, DirectGaussianObservationModel):
                H = torch.eye(self.state_size, device=device)
                R = observation_model.covariance_matrix
            elif isinstance(observation_model, RangefinderObservationModel):
                range = torch.sqrt(x[..., 0] ** 2 + x[..., 2] ** 2)
                H = (observation_model.weights[..., 0] * x / range.unsqueeze(-1)).unsqueeze(-2)
                H[..., [1, 3]] = 0
                R = batch_diag(observation_model.conditional_variance(x))
            elif isinstance(observation_model, GaussianAngleObservationModel):
                range2 = x[..., 0] ** 2 + x[..., 2] ** 2
                H = torch.zeros_like(x)
                H[..., 0] = x[..., 2] / range2
                H[..., 2] = -x[..., 0] / range2
                H = H.unsqueeze(-2)
                R = batch_diag(observation_model.conditional_variance(x))
            else:
                torch.set_grad_enabled(True)
                H = torch.stack([jacrev(observation_model.conditional_mean)(x_batch) for x_batch in x])
                R = getattr(observation_model, "covariance_matrix",
                            batch_diag(observation_model.conditional_variance(x)))
                torch.set_grad_enabled(False)
            S = H @ P @ H.mT + R
            K = P @ H.mT @ torch.linalg.inv(S)

            x = x + torch.einsum("...ij, ...j -> ...i", K, y)
            P = (torch.eye(self.state_size, device=device) - K @ H) @ P

            while P.det().min() <= 0:
                P[P.det() <= 0] += self.jitter * torch.eye(P.shape[-1], device=device)

        return {"loc": x, "covariance_matrix": P}
