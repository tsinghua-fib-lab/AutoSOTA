"""
Filter implementations of distribution transformers
"""

import torch
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal

from abc import ABC
from time import time

from model.distribution_transformer import DistributionTransformer
from dynamic.motion_models import LTIMotionModel
from distributions.utils import decode_gmm_sample, encode_gmm_sample


class Filter(ABC):
    """
    Abstract base class for filter implementation of distribution transformer
    """

    def filter(self, observation_series: dict[str, Tensor],
               x0_distribution: Distribution
               ) -> dict[str, Tensor]:
        """
        Filter the provided trajectory (assuming sequence in first dimension) assuming initial uncertainty
        x0_distribution.

        Args:
            observation_series: Dictionary of tensors containing observations of series to filter.
            x0_distribution: Initial state

        Returns:
            Dictionary of GMM parameters of shape (series_length | batch_shape | n_components | w mu sigma).

        """
        raise NotImplementedError


class LTIFilter(Filter):

    def __init__(self, model: DistributionTransformer,
                 motion_model: LTIMotionModel):
        """
        Distribution transformer filter for LTI system.

        Args:
            model: Distribution transformer model.
            motion_model: LTI motion model.

        """
        self.model = model

        self.state_transition_matrix = motion_model.state_transition_matrix
        self.process_noise_covariance_matrix = motion_model.process_noise_covariance_matrix
        self.constant_vector = motion_model.constant_vector

    def filter(self, observation_series: dict[str, Tensor],
               x0_distribution: MultivariateNormal
               ) -> tuple[dict[str, Tensor], float]:
        """
        Filter the provided series (assuming sequence in first dimension) assuming initial uncertainty
        x0_distribution.

        Args:
            observation_series: Dictionary of tensors containing observations of series to filter.
            x0_distribution: Initial state distribution.

        Returns:
            Tensor of GMM parameters of shape (series_length | batch_shape | n_components | w mu sigma).
            Inference time

        """
        device = list(observation_series.values())[0].device
        state_transition_matrix = self.state_transition_matrix.to(device)
        process_noise_covariance_matrix = self.process_noise_covariance_matrix.to(device)
        constant_vector = self.constant_vector.to(device)

        batched_series_shape = list(observation_series.values())[0].shape[:-1]
        batch_shape = batched_series_shape[1:]
        n_components = self.model.n_components
        n_params = 1 + self.model.state_size + self.model.state_size ** 2

        init_weights = torch.arange(1., n_components+1)
        init_weights = init_weights / init_weights.sum()
        init_weights = init_weights.unsqueeze(-1)
        init_weights = init_weights.broadcast_to(*x0_distribution.batch_shape, n_components, 1)
        init_loc = x0_distribution.loc.unsqueeze(-2)

        scale_parametrisation = self.model.component_embedding.scale_parametrisation
        init_scale = getattr(x0_distribution, scale_parametrisation).flatten(-2).unsqueeze(-2)
        prior_params = torch.hstack([init_weights, init_loc.broadcast_to(init_weights.shape[:-1]
                                                                         + init_loc.shape[-1:]),
                                     init_scale.broadcast_to(init_weights.shape[:-1] + init_scale.shape[-1:])]
                                    ).broadcast_to(*batch_shape, n_components, n_params).to(device)

        filtered_params = torch.empty(*batched_series_shape, n_components, n_params)
        inference_times = []

        for i, observations in enumerate(zip(*observation_series.values())):
            start_time = time()
            _, posterior_params = self.model(prior_params, **dict(zip(observation_series, observations)))
            filtered_params[i] = posterior_params
            prior_params_dict = decode_gmm_sample(posterior_params, scale_parametrisation)
            prior_params_dict["loc"] = (constant_vector +
                                        torch.einsum("...ij, ...j -> ...i",
                                                     state_transition_matrix, prior_params_dict["loc"]))

            match scale_parametrisation:
                case "covariance_matrix":
                    covariance_matrix = prior_params_dict["covariance_matrix"]
                case "precision_matrix":
                    covariance_matrix = prior_params_dict["precision_matrix"].inverse()
                case "scale_tril":
                    scale_tril = prior_params_dict["scale_tril"]
                    covariance_matrix = scale_tril @ scale_tril.mT
                case _:
                    raise ValueError

            covariance_matrix = (state_transition_matrix @ covariance_matrix @ state_transition_matrix.mT
                                 + process_noise_covariance_matrix)

            match scale_parametrisation:
                case "covariance_matrix":
                    prior_params_dict["covariance_matrix"] = covariance_matrix
                case "precision_matrix":
                    prior_params_dict["precision_matrix"] = covariance_matrix.inverse()
                case "scale_tril":
                    prior_params_dict["scale_tril"] = torch.linalg.cholesky(covariance_matrix)
                case _:
                    raise ValueError

            prior_params = encode_gmm_sample(prior_params_dict, scale_parametrisation)
            inference_times.append(time()-start_time)

        return decode_gmm_sample(filtered_params), sum(inference_times) / len(inference_times)
