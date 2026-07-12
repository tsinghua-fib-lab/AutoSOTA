import torch
from torch import Tensor
from torch.distributions import Distribution, InverseGamma, Wishart

import pytest
from typing import Callable

from distributions.distributions import GaussianMixtureModel, LinearGaussianObservationModel
from distributions.special import distribution_to_gmm, gmm_with_linear_gaussian_observations_posterior


@pytest.mark.parametrize(
    ("p", "n_components", "transform"),
    [
        (
            InverseGamma(1, 1),
            10,
            torch.log
        ),
        (
            InverseGamma(torch.ones(2, 2), torch.ones(2, 2)),
            10,
            torch.log
        ),
        (
            Wishart(3, torch.eye(2)),
            10,
            None
        )
    ]
)
def test_distribution_to_gmm(p: Distribution, n_components: int, transform: Callable[[Tensor], Tensor]):
    phi_gmm = distribution_to_gmm(p, n_components, transform)
    state_size = torch.prod(torch.tensor(p.event_shape)).to(torch.int).item()
    assert phi_gmm.shape == p.batch_shape + (n_components, 1 + state_size + state_size ** 2)


@pytest.mark.parametrize(
    ("prior", "observation_model", "observations", "expected_result"),
    [
        (
            GaussianMixtureModel(weights=torch.ones(1),
                                 loc=torch.zeros(1, 2),
                                 covariance_matrix=torch.eye(2).broadcast_to(1, 2, 2)),
            LinearGaussianObservationModel(observation_matrix=torch.eye(2),
                                           covariance_matrix=torch.eye(2)),
            torch.zeros(2),
            GaussianMixtureModel(weights=torch.ones(1),
                                 loc=torch.zeros(1, 2),
                                 covariance_matrix=0.5 * torch.eye(2).broadcast_to(1, 2, 2))
        ),
        (
            GaussianMixtureModel(weights=torch.tensor([1, 2])/3,
                                 loc=torch.tensor([[-1., -1.],
                                                   [1., 1.]]),
                                 covariance_matrix=torch.tensor([[[1., 0.5],
                                                                 [0.5, 2.]],
                                                                 [[1., -0.5],
                                                                 [-0.5, 1.]]])),
            LinearGaussianObservationModel(observation_matrix=torch.ones(1, 2),
                                           covariance_matrix=torch.eye(1)),
            torch.ones(1),
            GaussianMixtureModel(weights=torch.tensor([0.1417, 0.8583]),
                                 loc=torch.tensor([[-0.1, 0.5],
                                                   [0.75, 0.75]]),
                                 covariance_matrix=torch.tensor([[[0.55, -0.25],
                                                                  [-0.25, 0.75]],
                                                                 [[0.875, -0.625],
                                                                 [-0.625, 0.875]]])),
        ),
        (
            GaussianMixtureModel(weights=torch.ones(2, 1),
                                 loc=torch.zeros(2, 1, 2),
                                 covariance_matrix=torch.eye(2).broadcast_to(2, 1, 2, 2)),
            LinearGaussianObservationModel(observation_matrix=torch.eye(2),
                                           covariance_matrix=torch.eye(2)),
            torch.zeros(2, 2),
            GaussianMixtureModel(weights=torch.ones(2, 1),
                                 loc=torch.zeros(2, 1, 2),
                                 covariance_matrix=0.5 * torch.eye(2).broadcast_to(2, 1, 2, 2))
        )
    ]
)
def test_gmm_with_linear_gaussian_observations_posterior(prior: GaussianMixtureModel,
                                                         observation_model: LinearGaussianObservationModel,
                                                         observations: Tensor,
                                                         expected_result: GaussianMixtureModel):
    posterior = gmm_with_linear_gaussian_observations_posterior(prior, observation_model, observations)
    assert torch.allclose(posterior.weights, expected_result.weights, atol=1e-4)
    assert torch.allclose(posterior.loc, expected_result.loc, atol=1e-4)
    assert torch.allclose(posterior.covariance_matrix, expected_result.covariance_matrix, atol=1e-4)
