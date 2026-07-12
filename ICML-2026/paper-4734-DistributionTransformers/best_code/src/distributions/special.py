"""
Special methods and classes operating on distributions
"""

import torch
from torch import Tensor
from torch.distributions import Distribution, Wishart, MultivariateNormal, constraints

from typing import Optional, Callable
from sklearn.mixture import GaussianMixture
from functools import partial

from distributions.distributions import (GaussianMixtureModel, LinearGaussianObservationModel)


def distribution_to_gmm(p: Distribution, n_components: int,
                        transform: Optional[Callable[[Tensor], Tensor]] = None,
                        n_samples: int = 1000, scale_parametrisation: str = "scale_tril",
                        *args, **kwargs) -> Tensor:
    """
    Approximate an arbitrary distribution p with a Gaussian mixture model with a specified number of components.
    Samples n_samples from p and fits the GMM using Expectation Maximisation.

    Examples:
        >>> root_state_size = 2
        >>> p = Wishart(root_state_size + 1, torch.eye(root_state_size))
        >>> state_size = root_state_size ** 2
        >>> n_components = 4
        >>> transform = partial(torch.flatten, start_dim=-2)
        >>> phi = distribution_to_gmm(p, n_components, transform)
        >>> weights = phi[..., :n_components]
        >>> loc = phi[..., n_components:(1 + state_size) * n_components].unflatten(-1, (n_components, state_size))
        >>> scale = phi[..., -n_components * state_size ** 2:].unflatten(-1, (n_components, state_size, state_size))
        >>> gmm = GaussianMixtureModel(weights, loc, covariance_matrix=scale)

    Args:
        p: Probability distribution to approximate.
        n_components: Number of components in approximating GMM.
        transform: Transform under which to fit the GMM.
            Defaults to None.
        n_samples: Number of samples on which to perform EM.
            Defaults to 1000.
        scale_parametrisation: Parametrisation of scale parameter of GMM. One of "covariance_matrix", "precision_matrix"
            or "scale_tril".
            Defaults to "covariance_matrix".
        *args, **kwargs for sklearn GaussianMixture.

    Returns:
        Tensor of distribution parameters

    """
    samples = p.sample((n_samples,))
    if transform is not None:
        samples = transform(samples)
    device = samples.device
    samples = samples.cpu().detach()
    samples = torch.movedim(samples, 0, len(p.batch_shape))
    if len(p.batch_shape):
        samples = samples.flatten(end_dim=len(p.batch_shape)-1)
    else:
        samples = samples.unsqueeze(0)

    # Inner function for handling batched distributions
    def inner(sample):
        gmm = GaussianMixture(n_components=n_components, *args, **kwargs)
        gmm.fit(sample.reshape(n_samples, -1))
        weights = torch.tensor(gmm.weights_, dtype=torch.float32)
        loc = torch.tensor(gmm.means_, dtype=torch.float32)
        match scale_parametrisation:
            case "covariance_matrix":
                scale = torch.tensor(gmm.covariances_, dtype=torch.float32)
            case "precision_matrix":
                scale = torch.tensor(gmm.precisions_, dtype=torch.float32)
            case "scale_tril":
                scale = torch.cholesky(torch.tensor(gmm.covariances_, dtype=torch.float32))
            case _:
                raise ValueError('scale_parametrisation must be one of "covariance_matrix", "precision_matrix" or '
                                 '"scale_tril')
        return torch.cat([weights.unsqueeze(-1), loc, scale.flatten(-2)], dim=-1)

    phi = torch.cat([inner(sample) for sample in samples], dim=-1).to(device)
    return phi.reshape(*p.batch_shape, n_components, -1)


def gmm_with_linear_gaussian_observations_posterior(prior: GaussianMixtureModel,
                                                    observation_model: LinearGaussianObservationModel,
                                                    observations: Tensor,
                                                    device: str = "cuda:0",
                                                    jitter: float = 1e-6
                                                    ) -> GaussianMixtureModel:
    """
    Given a Gaussian mixture model prior, a linear Gaussian observation model, and a set of observations, return the
    analytical posterior; another Gaussian mixture model.

    Args:
        prior: Prior Gaussian mixture model.
        observation_model: Linear Gaussian observation model.
        observations: Tensor of observations.
        device: CUDA device.
            Defaults to "cuda:0".
        jitter: Small value to add to diagonal of poorly-conditioned covariance matrices. Raised to the power of
            1 / sqrt of dimensionality to ensure raw determinant increases by this amount each jitter addition.
            Defaults to 1e-6.

    Returns:
        Posterior Gaussian mixture model.

    """
    device = device if torch.cuda.is_available() else "cpu"

    match prior.scale_parametrisation:
        case "covariance_matrix":
            prior_covariance_matrix = prior.covariance_matrix
        case "precision_matrix":
            prior_covariance_matrix = torch.linalg.inv(prior.precision_matrix)
        case "scale_tril":
            prior_covariance_matrix = torch.einsum("...ij,...kj->...ik", prior.scale_tril, prior.scale_tril)
        case _:
            raise ValueError

    match observation_model.scale_parametrisation:
        case "covariance_matrix":
            observation_covariance_matrix = observation_model.covariance_matrix
        case "precision_matrix":
            observation_covariance_matrix = torch.linalg.inv(observation_model.precision_matrix)
        case "scale_tril":
            observation_covariance_matrix = torch.einsum("...ij,...kj->...ik", observation_model.scale_tril,
                                                         observation_model.scale_tril)
        case _:
            raise ValueError

    observation_matrix = observation_model.observation_matrix.to(device)

    schur_marginal_term = torch.einsum("ij,...jk,lk->...il", observation_matrix, prior_covariance_matrix,
                                       observation_matrix) + observation_covariance_matrix.to(device)
    schur_marginal_term = (schur_marginal_term.to(torch.float64) +
                           torch.transpose(schur_marginal_term.to(torch.float64), dim0=-2, dim1=-1)) / 2
    schur_marginal_term = schur_marginal_term.to(torch.float32)
    schur_inverse_term = torch.linalg.inv(schur_marginal_term)
    schur_covariance_term = torch.einsum("...ij,kj->...ik", prior_covariance_matrix, observation_matrix)
    observation_marginal_mean = torch.einsum("ij,...j->...i", observation_matrix, prior.loc)
    observations = observations.unsqueeze(-2)
    residual_term = observations - observation_marginal_mean

    posterior_loc = prior.loc + torch.einsum("...ij,...jk,...k->...i", schur_covariance_term, schur_inverse_term,
                                             residual_term)

    posterior_covariance_matrix = prior_covariance_matrix - torch.einsum("...ij,...jk,...lk->...il",
                                                                         schur_covariance_term, schur_inverse_term,
                                                                         schur_covariance_term)
    posterior_covariance_matrix = (posterior_covariance_matrix.to(torch.float64) +
                                   torch.transpose(posterior_covariance_matrix.to(torch.float64), dim0=-2, dim1=-1)) / 2
    posterior_covariance_matrix = posterior_covariance_matrix.to(torch.float32)

    observation_component_marginals = MultivariateNormal(loc=observation_marginal_mean,
                                                         covariance_matrix=schur_marginal_term)
    observation_component_evidences = torch.exp(observation_component_marginals.log_prob(observations))

    posterior_weights = prior.weights * observation_component_evidences
    posterior_weights /= posterior_weights.sum(dim=-1).unsqueeze(-1)

    # Add jitter where necessary
    cov_shape = posterior_covariance_matrix.shape
    jitter = jitter ** (1 / cov_shape[-1])
    posterior_covariance_matrix = posterior_covariance_matrix.flatten(0, -3)
    while not (constraints.positive_definite.check(posterior_covariance_matrix)).all():
        posterior_covariance_matrix[torch.logical_not(constraints.positive_definite.check(
            posterior_covariance_matrix))] += jitter * torch.eye(cov_shape[-1], device=device)
    posterior_covariance_matrix = posterior_covariance_matrix.reshape(cov_shape)

    posterior = GaussianMixtureModel(weights=posterior_weights, loc=posterior_loc,
                                     covariance_matrix=posterior_covariance_matrix, validate_args=True)
    return posterior
