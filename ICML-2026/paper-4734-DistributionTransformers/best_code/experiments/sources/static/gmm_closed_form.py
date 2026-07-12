"""
Experiment to validate method against closed form posterior of GMM prior with linear Gaussian observations
"""

import torch
from torch import Tensor

from functools import partial

from distributions.distributions import (GaussianMixtureModelConjugateMetaPrior, LinearGaussianObservationModel,
                                         CompleteDistribution, GaussianMixtureModel)
from distributions.utils import gmm_bounds_func
from distributions.special import gmm_with_linear_gaussian_observations_posterior
from model.embeddings import ComponentEmbedding, ObservationEmbedding
from model.distribution_transformer import DistributionTransformer
from workflows.train import train
from workflows.test import test_conjugate_prior


def run(n_components: int,
        state_size: int,
        meta_prior_kwargs: dict,
        observation_covariance_matrix: dict[str, list[list[float]]],
        observation_matrix: dict[str, list[list[float]]],
        component_embedding_kwargs: dict,
        observation_embedding_kwargs: dict[str, dict],
        transformer_kwargs: dict,
        training_kwargs: dict,
        testing_kwargs: dict,
        load_path: str | None = None,
        _run=None,
        *args, **kwargs):
    """
    Run an experiment comparing distribution transformers to the closed form posterior of a GMM prior under linear
    Gaussian observations.

    Args:
        n_components: Number of GMM components.
        state_size: Dimensionality of GMM.
        meta_prior_kwargs: Dictionary of parameters for the meta prior.
        observation_covariance_matrix: Dictionary of covariance matrices for gaussian observations.
        observation_matrix: Dictionary of observation matrices for gaussian observations.
        component_embedding_kwargs: Dictionary of component embedding parameters.
        observation_embedding_kwargs: Dictionary of dictionaries of observation embedding parameters.
        transformer_kwargs: Dictionary of parameters for the transformer model.
        training_kwargs: Dictionary of parameters for the training routine.
        testing_kwargs: Dictionary of parameters for the testing routine.
        load_path: Path from which to load model parameters
        _run: Sacred run object.

    Returns:

    """
    # Meta-prior
    meta_prior = GaussianMixtureModelConjugateMetaPrior(state_size=state_size, n_components=n_components,
                                                        **meta_prior_kwargs)

    # Observation model
    covariance_matrix_dict = {key: torch.tensor(val, dtype=torch.float32) * torch.eye(
        observation_embedding_kwargs[key]["observation_size"])
                              for key, val in observation_covariance_matrix.items()}
    observation_matrix_dict = {key: torch.tensor(val, dtype=torch.float32).broadcast_to(
        observation_embedding_kwargs[key]["observation_size"], state_size)
                               for key, val in observation_matrix.items()}
    observation_model = {key: LinearGaussianObservationModel(observation_matrix=observation_matrix_dict[key],
                                                             covariance_matrix=covariance_matrix_dict[key])
                         for key in covariance_matrix_dict}

    # Complete distribution
    complete_distribution = CompleteDistribution(meta_prior, **observation_model)

    # Distribution transformer
    d_model = transformer_kwargs["d_model"]
    component_embedding = ComponentEmbedding(state_size=state_size, d_model=d_model, **component_embedding_kwargs)
    observation_embedding = {key: ObservationEmbedding(d_model=d_model, **kwargs)
                             for key, kwargs in observation_embedding_kwargs.items()}
    model = DistributionTransformer(component_embedding=component_embedding,
                                    transformer_kwargs=transformer_kwargs,
                                    n_components=n_components,
                                    prior_embedding=None,
                                    sample_space_transform=None,
                                    **observation_embedding)

    if load_path is not None:
        model.load_state_dict(torch.load('experiments\\runs\\gmm_closed_form\\' + load_path, weights_only=True))
    else:
        model, last_epoch_metrics = train(model, complete_distribution, _run=_run, **training_kwargs)

    scale_parametrisation = component_embedding_kwargs["scale_parametrisation"]

    def conjugacy_update(phi: dict[str, Tensor],
                         z: dict[str, Tensor],
                         device: str
                         ) -> dict[str, Tensor]:
        dist = GaussianMixtureModel(**phi)
        for key in z:
            dist = gmm_with_linear_gaussian_observations_posterior(dist, observation_model[key], z[key], device)
        return {
            "weights": dist.weights,
            "loc": dist.loc,
            scale_parametrisation: getattr(dist, scale_parametrisation)
        }

    test_conjugate_prior(model, n_components, complete_distribution, conjugacy_update,
                         bounds_func=partial(gmm_bounds_func, scale_parametrisation=scale_parametrisation),
                         _run=_run, **testing_kwargs)
