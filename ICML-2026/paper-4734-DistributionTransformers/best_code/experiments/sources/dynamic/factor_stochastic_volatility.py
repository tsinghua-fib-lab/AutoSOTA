"""
Experiment testing method on time series filtering
"""

import torch
from torch.distributions import MultivariateNormal

from typing import Optional
from pathlib import Path

from distributions.distributions import (GaussianMixtureModelConjugateMetaPrior, CompleteDistribution,
                                         FactorStructureStochasticVolatility)
from model.embeddings import ComponentEmbedding, ObservationEmbedding
from model.distribution_transformer import DistributionTransformer
from workflows.train import train
from workflows.test import test_lti_filter
from dynamic.motion_models import LTIMotionModel


def run(n_components: int,
        state_size: int,
        meta_prior_kwargs: dict,
        observation_model_kwargs: dict[str, dict],
        component_embedding_kwargs: dict,
        observation_embedding_kwargs: dict[str, dict],
        transformer_kwargs: dict,
        motion_model_kwargs: dict,
        training_kwargs: dict,
        testing_kwargs: dict,
        load_path: Optional[str] = None,
        _run=None,
        *args, **kwargs):
    """
    Run an experiment comparing distribution transformers to baselines for a factor structure stochastic volatility
    model.

    Args:
        n_components: Number of GMM components.
        state_size: Dimensionality of GMM.
        meta_prior_kwargs: Dictionary of parameters for the meta prior.
        observation_model_kwargs: Dictionary of dictionaries of kwargs for observation models.
        component_embedding_kwargs: Dictionary of component embedding parameters.
        observation_embedding_kwargs: Dictionary of dictionaries of observation embedding parameters.
        transformer_kwargs: Dictionary of parameters for the transformer model.
        motion_model_kwargs: Dictionary of kwargs for the motion model.
        training_kwargs: Dictionary of parameters for the training routine.
        testing_kwargs: Dictionary of parameters for the testing routine.
        load_path: Path to load model from.
        _run: Sacred run object.

    Returns:

    """

    # Meta-prior
    # meta_prior = GaussianMixtureModelConjugateMetaPrior(state_size=state_size, n_components=n_components,
    #                                                     **meta_prior_kwargs)

    meta_prior = GaussianMixtureModelConjugateMetaPrior(state_size=state_size,
                                                        n_components=n_components,
                                                        loc_covariance_matrix=torch.eye(state_size)
                                                        + torch.ones(state_size, state_size),
                                                        **meta_prior_kwargs)


    # Observation model
    n_observations = observation_model_kwargs["obs_1"]["n_observations"]
    mean_return = torch.zeros(n_observations, dtype=torch.float32)
    factor_loadings = torch.randn(n_observations, state_size, dtype=torch.float32)
    residual_covariance = observation_model_kwargs["obs_1"]["residual_variance"] * torch.eye(n_observations, dtype=torch.float32)

    observation_model = {
        "obs_1": FactorStructureStochasticVolatility(mean_return, factor_loadings, residual_covariance)
    }

    # Complete distribution
    complete_distribution = CompleteDistribution(meta_prior, **observation_model)

    # Distribution transformer
    d_model = transformer_kwargs["d_model"]
    component_embedding = ComponentEmbedding(state_size=state_size, d_model=d_model, **component_embedding_kwargs)
    observation_embedding = {key: ObservationEmbedding(d_model=d_model, observation_size=n_observations, **kwargs)
                             for key, kwargs in observation_embedding_kwargs.items()}
    model = DistributionTransformer(component_embedding=component_embedding,
                                    transformer_kwargs=transformer_kwargs,
                                    n_components=n_components,
                                    prior_embedding=None,
                                    sample_space_transform=None,
                                    **observation_embedding)

    if load_path is not None:
        model.load_state_dict(torch.load(Path('experiments/runs/factor_stochastic_volatility/') / load_path, weights_only=True))
    else:
        model, _ = train(model, complete_distribution, _run=_run, **training_kwargs)

    state_transition_matrix = torch.diag_embed(torch.tensor(motion_model_kwargs["delta"], dtype=torch.float32))
    process_noise_scale_cholesky = torch.diag_embed(torch.tensor(motion_model_kwargs["sigma"], dtype=torch.float32))
    constant_vector = torch.tensor(motion_model_kwargs["alpha"], dtype=torch.float32) * (1 - torch.tensor(motion_model_kwargs["delta"], dtype=torch.float32))

    motion_model = LTIMotionModel(state_transition_matrix,
                                  process_noise_scale_cholesky,
                                  MultivariateNormal(
                                      constant_vector,
                                      torch.eye(len(constant_vector), dtype=torch.float32)),
                                  constant_vector)

    test_lti_filter(model, motion_model, observation_model, _run=_run, **testing_kwargs)
