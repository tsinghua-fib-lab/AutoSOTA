"""
Experiment testing method on time series filtering
"""

import torch
from torch.distributions import MultivariateNormal

from typing import Optional
import matplotlib.pyplot as plt

from distributions.distributions import (GaussianMixtureModelConjugateMetaPrior, CompleteDistribution,
                                         RangefinderObservationModel, GaussianAngleObservationModel)
from model.embeddings import ComponentEmbedding, ObservationEmbedding
from model.distribution_transformer import DistributionTransformer
from workflows.train import train
from workflows.test import test_lti_filter
from dynamic.motion_models import LTIMotionModel
from pathlib import Path


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
    Run an experiment comparing distribution transformers to the closed form posterior of a GMM prior under linear
    Gaussian observations.

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
    observation_model = {
        "obs_1": RangefinderObservationModel(torch.tensor(observation_model_kwargs["obs_1"]["scale"]),
                                             torch.tensor(observation_model_kwargs["obs_1"]["rate"]),
                                             torch.tensor(observation_model_kwargs["obs_1"]["max_range"]),
                                             torch.tensor(observation_model_kwargs["obs_1"]["weights"])),
        "obs_2": GaussianAngleObservationModel(torch.tensor(observation_model_kwargs["obs_2"]["scale"]))
    }

    # Plot observation model
    obs = observation_model["obs_1"]
    obs.condition_(torch.tensor([10, 0., 0., 0.]))

    plt.style.use(['seaborn-v0_8-paper'])

    points = torch.linspace(0, 20, steps=1000)
    p_density = torch.exp(obs.log_prob(points))
    fig, ax = plt.subplots()
    ax.plot(points, p_density)

    ax.set_ylabel("Probability Density")
    ax.set_xlabel("Range (km)")
    plt.show()
    fig.savefig(_run.observers[0].dir + "\\rangefinder_observation_model.pdf", format="pdf")

    # Complete distribution
    complete_distribution = CompleteDistribution(meta_prior, **observation_model)

    # Distribution transformer
    d_model = transformer_kwargs["d_model"]
    component_embedding = ComponentEmbedding(state_size=state_size, d_model=d_model, **component_embedding_kwargs)
    observation_embedding = {key: ObservationEmbedding(d_model=d_model, observation_size=1, **kwargs)
                             for key, kwargs in observation_embedding_kwargs.items()}
    model = DistributionTransformer(component_embedding=component_embedding,
                                    transformer_kwargs=transformer_kwargs,
                                    n_components=n_components,
                                    prior_embedding=None,
                                    sample_space_transform=None,
                                    **observation_embedding)

    if load_path is not None:
        model.load_state_dict(torch.load(Path('experiments/runs/lti_filter/') / load_path, weights_only=True))
    else:
        model, _ = train(model, complete_distribution, _run=_run, **training_kwargs)

    motion_model = LTIMotionModel(torch.tensor(motion_model_kwargs["state_transition_matrix"]),
                                  torch.tensor(motion_model_kwargs["process_noise_scale_cholesky"]),
                                  MultivariateNormal(
                                      torch.tensor(motion_model_kwargs["x0_loc"]),
                                      torch.tensor(motion_model_kwargs["x0_covariance_matrix"])
                                  ))

    test_lti_filter(model, motion_model, observation_model, _run=_run, **testing_kwargs)
