"""
Experiment to validate method against closed form posterior of inverse prior with linear Gaussian observations
"""

import torch
from torch import Tensor
from torch.func import vmap, jacrev

from typing import Union, Literal
import matplotlib.pyplot as plt

from distributions.distributions import (InverseGammaMetaPrior, MappedScaleGaussianObservationModel,
                                         GaussianMixtureModel, CompleteDistribution)
from distributions.utils import decode_gmm_sample
from model.distribution_transformer import DistributionTransformer
from competitor_methods.ace_dt_morphology import distribution_transformer_factory
from workflows.train import train
from workflows.test import test_conjugate_prior
from model.embeddings import ComponentEmbedding, GammaEmbedding, ObservationEmbedding


def run(n_components: int,
        meta_prior_kwargs: dict,
        observation_loc: dict[str, list[float]],
        distribution_embedding_kwargs: dict,
        component_embedding_kwargs: dict,
        observation_embedding_kwargs: dict[str, dict],
        transformer_kwargs: dict,
        training_kwargs: dict,
        testing_kwargs: dict,
        kind: Literal["DistributionTransformer", "DistributionTransformerWithEncoder",
                  "SingleChannelDistributionTransformer", "LatentDecodedDistributionTransformer"] = "DistributionTransformer",
        _run=None,
        *args, **kwargs) -> None:
    """
    Run an experiment comparing distribution transformers to the closed form posterior of a GMM prior under linear
    Gaussian observations.

    Args:
        n_components: Number of GMM components.
        meta_prior_kwargs: Dictionary of parameters for the meta prior.
        observation_loc: Dictionary of means of observation distributions.
        distribution_embedding_kwargs: Dictionary of distribution embedding parameters.
        component_embedding_kwargs: Dictionary of component embedding parameters.
        observation_embedding_kwargs: Dictionary of dictionaries of observation embedding parameters.
        transformer_kwargs: Dictionary of parameters for the transformer model.
        training_kwargs: Dictionary of parameters for the training routine.
        testing_kwargs: Dictionary of parameters for the testing routine.
        kind: Kind of distribution transformer to use.
        _run: Sacred run object.

    """

    meta_prior = InverseGammaMetaPrior(**meta_prior_kwargs)

    observation_model = {key: MappedScaleGaussianObservationModel(torch.tensor(loc, dtype=torch.float32),
                                                                  scale_parametrisation="covariance_matrix",
                                                                  mapping=None)
                         for key, loc in observation_loc.items()}

    complete_distribution = CompleteDistribution(meta_prior, **observation_model)

    d_model = transformer_kwargs["d_model"]

    prior_embedding = GammaEmbedding(
        d_model=d_model,
        n_components=1 if kind in ["SingleChannelDistributionTransformer", "LatentDecodedDistributionTransformer"] else n_components,
        **distribution_embedding_kwargs
    )

    model = distribution_transformer_factory(
        kind=kind,
        n_components=n_components,
        state_size=1,
        component_embedding_kwargs=component_embedding_kwargs,
        observation_embedding_kwargs=observation_embedding_kwargs,
        transformer_decoder_kwargs=transformer_kwargs,
        transformer_encoder_kwargs=transformer_kwargs,
        prior_embedding=prior_embedding,
        sample_space_transform=torch.log
    )

    model, _ = train(model, complete_distribution, _run=_run, **training_kwargs)

    def conjugacy_update(phi: dict[str, Tensor],
                         z: dict[str, Tensor],
                         device: str
                         ) -> dict[str, Tensor]:
        return {
            "concentration": phi["concentration"] + len(z) / 2,
            "rate": phi["rate"] + sum((z[key].squeeze() - torch.tensor(observation_loc[key]).to(device).squeeze()) ** 2
                                      for key in z) / 2
        }

    def bounds_func(params: dict[str, Tensor]) -> tuple[float, float]:
        concentration = params["concentration"].item()
        rate = params["rate"].item()
        return 1e-6, 4 * rate / concentration + 1 / rate

    if "test_meta_prior_kwargs" in testing_kwargs:
        test_meta_prior = InverseGammaMetaPrior(**testing_kwargs["test_meta_prior_kwargs"])
        complete_distribution = CompleteDistribution(test_meta_prior, **observation_model)

    test_conjugate_prior(model, complete_distribution, conjugacy_update,
                         bounds_func=bounds_func,
                         inverse_transform=torch.exp, _run=_run, **testing_kwargs)
