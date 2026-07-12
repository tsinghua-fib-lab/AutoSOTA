"""
Experiment to validate method against closed form posterior of GMM prior with linear Gaussian observations
"""

from torch.distributions import Bernoulli

from distributions.distributions import (GaussianMixtureModelConjugateMetaPrior, NormalisedDatasetGLMObservationModel,
                                         CompleteDistribution)
from model.embeddings import ComponentEmbedding, ObservationEmbedding
from model.distribution_transformer import DistributionTransformer
from workflows.train import train
from workflows.test import test


def run(n_components: int,
        meta_prior_kwargs: dict,
        observation_model_kwargs: dict,
        component_embedding_kwargs: dict,
        observation_embedding_kwargs: dict[str, dict],
        transformer_kwargs: dict,
        training_kwargs: dict,
        testing_kwargs: dict,
        _run=None,
        *args, **kwargs):
    """
    Run an experiment comparing distribution transformers to the closed form posterior of a GMM prior under linear
    Gaussian observations.

    Args:
        n_components: Number of GMM components.
        meta_prior_kwargs: Dictionary of parameters for the meta prior.
        observation_model_kwargs: Dictionary of dictionaries of observation model kwargs.
        component_embedding_kwargs: Dictionary of component embedding parameters.
        observation_embedding_kwargs: Dictionary of dictionaries of observation embedding parameters.
        transformer_kwargs: Dictionary of parameters for the transformer model.
        training_kwargs: Dictionary of parameters for the training routine.
        testing_kwargs: Dictionary of parameters for the testing routine.
        _run: Sacred run object.

    Returns:

    """

    state_size = list(observation_model_kwargs.values())[0]["n_features"] + 1

    # Meta-prior
    meta_prior = GaussianMixtureModelConjugateMetaPrior(state_size=state_size, n_components=n_components,
                                                        **meta_prior_kwargs)

    # Observation model
    observation_model = {key: NormalisedDatasetGLMObservationModel(distribution=Bernoulli,
                                                                   inverse_link=lambda x: {"logits": x},
                                                                   **kwargs)
                         for key, kwargs in observation_model_kwargs.items()}

    # Complete distribution
    complete_distribution = CompleteDistribution(meta_prior, **observation_model)

    # Distribution transformer
    d_model = transformer_kwargs["d_model"]
    component_embedding = ComponentEmbedding(state_size=state_size, d_model=d_model, **component_embedding_kwargs)
    observation_embedding = {key: ObservationEmbedding(d_model=d_model, observation_size=state_size,
                                                       sequential=True, **kwargs)
                             for key, kwargs in observation_embedding_kwargs.items()}
    model = DistributionTransformer(component_embedding=component_embedding,
                                    transformer_kwargs=transformer_kwargs,
                                    n_components=n_components,
                                    prior_embedding=None,
                                    sample_space_transform=None,
                                    **observation_embedding)

    model, last_epoch_metrics = train(model, complete_distribution, _run=_run, **training_kwargs)

    test(model, n_components, complete_distribution, _run=_run, **testing_kwargs)
