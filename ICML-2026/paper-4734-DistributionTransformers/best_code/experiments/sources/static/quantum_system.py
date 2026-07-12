"""
Experiment inferring parameters of an LTI mixed Pauli X and Z system.
"""

import torch
from torch import Tensor
from torch.distributions import Bernoulli, Normal, MultivariateNormal
from torch.types import _size

from typing import Optional

from distributions.distributions import BetaMetaPrior, ObservationModel, CompleteDistribution
from model.distribution_transformer import DistributionTransformer
from model.embeddings import ComponentEmbedding, GammaEmbedding, ObservationEmbedding
from workflows.train import train
from workflows.test import test_quantum


class QuantumSystemObservationModel(ObservationModel):
    _validate_args = False

    sigma_x = torch.tensor([[0, 1], [1, 0]])    # Pauli X matrix
    sigma_z = torch.tensor([[1, 0], [0, -1]])   # Pauli Z matrix

    def __init__(self, initial_state_loc: Tensor,
                 initial_state_covariance_matrix: Tensor,
                 t_loc: float = 1,
                 t_std: float = 0.05):
        """
        Observation model based on a simple 2 level quantum system undergoing evolution for t * h_bar seconds according
        to the Hamiltonian H = delta * sigma_x + (1 - delta) * sigma_z where sigma_x and sigma_z are the Pauli X and Z
        matrices respectively.

        Args:
            initial_state_loc: Mean of initial quantum state, expressed as a real vector.
            initial_state_covariance_matrix: Covariance matrix of initial quantum state.
            t_loc: Mean of evolution time normalised by the reduced Planck's constant.
                Defaults to 1 J^-1.
            t_std: Standard deviation of evolution time normalised by the reduced Planck's constant.
                Defaults to 0.05.

        """
        super().__init__()
        self.initial_state_dist = MultivariateNormal(initial_state_loc, initial_state_covariance_matrix)
        self.t_dist = Normal(t_loc, t_std)
        self.delta: Optional[Tensor] = None

    def condition_(self, x: Tensor) -> None:
        """
        Condition on a value for delta.

        Args:
            x: Delta for Hamiltonian.

        """
        self.device = x.device
        self.delta = x.cpu().sigmoid().unsqueeze(-1).unsqueeze(-1)

    def _distribution(self, sample_shape: _size = torch.Size()) -> Bernoulli:
        extended_sample_shape = sample_shape + self.delta.shape[:-2]
        initial_state = self.initial_state_dist.sample(extended_sample_shape)
        initial_state /= (initial_state.abs() ** 2).sum(dim=-1, keepdim=True).sqrt()
        t = self.t_dist.sample(extended_sample_shape).reshape(*extended_sample_shape, 1, 1)
        sigma_x = self.sigma_x
        sigma_z = self.sigma_z
        hamiltonian = self.delta * sigma_x + (1 - self.delta) * sigma_z
        transition = torch.matrix_exp(-1j * t * hamiltonian)
        state = torch.einsum("...ij,...j->...i", transition, initial_state.to(torch.complex64))
        probabilities = state.abs() ** 2
        probabilities /= probabilities.sum(dim=-1, keepdim=True)
        return Bernoulli(probs=probabilities[..., 0:1])

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        distribution = self._distribution(sample_shape)
        return distribution.sample().to(self.device)

    def log_prob(self, value: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        # Stochastic approximation
        distribution = self._distribution((n_samples,))
        log_prob = distribution.log_prob(value.cpu()).logsumexp(dim=0) - torch.tensor(n_samples).log()
        return log_prob.to(value.device)


def run(n_components: int,
        meta_prior_kwargs: dict,
        initial_states_loc: dict[str, list[float]],
        initial_states_covariance_matrix: dict[str, list[list[float]]],
        measurement_times_loc: dict[str, float],
        measurement_times_std: dict[str, float],
        distribution_embedding_kwargs: dict,
        component_embedding_kwargs: dict,
        observation_embedding_kwargs: dict[str, dict],
        transformer_kwargs: dict,
        training_kwargs: dict,
        testing_kwargs: dict,
        _run=None,
        *args, **kwargs):
    """
    Run an experiment inferring the parameter of a simple quantum system.

    Args:
        n_components: Number of GMM components.
        meta_prior_kwargs: Dictionary of parameters for the meta prior.
        initial_states_loc: Dictionary of mean initial states for observations.
        initial_states_covariance_matrix: Dictionary of covariance matrix for initial states.
        measurement_times_loc: Dictionary of mean measurement times for observations.
        measurement_times_std: Dictionary of standard deviation for measurement times.
        distribution_embedding_kwargs: Dictionary of distribution embedding parameters.
        component_embedding_kwargs: Dictionary of component embedding parameters.
        observation_embedding_kwargs: Dictionary of dictionaries of observation embedding parameters.
        transformer_kwargs: Dictionary of parameters for the transformer model.
        training_kwargs: Dictionary of parameters for the training routine.
        testing_kwargs: Dictionary of parameters for the testing routine.
        _run: Sacred run object.

    Returns:

    """

    # Meta-prior
    meta_prior = BetaMetaPrior(**meta_prior_kwargs)

    # Observation model
    observation_model = {key: QuantumSystemObservationModel(torch.tensor(initial_state),
                                                            torch.tensor(initial_states_covariance_matrix[key]),
                                                            measurement_times_loc[key],
                                                            measurement_times_std[key])
                         for key, initial_state in initial_states_loc.items()}

    complete_distribution = CompleteDistribution(meta_prior, **observation_model)

    d_model = transformer_kwargs["d_model"]
    prior_embedding = GammaEmbedding(d_model=d_model, n_components=n_components, **distribution_embedding_kwargs)
    component_embedding = ComponentEmbedding(state_size=1, d_model=d_model, **component_embedding_kwargs)
    observation_embedding = {key: ObservationEmbedding(d_model=d_model, observation_size=1, **kwargs)
                             for key, kwargs in observation_embedding_kwargs.items()}
    model = DistributionTransformer(component_embedding=component_embedding,
                                    transformer_kwargs=transformer_kwargs,
                                    n_components=n_components,
                                    prior_embedding=prior_embedding,
                                    sample_space_transform=torch.logit,
                                    **observation_embedding)

    model, last_epoch_metrics = train(model, complete_distribution, _run=_run, **training_kwargs)

    def bounds_func(phi: dict[str, Tensor]) -> tuple[float, float]:
        return 1e-6, 1.-1e-6

    test_quantum(model, complete_distribution, inverse_transform=torch.sigmoid, bounds_func=bounds_func,
         _run=_run, **testing_kwargs)
