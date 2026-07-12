"""
Experiment evaluating method on problem of finding posterior for GP hyperparameters
"""

import torch
from torch import Tensor
from torch.distributions import Normal, MultivariateNormal, constraints
from torch.distributions.utils import lazy_property
from torch.types import _size

from gpytorch import add_jitter
from gpytorch.kernels import RBFKernel, ScaleKernel


from distributions.distributions import (InverseGammaMetaPrior, ObservationModel, CompleteDistribution,
                                         GaussianMixtureModel)
from model.distribution_transformer import DistributionTransformer
from workflows.train import train
from workflows.test import test_conjugate_prior
from model.embeddings import ComponentEmbedding, GammaEmbedding, ObservationEmbedding


class GPHyperparameterObservationModel(ObservationModel):
    arg_constraints = {
        "x_covariance_matrix": constraints.positive_definite,
    }

    def __init__(self, dataset_size: int, x_loc: Tensor = torch.zeros(1), x_covariance_matrix: Tensor = torch.eye(1),
                 output_scale: Tensor = torch.ones(torch.Size())):
        """
        Dataset drawn from marginal GP for y over points x drawn from normal distribution,
        conditioned on scale parameter.

        Args:
            dataset_size: Dataset size.
            x_loc: Mean of distribution over x.
                Defaults to 0.
            x_covariance_matrix: Variance of distribution over x.
                Defaults to 1.
            output_scale: Variance of marginal distribution over y.
                Defaults to 1.
        """
        super().__init__()
        self.dataset_size = dataset_size
        self.x_distribution = MultivariateNormal(loc=x_loc, covariance_matrix=x_covariance_matrix)
        self.kernel = RBFKernel()
        self.output_scale = output_scale
        self.n_observations = x_loc.shape[-1] + 1

    def condition_(self, x: Tensor) -> None:
        """
        Condition observation model on lengthscale.

        Args:
            x: Lengthscale.

        """
        self.kernel.lengthscale = x

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        kernel = ScaleKernel(self.kernel)
        kernel.outputscale = self.output_scale
        x = self.x_distribution.sample(sample_shape + (self.dataset_size,))
        y = MultivariateNormal(loc=torch.zeros(self.dataset_size),
                               covariance_matrix=add_jitter(kernel(x).to_dense())).sample()
        return torch.hstack([x, y.unsqueeze(-1)])

    @lazy_property
    def x_covariance_matrix(self):
        return self.x_covariance_matrix
