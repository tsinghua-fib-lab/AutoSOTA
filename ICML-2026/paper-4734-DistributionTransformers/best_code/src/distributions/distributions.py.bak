"""
Custom distributions used by the model for meta-priors, priors, likelihoods and sampling
"""

import torch
from torch import Tensor
from torch.distributions import (Distribution, Wishart, MultivariateNormal, Independent, Categorical,
                                 MixtureSameFamily, Dirichlet, InverseGamma, Beta, Normal, Exponential,
                                 Uniform, constraints)
from torch.distributions.utils import lazy_property
from torch.types import _size

from typing import Optional, Callable

from distributions.utils import decode_gmm_sample, encode_gmm_sample


class GaussianMixtureModel(MixtureSameFamily):
    arg_constraints = {
        "weights": constraints.simplex,
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector

    def __init__(self, weights: Tensor,
                 loc: Tensor,
                 covariance_matrix: Optional[Tensor] = None,
                 precision_matrix: Optional[Tensor] = None,
                 scale_tril: Optional[Tensor] = None,
                 validate_args: bool = True):
        """
        Gaussian Mixture Model distribution. Supports sampling with different parameters in dimension 0.

        Example - no batching:
            >>> weights = torch.ones(3) / 3
            >>> loc = torch.zeros(3, 2, dtype=torch.float32)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32).broadcast_to(3, 2, 2)
            >>> gmm = GaussianMixtureModel(weights, loc, covariance_matrix)
            >>> print(gmm.sample())

        Example - batching:
            >>> weights = torch.ones(4, 3) / 3
            >>> loc = torch.zeros(4, 3, 2, dtype=torch.float32)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32).broadcast_to(4, 3, 2, 2)
            >>> gmm = GaussianMixtureModel(weights, loc, covariance_matrix)
            >>> print(gmm.sample())

        Args:
            weights: Weighting of each mixture component. Must all be positive and sum to 1.
            loc: Tensor of means of each mixture component.
            covariance_matrix: Tensor of covariance matrices of each mixture component. Must all be positive definite.
            precision_matrix:  Tensor of precision matrices of each mixture component. Must all be positive definite.
            scale_tril: Tensor of lower triangular representation of scale matrix, i.e. Cholesky decomposition of
                covariance matrix. Must have positive diagonal elements.
            validate_args: Whether to validate model parameters obey constraints.
        """
        super().__init__(Categorical(weights, validate_args=False),
                         Independent(MultivariateNormal(loc,
                                                        covariance_matrix=covariance_matrix,
                                                        precision_matrix=precision_matrix,
                                                        scale_tril=scale_tril,
                                                        validate_args=validate_args),
                                     0),
                         validate_args=validate_args)
        self.weights = weights
        self.loc = loc
        self.covariance_matrix = self.component_distribution.base_dist.covariance_matrix
        self.precision_matrix = self.component_distribution.base_dist.precision_matrix
        self.scale_tril = self.component_distribution.base_dist.scale_tril
        self.n_components = weights.shape[-1]
        self.state_size = loc.shape[-1]
        if (covariance_matrix is not None) + (scale_tril is not None) + (
                precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.scale_parametrisation = "scale_tril"
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.scale_parametrisation = "covariance_matrix"
        elif precision_matrix is not None:
            if precision_matrix.dim() < 2:
                raise ValueError(
                    "precision_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.scale_parametrisation = "precision_matrix"

    @lazy_property
    def weights(self):
        return self.weights

    @lazy_property
    def loc(self):
        return self.loc

    @lazy_property
    def covariance_matrix(self):
        return self.covariance_matrix

    @lazy_property
    def precision_matrix(self):
        return self.precision_matrix

    @lazy_property
    def scale_tril(self):
        return self.scale_tril


class MetaPrior(Distribution):
    """
    Abstract base class of meta-priors p(phi) where phi parametrises a prior over x, that is to say priors over priors.
    """

    def __init__(self, prior: type[Distribution]):
        super().__init__()
        self.prior = prior
        self.prior_size: Optional[int] = None

    def decode_sample(self, sample: Tensor) -> dict[str, Tensor]:
        """
        Decode tensor of sampled parameters to dictionary of tensors keyed by parameter.

        Args:
            sample: Sampled tensor.

        Returns:
            Decoded sample.
        """
        raise NotImplementedError

    def encode_sample(self, decoded_sample: dict[str, Tensor]) -> Tensor:
        """
        Encode dictionary of sampled parameters to a singular tensor. Inverse operation of decode_sample.

        Args:
            decoded_sample:  Dictionary of decoded sample.

        Returns:
            Tensor encoding sample.
        """
        raise NotImplementedError


class GaussianMixtureModelConjugateMetaPrior(MetaPrior):
    arg_constraints = {
        "weights_concentration": constraints.simplex,
        "loc_loc": constraints.real_vector,
        "loc_covariance_matrix": constraints.positive_definite,
        "loc_precision_matrix": constraints.positive_definite,
        "loc_scale_tril": constraints.lower_cholesky,
        "scale_df": constraints.greater_than(0),
        "scale_loc": constraints.real_vector,
        "scale_covariance_matrix": constraints.positive_definite,
        "scale_precision_matrix": constraints.positive_definite,
        "scale_scale_tril": constraints.lower_cholesky,
        "scale_eps": constraints.greater_than_eq(0)
    }

    def __init__(self,
                 weights_concentration: Optional[Tensor] = None,
                 loc_loc: Optional[Tensor] = None,
                 loc_covariance_matrix: Optional[Tensor] = None,
                 loc_precision_matrix: Optional[Tensor] = None,
                 loc_scale_tril: Optional[Tensor] = None,
                 scale_parametrisation: Optional[str] = None,
                 scale_df: Optional[Tensor] = None,
                 scale_covariance_matrix: Optional[Tensor] = None,
                 scale_precision_matrix: Optional[Tensor] = None,
                 scale_scale_tril: Optional[Tensor] = None,
                 scale_eps: Optional[float] = None,
                 n_components: Optional[int] = None,
                 state_size: Optional[int] = None,
                 default_scale_multiple: float = 1.
                 ):
        """
        Meta-prior for a Gaussian Mixture Model. Uses conjugate priors for all parameters. Each parameter is referred to
        as for a single distribution, but must be provided as a tensor of parameters for each component.

        Example - default values:

            >>> gmm_conjugate_meta_prior = GaussianMixtureModelConjugateMetaPrior(n_components=3, state_size=2)
            >>> sample = gmm_conjugate_meta_prior.sample((1,))
            >>> print(gmm_conjugate_meta_prior.decode_sample(sample))

        Args:
            weights_concentration: Concentration parameter for Dirichlet prior over weights.
                Defaults to uniform.
            loc_loc: Loc parameter for Multivariate Normal prior over loc.
                Defaults to zero for all components.
            loc_covariance_matrix: Covariance matrix parameter for Multivariate Normal prior over loc.
                Defaults to identity for all components.
            loc_precision_matrix: Precision matrix parameter for Multivariate Normal prior over loc.
                Defaults to identity for all components.
            loc_scale_tril: Lower triangular scale parameter for Multivariate Normal prior over loc.
                Defaults to identity for all components.
            scale_parametrisation: Parametrisation used for scale parameter, common between all components.
                Defaults to "covariance_matrix".
            scale_df: Degrees of freedom of Wishart prior for precision_matrix.
                Defaults to state_size + 1 for all components.
            scale_covariance_matrix: Covariance matrix of Wishart prior for precision_matrix.
                Defaults to identity for all components.
            scale_precision_matrix: Precision matrix of Wishart prior for precision_matrix.
                Defaults to identity for all components.
            scale_scale_tril: Lower triangular scale matrix of Wishart prior for precision_matrix.
                Defaults to identity for all components.
            scale_eps: Small constant to add to diagonal of sampled scale matrix.
                Defaults to 1e-6.
            n_components: Number of components in mixture model. Only specify if using default value for
                weights_concentration else overridden.
            state_size: Size of state in mixture model. Only specify if using default value for loc_loc else overridden.
            default_scale_multiple: Multiplier on default scale parameters.
                Defaults to 1.
        """
        super().__init__(GaussianMixtureModel)

        if n_components is not None or state_size is not None:
            assert n_components is not None and state_size is not None, \
                "both n_components and state_size must be specified if using default values"
        self.n_components = weights_concentration.shape[-1] if weights_concentration is not None else n_components
        self.state_size = loc_loc.shape[-1] if loc_loc is not None else state_size
        self.weights_concentration = torch.ones(self.n_components) / self.n_components \
            if weights_concentration is None else weights_concentration
        self.loc_loc = torch.zeros((self.n_components, self.state_size), dtype=torch.float32) \
            if loc_loc is None else loc_loc
        if loc_covariance_matrix is None and loc_precision_matrix is None and loc_scale_tril is None:
            self.loc_covariance_matrix = (torch.eye(self.state_size, dtype=torch.float32
                                                    ).broadcast_to(self.n_components, self.state_size, self.state_size)
                                          * default_scale_multiple)
            self.loc_precision_matrix = loc_precision_matrix
            self.loc_scale_tril = loc_scale_tril
        else:
            self.loc_covariance_matrix = loc_covariance_matrix
            self.loc_precision_matrix = loc_precision_matrix
            self.loc_scale_tril = loc_scale_tril
        self.scale_parametrisation = "covariance_matrix" if scale_parametrisation is None else scale_parametrisation
        self.scale_df = (self.state_size + 1) // min(default_scale_multiple, 1) if scale_df is None else scale_df
        if scale_covariance_matrix is None and scale_precision_matrix is None and scale_scale_tril is None:
            self.scale_covariance_matrix = (torch.eye(self.state_size, dtype=torch.float32
                                                      ).broadcast_to(self.n_components, self.state_size,
                                                                     self.state_size)
                                            * default_scale_multiple)
            self.scale_precision_matrix = scale_precision_matrix
            self.scale_scale_tril = scale_scale_tril
        else:
            self.scale_covariance_matrix = scale_covariance_matrix
            self.scale_precision_matrix = scale_precision_matrix
            self.scale_scale_tril = scale_scale_tril
        self.scale_eps = 1e-6 if scale_eps is None else scale_eps
        self.prior_size = self.n_components * (1 + self.state_size + self.state_size ** 2)

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        weights = Dirichlet(self.weights_concentration).sample(sample_shape)
        loc = Independent(MultivariateNormal(self.loc_loc,
                                             covariance_matrix=self.loc_covariance_matrix,
                                             precision_matrix=self.loc_precision_matrix,
                                             scale_tril=self.loc_scale_tril
                                             ), 0).sample(sample_shape)
        precision_matrix = Independent(Wishart(self.scale_df,
                                               covariance_matrix=self.scale_covariance_matrix,
                                               precision_matrix=self.scale_precision_matrix,
                                               scale_tril=self.scale_scale_tril
                                               ), 0).sample(sample_shape)
        # Add jitter of magnitude scale_eps to diagonal
        precision_matrix += self.scale_eps * torch.eye(self.state_size)
        match self.scale_parametrisation:
            case "covariance_matrix":
                scale = torch.linalg.inv(precision_matrix.to(torch.float64)).to(torch.float32)
            case "precision_matrix":
                scale = precision_matrix
            case "scale_tril":
                scale = torch.linalg.cholesky(precision_matrix, upper=True).inverse().mT
            case _:
                raise AssertionError('scale_parametrisation must be one of "covariance_matrix", "precision_matrix" or '
                                     '"scale_tril"')
        return torch.cat([weights.unsqueeze(-1), loc, scale.flatten(start_dim=-2)], dim=-1)

    def log_prob(self, value: Tensor) -> Tensor:
        """
        Calculate the logarithm of the probability density function evaluated at the input value.

        Args:
            value: Value to query probability density function.

        Returns:
            Log probability of value.

        """
        batch_shape = value.shape[:-2]
        params_dict = self.decode_sample(value)

        match self.scale_parametrisation:
            case "covariance_matrix":
                precision_matrix = torch.linalg.inv(params_dict[self.scale_parametrisation])
            case "precision_matrix":
                precision_matrix = params_dict[self.scale_parametrisation]
            case "scale_tril":
                covariance_matrix = torch.einsum("...ij, ...kj -> ...ik", params_dict[self.scale_parametrisation],
                                                 params_dict[self.scale_parametrisation])
                precision_matrix = torch.linalg.inv(covariance_matrix)
            case _:
                raise AssertionError('scale_parametrisation must be one of "covariance_matrix", "precision_matrix" or '
                                     '"scale_tril"')
        loc = params_dict["loc"].reshape(batch_shape + (self.n_components, self.state_size))
        precision_matrix = precision_matrix.reshape(batch_shape + (self.n_components, self.state_size,
                                                                        self.state_size))
        return (Dirichlet(self.weights_concentration).log_prob(params_dict["weights"]) +
                Independent(MultivariateNormal(self.loc_loc,
                                               covariance_matrix=self.loc_covariance_matrix,
                                               precision_matrix=self.loc_precision_matrix,
                                               scale_tril=self.loc_scale_tril
                                               ), 1).log_prob(loc) +
                Independent(Wishart(self.scale_df,
                                    covariance_matrix=self.scale_covariance_matrix,
                                    precision_matrix=self.scale_precision_matrix,
                                    scale_tril=self.scale_scale_tril
                                    ), 1).log_prob(precision_matrix))

    def decode_sample(self, sample: Tensor) -> dict[str, Tensor]:
        """
        Decode tensor of sampled parameters to dictionary of tensors keyed by GMM parameter.

        Args:
            sample: Sampled tensor.

        Returns:
            Decoded sample.

        """
        return decode_gmm_sample(sample, self.scale_parametrisation)

    def encode_sample(self, decoded_sample: dict[str, Tensor]) -> Tensor:
        """
        Encode dictionary of sampled parameters to a singular tensor. Inverse operation of decode_sample.

        Args:
            decoded_sample:  Dictionary of decoded sample.

        Returns:
            Tensor encoding sample.

        """
        return encode_gmm_sample(decoded_sample, self.scale_parametrisation)

    @lazy_property
    def weights_concentration(self):
        return self.weights_concentration

    @lazy_property
    def loc_loc(self):
        return self.loc_loc

    @lazy_property
    def loc_covariance_matrix(self):
        return self.loc_covariance_matrix

    @lazy_property
    def loc_precision_matrix(self):
        return self.loc_precision_matrix

    @lazy_property
    def loc_scale_tril(self):
        return self.loc_scale_tril

    @lazy_property
    def scale_df(self):
        return self.scale_df

    @lazy_property
    def scale_loc(self):
        return self.scale_loc

    @lazy_property
    def scale_covariance_matrix(self):
        return self.scale_covariance_matrix

    @lazy_property
    def scale_precision_matrix(self):
        return self.scale_precision_matrix

    @lazy_property
    def scale_scale_tril(self):
        return self.scale_scale_tril

    @lazy_property
    def scale_eps(self):
        return self.scale_eps


class InverseGammaMetaPrior(MetaPrior):
    arg_constraints = {
        "concentration_concentration": constraints.positive,
        "concentration_rate": constraints.positive,
        "rate_concentration": constraints.positive,
        "rate_rate": constraints.positive
    }

    def __init__(self,
                 concentration_concentration: float = 1,
                 concentration_rate: float = 1,
                 rate_concentration: int = 1,
                 rate_rate: int = 1):
        """
        Meta prior for an inverse gamma distribution.

        Args:
            concentration_concentration: Concentration parameter for inverse gamma meta-prior of concentration
                parameter.
            concentration_rate: Rate parameter for inverse gamma meta-prior of concentration parameter.
            rate_concentration: Concentration parameter for inverse gamma meta-prior of rate parameter.
            rate_rate: Rate parameter for inverse gamma meta-pior of rate parameter.

        """
        super().__init__(InverseGamma)
        self.concentration_concentration = concentration_concentration
        self.concentration_rate = concentration_rate
        self.rate_concentration = rate_concentration
        self.rate_rate = rate_rate

        self.prior_size = 2

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        concentration = InverseGamma(self.concentration_concentration, self.concentration_rate).sample(sample_shape)
        rate = InverseGamma(self.rate_concentration, self.rate_rate).sample(sample_shape)
        return torch.stack([concentration, rate], dim=-1)

    def decode_sample(self, sample: Tensor) -> dict[str, Tensor]:
        """
        Decode tensor of sampled parameters to dictionary of tensors keyed by GMM parameter.

        Args:
            sample: Sampled tensor.

        Returns:
            Decoded sample.
        """
        concentration = sample[..., 0]
        rate = sample[..., 1]
        return {"concentration": concentration,
                "rate": rate}

    def encode_sample(self, decoded_sample: dict[str, Tensor]) -> Tensor:
        """
        Encode dictionary of sampled parameters to a singular tensor. Inverse operation of decode_sample.

        Args:
            decoded_sample:  Dictionary of decoded sample.

        Returns:
            Tensor encoding sample.
        """
        concentration = decoded_sample["concentration"]
        rate = decoded_sample["rate"]
        return torch.stack([concentration, rate], dim=-1)

    @lazy_property
    def concentration_concentration(self):
        return self.concentration_concentration

    @lazy_property
    def concentration_rate(self):
        return self.concentration_rate

    @lazy_property
    def rate_concentration(self):
        return self.rate_concentration

    @lazy_property
    def rate_rate(self):
        return self.rate_rate


class BetaMetaPrior(MetaPrior):
    arg_constraints = {
        "concentration1_concentration": constraints.positive,
        "concentration1_rate": constraints.positive,
        "concentration0_concentration": constraints.positive,
        "concentration0_rate": constraints.positive
    }

    def __init__(self,
                 concentration1_concentration: float = 1,
                 concentration1_rate: float = 1,
                 concentration0_concentration: float = 1,
                 concentration0_rate: float = 1,):
        """
        Meta prior for a Beta distribution.

        Args:
            concentration1_concentration: Concentration parameter for inverse gamma meta-prior of concentration1
            parameter.
            concentration1_rate: Rate parameter for gamma meta-prior of concentration1 parameter.
            concentration0_concentration: Concentration parameter for inverse gamma meta-prior of concentration0
            parameter.
            concentration0_rate: Rate parameter for gamma meta-pior of concentration0 parameter.

        """
        super().__init__(Beta)
        self.concentration1_concentration = concentration1_concentration
        self.concentration1_rate = concentration1_rate
        self.concentration0_concentration = concentration0_concentration
        self.concentration0_rate = concentration0_rate

        self.prior_size = 2

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        concentration1 = InverseGamma(self.concentration1_concentration, self.concentration1_rate).sample(sample_shape)
        concentration0 = InverseGamma(self.concentration0_concentration, self.concentration0_rate).sample(sample_shape)
        return torch.stack([concentration1, concentration0], dim=-1)

    def decode_sample(self, sample: Tensor) -> dict[str, Tensor]:
        """
        Decode tensor of sampled parameters to dictionary of tensors keyed by GMM parameter.

        Args:
            sample: Sampled tensor.

        Returns:
            Decoded sample.
        """
        concentration1 = sample[..., 0]
        concentration0 = sample[..., 1]
        return {"concentration1": concentration1,
                "concentration0": concentration0}

    def encode_sample(self, decoded_sample: dict[str, Tensor]) -> Tensor:
        """
        Encode dictionary of sampled parameters to a singular tensor. Inverse operation of decode_sample.

        Args:
            decoded_sample:  Dictionary of decoded sample.

        Returns:
            Tensor encoding sample.
        """
        concentration1 = decoded_sample["concentration1"]
        concentration0 = decoded_sample["concentration0"]
        return torch.stack([concentration1, concentration0], dim=-1)

    @lazy_property
    def concentration1_concentration(self):
        return self.concentration1_concentration

    @lazy_property
    def concentration1_rate(self):
        return self.concentration1_rate

    @lazy_property
    def concentration0_concentration(self):
        return self.concentration0_concentration

    @lazy_property
    def concentration0_rate(self):
        return self.concentration0_rate


class ObservationModel(Distribution):

    def __init__(self):
        """
        Abstract base class of observation models p(z|x).
        """
        super().__init__()
        self.distribution: Optional[Distribution] = None
        self.n_observations: Optional[int] = None
        self.mapping = torch.nn.Identity()
        self.device: torch.device = torch.device("cpu")

    def condition_(self, x: Tensor):
        """
        Condition observation distribution on state in place.

        Args:
            x: State to condition sample on. Can be batched or not.

        """
        raise NotImplementedError

    def conditional_mean(self, x: Tensor):
        """
        Get mean of distribution conditioned on state. Also conditions self in place.
        Args:
            x: State on which to condition.

        Returns:
            Mean of distribution conditioned on x.

        """
        self.condition_(x)
        return self.mean.unsqueeze(-1)

    def conditional_variance(self, x: Tensor):
        """
        Get variance of distribution conditioned on state. Also conditions self in place.
        Args:
            x: State on which to condition.

        Returns:
            Variance of distribution conditioned on x.

        """
        self.condition_(x)
        return self.variance.unsqueeze(-1)

    @property
    def mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def variance(self) -> torch.Tensor:
        return self.distribution.variance

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        sample = self.distribution.sample(sample_shape=sample_shape).to(self.device)
        if self.distribution.event_shape == torch.Size():
            sample = sample.unsqueeze(-1)
        return sample

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        device = value.device
        return self.distribution.log_prob(value.to(self.mean.device)).to(device)


class DirectGaussianObservationModel(ObservationModel):
    arg_constraints = {
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }

    def __init__(self,
                 covariance_matrix: Optional[Tensor] = None,
                 precision_matrix: Optional[Tensor] = None,
                 scale_tril: Optional[Tensor] = None):
        """
        Observation model for direction observation of state subject to Gaussian noise.

        Example - no batching:
            >>> x = torch.ones(2, dtype=torch.float32)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32)
            >>> dgom = DirectGaussianObservationModel(covariance_matrix=covariance_matrix)
            >>> dgom.condition_(x)
            >>> print(dgom.sample())

        Example - batching:
            >>> x = torch.ones((2, 2), dtype=torch.float32)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32).broadcast_to(2, 2, 2)
            >>> dgom = DirectGaussianObservationModel(covariance_matrix=covariance_matrix)
            >>> dgom.condition_(x)
            >>> print(dgom.sample())

        Args:
            covariance_matrix: Covariance matrix for Gaussian noise.
            precision_matrix: Precision matrix for Gaussian noise.
            scale_tril: Lower triangular scale parameter for Gaussian noise.

        """
        super().__init__()
        if (covariance_matrix is not None) + (scale_tril is not None) + (
                precision_matrix is not None
        ) != 1:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError(
                    "scale_tril matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.scale_parametrisation = "scale_tril"
            self.n_observations = scale_tril.shape[-1]
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError(
                    "covariance_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.scale_parametrisation = "covariance_matrix"
            self.n_observations = covariance_matrix.shape[-1]
        elif precision_matrix is not None:
            if precision_matrix.dim() < 2:
                raise ValueError(
                    "precision_matrix must be at least two-dimensional, "
                    "with optional leading batch dimensions"
                )
            self.scale_parametrisation = "precision_matrix"
            self.n_observations = precision_matrix.shape[-1]

        self.distribution: Optional[MultivariateNormal] = None
        self.covariance_matrix = covariance_matrix
        self.precision_matrix = precision_matrix
        self.scale_tril = scale_tril

    def condition_(self, x: Tensor):
        """
        Condition observation distribution on state.
        Args:
            x: State to condition sample on. Can be batched or not.

        """
        self.device = x.device
        self.distribution = MultivariateNormal(loc=x.cpu(), covariance_matrix=self.covariance_matrix,
                                               precision_matrix=self.precision_matrix, scale_tril=self.scale_tril)

    @lazy_property
    def covariance_matrix(self):
        return self.covariance_matrix

    @lazy_property
    def precision_matrix(self):
        return self.precision_matrix

    @lazy_property
    def scale_tril(self):
        return self.scale_tril


class MappedGaussianObservationModel(DirectGaussianObservationModel):
    arg_constraints = {
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }

    def __init__(self,
                 covariance_matrix: Optional[Tensor] = None,
                 precision_matrix: Optional[Tensor] = None,
                 scale_tril: Optional[Tensor] = None,
                 mapping: Optional[Callable[[Tensor], Tensor]] = None):
        """
        Observation model for observation of mapping of state subject to Gaussian noise.

        Example - no batching:
            >>> x = torch.ones(2, dtype=torch.float32)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32)
            >>> mapping = lambda x: x ** 2
            >>> mgom = MappedGaussianObservationModel(covariance_matrix=covariance_matrix, mapping=mapping)
            >>> mgom.condition_(x)
            >>> print(mgom.sample())

        Example - batching:
            >>> x = torch.ones((2, 2), dtype=torch.float32)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32).broadcast_to(2, 2, 2)
            >>> mapping = lambda x: x ** 3
            >>> mgom = MappedGaussianObservationModel(covariance_matrix=covariance_matrix, mapping=mapping)
            >>> mgom.condition_(x)
            >>> print(mgom.sample())

        Args:
            covariance_matrix: Covariance matrix for Gaussian noise.
            precision_matrix: Precision matrix for Gaussian noise.
            scale_tril: Lower triangular scale parameter for Gaussian noise.
            mapping: Mapping from state to observation.
                Defaults to identity.

        """
        super().__init__(covariance_matrix, precision_matrix, scale_tril)
        self.mapping = torch.nn.Identity() if mapping is None else mapping

    def condition_(self, x: Tensor):
        self.device = x.device
        self.distribution = MultivariateNormal(loc=self.mapping(x.cpu()), covariance_matrix=self.covariance_matrix,
                                               precision_matrix=self.precision_matrix, scale_tril=self.scale_tril)


class LinearGaussianObservationModel(MappedGaussianObservationModel):
    arg_constraints = {
        "covariance_matrix": constraints.positive_definite,
        "precision_matrix": constraints.positive_definite,
        "scale_tril": constraints.lower_cholesky,
    }

    def __init__(self,
                 observation_matrix: Tensor = None,
                 covariance_matrix: Optional[Tensor] = None,
                 precision_matrix: Optional[Tensor] = None, scale_tril: Optional[Tensor] = None):
        """
        Observation model for observation of mapping of state subject to Gaussian noise.

        Example - no batching:
            >>> x = torch.ones(2, dtype=torch.float32)
            >>> covariance_matrix = torch.eye(3, dtype=torch.float32)
            >>> observation_matrix = torch.ones((3, 2), dtype=torch.float32)
            >>> mgom = LinearGaussianObservationModel(observation_matrix, covariance_matrix=covariance_matrix)
            >>> mgom.condition_(x)
            >>> print(mgom.sample())

        Example - batching:
            >>> x = torch.ones((2, 2), dtype=torch.float32)
            >>> covariance_matrix = torch.eye(3, dtype=torch.float32).broadcast_to(2, 3, 3)
            >>> observation_matrix = torch.ones((3, 2), dtype=torch.float32)
            >>> mgom = LinearGaussianObservationModel(observation_matrix, covariance_matrix=covariance_matrix)
            >>> mgom.condition_(x)
            >>> print(mgom.sample())

        Args:
            observation_matrix: Matrix mapping from state to observation.
            covariance_matrix: Covariance matrix for Gaussian noise.
            precision_matrix: Precision matrix for Gaussian noise.
            scale_tril: Lower triangular scale parameter for Gaussian noise.

        """
        if covariance_matrix is not None:
            n_obs = covariance_matrix.shape[-1]
        elif precision_matrix is not None:
            n_obs = precision_matrix.shape[-1]
        elif scale_tril is not None:
            n_obs = scale_tril.shape[-1]
        else:
            raise ValueError(
                "Exactly one of covariance_matrix or precision_matrix or scale_tril may be specified."
            )

        assert observation_matrix.dim() == 2, \
            "Observation matrix must be exactly 2 dimensional, batching is not supported."
        assert observation_matrix.shape[-2] == n_obs
        self.observation_matrix = observation_matrix
        mapping = lambda x: torch.einsum("ij,...j->...i", self.observation_matrix, x)
        super().__init__(covariance_matrix=covariance_matrix,
                         precision_matrix=precision_matrix,
                         scale_tril=scale_tril,
                         mapping=mapping)


class ScaleGaussianObservationModel(ObservationModel):
    arg_constraints = {
        "loc": constraints.real_vector
    }

    def __init__(self,
                 loc: Tensor,
                 scale_parametrisation: Optional[str] = None):
        """
        Observation model for observing state via its parametrisation of a Gaussian's scale parameter.

        Example - no batching:
            >>> loc = torch.ones(2, dtype=torch.float32)
            >>> x = torch.eye(2, dtype=torch.float32)
            >>> sgom = ScaleGaussianObservationModel(loc=loc)
            >>> sgom.condition_(x)
            >>> print(sgom.sample())

        Example - batching:
            >>> loc = torch.ones((3, 2), dtype=torch.float32)
            >>> x = torch.eye(2, dtype=torch.float32).broadcast_to((3, 2, 2))
            >>> sgom = ScaleGaussianObservationModel(loc=loc)
            >>> sgom.condition_(x)
            >>> print(sgom.sample())

        Args:
            loc: Mean of observation distribution.
            scale_parametrisation: Scale parametrisation style of Gaussian.

        """
        super().__init__()
        assert scale_parametrisation in {None, "covariance_matrix", "precision_matrix", "scale_tril"}, \
            'scale_parametrisation must be one of "covariance_matrix", "precision_matrix" or "scale_tril"'
        self.distribution: Optional[MultivariateNormal] = None
        self.loc = loc
        self.n_observations = loc.shape[-1]
        self.scale_parametrisation = "covariance_matrix" if scale_parametrisation is None else scale_parametrisation

    def condition_(self, x: Tensor):
        """
        Condition observation distribution on state.
        Args:
            x: State to condition sample on. Can be batched or not.

        """
        self.device = x.device
        self.distribution = MultivariateNormal(loc=self.loc, **{self.scale_parametrisation:
                                                                x.cpu().unsqueeze(-1).unsqueeze(-1)})

    @lazy_property
    def loc(self):
        return self.loc


class MappedScaleGaussianObservationModel(ScaleGaussianObservationModel):

    def __init__(self,
                 loc: Tensor,
                 scale_parametrisation: Optional[str] = None,
                 mapping: Optional[Callable[[Tensor], Tensor]] = None):
        """
        Observation model for observing state via its parametrisation of a Gaussian's scale parameter.

        Example - no batching:
            >>> loc = torch.ones(1, dtype=torch.float32)
            >>> mapping = torch.exp
            >>> x = torch.eye(1, dtype=torch.float32)
            >>> msgom = MappedScaleGaussianObservationModel(loc=loc, mapping=mapping)
            >>> msgom.condition_(x)
            >>> print(msgom.sample())

        Example - batching:
            >>> loc = torch.ones((3, 2), dtype=torch.float32)
            >>> x = torch.eye(2, dtype=torch.float32).broadcast_to((3, 2, 2))
            >>> msgom = MappedScaleGaussianObservationModel(loc=loc)
            >>> msgom.condition_(x)
            >>> print(msgom.sample())

        Args:
            loc: Mean of observation distribution.
            scale_parametrisation: Scale parametrisation style of Gaussian.

        """
        super().__init__(loc=loc, scale_parametrisation=scale_parametrisation)
        self.mapping = torch.nn.Identity() if mapping is None else mapping

    def condition_(self, x: Tensor):
        """
        Condition observation distribution on state.
        Args:
            x: State to condition sample on. Can be batched or not.

        """
        self.device = x.device
        self.distribution = MultivariateNormal(loc=self.loc, **{self.scale_parametrisation:
                                                                self.mapping(x.cpu()).unsqueeze(-1).unsqueeze(-1)})


class NormalisedDatasetGLMObservationModel(ObservationModel):
    arg_constraints = {}

    def __init__(self, n_features: int,
                 n_datapoints: int,
                 distribution: type[Distribution],
                 inverse_link: Callable[[Tensor], dict[str, Tensor]] = lambda x: {"loc": x},
                 **auxillary_params: Tensor):
        """
        Observation model for dataset and target modelled by a GLM with standardised features.

        Args:
            n_features: Number of features in the GLM.
            n_datapoints: Number of datapoints in the conditioning dataset.
            distribution: Distribution for target.
            inverse_link: Inverse of link function, returning dictionary of parameters as a function of the evaluated
                linear function of the features.
            **auxillary_params: Any parameters not determined by inverse_link.
        """
        super().__init__()
        self.n_features = n_features
        self.n_datapoints = n_datapoints
        self.target_distribution = distribution
        self.inverse_link = inverse_link
        self.auxillary_params = auxillary_params

        self.weights: Optional[Tensor] = None
        self.bias: Optional[Tensor] = None

    def condition_(self, x: Tensor):
        assert x.shape[-1] == self.n_features + 1, "number of weights + bias must match number of features + 1"
        self.weights = x[..., :-1].broadcast_to(self.n_datapoints, *x.shape[:-1], -1).movedim(0, -2)
        self.bias = x[..., -1].broadcast_to(self.n_datapoints, *x.shape[:-1]).movedim(0, -1)

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        features = Normal(loc=torch.zeros(self.n_datapoints, self.n_features),
                          scale=torch.ones(self.n_datapoints, self.n_features)
                          ).sample(self.weights.shape[:-2] + sample_shape)
        linked_mean = torch.einsum("...i, ...i -> ...", features, self.weights) + self.bias
        targets = self.target_distribution(**self.inverse_link(linked_mean), **self.auxillary_params).sample()
        return torch.cat([features, targets.unsqueeze(-1)], dim=-1)

    def log_prob(self, value: torch.Tensor) -> Tensor:
        features = value[..., :-1]
        targets = value[..., -1]
        linked_mean = torch.einsum("...i, ...i -> ...", features, self.weights) + self.bias
        return self.target_distribution(**self.inverse_link(linked_mean), **self.auxillary_params
                                        ).log_prob(targets).sum(dim=-1)


class CompleteDistribution(Distribution):
    _validate_args = False

    def __init__(self,
                 meta_prior: MetaPrior,
                 **observation_model: ObservationModel):
        """
        Complete distribution over prior, state and observation, p(phi, x, z). We implicitly decompose this
        hierarchically as p(phi) p(x|phi) p(z|x).

        Example - no batching:
            >>> meta_prior = GaussianMixtureModelConjugateMetaPrior(n_components=4, state_size=2)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32)
            >>> observation_model = {"obs_1": DirectGaussianObservationModel(covariance_matrix=covariance_matrix)}
            >>> complete_distribution = CompleteDistribution(meta_prior, **observation_model)
            >>> print(complete_distribution.sample())

        Example - batching:
            >>> meta_prior = GaussianMixtureModelConjugateMetaPrior(n_components=4, state_size=2)
            >>> covariance_matrix = torch.eye(2, dtype=torch.float32)
            >>> observation_model = {"obs_1": DirectGaussianObservationModel(covariance_matrix=covariance_matrix)}
            >>> complete_distribution = CompleteDistribution(meta_prior, **observation_model)
            >>> print(complete_distribution.sample((2, 2)))

        Args:
            meta_prior: Meta-prior distribution, p(phi).
            observation_model: Observation model, p(z|x).

        """
        super().__init__()
        self.meta_prior = meta_prior
        self.prior = meta_prior.prior
        self.observation_model: dict[str, ObservationModel] = observation_model
        self.prior_sample: Optional[Tensor] = None

    def sample(self,
               sample_shape: _size = torch.Size(),
               cache_prior: bool = False
               ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if not cache_prior or self.prior_sample is None:
            phi = self.meta_prior.sample(sample_shape)
            self.prior_sample = phi
        else:
            phi = self.prior_sample
        phi_decoded = self.meta_prior.decode_sample(phi)
        x = self.prior(**phi_decoded).sample()
        for observation_model in self.observation_model.values():
            observation_model.condition_(x)
        z = {key: observation_model.sample() for key, observation_model in self.observation_model.items()}
        return phi, x, z


class FactorStructureStochasticVolatility(ObservationModel):
    arg_constraints = {}

    def __init__(self, mean_return: Tensor,
                 factor_loadings: Tensor,
                 residual_covariance: Tensor):
        """
        Multivariate Factor Structure Stochastic Volatility model, as laid out in
        https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/04/9519.pdf

        Args:
            mean_return: Vector of mean returns
            factor_loadings: Factor loadings matrix
            residual_covariance: Covariance matrix of stock noise not explained by factors
        """
        super().__init__()
        self.mean_return = mean_return
        self.factor_loadings = factor_loadings
        self.residual_covariance = residual_covariance

        self.n_observations = mean_return.shape[-1]

    def condition_(self, x: Tensor):
        self.device = x.device
        covariance_matrix = (torch.einsum("...ij, ...j, ...kj -> ...ik",
                                         self.factor_loadings.to(self.device), torch.exp(x),
                                         self.factor_loadings.to(self.device))
                             + self.residual_covariance.to(self.device))
        scale_tril = torch.linalg.cholesky(covariance_matrix)
        self.distribution = MultivariateNormal(self.mean_return.to(self.device), scale_tril=scale_tril)


class RangefinderObservationModel(ObservationModel):
    arg_constraints = {"scale": constraints.positive,
                       "rate": constraints.positive,
                       "max_range": constraints.positive,
                       "weights": constraints.simplex}

    def __init__(self, scale: Tensor,
                 rate: Tensor,
                 max_range: Tensor,
                 weights: Tensor):
        """
        Radar / Sonar rangefinder model. Comprised of mixture of Gaussian centered at true observation with standard
        deviation proportional to range + 1, uniform noise, modelling sensor failure, exponential noise representing
        unexpected interruptions and a maximum range term.

        Args:
            scale: Standard deviation of Gaussian component / range.
            rate: Decay constant of exponential component.
            max_range: Maximum sensor range.
            weights: Weights between Gaussian, exponential and uniform components.
        """
        super().__init__()
        self.scale = scale
        self.rate = rate
        self.max_range = max_range
        self.weights = weights

        self.mixture_distribution = Categorical(weights)
        self.exponential_distribution = Exponential(rate)
        self.uniform_distribution = Uniform(torch.tensor(0.), max_range)

        self.normal_distribution: Optional[Normal] = None

    def condition_(self, x: Tensor):
        self.device = x.device
        self.to_device()
        range = torch.sqrt(x[..., 0] ** 2 + x[..., 2] ** 2)
        self.normal_distribution = Normal(range, self.scale * (range + 1))

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        expanded_sample_shape = sample_shape + self.normal_distribution.batch_shape
        mix_sample = self.mixture_distribution.sample(expanded_sample_shape)
        normal_sample = self.normal_distribution.sample(sample_shape)
        exponential_sample = self.exponential_distribution.sample(expanded_sample_shape)
        uniform_sample = self.uniform_distribution.sample(expanded_sample_shape)
        samples = torch.stack([normal_sample, exponential_sample, uniform_sample], -1)

        mix_shape = mix_sample.shape
        mix_sample_r = mix_sample.unsqueeze(-1)
        mix_sample_r = mix_sample_r.repeat(torch.Size([1] * (len(mix_shape) + 1)))

        samples = samples.gather(-1, mix_sample_r)
        samples = torch.maximum(samples, torch.tensor(0.))
        samples = torch.minimum(samples, self.max_range)
        return samples

    def cdf(self, value: Tensor) -> Tensor:
        mix_prob = self.mixture_distribution.probs
        cdf_normal = self.normal_distribution.cdf(value)
        cdf_exponential = self.exponential_distribution.cdf(value)
        cdf_uniform = value / self.max_range
        cdf_stack = torch.stack([cdf_normal, cdf_exponential, cdf_uniform], dim=-1)
        cdf = torch.sum(cdf_stack * mix_prob, dim=-1)
        cdf[value >= self.max_range] = 1.
        return cdf

    def log_prob(self, value: Tensor) -> Tensor:
        log_mix_prob = torch.log_softmax(
            self.mixture_distribution.logits, dim=-1
        )
        log_normal_prob = self.normal_distribution.log_prob(value)
        log_exponential_prob = self.exponential_distribution.log_prob(value).broadcast_to(log_normal_prob.shape)
        log_uniform_prob = torch.log((value <= self.max_range) / self.max_range).broadcast_to(log_normal_prob.shape)
        log_probs = torch.stack([log_normal_prob, log_uniform_prob, log_exponential_prob], dim=-1)
        return torch.logsumexp(log_probs + log_mix_prob, dim=-1)

    @property
    def mean(self) -> Tensor:
        probs = self.mixture_distribution.probs
        mean_normal = self.normal_distribution.mean
        mean_exponential = self.exponential_distribution.mean.broadcast_to(mean_normal.shape)
        mean_uniform = self.uniform_distribution.mean.broadcast_to(mean_normal.shape)
        mean_stack = torch.stack([mean_normal, mean_exponential, mean_uniform], dim=-1)
        return torch.sum(mean_stack * probs, dim=-1)

    @property
    def variance(self) -> Tensor:
        probs = self.mixture_distribution.probs
        mean_normal = self.normal_distribution.mean
        mean_exponential = self.exponential_distribution.mean.broadcast_to(mean_normal.shape)
        mean_uniform = self.uniform_distribution.mean.broadcast_to(mean_normal.shape)
        mean_stack = torch.stack([mean_normal, mean_exponential, mean_uniform], dim=-1)
        var_cond_mean = torch.sum(probs * (mean_stack -
                                           self.mean.broadcast_to(3, *mean_normal.shape).swapdims(0, -1)) ** 2,
                                  dim=-1)

        mean_cond_var_normal = self.normal_distribution.variance.broadcast_to(mean_normal.shape)
        mean_cond_var_exponential = self.exponential_distribution.variance.broadcast_to(mean_normal.shape)
        mean_cond_var_uniform = self.uniform_distribution.variance.broadcast_to(mean_normal.shape)
        mean_cond_var_stack = torch.stack([mean_cond_var_normal, mean_cond_var_exponential,
                                           mean_cond_var_uniform], dim=-1)
        mean_cond_var = torch.sum(probs * mean_cond_var_stack, dim=-1)

        return mean_cond_var + var_cond_mean

    def to_device(self):
        device = self.device
        self.scale = self.scale.to(device)
        self.rate = self.rate.to(device)
        self.max_range = self.max_range.to(device)
        self.weights = self.weights.to(device)

        self.mixture_distribution = Categorical(self.weights)
        self.exponential_distribution = Exponential(self.rate)
        self.uniform_distribution = Uniform(torch.tensor(0., device=device), self.max_range)

    @lazy_property
    def scale(self):
        return self.scale

    @lazy_property
    def rate(self):
        return self.rate

    @lazy_property
    def max_range(self):
        return self.max_range

    @lazy_property
    def weights(self):
        return self.weights


class GaussianAngleObservationModel(ObservationModel):
    arg_constraints = {"scale": constraints.positive}

    def __init__(self, scale: Tensor):
        """
        Noisy angle sensor.

        Args:
            scale: Noise standard deviation.
        """
        super().__init__()
        self.scale = scale

    def condition_(self, x: Tensor):
        self.device = x.device
        self.scale = self.scale.to(x.device)
        angle = torch.atan2(x[..., 0], x[..., 2])
        self.distribution = Normal(angle, self.scale)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.distribution.log_prob(value),
                            self.distribution.log_prob(value + 2 * torch.pi),
                            self.distribution.log_prob(value - 2 * torch.pi)]).logsumexp(0)

    def sample(self, sample_shape: _size = torch.Size()) -> Tensor:
        sample = self.distribution.sample(sample_shape)
        return ((sample + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)

    @lazy_property
    def scale(self):
        return self.scale
