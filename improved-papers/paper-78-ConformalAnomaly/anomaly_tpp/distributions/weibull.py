import torch

from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all


class Weibull(Distribution):
    """Weibull distribution with scale-shape parametrization.

    See (k, b) parametrization in
    https://en.wikipedia.org/wiki/Weibull_distribution#Alternative_parameterizations
    """
    arg_constraints = {"log_scale": constraints.real, "log_shape": constraints.real}

    def __init__(
        self, log_scale: torch.Tensor, log_shape: torch.Tensor, eps=1e-10, validate_args=None
    ):
        self.log_scale, self.log_shape = broadcast_all(log_scale, log_shape)
        self.eps = eps
        batch_shape = self.log_scale.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def log_hazard(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logarithm of the hazard function h(x) = p(x) / S(x)."""
        x = torch.clamp_min(x, self.eps)  # ensure x > 0 for numerical stability
        return self.log_scale + self.log_shape + (self.log_shape.exp() - 1) * x.log()

    def log_survival(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute logarithm of the survival function S(x) = \int_0^x p(u) du."""
        x = torch.clamp_min(x, self.eps)  # ensure x > 0 for numerical stability
        return -self.log_scale.exp() * torch.pow(x, self.log_shape.exp())

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log likelihood of the observation."""
        return self.log_hazard(x) + self.log_survival(x)

    def rsample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """Draw a batch of samples from the distribution with reparametrization."""
        shape = torch.Size(sample_shape) + self.batch_shape
        z = torch.empty(
            shape, device=self.log_scale.device, dtype=self.log_scale.dtype
        ).exponential_(1.0)
        samples = (z * self.log_scale.neg().exp() + self.eps).pow(self.log_shape.neg().exp())
        return samples

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)
