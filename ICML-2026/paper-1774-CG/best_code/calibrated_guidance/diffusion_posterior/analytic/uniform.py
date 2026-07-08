import torch
from torch.distributions.utils import lazy_property

from calibrated_guidance.diffusion_posterior.analytic.base import AnalyticDiffusionPosterior
from calibrated_guidance.utils import broadcast_time


class TruncatedNormal(torch.distributions.Distribution):
    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real

    def __init__(self, loc, scale, low, high, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self.normal = torch.distributions.Normal(loc, scale, validate_args=validate_args)
        self.low_cdf = self.normal.cdf(self.low)
        self.high_cdf = self.normal.cdf(self.high)
        super().__init__(batch_shape=torch.broadcast_shapes(
            loc.shape, scale.shape, low.shape, high.shape
        ), validate_args=validate_args)

    def sample(self, sample_shape = torch.Size()):
        shape = torch.Size(sample_shape) + self.batch_shape
        # Inverse-CDF sampling, clamped for numerical stability when the
        # (untruncated) mean lies far outside [low, high] -- e.g. the diffusion
        # posterior at t close to 1, where the boundary CDFs collapse to 0/1 and
        # icdf would return +-inf. Mirrors the robust truncated-normal sampler in
        # the original SBI eval scripts.
        eps = 1e-12
        low_cdf = self.low_cdf.clamp(eps, 1.0 - eps)
        high_cdf = torch.maximum(self.high_cdf.clamp(eps, 1.0 - eps), low_cdf + eps)
        u = torch.rand(shape, device=self.loc.device, dtype=self.loc.dtype)
        samples = self.normal.icdf(low_cdf + u * (high_cdf - low_cdf))
        return samples.clamp(self.low, self.high)

    def log_prob(self, value):
        logp = self.normal.log_prob(value)
        logZ = torch.log(
            self.high_cdf - self.low_cdf
        )
        logp = logp - logZ
        return torch.where(
            (value < self.low) | (value > self.high),
            torch.tensor(-float("inf"), device=value.device),
            logp,
        )

    @lazy_property
    def mean(self):
        a = (self.low - self.loc) / self.scale
        b = (self.high - self.loc) / self.scale

        std_normal = torch.distributions.Normal(
            torch.zeros_like(self.loc),
            torch.ones_like(self.scale),
            validate_args=self._validate_args
        )

        Z = std_normal.cdf(b) - std_normal.cdf(a)
        mean = self.loc + self.scale * (
            std_normal.log_prob(a).exp() - std_normal.log_prob(b).exp()
        ) / Z

        return mean


class UniformDiffusionPosterior(AnalyticDiffusionPosterior):
    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        super().__init__()
        self.low = low
        self.high = high

    def diffusion_posterior(self, xt, t) -> torch.distributions.Distribution:
        if not torch.is_tensor(t) and t == 1.0:
            return torch.distributions.Independent(
                torch.distributions.Uniform(self.low[None], self.high[None]),
                reinterpreted_batch_ndims=len(xt.shape[1:]),
                validate_args=False
            )
        t = broadcast_time(t, xt)
        a_t = (1 - t)
        b_t = t
        mean = xt / a_t
        if torch.is_tensor(t):
            scale = b_t / a_t
        else:
            scale = torch.full_like(mean, b_t / a_t)
        return torch.distributions.Independent(
            TruncatedNormal(
                mean, scale,
                self.low, self.high, validate_args=False
            ),
            reinterpreted_batch_ndims=len(xt.shape[1:]),
            validate_args=False
        )
