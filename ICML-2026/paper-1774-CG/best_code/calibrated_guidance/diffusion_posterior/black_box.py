import torch

from .base import DiffusionPosterior, SampleOutput
import torch


class BlackBoxDiffusionPosterior(DiffusionPosterior):
    def __init__(self, sample_fn, mean_fn=None):
        self._dist_cache = {}
        self.sample_fn = sample_fn
        self.mean_fn = mean_fn

    def sample(self, x_t, t, num_samples) -> SampleOutput:
        output = self.sample_fn(x_t, t, num_samples)
        if not isinstance(output, SampleOutput):
            output = SampleOutput(samples=output)
        return output

    def log_prob(self, x0, xt, t):
        a_t = 1 - t
        b_t = t

        eps_t = (xt - a_t * x0) / b_t
        return torch.distributions.Independent(
            torch.distributions.Normal(
                torch.zeros_like(xt[0]),
                torch.ones_like(xt[0])
            ), len(xt.shape[1:])
        ).log_prob(eps_t)
