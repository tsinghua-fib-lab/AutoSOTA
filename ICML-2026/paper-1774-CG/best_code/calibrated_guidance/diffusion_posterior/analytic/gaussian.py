import torch

from calibrated_guidance.diffusion_posterior.analytic.base import AnalyticDiffusionPosterior
from calibrated_guidance.utils import broadcast_time


class GaussianDiffusionPosterior(AnalyticDiffusionPosterior):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.loc = loc
        self.scale = scale

    def diffusion_posterior(self, xt, t) -> torch.distributions.Distribution:
        t = broadcast_time(t, xt)
        a_t = (1 - t)
        b_t = t
        var_posterior = 1 / (a_t ** 2 / b_t ** 2 + 1 / self.scale ** 2)
        posterior_loc = var_posterior * (a_t / b_t ** 2 * xt + self.loc / self.scale ** 2)
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=posterior_loc, scale=var_posterior.sqrt()),
            reinterpreted_batch_ndims=len(xt.shape[1:])
        )
