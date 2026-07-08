import torch

from calibrated_guidance.diffusion_posterior.base import DiffusionPosterior, SampleOutput


class AnalyticDiffusionPosterior(DiffusionPosterior):
    def diffusion_posterior(self, xt, t) -> torch.distributions.Distribution:
        raise NotImplementedError

    def sample(self, xt: torch.Tensor, t: torch.Tensor, num_samples: int) -> SampleOutput:
        posterior_dist = self.diffusion_posterior(xt, t)
        if posterior_dist.has_rsample:
            samples = posterior_dist.rsample([num_samples])  # [num_samples, B, ...]
        else:
            samples = posterior_dist.sample([num_samples])  # [num_samples, B, ...]
        # [B, num_samples, ...]
        log_probs = posterior_dist.log_prob(samples).swapaxes(1, 0)
        samples = samples.swapaxes(1, 0)
        return SampleOutput(samples, log_probs=log_probs)

    def log_prob(self, x: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        posterior_dist = self.diffusion_posterior(xt, t)
        # Need to move num_samples to first axis for log_prob
        return posterior_dist.log_prob(x.swapaxes(0, 1)).swapaxes(0, 1)

    def mean(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        posterior_dist = self.diffusion_posterior(xt, t)
        return posterior_dist.mean
