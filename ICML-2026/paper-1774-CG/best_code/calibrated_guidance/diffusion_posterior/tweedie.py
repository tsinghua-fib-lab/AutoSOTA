import torch

from calibrated_guidance import DiffusionPosterior, SampleOutput
from calibrated_guidance.utils import broadcast_time


class TweedieDiffusionPosterior(DiffusionPosterior):
    def __init__(self, diffusion_posterior: DiffusionPosterior, noise_factor: float = 0.0):
        super().__init__()
        self.diffusion_posterior = diffusion_posterior
        self.noise_factor = noise_factor

    def sample(self, xt, t, num_samples):
        x0_mean = self.diffusion_posterior.mean(xt, t)
        if self.noise_factor > 0.0:
            t = broadcast_time(t, xt.unsqueeze(1))
            a_t = (1 - t)
            b_t = t
            noise_shape = torch.randn(x0_mean.shape[0], num_samples, *x0_mean.shape[1:], device=x0_mean.device)
            x0_samples = x0_mean[:, None] + noise_shape * b_t / (a_t ** 2 + b_t ** 2) ** (1 / 2) * self.noise_factor
        else:
            assert num_samples == 1, "Cannot sample multiple samples without noise"
            x0_samples = x0_mean[:, None]
        return SampleOutput(x0_samples)

    def mean(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.diffusion_posterior.mean(xt, t)
