from typing import Callable
import pyro.distributions as dist
import torch
from torch import Tensor

from simulator.base import Simulator


class Gaussian(Simulator):
    def __init__(
        self,
        theta_dim: int = 1,
        obs_dim: int = 2,
        x_raw_dim: int = 100,
        prior_scale: float = 5.0,
        likelihood_scale: float = 1.0,
        misspecified_scale: float = 2.0,
    ):
        theta_dim = int(theta_dim)
        obs_dim = int(obs_dim)
        assert theta_dim == 1, "Only 1D theta is supported for Gaussian simulator"
        super().__init__(obs_dim=obs_dim, theta_dim=theta_dim, name="gaussian")

        # Prior parameters
        self.prior_params = {"loc": torch.zeros((theta_dim,)), "scale": prior_scale}
        self.prior_dist = dist.Normal(**self.prior_params)

        # Likelihood parameters
        self.likelihood_scale = likelihood_scale
        self.x_raw_dim = int(x_raw_dim)
        self.likelihood_dist = None

        # Noisy process parameters
        self.misspecified_scale = misspecified_scale
        self.post_dist = None

        self.denoise_dist = None

    def get_simulator(
        self, misspecified: bool, return_raw: bool = False
    ) -> Callable[[Tensor], Tensor]:
        def simulator(theta: Tensor) -> Tensor:
            scale = self.likelihood_scale if misspecified else self.misspecified_scale
            xdist = dist.Normal(loc=theta, scale=scale)
            x_raw = xdist.sample(sample_shape=(self.x_raw_dim,))

            # Return mean and variance of the raw data
            xobs = torch.stack([x_raw.mean(dim=0), x_raw.var(dim=0)])
            if return_raw:
                return xobs.transpose(0, 1).squeeze(), x_raw
            return xobs.transpose(0, 1).squeeze()

        return simulator

    def sample_reference_posterior(
        self, num_samples: int, observations: Tensor, misspecified: bool, **kwargs
    ) -> Tensor:
        raise NotImplementedError

    def get_noisy_process(self):
        def noisy_process(x: Tensor) -> Tensor:
            assert self.misspecified_scale > self.likelihood_scale
            y_raw = x + torch.randn_like(x) * (self.misspecified_scale - self.likelihood_scale)
            yobs = torch.stack([y_raw.mean(dim=0), y_raw.var(dim=0)])
            return yobs.transpose(0, 1).squeeze()

        return noisy_process

    def sample_denoiser(self, num_samples: int, y: Tensor) -> Tensor:
        raise NotImplementedError
