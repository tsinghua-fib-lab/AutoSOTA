"""
Extended Kalman filter competitor for filtering tasks
"""

import torch
from torch import Tensor
from torch.distributions import Distribution, Categorical, MultivariateNormal
from torch.func import vmap

from time import time

from distributions.distributions import ObservationModel, GaussianMixtureModel
from dynamic.motion_models import MotionModel
from dynamic.filters import Filter


class ParticleFilter(Filter):

    def __init__(self, state_size: int,
                 motion_model: MotionModel,
                 n_particles: int,
                 **observation_model: ObservationModel):
        """
        Particle filter. Evolves population of particles through motion model, weights by likelihood and resamples.

        Args:
            state_size: Size of state.
            motion_model: Motion model. Linearises mapping from state to mean of next state distribution.
            n_particles: Number of particles with which to track the state.
            observation_model: Observation model. Linearises mapping from state to mean of observation distribution.
        """
        self.state_size = state_size
        self.motion_model = motion_model
        self.n_particles = n_particles
        self.observation_model = observation_model

    def filter(self, observation_series: dict[str, Tensor],
               x0_distribution: Distribution
               ) -> tuple[Tensor, float]:
        """
        Filter the provided series (assuming sequence in first dimension) assuming initial uncertainty
        x0_distribution.

        Args:
            observation_series: Dictionary of tensors containing observations of series to filter.
            x0_distribution: Initial state distribution.

        Returns:
            Tensor of multivariate Gaussian parameters of shape (series_length | batch_shape | mu sigma).

        """
        device = list(observation_series.values())[0].device

        batched_series_shape = list(observation_series.values())[0].shape[:-1]
        batch_shape = batched_series_shape[1:]
        if batch_shape == torch.Size():
            batch_shape = (1,)

        particles = x0_distribution.sample((self.n_particles, *batch_shape)).to(device).movedim(0, -2)
        particles_all = torch.empty(*batched_series_shape, self.n_particles, self.state_size)

        inference_times = []
        for i, observations in enumerate(zip(*observation_series.values())):
            start_time = time()
            particles = self._update(particles, **dict(zip(observation_series, observations)))
            particles_all[i] = particles.cpu()
            particles = self._predict(particles, i)
            inference_times.append(time() - start_time)

        return (particles_all,
                sum(inference_times) / len(inference_times))

    def _predict(self, particles: Tensor,
                 k: int
                 ) -> Tensor:
        """
        Prediction step of Particle Filter.

        Args:
            particles: Particles of previous step
            k: Time index of current step.

        Returns:
            Prior particles for current step.

        """
        device = particles.device
        noise = self.motion_model.noise_distribution.sample(particles.shape[:-1])
        return self.motion_model.f(particles.cpu(), noise, k).to(device)

    def _update(self, particles: Tensor,
                **observations: Tensor
                ) -> Tensor:
        """
        Update step of Particle Filter.

        Args:
            particles: Particles for current step.

        Returns:
            Posterior particles for current step.

        """
        device = particles.device

        log_weights = torch.zeros(particles.shape[:-1], device=device)

        for key, observation in observations.items():
            observation_model = self.observation_model[key]
            observation_model.condition_(particles.movedim(-2, 0))
            log_weights += observation_model.log_prob(observation.squeeze(-1)).movedim(0, -1)

        resampling_dist = Categorical(logits=log_weights)
        indexes = resampling_dist.sample((particles.shape[-2],)).movedim(0, -1)
        return torch.gather(particles, -2, indexes.unsqueeze(-1).expand(particles.shape))

    def fit_density(self, particles: Tensor) -> MultivariateNormal:
        cov_func = lambda x: torch.cov(x).reshape(x.shape[0], x.shape[0])
        for _ in range(particles.dim()-2):
            cov_func = vmap(cov_func)
        # sigma = (cov_func(particles.mT).unsqueeze(-3).expand(*particles.shape, self.state_size)
        #              + 0.001 * torch.eye(self.state_size, device=particles.device))
        # scale_tril = torch.linalg.cholesky(sigma * 1.06 * self.n_particles ** -0.2)
        # return GaussianMixtureModel(weights=torch.ones(particles.shape[:-1], device=particles.device),
        #                             loc=particles,
        #                             scale_tril=scale_tril)
        scale_tril = torch.linalg.cholesky(cov_func(particles.mT) + 1e-3 * torch.eye(self.state_size, device=particles.device))
        return MultivariateNormal(particles.mean(dim=-2), scale_tril=scale_tril)

