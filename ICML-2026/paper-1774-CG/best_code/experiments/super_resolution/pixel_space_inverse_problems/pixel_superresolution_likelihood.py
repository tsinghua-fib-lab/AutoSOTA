import torch

from pixel_space_inverse_problems.superresolution_likelihood import (
    downsample_pixels_by_bicubic_4x,
)


class PixelSpaceSuperResolutionLikelihood:
    """Gaussian likelihood of y = bicubic_down_4x(x) + noise_std * N(0, I)."""

    def __init__(self, observation_low_res_m1to1, noise_std):
        self._observation = observation_low_res_m1to1
        self._noise_std = float(noise_std)

    def __call__(self, pixels_bnchw):
        batch_size, num_particles = pixels_bnchw.shape[:2]
        spatial_shape = pixels_bnchw.shape[2:]
        flat_pixels = pixels_bnchw.reshape(
            batch_size * num_particles, *spatial_shape
        ).clamp(-1.0, 1.0)
        predicted_observation = downsample_pixels_by_bicubic_4x(flat_pixels)
        observation_expanded = (
            self._observation.unsqueeze(1)
            .expand(batch_size, num_particles, *self._observation.shape[1:])
            .reshape(batch_size * num_particles, *self._observation.shape[1:])
        )
        squared_error = (predicted_observation - observation_expanded).pow(2)
        sum_squared_error_per_sample = squared_error.flatten(1).sum(dim=1)
        log_likelihood_per_sample = (
            -0.5 * sum_squared_error_per_sample / (self._noise_std ** 2)
        )
        return log_likelihood_per_sample.reshape(batch_size, num_particles)
