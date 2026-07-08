import torch


# ITU-R BT.601 luminance weights; sum to 1 so values stay in [-1, 1].
_LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def rgb_to_grayscale_m1to1(pixels_nchw_m1to1):
    w = _LUMA_WEIGHTS.to(dtype=pixels_nchw_m1to1.dtype, device=pixels_nchw_m1to1.device)
    return (pixels_nchw_m1to1 * w).sum(dim=1, keepdim=True)


class PixelSpaceColorizationLikelihood:
    """Gaussian likelihood of y = rgb_to_grayscale(x) + noise_std * N(0, I)."""

    def __init__(self, observation_gray_m1to1, noise_std):
        self._observation = observation_gray_m1to1
        self._noise_std = float(noise_std)

    def __call__(self, pixels_bnchw):
        batch_size, num_particles = pixels_bnchw.shape[:2]
        spatial_shape = pixels_bnchw.shape[2:]
        flat_pixels = pixels_bnchw.reshape(
            batch_size * num_particles, *spatial_shape
        ).clamp(-1.0, 1.0)
        predicted_observation = rgb_to_grayscale_m1to1(flat_pixels)
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


def build_tiled_pixel_colorization_log_likelihood(observation_gray_m1to1, noise_std, tile_count):
    # Mirror of build_tiled_pixel_sr_log_likelihood but with grayscale as the
    # forward operator. Tiles a k x k grid over the full-res 1-channel
    # observation; sum_over_tiles(log_lik_per_tile) == whole-image log-lik.
    k = int(round(tile_count ** 0.5))
    assert k * k == tile_count, f"tile_count={tile_count} must be a perfect square"
    noise_std_value = float(noise_std)

    def _tiled_log_likelihoods(samples_bnchw):
        batch_size, num_particles, _channels, height, width = samples_bnchw.shape
        flat_samples = samples_bnchw.reshape(
            batch_size * num_particles, _channels, height, width
        ).clamp(-1.0, 1.0)
        predicted_obs_flat = rgb_to_grayscale_m1to1(flat_samples)  # (B*N, 1, H, W)
        _, obs_c, obs_h, obs_w = predicted_obs_flat.shape
        assert obs_h % k == 0 and obs_w % k == 0, f"observation {obs_h}x{obs_w} not divisible by k={k}"
        obs_tile_h, obs_tile_w = obs_h // k, obs_w // k

        sq_err = (predicted_obs_flat - observation_gray_m1to1).pow(2)
        sq_err_tiled = sq_err.reshape(batch_size * num_particles, obs_c, k, obs_tile_h, k, obs_tile_w)
        sq_err_tiled = sq_err_tiled.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(
            batch_size * num_particles, tile_count, obs_c * obs_tile_h * obs_tile_w
        )
        sq_err_per_tile = sq_err_tiled.sum(dim=-1)
        return (-0.5 * sq_err_per_tile / (noise_std_value ** 2)).reshape(batch_size, num_particles, tile_count)

    return _tiled_log_likelihoods
