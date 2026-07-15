import math
from functools import partial

import lightning as L
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import nn
from torch.optim import Adam
from torch.special import expm1
from tqdm.auto import tqdm

from sed.models.modules.unet import MLPUnet
from sed.utils import default


def log(t, eps=1e-20):
    # Safe logarithm function to avoid numerical issues (e.g., log(0))
    return torch.log(t.clamp(min=eps))


def right_pad_dims_to(x, t):
    # Pads tensor t with rightmost singleton dimensions until it matches x.ndim.
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def beta_linear_log_snr(t):
    # Computes log SNR according to a linear noise schedule (DDPM style)
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))


def alpha_cosine_log_snr(t, s: float = 0.008):
    # Computes log SNR based on a cosine schedule (Improved DDPMs, common default)
    # Comment reflects uncertainty about strict equivalence to discrete beta clip
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


def log_snr_to_alpha_sigma(log_snr):
    # Derives signal and noise weights from log SNR
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


class Diffusion(L.LightningModule):
    def __init__(
        self,
        unet_config: dict,
        *,
        image_size: int,
        timesteps: int = 1000,
        use_ddim: bool = False,
        noise_schedule: str = 'cosine',
        time_difference: float = 0.,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Backbone is always a ConvUnet for this configuration
        self.unet_model = MLPUnet(**unet_config, input_dim=image_size)
        self.channels = self.unet_model.channels
        self.data_channels = self.unet_model.data_channels

        self.image_size = image_size

        # Chooses the log SNR function based on schedule type ('linear' or 'cosine')
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.timesteps = timesteps
        self.use_ddim = use_ddim
        self.time_difference = time_difference

    @property
    def device(self):
        # Returns which device the UNet is on for convenience
        return next(self.unet_model.parameters()).device

    def get_sampling_timesteps(self, batch, *, device):
        # Returns a sequence of time intervals for sampling: [(t_0, t_1), (t_1, t_2), ...]
        times = torch.linspace(1., 0., self.timesteps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, shape, time_difference=None):
        # Generates a batch of samples from noise by iteratively denoising using DDPM (stochastic) algorithm.
        batch, device = shape[0], self.device
        time_difference = default(time_difference, self.time_difference)
        time_pairs = self.get_sampling_timesteps(batch, device=device)
        img = torch.randn(shape, device=device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step', total=self.timesteps):
            # Step through time intervals

            # Apply any time delay specified
            time_next = (time_next - time_difference).clamp(min=0.)

            # Calculate noise conditional (log SNR)
            noise_cond = self.log_snr(time)

            # Model predicts x_start (clean image estimate)
            x_start = self.unet_model(img, noise_cond, x_start)

            # Compute (possibly broadcasted) log SNRs for this and next step
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            # Compute alpha/sigma terms for diffusion process
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # Compute constants for posterior mean/variance
            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha +
                                 c * x_start)  # posterior mean
            variance = (sigma_next ** 2) * c  # posterior variance
            log_variance = log(variance)

            # Determine masking for whether to add noise (not last step)
            not_last_time = rearrange(time_next > 0, 'b -> b 1')

            noise = torch.where(
                not_last_time,
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            # Update img for next step using reparameterization
            img = mean + (0.5 * log_variance).exp() * noise

        return self.return_sampled(img)

    @torch.no_grad()
    def ddim_sample(self, shape, time_difference=None):
        # Generates samples deterministically using the DDIM method (faster, no random noise)
        batch, device = shape[0], self.device
        time_difference = default(time_difference, self.time_difference)
        time_pairs = self.get_sampling_timesteps(batch, device=device)
        img = torch.randn(shape, device=device)
        x_start = None

        for times, times_next in tqdm(time_pairs, desc='sampling loop time step'):
            # Apply time delay to next time
            times_next = (times_next - time_difference).clamp(min=0.)

            # Calculate log SNRs for current and next step, padded to img shape
            log_snr = self.log_snr(times)
            log_snr_next = self.log_snr(times_next)
            padded_log_snr, padded_log_snr_next = map(
                partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(
                padded_log_snr_next)

            # Model predicts clean image
            x_start = self.unet_model(img, log_snr, x_start)

            # Compute predicted noise (epsilon)
            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)

            # Update image for next step using deterministic transition
            img = x_start * alpha_next + pred_noise * sigma_next

        return self.return_sampled(img)

    @torch.no_grad()
    def return_sampled(self, img):
        # Post-processes the result to the normalized range [0, 1]
        return ((img[:, :self.data_channels] + 1) / 2)

    @torch.no_grad()
    def sample(self, batch_size=16):
        # Convenience method to dispatch to the appropriate sampling method
        image_size, channels = self.image_size, self.channels
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        return sample_fn((batch_size, channels))

    def forward(self, batch, *args, **kwargs):
        # Forward pass for training. Adds noise, possibly uses self-conditioning.
        times = torch.zeros((batch.shape[0],)).to(
            batch).float().uniform_(0, 1.)
        noise = torch.randn_like(batch)
        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(batch, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_img = alpha * batch + sigma * noise

        # Optionally use self-conditioning (50% chance)
        self_cond = None
        if torch.rand((1)) < 0.5:
            with torch.no_grad():
                self_cond = self.unet_model(noised_img, noise_level).detach_()

        # Predict the denoised image
        pred = self.unet_model(noised_img, noise_level, self_cond)
        return pred

    def training_step(self, batch, batch_idx):
        # Training step: normalizes, runs forward, backprop on MSE to original image
        batch = (batch * 2) - 1
        pred = self(batch)
        loss = F.mse_loss(pred, batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Standard Adam optimizer (betas per diffusion literature)
        return Adam(self.unet_model.parameters(), lr=1e-4, betas=(0.9, 0.99))
