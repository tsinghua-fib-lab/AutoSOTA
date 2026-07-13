"""Minimal PeakVI-style ATAC VAE."""

import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from torch import nn

from MLP_variants import MLP
from VAE_mixins import _CovariateMixin
from VAE_variants import DLVAE, VAE


class _AtacBaseMixin:
    """Helpers for ATAC-specific depth, region, and probability logic."""

    def __init__(self, hidden_dim, num_layers):
        self.d_encoder = MLP(
            self.in_dim,
            [hidden_dim] * num_layers,
            1,
        )
        self.region_factors = nn.Parameter(torch.zeros(self.in_dim))

    def get_library_size_factors(self, *args):
        return torch.sigmoid(self.d_encoder(self._get_input_args(*args)))

    def get_region_factors(self, ref=None):
        region_factors = torch.sigmoid(self.region_factors).unsqueeze(0)
        if ref is None:
            return region_factors

        return region_factors.type_as(ref)

    def get_atac_dist(self, decoder, latent, *args):
        px = torch.sigmoid(decoder(self._get_latent_args(latent, *args)))
        d = self.get_library_size_factors(*args).type_as(px)
        r = self.get_region_factors(px)
        probs = (px * d * r).clamp(min=1e-6, max=1 - 1e-6)

        x_dist = dist.Bernoulli(
            probs=probs,
            validate_args=False,
        )

        return x_dist, probs, px, d, r

    def get_atac_obs(self, ref, *args):
        return (self._get_output_args(*args) > 0).type_as(ref)


class PeakVAE(_AtacBaseMixin, _CovariateMixin, VAE):
    """ATAC VAE with Bernoulli reconstruction.

    Pass (x,), or (x, s) when covariate_dim > 0
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,  # sqrt(n_regions)
        num_layers: int = 2,
        latent_dim: int = 10,   # sqrt(hidden_dim)
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
        covariate_dim: int = 0,
        linear_decoder: bool = False,
    ):
        VAE.__init__(
            self,
            in_dim,
            hidden_dim,
            num_layers,
            latent_dim,
            recon_weight,
            kl_weight,
        )
        _CovariateMixin.__init__(self, covariate_dim, covariate_arg_index=1)
        _AtacBaseMixin.__init__(self, hidden_dim, num_layers)
        decoder_hidden_dims = [hidden_dim] * num_layers
        decoder_bias = True

        if linear_decoder:
            decoder_hidden_dims = []
            decoder_bias = False

        self.decoder = MLP(
            self.latent_dim + self.covariate_dim,
            decoder_hidden_dims,
            self.in_dim,
            bias=decoder_bias,
        )

    def _reconstruct(self, z, *args):
        x_dist, probs, px, d, r = self.get_atac_dist(self.decoder, z, *args)
        x_obs = self.get_atac_obs(probs, *args)

        pyro.deterministic("px", px)
        pyro.deterministic("d", d)
        pyro.deterministic("r", r)
        pyro.deterministic("rec_probs", probs)

        with poutine.scale(scale=self.recon_weight):
            pyro.sample("rec", x_dist.to_event(1), obs=x_obs)


class PeakDLVAE(_AtacBaseMixin, _CovariateMixin, DLVAE):
    """ATAC DLVAE with Bernoulli reconstruction.

    Pass (x, y) or (x, y, s) where x is the binarized accessibility matrix, y
    is the condition label, and s when covariate_dim > 0.
    """

    def __init__(
        self,
        in_dim: int,
        label_dims,
        hidden_dim: int = 128,
        num_layers: int = 2,
        latent_dim: int = 10,
        w_dim: int = 10,
        recon_weight: float = 1.0,
        recon_weight_z: float = 1.0,
        z_kl_weight: float = 1.0,
        w_kl_weight: float = 1.0,
        adversarial_weight: float = 1.0,
        classifier_layers: int = 1,
        learnable_prior: bool = False,
        covariate_dim: int = 0,
        linear_decoder: bool = False,
    ):
        DLVAE.__init__(
            self, in_dim, label_dims, hidden_dim, num_layers, latent_dim, w_dim,
            recon_weight, recon_weight_z, z_kl_weight, w_kl_weight,
            adversarial_weight, classifier_layers, learnable_prior,
        )
        _CovariateMixin.__init__(self, covariate_dim, covariate_arg_index=2)
        _AtacBaseMixin.__init__(self, hidden_dim, num_layers)
        decoder_hidden_dims = [hidden_dim] * num_layers
        decoder_bias = True

        if linear_decoder:
            decoder_hidden_dims = []
            decoder_bias = False

        self.decoder = MLP(
            self.w_dim + self.latent_dim + self.covariate_dim,
            decoder_hidden_dims,
            self.in_dim,
            bias=decoder_bias,
        )
        self.decoder_z = MLP(
            self.latent_dim + self.covariate_dim,
            decoder_hidden_dims,
            self.in_dim,
            bias=decoder_bias,
        )

    def _reconstruct(self, w_z, *args):
        x_dist, probs, px, d, r = self.get_atac_dist(self.decoder, w_z, *args)
        x_obs = self.get_atac_obs(probs, *args)

        pyro.deterministic("px_w", px)
        pyro.deterministic("d", d)
        pyro.deterministic("r", r)
        pyro.deterministic("rec_probs_w", probs)

        with poutine.scale(scale=self.recon_weight):
            pyro.sample("rec_w", x_dist.to_event(1), obs=x_obs)

    def _reconstruct_z(self, z, *args):
        x_dist, probs, px, _, _ = self.get_atac_dist(self.decoder_z, z, *args)
        x_obs = self.get_atac_obs(probs, *args)

        pyro.deterministic("px_z", px)
        pyro.deterministic("rec_probs_z", probs)

        with poutine.scale(scale=self.recon_weight_z):
            pyro.sample("rec_z", x_dist.to_event(1), obs=x_obs)

        return probs
