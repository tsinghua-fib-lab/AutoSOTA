# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------------
# Portions of this file are derived from TSDiff
# - Source: https://github.com/amazon-science/unconditional-time-series-diffusion
# - Paper: Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting
# - License: Apache-2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from probts.utils import extract
from probts.model.forecaster import Forecaster
from probts.model.nn.arch.S4.s4_backbones import BackboneModel
from probts.utils import repeat
import sys

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.1
    return torch.linspace(beta_start, beta_end, timesteps)


class TSDiffCond(Forecaster):
    def __init__(
        self,
        hidden_dim: int,
        step_emb: int,
        timesteps: int,
        num_residual_blocks: int,
        dropout: float = 0,
        # use_features: bool = False,
        init_skip=True,
        noise_observed=False, # reconstruct past
        mode="diag",
        measure="diag",
        **kwargs
    ):
        super().__init__(**kwargs)
        backbone_parameters = {
            "input_dim": self.target_dim,
            "hidden_dim": hidden_dim,
            "output_dim": self.target_dim,
            "step_emb": step_emb,
            "num_residual_blocks": num_residual_blocks,
            "residual_block": "s4",
            "mode": mode,
            'measure': measure,
        }
        # self.use_features=use_features
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps)
        self.sqrt_one_minus_beta = torch.sqrt(1.0 - self.betas)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.backbone = BackboneModel(
            **backbone_parameters,
            num_features=self.target_dim,
            init_skip=init_skip,
            dropout=dropout,
        )
        self.noise_observed = noise_observed

    def _extract_features(self, batch_data):
        inputs = self.get_inputs(batch_data, 'all')
        x = inputs[:,:, :self.target_dim]
        features = inputs.clone()
        
        if self.use_time_feat:
            features[:,self.context_length:, :self.target_dim] = 0
        else:
            features = features[:,:, :self.target_dim]
            features[:,self.context_length:] = 0
        
        observation_mask = torch.zeros_like(x, device=x.device)
        observation_mask[:,:self.context_length] = 1
        
        return x, features, observation_mask

    def q_sample(self, x_start, t, noise=None):
        device = next(self.backbone.parameters()).device
        if noise is None:
            noise = torch.randn_like(x_start, device=device)
        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def p_losses(
        self,
        x_start,
        t,
        features=None,
        noise=None,
        loss_type="l2",
        reduction="none",
    ):
        device = next(self.backbone.parameters()).device
        if noise is None:
            noise = torch.randn_like(x_start, device=device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.backbone(x_noisy, t, features)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise, reduction=reduction)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise, reduction=reduction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(
                noise, predicted_noise, reduction=reduction
            )
        else:
            raise NotImplementedError()

        return loss, x_noisy, predicted_noise

    @torch.no_grad()
    def p_sample(self, x, t, t_index, features=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)


        predicted_noise = self.backbone(x, t, features)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def step(self, x, t, features, loss_mask):
        noise = torch.randn_like(x)
        if not self.noise_observed:
            noise = (1 - loss_mask) * x + noise * loss_mask

        num_eval = loss_mask.sum()
        sq_err, _, _ = self.p_losses(
            x,
            t,
            features,
            loss_type="l2",
            reduction="none",
            noise=noise,
        )

        if self.noise_observed:
            elbo_loss = sq_err.mean()
        else:
            sq_err = sq_err * loss_mask
            elbo_loss = sq_err.sum() / (num_eval if num_eval else 1)
        return elbo_loss



    def loss(self, batch_data):
        # [b l k 1], [b l k 2]
        x, features, observation_mask = self._extract_features(batch_data)
        loss_mask = 1 - observation_mask

        t = torch.randint(
            0, self.timesteps, [x.shape[0]], device=x.device
        ).long()
        
        loss = self.step(x, t, features, loss_mask)

        if torch.isnan(loss):
            print("Loss is NaN, exiting.")
            sys.exit(1)
        return loss

    def forecast(self, batch_data, num_samples):
        observation, features, observation_mask = self._extract_features(batch_data)

        observation = observation.to(observation.device)

        pred = self.sample(
            observation=observation,
            observation_mask=observation_mask,
            n_samples=num_samples,
            features=features,
        )  

        return pred[:,:,-self.prediction_length:,:]

    @torch.no_grad()
    def sample(self, observation, observation_mask, n_samples, features=None):

        repeated_observation = repeat(observation, n_samples)
        repeated_observation_mask = repeat(observation_mask, n_samples)
        repeated_features = repeat(features, n_samples)
        
        batch_size, length, ch = repeated_observation.shape
        seq = torch.randn_like(repeated_observation)

        for i in reversed(range(0, self.timesteps)):
            if not self.noise_observed:
                seq = repeated_observation_mask * repeated_observation + seq * (1 - repeated_observation_mask)

            seq = self.p_sample(
                seq,
                torch.full((batch_size,), i, device=repeated_observation.device, dtype=torch.long),
                i,
                repeated_features,
            )

        seq = seq.reshape(-1, n_samples, length, ch)
        return seq 
