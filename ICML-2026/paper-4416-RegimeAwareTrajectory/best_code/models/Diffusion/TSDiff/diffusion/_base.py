# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.modules.feature import FeatureEmbedder

import sys
sys.path.extend(['./', '../', '../../'])
from models.Diffusion.TSDiff.utils import extract
from models.Diffusion.TSDiff.diffusion._scaler import MeanScaler, NOPScaler, StdScaler

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "future_time_feat",
]

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.1
    return torch.linspace(beta_start, beta_end, timesteps)

class TSDiffBase(pl.LightningModule):
    def __init__(
        self,
        backbone_parameters,
        timesteps,
        diffusion_scheduler,
        context_length,
        prediction_length,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinalities=None,
        freq=None,
        normalization="none",
        use_features=False,
        use_lags=True,
        lr: float = 1e-3,

        **kwargs, ## add this
    ):
        super().__init__()
        self.timesteps = timesteps
        self.betas = eval(diffusion_scheduler)(timesteps)
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
        self.logs = {}
        self.normalization = normalization

        if normalization == "mean":
            self.scaler = MeanScaler(dim=1, keepdim=True)
        elif normalization == "zscore": 
            self.scaler = StdScaler(dim=1, keepdim=True, minimum_scale=1)
        else:
            self.scaler = NOPScaler(dim=1, keepdim=True)
        if cardinalities is None:
            cardinalities = [1]
        self.embedder = FeatureEmbedder(
            cardinalities=cardinalities,
            embedding_dims=[min(50, (cat + 1) // 2) for cat in cardinalities],
        )
        self.time_features = (
            time_features_from_frequency_str(freq) if freq is not None else []
        )

        self.num_feat_dynamic_real = (
            1 + num_feat_dynamic_real + len(self.time_features)
        )
        self.num_feat_static_cat = max(num_feat_static_cat, 1)
        self.num_feat_static_real = max(num_feat_static_real, 1)

        self.use_features = use_features
        self.use_lags = use_lags

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.losses_running_mean = torch.ones(timesteps, requires_grad=False)
        self.lr = lr
        self.best_crps = np.inf

    def _extract_features(self, data):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=int(1e12)
        )
        return [optimizer], {"scheduler": scheduler, "monitor": "train/loss"}

    def log(self, name, value, **kwargs):
        super().log(name, value, **kwargs)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        if name not in self.logs:
            self.logs[name] = [value]
        else:
            self.logs[name].append(value)

    def get_logs(self):
        logs = self.logs
        logs["epochs"] = list(range(self.current_epoch))
        return pd.DataFrame.from_dict(logs)

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
        reduction="mean",
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

    @torch.no_grad()
    def p_sample_ddim(self, x, t, features=None, noise=None):
        if noise is None:
            noise = self.backbone(x, t, features)
        sqrt_alphas_cumprod_prev_t = extract(
            self.alphas_cumprod_prev, t, x.shape
        ).sqrt()
        sqrt_one_minus_alphas_cumprod_prev_t = extract(
            1 - self.alphas_cumprod_prev, t, x.shape
        ).sqrt()
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        x0pointer = (
            sqrt_alphas_cumprod_prev_t
            * (x - sqrt_one_minus_alphas_cumprod_t * noise)
            / sqrt_alphas_cumprod_t
        )
        xtpointer = sqrt_one_minus_alphas_cumprod_prev_t * noise
        return x0pointer + xtpointer

    @torch.no_grad()
    def p_sample_genddim(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        t_prev: Optional[torch.Tensor] = None,
        eta: float = 0.0,
        features=None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generalized DDIM step that interpolates between
        DDPM (eta=1) and DDIM (eta=0).

        Args:
            x (torch.Tensor): _description_
            t (torch.Tensor): _description_
            features (_type_, optional): _description_. Defaults to None.
            noise (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """
        if noise is None:
            noise = self.backbone(x, t, features)
        if t_prev is None:
            t_prev = t - 1

        alphas_cumprod_t = extract(self.alphas_cumprod, t, x.shape)
        alphas_cumprod_prev_t = (
            extract(self.alphas_cumprod, t_prev, x.shape)
            if t_index > 0
            else torch.ones_like(alphas_cumprod_t)
        )
        sqrt_alphas_cumprod_prev_t = alphas_cumprod_prev_t.sqrt()

        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)

        x0pointer = (
            sqrt_alphas_cumprod_prev_t
            * (x - sqrt_one_minus_alphas_cumprod_t * noise)
            / sqrt_alphas_cumprod_t
        )
        c1 = (
            eta
            * (
                (1 - alphas_cumprod_t / alphas_cumprod_prev_t)
                * (1 - alphas_cumprod_prev_t)
                / (1 - alphas_cumprod_t)
            ).sqrt()
        )
        c2 = ((1 - alphas_cumprod_prev_t) - c1**2).sqrt()
        return x0pointer + c1 * torch.randn_like(x) + c2 * noise

    @torch.no_grad()
    def sample(self, noise, features=None):
        device = next(self.backbone.parameters()).device
        batch_size, length, ch = noise.shape
        seq = noise
        seqs = [seq.cpu()]

        for i in reversed(range(0, self.timesteps)):
            seq = self.p_sample(
                seq,
                torch.full((batch_size,), i, device=device, dtype=torch.long),
                i,
                features,
            )
            seqs.append(seq.cpu().numpy())

        return np.stack(seqs, axis=0)

    def fast_denoise(self, xt, t, features=None, noise=None):
        if noise is None:
            noise = self.backbone(xt, t, features)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, xt.shape
        )
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, xt.shape)
        return (
            xt - sqrt_one_minus_alphas_cumprod_t * noise
        ) / sqrt_alphas_cumprod_t

    def forward(self, x, mask):
        raise NotImplementedError()

    def training_step(self, data, idx):
        assert self.training is True
        device = next(self.backbone.parameters()).device
        if isinstance(data, dict):
            x, _, features = self._extract_features(data)
        else:
            x, _ = self.scaler(data, torch.ones_like(data))

        t = torch.randint(
            0, self.timesteps, (x.shape[0],), device=device
        ).long()
        elbo_loss, xt, noise = self.p_losses(x, t, features, loss_type="l2")
        return {
            "loss": elbo_loss,
            "elbo_loss": elbo_loss,
        }

    '''
    def on_train_epoch_end(self, outputs):
        epoch_loss = sum(x["loss"] for x in outputs) / len(outputs)
        elbo_loss = sum(x["elbo_loss"] for x in outputs) / len(outputs)
        self.log("train_loss", epoch_loss)
        self.log("train_elbo_loss", elbo_loss)

    def validation_step(self, data, idx):
        device = next(self.backbone.parameters()).device
        if isinstance(data, dict):
            x, _, features = self._extract_features(data)
        else:
            x, features = data, None
        t = torch.randint(
            0, self.timesteps, (x.shape[0],), device=device
        ).long()
        elbo_loss, xt, noise = self.p_losses(x, t, features, loss_type="l2")
        return {
            "loss": elbo_loss,
            "elbo_loss": elbo_loss,
        }

    def on_validation_epoch_end(self, outputs):
        epoch_loss = sum(x["loss"] for x in outputs) / len(outputs)
        elbo_loss = sum(x["elbo_loss"] for x in outputs) / len(outputs)
        self.log("valid_loss", epoch_loss)
        self.log("valid_elbo_loss", elbo_loss)
    '''
