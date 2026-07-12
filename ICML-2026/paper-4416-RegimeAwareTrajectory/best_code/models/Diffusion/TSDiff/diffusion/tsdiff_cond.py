# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import lagged_sequence_values

import sys
sys.path.extend(['./', '../', '../../'])
from models.Diffusion.TSDiff.arch import BackboneModel
from models.Diffusion.TSDiff.diffusion._base import TSDiffBase
from models.Diffusion.TSDiff.diffusion._base import PREDICTION_INPUT_NAMES
from models.Diffusion.TSDiff.utils import get_lags_for_freq

PREDICTION_INPUT_NAMES = PREDICTION_INPUT_NAMES + ["orig_past_target"]


class TSDiffCond(TSDiffBase):
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
        lr=1e-3,
        init_skip=True,
        noise_observed=True,
    ):
        super().__init__(
            backbone_parameters,
            timesteps=timesteps,
            diffusion_scheduler=diffusion_scheduler,
            context_length=context_length,
            prediction_length=prediction_length,
            num_feat_dynamic_real=num_feat_dynamic_real,
            num_feat_static_cat=num_feat_static_cat,
            num_feat_static_real=num_feat_static_real,
            cardinalities=cardinalities,
            freq=freq,
            normalization=normalization,
            use_features=use_features,
            use_lags=use_lags,
            lr=lr,
        )

        num_features = (
            (
                self.num_feat_dynamic_real
                + self.num_feat_static_cat
                + self.num_feat_static_real
                + 1
            )
            if use_features
            else 0
        )
        self.freq = freq
        self.lags_seq = get_lags_for_freq(freq) if use_lags else [0]
        self.backbone = BackboneModel(
            **backbone_parameters,
            num_features=(
                num_features + 2 + (len(self.lags_seq) if use_lags else 0)
            ),
            init_skip=init_skip,
        )
        self.noise_observed = noise_observed

    def _extract_features(self, data):
        device = next(self.parameters()).device
        prior = data["past_target"][:, : -self.context_length]
        context = data["past_target"][:, -self.context_length :]
        context_observed = data["past_observed_values"][
            :, -self.context_length :
        ]
        scaled_context, scale = self.scaler(context, context_observed)
        features = []

        scaled_prior = prior / scale
        scaled_future = data["future_target"] / scale
        scaled_orig_context = (
            data["orig_past_target"][:, -self.context_length :]
        ) / scale

        x = torch.cat([scaled_orig_context, scaled_future], dim=1)
        observation_mask = torch.zeros_like(x, device=device)
        observation_mask[:, : -self.prediction_length] = data[
            "past_observed_values"
        ][:, -self.context_length :].data
        x_past = torch.cat(
            [scaled_context, torch.zeros_like(scaled_future)], dim=1
        ).clone()

        assert x.size() == x_past.size()

        if data["feat_static_cat"] is not None:
            features.append(self.embedder(data["feat_static_cat"]))
        if data["feat_static_real"] is not None:
            features.append(data["feat_static_real"])
        static_feat = torch.cat(
            features,
            dim=1,
        )
        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, x.shape[1], -1
        )
        features = []
        if self.use_features:
            features.append(expanded_static_feat)

            time_features = []
            if data["past_time_feat"] is not None:
                time_features.append(
                    data["past_time_feat"][:, -self.context_length :]
                )
            if data["future_time_feat"] is not None:
                time_features.append(data["future_time_feat"])
            features.append(torch.cat(time_features, dim=1))
        lags = lagged_sequence_values(
            self.lags_seq,
            scaled_prior,
            torch.cat([scaled_context, scaled_future], dim=1),
            dim=1,
        )
        if self.use_lags:
            features.append(lags)
        features.append(x_past[..., None])
        features.append(observation_mask[..., None])
        features = torch.cat(features, dim=-1)
        return x[..., None], scale[..., None], features

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

    def training_step(self, data, idx):
        assert self.training is True
        device = next(self.parameters()).device

        x, _, features = self._extract_features(data)

        # Last dim of features has the observation mask
        observation_mask = features[..., -1:]
        loss_mask = 1 - observation_mask

        t = torch.randint(
            0, self.timesteps, (x.shape[0],), device=device
        ).long()
        elbo_loss = self.step(x, t, features, loss_mask)
        return {
            "loss": elbo_loss,
            "elbo_loss": elbo_loss,
        }

    def validation_step(self, data, idx):
        device = next(self.parameters()).device

        x, _, features = self._extract_features(data)

        # Last dim of features has the observation mask
        observation_mask = features[..., -1:]
        loss_mask = 1 - observation_mask

        val_loss = 0.0
        for i in range(self.timesteps):
            t = torch.full((x.shape[0],), i, device=device).long()
            val_loss += self.step(x, t, features, loss_mask)

        val_loss /= self.timesteps

        return {
            "loss": val_loss,
            "elbo_loss": val_loss,
        }

    @torch.no_grad()
    def forecast(self, observation, observation_mask, features=None):
        device = next(self.backbone.parameters()).device
        batch_size, length, ch = observation.shape

        seq = torch.randn_like(observation)

        for i in reversed(range(0, self.timesteps)):
            if not self.noise_observed:
                seq = observation_mask * observation + seq * (
                    1 - observation_mask
                )

            seq = self.p_sample(
                seq,
                torch.full((batch_size,), i, device=device, dtype=torch.long),
                i,
                features,
            )

        return seq

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        feat_static_cat: torch.Tensor = None,
        feat_static_real: torch.Tensor = None,
        past_time_feat: torch.Tensor = None,
        future_time_feat: torch.Tensor = None,
        orig_past_target: torch.Tensor = None,
    ):
        # This is only used during prediction
        device = next(self.backbone.parameters()).device
        data = dict(
            feat_static_cat=feat_static_cat.to(device)
            if feat_static_cat is not None
            else None,
            feat_static_real=feat_static_real.to(device)
            if feat_static_real is not None
            else None,
            past_time_feat=past_time_feat.to(device)
            if past_time_feat is not None
            else None,
            past_target=past_target.to(device),
            orig_past_target=orig_past_target.to(device),
            future_target=torch.zeros(
                past_target.shape[0], self.prediction_length, device=device
            ),
            past_observed_values=past_observed_values.to(device)
            if past_observed_values is not None
            else None,
            future_time_feat=future_time_feat.to(device)
            if future_time_feat is not None
            else None,
        )

        observation, scale, features = self._extract_features(data)
        observation = observation.to(device)
        batch_size, length, ch = observation.shape
        observation_mask = features[..., -1:]

        pred = self.forecast(
            observation=observation,
            observation_mask=observation_mask,
            features=features,
        )

        pred = pred * scale

        return pred[:, None, length - self.prediction_length :, 0]

    def get_predictor(self, input_transform, batch_size=40, device=None):
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            device=device,
        )
