# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import copy

import torch
from gluonts.torch.util import lagged_sequence_values

import sys
sys.path.extend(['./', '../', '../../'])
from models.Diffusion.TSDiff.arch import BackboneModel
from models.Diffusion.TSDiff.diffusion._base import TSDiffBase
from models.Diffusion.TSDiff.utils import get_lags_for_freq


class TSDiff(TSDiffBase):
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
        init_skip=True,
        lr=1e-3,
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

        self.freq = freq
        if use_lags:
            self.lags_seq = get_lags_for_freq(freq)
            backbone_parameters = backbone_parameters.copy()
            backbone_parameters["input_dim"] += len(self.lags_seq)
            backbone_parameters["output_dim"] += len(self.lags_seq)
        else:
            self.lags_seq = [0]
        self.input_dim = backbone_parameters["input_dim"]
        self.backbone = BackboneModel(
            **backbone_parameters,
            num_features=(
                self.num_feat_static_real
                + self.num_feat_static_cat
                + self.num_feat_dynamic_real
                + 1  # log_scale
            ),
            init_skip=init_skip,
        )
        self.ema_rate = []  # [0.9999]
        self.ema_state_dicts = [
            copy.deepcopy(self.backbone.state_dict())
            for _ in range(len(self.ema_rate))
        ]

    def _extract_features(self, data):
        prior = data["past_target"][:, : -self.context_length]
        context = data["past_target"][:, -self.context_length :]
        context_observed = data["past_observed_values"][
            :, -self.context_length :
        ]
        if self.normalization == "zscore":
            scaled_context, scale = self.scaler(
                context, context_observed, data["stats"]
            )
        else:
            scaled_context, scale = self.scaler(context, context_observed)
        features = []

        scaled_prior = prior / scale
        scaled_future = data["future_target"] / scale
        features.append(scale.log())

        x = torch.cat([scaled_context, scaled_future], dim=1)
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

        features = [expanded_static_feat]

        time_features = []
        if data["past_time_feat"] is not None:
            time_features.append(
                data["past_time_feat"][:, -self.context_length :]
            )
        if data["future_time_feat"] is not None:
            time_features.append(data["future_time_feat"])
        features.append(torch.cat(time_features, dim=1))
        features = torch.cat(features, dim=-1)

        if self.use_lags:
            lags = lagged_sequence_values(
                self.lags_seq,
                scaled_prior,
                torch.cat([scaled_context, scaled_future], dim=1),
                dim=1,
            )
            x = torch.cat([x[:, :, None], lags], dim=-1)
        else:
            x = x[:, :, None]
        if not self.use_features:
            features = None

        return x, scale[:, :, None], features

    @torch.no_grad()
    def sample_n(
        self,
        num_samples: int = 1,
        return_lags: bool = False,
    ):
        device = next(self.backbone.parameters()).device
        seq_len = self.context_length + self.prediction_length

        samples = torch.randn(
            (num_samples, seq_len, self.input_dim), device=device
        )

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            samples = self.p_sample(samples, t, i, features=None)

        samples = samples.cpu().numpy()

        if return_lags:
            return samples

        return samples[..., 0]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        for rate, state_dict in zip(self.ema_rate, self.ema_state_dicts):
            update_ema(state_dict, self.backbone.state_dict(), rate=rate)


def update_ema(target_state_dict, source_state_dict, rate=0.99):
    with torch.no_grad():
        for key, value in source_state_dict.items():
            ema_value = target_state_dict[key]
            ema_value.copy_(
                rate * ema_value + (1.0 - rate) * value.cpu(),
                non_blocking=True,
            )
