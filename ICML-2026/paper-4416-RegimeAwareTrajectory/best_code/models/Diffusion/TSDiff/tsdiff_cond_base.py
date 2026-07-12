# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
import torch
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import lagged_sequence_values
import torch.nn.functional as F
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import sys
sys.path.extend(['./', '../', '../../'])
from models.Diffusion.TSDiff.arch import BackboneModel
from models.Diffusion.TSDiff.diffusion._base import TSDiffBase
from models.Diffusion.TSDiff.diffusion._base import PREDICTION_INPUT_NAMES
from models.Diffusion.TSDiff.utils import get_lags_for_freq

from models._base.base_module import BaseModule

PREDICTION_INPUT_NAMES = PREDICTION_INPUT_NAMES + ["orig_past_target"]


class TSDiffCond(TSDiffBase, BaseModule):
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
        noise_observed=True,
        # Training parameters
        lr: float = 1e-4,
        min_lr: float = 1e-6,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10,
        max_epochs: int = 200,
        # visualization
        plot_n_samples: int = 4,
        plot_every_n_epochs: int = 5,
        save_predictions_to_file_path: Optional[str] = None,
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
            # visualization
            log_val_plots=True,
            plot_n_samples=plot_n_samples, 
            plot_every_n_epochs=plot_every_n_epochs,             
            save_predictions_to_file_path=save_predictions_to_file_path
        )
        self.save_hyperparameters()

        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

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
        past, future, condition_dict = data[:3]
        prior = past[:, : -self.context_length]
        context = past[:, -self.context_length :]
        context_observed = torch.ones_like(context)
        scaled_context, loc, scale = self.scaler(context, context_observed)
        features = []

        scaled_prior = (prior - loc) / scale
        scaled_future = (future - loc) / scale
        scaled_orig_context = (
            past[:, -self.context_length :] - loc
        ) / scale

        x = torch.cat([scaled_orig_context, scaled_future], dim=1)
        observation_mask = torch.zeros_like(x, device=device)
        observation_mask[:, : -self.prediction_length] = torch.ones_like(
            observation_mask[:, : -self.prediction_length]
        )
        x_past = torch.cat(
            [scaled_context, torch.zeros_like(scaled_future)], dim=1
        ).clone()

        assert x.size() == x_past.size()

        if condition_dict.get('traj_pattern', None) is not None:
            features.append(self.embedder(condition_dict["traj_pattern"].long()))
        if condition_dict.get('period', None) is not None:
            features.append(condition_dict["period"])
        static_feat = torch.cat(
            features,
            dim=1,
        )
        expanded_static_feat = static_feat.unsqueeze(1).expand(
            -1, x.shape[1], -1
        )
        features = []
        if self.use_features: # Only expanded_static_feat
            features.append(expanded_static_feat)

            # time_features = []
            # if condition_dict.get('past_mark', None) is not None:
            #     time_features.append(
            #         condition_dict["past_mark"][:, -self.context_length :]
            #     )
            # if condition_dict.get('future_mark', None) is not None:
            #     time_features.append(condition_dict["future_mark"][:, -self.prediction_length :])
            
            # features.append(torch.cat(time_features, dim=1))
        lags = lagged_sequence_values(
            self.lags_seq,
            scaled_prior,
            torch.cat([scaled_context, scaled_future], dim=1),
            dim=1,
        )
        if self.use_lags: # default False
            features.append(lags)
        features.append(x_past)
        features.append(observation_mask)
        features = torch.cat(features, dim=-1)

        return x, loc, scale, features

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

        x, _, _, features = self._extract_features(data)

        # Last dim of features has the observation mask
        observation_mask = features[..., -1:]
        loss_mask = 1 - observation_mask

        t = torch.randint(
            0, self.timesteps, (x.shape[0],), device=device
        ).long()
        elbo_loss = self.step(x, t, features, loss_mask)
        self.log('train/loss', elbo_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return elbo_loss

    def val_test_step(self, batch, batch_idx, prefix):
        device = next(self.parameters()).device
        x, loc, scale, features = self._extract_features(batch)

        # Last dim of features has the observation mask
        observation_mask = features[..., -1:]
        loss_mask = 1 - observation_mask

        past = x[:, : -self.prediction_length]
        future = x[:, -self.prediction_length :]
        future_pred = self.forecast(x, observation_mask, features) * scale + loc
        future_pred = future_pred[:, -self.prediction_length :]

        mse = F.mse_loss(future_pred, future)
        mae = F.l1_loss(future_pred, future)
        
        self.log(f'{prefix}/mse', mse, sync_dist=True)
        self.log(f'{prefix}/mae', mae, sync_dist=True)
        self.log(f'{prefix}/loss', mse, prog_bar=True, sync_dist=True)
        
        if batch_idx == 0 and self.log_val_plots and prefix == 'val':
            self._val_vis_data = {
                'input': past.detach().cpu(),
                'target': future.detach().cpu(),
                'pred': future_pred.detach().cpu(),
                'uncertainty': None,
            }
        elif batch_idx == 0 and self.log_val_plots and prefix == 'test':
            self._test_vis_data = {
                'input': past.detach().cpu(),
                'target': future.detach().cpu(),
                'pred': future_pred.detach().cpu(),
                'uncertainty': None,
            }

        if prefix == 'test':
            self.test_metrics['past'].append(past.cpu().detach().numpy())
            self.test_metrics['true'].append(future.cpu().detach().numpy())
            self.test_metrics['pred'].append(future_pred.cpu().detach().numpy())
            self.test_metrics['ucty'].append(None)
        
        return {f'{prefix}/mse': mse, f'{prefix}/mae': mae}
    
    def validation_step(self, data, idx):
        return self.val_test_step(data, idx, prefix='val')
    def test_step(self, batch, batch_idx):
        return self.val_test_step(batch, batch_idx, prefix='test')

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

        observation, loc, scale, features = self._extract_features(data)
        observation = observation.to(device)
        batch_size, length, ch = observation.shape
        observation_mask = features[..., -1:]

        pred = self.forecast(
            observation=observation,
            observation_mask=observation_mask,
            features=features,
        )

        pred = pred * scale + loc

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
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch + 1) / float(max(1, self.warmup_epochs))
            else:
                progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
                return max(self.min_lr / self.lr, 0.5 * (1 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
