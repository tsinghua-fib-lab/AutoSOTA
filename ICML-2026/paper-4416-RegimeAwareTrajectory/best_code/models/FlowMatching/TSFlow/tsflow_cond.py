from typing import Tuple

import torch
from einops import rearrange
from ema_pytorch import EMA
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import lagged_sequence_values
from gluonts.transform.split import InstanceSplitter
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
import torch.nn.functional as F

import sys
sys.path.extend(['./', '../', '../../'])
from models.FlowMatching.TSFlow.arch import BackboneModel
from models.FlowMatching.TSFlow.model._base import PREDICTION_INPUT_NAMES, TSFlowBase
from models.FlowMatching.TSFlow.utils.gaussian_process import Q0Dist
from models.FlowMatching.TSFlow.utils.util import LongScaler
from models.FlowMatching.TSFlow.utils.variables import Prior, Setting
from utils.metrics import CRPS_ensemble

patch_typeguard()


class TSFlowCond(TSFlowBase):
    def __init__(
        self,
        target_dim: int,
        context_length: int,
        prediction_length: int,
        backbone_params: dict,
        prior_params: dict,
        optimizer_params: dict,
        ema_params: dict,
        frequency: str,
        normalization: str | None = None,
        use_lags: bool = True,
        use_ema: bool = False,
        num_steps: int = 16,
        solver: str = "euler",
        matching: str = "random",

        # using traj pattern and period as conditions
        use_conditions: bool = False,
        # visualization
        plot_n_samples: int = 10,
        plot_every_n_epochs: int = 10,
        save_predictions_to_file_path: str=None,
        # Uncertainty prediction:
        pred_n_samples: int=1,
    ):
        super().__init__(
            context_length=context_length,
            prediction_length=prediction_length,
            prior_params=prior_params,
            optimizer_params=optimizer_params,
            frequency=frequency,
            normalization=normalization,
            use_lags=use_lags,
            use_ema=use_ema,
            num_steps=num_steps,
            solver=solver,
            matching=matching,
        )
        self.plot_n_samples = plot_n_samples
        self.plot_every_n_epochs = plot_every_n_epochs
        self.save_predictions_to_file_path = save_predictions_to_file_path
        self.use_conditions = use_conditions
        num_features = 2 + (len(self.lags_seq) if use_lags else 0) + (2 if self.use_conditions else 0)

        self.pred_n_samples = pred_n_samples

        # must univariate
        target_dim = 1

        self.backbone = BackboneModel(
            **backbone_params,
            num_features=num_features,
            target_dim=target_dim,
        )
        self.ema_backbone = EMA(self.backbone, **ema_params)
        self.guidance_scale = 0
        self.sigmax = self.sigmin
        self.q0 = Q0Dist(
            **prior_params,
            prediction_length=prediction_length,
            freq=self.freq,
            iso=1e-1 if self.prior != Prior.ISO else 0,
        )

        self._val_vis_data = None
        self._test_vis_data = None
        self.test_metrics = {'past': [], 'true': [], 'pred': [], 'ucty': [], 'n_pred': []}

    @typechecked
    def _transform_data(self, data: list) -> dict:
        past_target, future_target, conditions = data
        return {'x': past_target, 'x_target': future_target, 'conditions': conditions}

    @typechecked
    def _extract_features(
        self, data: dict
    ) -> Tuple[
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", 1, "num_series"],
        TensorType[float, "batch", 1, "num_series"],
        TensorType[float, "batch", "length", "num_series", "num_features"],
    ]:
        ####### Add for biological Dataset #######
        if 'x' in data.keys():
            x = data['x']
            x_target = data['x_target']
            past_obs = data.get('past_obs', None)
            data['past_target'] = past_obs if past_obs is not None else x
            data["future_target"] = x_target
            data["past_observed_values"] = torch.ones_like(data['past_target'])
            data['mean'] = None
        ####### Add for biological Dataset #######
        past = data["past_target"]
        future = data["future_target"]
        context_observed = data["past_observed_values"]
        mean = data["mean"]
        conditions = data.get('conditions', None)
        # default input: shape = [B, L, 1]

        context = past[:, -self.context_length :]
        long_context = past[:, : -self.context_length]
        prior_context = past[:, -self.prior_context_length :]

        if isinstance(self.scaler, LongScaler):
            scaled_context, loc, scale = self.scaler(context, scale=mean)
        else:
            _, loc, scale = self.scaler(past, context_observed)
            scaled_context = context / scale
        scaled_long_context = (long_context - loc) / scale
        scaled_prior_context = (prior_context - loc) / scale
        scaled_future = (future - loc) / scale

        # print(scaled_long_context.shape, scaled_prior_context.shape, scaled_future.shape)
        # print('past.shape:', past.shape)
        # print('future.shape:', future.shape)
        

        x1 = torch.cat([scaled_context, scaled_future], dim=-2)
        batch_size, length, c = x1.shape
        # print(x1.shape)

        observation_mask = torch.zeros_like(x1)
        observation_mask[:, : -self.prediction_length] = context_observed[:, -self.context_length :]

        features = []
        if self.use_lags:
            lags = lagged_sequence_values(
                self.lags_seq,
                scaled_long_context,
                x1,
                dim=1,
            )
            features.append(lags)

        dist = self.q0.gp_regression(rearrange(scaled_prior_context, "b l c -> (b c) l"), self.prediction_length)

        fut = rearrange(dist.sample(), "(b c) l -> b l c", c=c)
        fut_mean = rearrange(dist.mean, "(b c) l -> b l c", c=c)
        fut_std = torch.diagonal(dist.covariance_matrix, dim1=-2, dim2=-1)
        fut_std = rearrange(fut_std, "(b c) ... -> b ... c", c=c)
        features.append(torch.cat([scaled_context, fut_mean], dim=-2).unsqueeze(-1))
        features.append(observation_mask.unsqueeze(-1))
        x0 = torch.cat([scaled_context, fut], dim=-2)

        traj_pattern = conditions.get('traj_pattern', None) if conditions is not None else None # [B, C]
        period = conditions.get('period', None) if conditions is not None else None # [B, C]
        if traj_pattern is not None and self.use_conditions:
            traj_pattern = traj_pattern.unsqueeze(1).repeat(1, length, 1)  # [B, L, C]
            features.append(traj_pattern.unsqueeze(-1))
        if period is not None and self.use_conditions:
            period = period.unsqueeze(1).repeat(1, length, 1)  # [B, L, C]
            features.append(period.unsqueeze(-1))
        
        features = torch.cat(features, dim=-1)
        return x1, x0, observation_mask, loc, scale, features

    @typechecked
    def training_step(self, data: list, idx: int) -> dict:
        assert self.training is True
        x1, x0, _, _, _, features = self._extract_features(self._transform_data(data))
        t = torch.rand((x1.shape[0], 1), device=self.device)
        loss = self.p_losses(x1, x0, t, features)
        self.log(
            "train/loss",
            loss,
            batch_size=x1.shape[0],
            logger=True,
            prog_bar=True
        )
        return {"loss": loss}
    
    def val_test_step(self, data: list, idx: int, prefix='val'):
        assert self.training is False
        data = self._transform_data(data)

        past_target = data['x']
        past_observed_values = torch.ones_like(past_target)
        future_target = data['x_target']
        conditions = data.get('conditions', None)
        
        pred_target_list = []
        for _ in range(self.pred_n_samples):
            pred_target = self(past_target, past_observed_values, None, conditions) # [B, N, L, 1]
            pred_target_list.append(pred_target)
        pred_target = torch.concatenate(pred_target_list, dim=1)  # [B, N, L, 1]

        crps = CRPS_ensemble(pred_target, future_target) # [B, N, L, 1]; [B, L, 1]
        mean_pred_target = pred_target.mean(dim=1, keepdim=False)
        std_pred_target = pred_target.std(dim=1, keepdim=False) # uncertainty

        if idx == 0 and prefix == 'val':
            self._val_vis_data = {'input': past_target, 'target': future_target, 'pred': mean_pred_target}
        elif idx == 0 and prefix == 'test':
            self._test_vis_data = {'input': past_target, 'target': future_target, 'pred': mean_pred_target}

        loss = F.mse_loss(mean_pred_target, future_target)
        mae = F.l1_loss(mean_pred_target, future_target)
        self.log(
            f"{prefix}/loss", loss, on_step=False, batch_size=past_target.shape[0], on_epoch=True, logger=True, prog_bar=True)
        self.log(
            f"{prefix}/mse", loss, on_step=False, batch_size=past_target.shape[0], on_epoch=True, logger=True)
        self.log(
            f"{prefix}/mae", mae, on_step=False, batch_size=past_target.shape[0], on_epoch=True, logger=True)
        self.log(
            f"{prefix}/CRPS", crps, on_step=False, batch_size=past_target.shape[0], on_epoch=True, logger=True)
        
        # save predictions for test
        if prefix == 'test':
            self.test_metrics['past'].append(past_target.cpu().detach().numpy())
            self.test_metrics['true'].append(future_target.cpu().detach().numpy())
            self.test_metrics['pred'].append(mean_pred_target.cpu().detach().numpy())

            self.test_metrics['ucty'].append(std_pred_target.cpu().detach().numpy())
            self.test_metrics['n_pred'].append(pred_target.cpu().detach().numpy())
        return {f"{prefix}/loss": loss}
    
    @typechecked
    def validation_step(self, data: list, idx: int):
        return self.val_test_step(data, idx, prefix='val')
    
    @typechecked
    def test_step(self, data: list, idx: int):
        return self.val_test_step(data, idx, prefix='test')
    
    def plot_trajectory_predictions(self, x_input, x_target, predictions, uncertainties=None, prefix='test'):
        """Plot trajectory predictions for given inputs."""
        try:
            from models._base.trajectory_visualization import (
                plot_trajectory_grid, 
                fig_to_tensor
            )
            import matplotlib.pyplot as plt
            
            fig = plot_trajectory_grid(
                input_seqs=x_input.detach().cpu(),
                target_seqs=x_target.detach().cpu(),
                pred_seqs=predictions.detach().cpu(),
                uncertainties=uncertainties.detach().cpu() if uncertainties is not None else None,
                n_samples=self.plot_n_samples,
                n_cols=2,
                title=f'{prefix.capitalize()} Predictions',
            )
            
            # Log to logger (WandB)
            if self.logger is not None:
                try:
                    if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
                        import wandb
                        self.logger.experiment.log({
                            f'{prefix}/trajectory_predictions': wandb.Image(fig),
                        })
                except Exception as e:
                    print(f"Warning: Failed to log to wandb: {e}")
            
            plt.close(fig)
            
        except ImportError:
            print("Warning: trajectory_visualization module not found. Skipping visualization.")
        except Exception as e:
            print(f"Warning: {prefix.capitalize()} visualization failed: {e}")

    def on_validation_epoch_end(self):
        # Check if should plot this epoch
        if (self.current_epoch + 1) % self.plot_every_n_epochs != 0:
            self._val_vis_data = None
            return
        
        if self._val_vis_data is None:
            return
        
        self.plot_trajectory_predictions(
            x_input=self._val_vis_data['input'],
            x_target=self._val_vis_data['target'],
            predictions=self._val_vis_data['pred'],
            uncertainties=None,
            prefix='val'
        )
        self._val_vis_data = None

    def on_test_epoch_end(self):
        self._save_predictions_to_file(self.save_predictions_to_file_path)
        self.plot_trajectory_predictions(
            x_input=self._test_vis_data['input'],
            x_target=self._test_vis_data['target'],
            predictions=self._test_vis_data['pred'],
            uncertainties=None,
            prefix='test'
        )
        self._test_vis_data = None

    def _save_predictions_to_file(self, save_path: str):
        """Save predictions and targets to file."""
        if not save_path:
            print("Warning: No save path provided for predictions.")
            return
        
        import os
        import numpy as np
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save test metrics
        np.savez_compressed(
            save_path + '/test_predictions.npz',
            inputs=np.concatenate(self.test_metrics['past'], axis=0),
            predictions=np.concatenate(self.test_metrics['pred'], axis=0),
            targets=np.concatenate(self.test_metrics['true'], axis=0),
            uncertainties=np.concatenate(self.test_metrics['ucty'], axis=0) if self.test_metrics['ucty'] else None,
            n_pred=np.concatenate(self.test_metrics['n_pred'], axis=0),
        )
        print(f"Test predictions saved to: {save_path}")


    @typechecked
    def forward(
        self,
        past_target: TensorType[float, "batch", "length"] | TensorType[float, "batch", "length", "num_series"],
        past_observed_values: TensorType[float, "batch", "length"] | TensorType[float, "batch", "length", "num_series"],
        mean: TensorType[float, "batch", 1] | TensorType[float, "batch", 1, "num_series"] = None,
        conditions=None,
    ) -> (
        TensorType[float, "batch", "num_samples", "prediction_length"]
        | TensorType[float, "batch", "num_samples", "prediction_length", "num_series"]
    ):
        B, L, C = past_target.shape
        # This is only used during prediction
        past_target = past_target.to(self.device).repeat_interleave(self.num_samples, dim=0)
        past_observed_values = past_observed_values.to(self.device).repeat_interleave(self.num_samples, dim=0)
        if mean: mean = mean.to(self.device).repeat_interleave(self.num_samples, dim=0)

        future_target = torch.zeros((past_target.shape[0], self.prediction_length, past_target.shape[2]), device=self.device, dtype=past_target.dtype)
        data = dict(
            past_target=past_target,
            past_observed_values=past_observed_values,
            mean=mean,
            future_target=future_target,
            conditions=conditions,
        )
        observation, x0, observation_mask, loc, scale, features = self._extract_features(data)
        x0 = x0 + self.sigmax * torch.randn_like(x0)
        pred = self.sample(
            x0.to(self.device),
            features=features,
            observation=observation,
            observation_mask=observation_mask,
            guidance_scale=self.guidance_scale,
        )
        # must univariate
        pred = rearrange(pred * scale + loc, "(b n) l 1 -> b n l 1", n=self.num_samples)
        return pred[:, :, observation.shape[1] - self.prediction_length :]
