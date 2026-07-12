from typing import Tuple, Optional

import torch
from einops import rearrange
from torchtyping import TensorType
import torch.nn.functional as F

import sys
sys.path.extend(['./', '../', '../../'])
from models.FlowMatching.RegimeFlow.arch.backbone import BackboneModel
from models.FlowMatching.RegimeFlow.arch._base import RegimeFlowBase
from models.FlowMatching.RegimeFlow.arch.bio_cond_layers import ConditionEncoder
from models.FlowMatching.RegimeFlow.arch.source_BLR import BLRTemplatePriorGenerator, BLRPriorConfig
from utils.metrics import CRPS_ensemble

# ===================== RegimeFlowCond with BLR Prior =====================
class RegimeFlowCond(RegimeFlowBase):
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        backbone_params: dict,
        prior_params: dict,
        optimizer_params: dict,
        normalization: str | None = None,
        num_steps: int = 16,
        solver: str = "euler",
        matching: str = "random",
        
        # Condition encoder parameters
        use_condition: bool = True,
        condition_dropout: float = 0.1,
        cond_dim: int = 128,
        num_patterns: int = 6,
        num_freqs: int = 64,

        # Visualization
        plot_n_samples: int = 10,
        plot_every_n_epochs: int = 10,
        save_predictions_to_file_path: str = None,
        # uncertainty prediction
        pred_n_samples: int=1,
        guidance_scale: float = 0,
        beta_alpha: float = 1.0,  # 1.0=uniform, <1.0=U-shaped Beta
    ):
        super().__init__(
            context_length=context_length,
            prediction_length=prediction_length,
            prior_params=prior_params,
            optimizer_params=optimizer_params,
            normalization=normalization,
            num_steps=num_steps,
            solver=solver,
            matching=matching,
        )
        
        self.plot_n_samples = plot_n_samples
        self.plot_every_n_epochs = plot_every_n_epochs
        self.save_predictions_to_file_path = save_predictions_to_file_path
        self.use_condition = use_condition
        self.condition_dropout = condition_dropout
        self.cond_dim = cond_dim

        self.pred_n_samples = pred_n_samples

        # Condition encoder
        if use_condition:
            self.condition_encoder = ConditionEncoder(
                d_model=cond_dim,
                num_patterns=num_patterns,
                num_freqs=num_freqs,
            )
        else:
            self.condition_encoder = None

        # Backbone model
        self.backbone = BackboneModel(
            **backbone_params,
            cond_dim=cond_dim if use_condition else 0,
            use_adaLN=use_condition,
        )
        
        self.guidance_scale = guidance_scale
        self.beta_alpha = beta_alpha
        self.sigmax = self.sigmin

        # Prior
        self.prior_params = prior_params
        self.prior_type = prior_params.pop('name', 'BLRTemplatePriorGenerator')
        self.prior = None

        self._val_vis_data = None
        self._test_vis_data = None
        self.test_metrics = {'past': [], 'true': [], 'pred': [], 'ucty': [], 'n_pred': [], 'prior': []}

    def setup(self, stage=None):
        """Initialize BLR prior."""
        if self.prior is not None: return

        if self.prior_type == 'BLRTemplatePriorGenerator':
            blr_config = BLRPriorConfig(
                alpha=self.prior_params.get('alpha', 1.0),
                beta=self.prior_params.get('beta', 20.0),
                noise_scale=self.prior_params.get('noise_scale', 0.1),
                saturation_rate=self.prior_params.get('saturation_rate', 3.0),
                slope_window=self.prior_params.get('slope_window', 10),
                min_variance=self.prior_params.get('min_variance', 1e-6),
                num_harmonics=self.prior_params.get('num_harmonics', 4),
                saturation_scales=self.prior_params.get('saturation_scales', (0.2, 0.5, 1.0, 2.0, 5.0)),
            )
            self.prior = BLRTemplatePriorGenerator(blr_config)
        elif self.prior_type in ['ISO', 'OU', 'SE', 'PE']:
            from models.FlowMatching.TSFlow.utils.gaussian_process import Q0Dist
            self.prior = Q0Dist(
                **self.prior_params,
                prediction_length=self.prediction_length,
                freq=1,
                iso=1e-1 if self.prior_type != 'ISO' else 0,
            )
        else: raise NotImplementedError(f"Prior type {self.prior_type} not implemented.")

    def _transform_data(self, data: list) -> dict:
        past_target, future_target, condition_dict = data
        return {
            'x': past_target, 
            'x_target': future_target, 
            'condition': condition_dict
        }

    def _extract_fut_from_prior_blr(
        self,
        device: torch.device,
        batch_size: int,
        c: int,
        scaled_prior_context: TensorType[float, "batch", "prior_context_length"],
        features: dict,
        condition_dropout_mask: Optional[dict] = None,
    ) -> TensorType[float, "batch", "prediction_length", "num_series"]:
        # Construct time indices
        past_times = torch.arange(
            self.prior_context_length, 
            device=device, 
            dtype=torch.float32
        ).unsqueeze(0).repeat(batch_size, 1)
        
        future_times = torch.arange(
            self.prior_context_length, 
            self.prior_context_length + self.prediction_length,
            device=device, 
            dtype=torch.float32
        ).unsqueeze(0).repeat(batch_size, 1)
        
        # Get pattern and period from features
        pattern_ids = features.get('traj_pattern', None)
        if pattern_ids is None:
            # FIXED: Default to 0 (DIRECTLY_STABLE) when not provided
            pattern_ids = torch.zeros(batch_size, device=device, dtype=torch.long)
        elif len(pattern_ids.shape) > 1:
            pattern_ids = pattern_ids.squeeze(-1)
        
        periods = features.get('period', None)
        if periods is None:
            # FIXED: Default to 0 (consistent with pattern=0)
            periods = torch.zeros(batch_size, device=device, dtype=torch.float32)
        elif len(periods.shape) > 1:
            periods = periods.squeeze(-1)
        
        # CRITICAL: Apply condition dropout for CFG training
        # When a condition is dropped, BLR should use default values (pattern=0, period=0)
        # This allows the model to learn inference without conditions
        if condition_dropout_mask is not None:
            pattern_dropout = condition_dropout_mask.get('pattern', None)
            period_dropout = condition_dropout_mask.get('period', None)
            
            if pattern_dropout is not None:
                # Set to default (0) for dropped samples
                pattern_ids = torch.where(
                    pattern_dropout,
                    torch.zeros_like(pattern_ids),
                    pattern_ids
                )
            
            if period_dropout is not None:
                # Set to default (0) for dropped samples
                periods = torch.where(
                    period_dropout,
                    torch.zeros_like(periods),
                    periods
                )
        
        # Sample from BLR for each channel
        fut_list = []
        for ch in range(c):
            x0_ch, mu_ch, info_ch = self.prior(
                past_times=past_times,
                past_values=scaled_prior_context[:, :, ch],
                future_times=future_times,
                pattern_ids=pattern_ids,
                period=periods
            )
            fut_list.append(x0_ch.unsqueeze(-1))
        
        # Concatenate channels
        fut = torch.cat(fut_list, dim=-1)  # [B, L_pred, C]
        return fut

    def _extract_fut_from_prior_Q0(
            self,
            device: torch.device,
            batch_size: int,
            c: int,
            scaled_prior_context: TensorType[float, "batch", "prior_context_length"],
        ):
        dist = self.prior.gp_regression(rearrange(scaled_prior_context, "b l c -> (b c) l"), self.prediction_length)
        fut = rearrange(dist.sample(), "(b c) l -> b l c", c=c)
        return fut

    def _extract_features(
        self, 
        data: dict,
        condition_dropout_mask: Optional[dict] = None,  # NEW: for CFG training
    ) -> Tuple[
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", 1, "num_series"],
        TensorType[float, "batch", 1, "num_series"],
        dict
    ]:
        """
        Extract features and conditions from data.
        
        Args:
            data: Input data dictionary
            condition_dropout_mask: Optional dict with 'pattern' and 'period' dropout masks
                                   for CFG training. When a sample is dropped, BLR will use
                                   default values (pattern=0, period=0).
        
        Returns:
            x1: Target trajectory
            x0: Initial noisy trajectory (from BLR prior)
            observation_mask: Mask for observations
            loc: Normalization location
            scale: Normalization scale
            condition_dict: Dictionary containing traj_pattern, period, etc.
        """
        # Process biological dataset
        if 'x' in data.keys():
            x = data['x']
            x_target = data['x_target']
            past_obs = data.get('past_obs', None)
            data['past_target'] = past_obs if past_obs is not None else x
            data["future_target"] = x_target
            data["past_observed_values"] = torch.ones_like(data['past_target'])
            data['mean'] = None
        
        past = data["past_target"]
        future = data["future_target"]
        context_observed = data["past_observed_values"]
        condition_dict = data.get('condition', {})

        # Extract context windows
        context = past[:, -self.context_length:]
        prior_context = past[:, -self.prior_context_length:]

        # Normalization
        _, loc, scale = self.scaler(past, context_observed)
        scaled_context = context / scale
        
        scaled_prior_context = (prior_context - loc) / scale
        scaled_future = (future - loc) / scale

        # Construct x1 (target)
        x1 = torch.cat([scaled_context, scaled_future], dim=-2)
        batch_size, length, c = x1.shape

        # Observation mask
        observation_mask = torch.zeros_like(x1)
        observation_mask[:, :-self.prediction_length] = context_observed[:, -self.context_length:]

        # Extract conditions
        features = {}
        if 'traj_pattern' in condition_dict:
            features['traj_pattern'] = condition_dict['traj_pattern']
        if 'period' in condition_dict:
            features['period'] = condition_dict['period']

        if self.prior_type == 'BLRTemplatePriorGenerator':
            fut = self._extract_fut_from_prior_blr(
                device=self.device,
                batch_size=batch_size,
                c=c,
                scaled_prior_context=scaled_prior_context,
                features=features,
                condition_dropout_mask=condition_dropout_mask,
            )
        else:
            fut = self._extract_fut_from_prior_Q0(
                device=self.device, batch_size=batch_size, c=c,
                scaled_prior_context=scaled_prior_context,
            )

        # Construct x0
        x0 = torch.cat([scaled_context, fut], dim=-2)

        return x1, x0, observation_mask, loc, scale, features

    def _prepare_condition_embedding(
        self, 
        features: dict, 
        batch_size: int, 
        device: torch.device,
        pattern_dropout: Optional[torch.Tensor] = None,  # NEW: explicit dropout masks
        period_dropout: Optional[torch.Tensor] = None,   # NEW: explicit dropout masks
    ) -> Optional[torch.Tensor]:
        """
        Prepare condition embedding from features dict.
        
        Args:
            features: Dict containing 'traj_pattern' and/or 'period'
            batch_size: Batch size
            device: Device
            pattern_dropout: Boolean mask (B,) indicating which samples have pattern dropped
            period_dropout: Boolean mask (B,) indicating which samples have period dropped
            
        Returns:
            Condition embedding (B, cond_dim) or None if no conditions
        """
        if not self.use_condition or self.condition_encoder is None:
            return None
        
        # Check if we have any conditions
        has_pattern = 'traj_pattern' in features and features['traj_pattern'] is not None
        has_period = 'period' in features and features['period'] is not None
        
        if not has_pattern and not has_period:
            # No conditions provided, use null embeddings
            return self.condition_encoder(
                batch_size=batch_size,
                device=device,
                use_null=True
            )
        
        # Encode conditions with provided dropout masks
        cond_emb = self.condition_encoder(
            batch_size=batch_size,
            device=device,
            traj_pattern=features.get('traj_pattern', None),
            period=features.get('period', None),
            traj_pattern_dropout=pattern_dropout,  # Use provided masks
            period_dropout=period_dropout,          # Use provided masks
            use_null=False,
        )
        
        return cond_emb

    def training_step(self, data: list, idx: int) -> dict:
        """
        Training step with condition encoding and CFG dropout.
        
        CRITICAL: Dropout is applied consistently to both:
        1. BLR prior (uses pattern=0, period=0 for dropped samples)
        2. Condition encoder (uses null embeddings for dropped samples)
        
        This ensures the model learns to infer without conditions.
        """
        assert self.training is True
        
        data_dict = self._transform_data(data)
        batch_size = data_dict['x'].shape[0]
        
        # Generate condition dropout masks for CFG training
        condition_dropout_mask = None
        pattern_dropout = None
        period_dropout = None
        
        if self.condition_dropout > 0:
            # Random dropout: True where we should drop
            pattern_dropout = torch.rand(batch_size, device=self.device) < self.condition_dropout
            period_dropout = torch.rand(batch_size, device=self.device) < self.condition_dropout
            # If pattern is dropped, period must also be dropped
            period_dropout = period_dropout | pattern_dropout
            
            condition_dropout_mask = {
                'pattern': pattern_dropout,
                'period': period_dropout,
            }
        
        # Extract features with dropout applied to BLR
        x1, x0, _, _, _, features = self._extract_features(
            data_dict, 
            condition_dropout_mask=condition_dropout_mask
        )
        
        # Sample time (Beta distribution for improved flow matching)
        if self.beta_alpha != 1.0:
            t = torch.distributions.Beta(self.beta_alpha, self.beta_alpha).sample((x1.shape[0], 1)).to(self.device)
        else:
            t = torch.rand((x1.shape[0], 1), device=self.device)
        
        # Prepare condition embedding with same dropout masks
        cond_emb = self._prepare_condition_embedding(
            features=features,
            batch_size=x1.shape[0],
            device=self.device,
            pattern_dropout=pattern_dropout,
            period_dropout=period_dropout,
        )
        
        # Compute loss
        loss = self.p_losses(x1, x0, t, cond_emb=cond_emb)
        
        self.log(
            "train/loss",
            loss,
            batch_size=x1.shape[0],
            logger=True,
            prog_bar=True
        )
        
        return {"loss": loss}
    
    def val_test_step(self, data: list, idx: int, prefix='val'):
        """Validation/test step - no dropout applied."""
        assert self.training is False
        data = self._transform_data(data)

        past_target = data['x']
        past_observed_values = torch.ones_like(past_target)
        future_target = data['x_target']
        condition_dict = data.get('condition', {})
        
        # Forward pass with conditions (no dropout)
        pred_target_list = []
        future_prior_list = []
        for _ in range(self.pred_n_samples):
            pred_target, x0 = self(
                past_target, 
                past_observed_values, 
                None,
                condition_dict=condition_dict,
                return_prior=True
            )
            pred_target_list.append(pred_target)
            future_prior_list.append(x0)
        pred_target = torch.concatenate(pred_target_list, dim=1)  # [B, N, L, 1]
        future_prior = torch.stack(future_prior_list, dim=1)  # [B, N, L, 1]

        crps = CRPS_ensemble(pred_target, future_target) # [B, N, L, 1]; [B, L, 1]

        mean_pred_target = pred_target.mean(dim=1, keepdim=False)
        std_pred_target = pred_target.std(dim=1, keepdim=False)

        # Store visualization data
        if idx == 0 and prefix == 'val':
            self._val_vis_data = {
                'input': past_target, 
                'target': future_target, 
                'pred': mean_pred_target
            }
        elif idx == 0 and prefix == 'test':
            self._test_vis_data = {
                'input': past_target, 
                'target': future_target, 
                'pred': mean_pred_target
            }

        # Compute metrics
        loss = F.mse_loss(mean_pred_target, future_target)
        mae = F.l1_loss(mean_pred_target, future_target)
        
        self.log(
            f"{prefix}/loss", loss, 
            on_step=False, batch_size=past_target.shape[0], 
            on_epoch=True, logger=True, prog_bar=True
        )
        self.log(
            f"{prefix}/mse", loss, 
            on_step=False, batch_size=past_target.shape[0], 
            on_epoch=True, logger=True
        )
        self.log(
            f"{prefix}/mae", mae, 
            on_step=False, batch_size=past_target.shape[0], 
            on_epoch=True, logger=True
        )
        self.log(
            f"{prefix}/CRPS", crps, 
            on_step=False, batch_size=past_target.shape[0], 
            on_epoch=True, logger=True
        )
        
        # Save predictions for test
        if prefix == 'test':
            self.test_metrics['past'].append(past_target.cpu().detach().numpy())
            self.test_metrics['true'].append(future_target.cpu().detach().numpy())
            self.test_metrics['pred'].append(mean_pred_target.cpu().detach().numpy())
            if std_pred_target is not None:
                self.test_metrics['ucty'].append(std_pred_target.cpu().detach().numpy())
            self.test_metrics['n_pred'].append(pred_target.cpu().detach().numpy())
            self.test_metrics['prior'].append(future_prior.cpu().detach().numpy())
        
        return {f"{prefix}/loss": loss}
    
    def validation_step(self, data: list, idx: int):
        return self.val_test_step(data, idx, prefix='val')
    
    def test_step(self, data: list, idx: int):
        return self.val_test_step(data, idx, prefix='test')
    
    def plot_trajectory_predictions(
        self, 
        x_input, 
        x_target, 
        predictions, 
        uncertainties=None, 
        prefix='test'
    ):
        """Plot trajectory predictions."""
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
        """Validation epoch end with visualization."""
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
        """Test epoch end with saving and visualization."""
        self._save_predictions_to_file(self.save_predictions_to_file_path)
        
        if self._test_vis_data is not None:
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
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        ucty_data = None
        if self.test_metrics['ucty'] and len(self.test_metrics['ucty']) > 0:
            ucty_data = np.concatenate(self.test_metrics['ucty'], axis=0)
        
        np.savez_compressed(
            os.path.join(save_path, 'test_predictions.npz'),
            inputs=np.concatenate(self.test_metrics['past'], axis=0),
            predictions=np.concatenate(self.test_metrics['pred'], axis=0),
            targets=np.concatenate(self.test_metrics['true'], axis=0),
            uncertainties=ucty_data,
            n_pred=np.concatenate(self.test_metrics['n_pred'], axis=0),
            prior=np.concatenate(self.test_metrics['prior'], axis=0),
        )
        print(f"Test predictions saved to: {save_path}")

    def forward(
        self,
        past_target: TensorType[float, "batch", "length"] | TensorType[float, "batch", "length", "num_series"],
        past_observed_values: TensorType[float, "batch", "length"] | TensorType[float, "batch", "length", "num_series"],
        mean: TensorType[float, "batch", 1] | TensorType[float, "batch", 1, "num_series"] = None,
        condition_dict: dict = None,
        return_prior: bool=False,
    ) -> (
        TensorType[float, "batch", "num_samples", "prediction_length"]
        | TensorType[float, "batch", "num_samples", "prediction_length", "num_series"]
    ):
        """
        Forward pass for inference - no dropout applied.
        
        Args:
            past_target: Past observations
            past_observed_values: Mask for past observations
            mean: Mean for normalization
            condition_dict: Dictionary containing 'traj_pattern' and/or 'period'
        """
        B, L, C = past_target.shape
        
        # Repeat for multiple samples
        past_target = past_target.to(self.device).repeat_interleave(self.num_samples, dim=0)
        past_observed_values = past_observed_values.to(self.device).repeat_interleave(self.num_samples, dim=0)
        if mean is not None: 
            mean = mean.to(self.device).repeat_interleave(self.num_samples, dim=0)

        # Prepare data
        future_target = torch.zeros(
            (past_target.shape[0], self.prediction_length, past_target.shape[2]), 
            device=self.device, 
            dtype=past_target.dtype
        )
        
        data = dict(
            past_target=past_target,
            past_observed_values=past_observed_values,
            mean=mean,
            future_target=future_target,
            condition=condition_dict if condition_dict is not None else {},
        )
        
        # Extract features (no dropout for inference)
        observation, x0, observation_mask, loc, scale, features = self._extract_features(
            data,
            condition_dropout_mask=None  # No dropout during inference
        )
        
        # Prepare condition embedding (no dropout for inference)
        cond_emb = self._prepare_condition_embedding(
            features=features,
            batch_size=observation.shape[0],
            device=self.device,
            pattern_dropout=None,  # No dropout during inference
            period_dropout=None,
        )
        
        
        # Compute null embedding for CFG (classifier-free guidance)
        null_cond_emb = None
        if self.guidance_scale > 0 and self.use_condition and self.condition_encoder is not None:
            null_cond_emb = self.condition_encoder(
                batch_size=observation.shape[0],
                device=self.device,
                use_null=True
            )
        # Add noise
        x0 = x0 + self.sigmax * torch.randn_like(x0)
        
        # Sample
        pred = self.sample(
            x0.to(self.device),
            cond_emb=cond_emb,
            observation=observation,
            observation_mask=observation_mask,
            guidance_scale=self.guidance_scale,
            null_cond_emb=null_cond_emb,
        )
        
        # Denormalize and reshape
        pred = rearrange(pred * scale + loc, "(b n) l 1 -> b n l 1", n=self.num_samples)
        
        if return_prior:
            return pred[:, :, observation.shape[1] - self.prediction_length:], x0
        return pred[:, :, observation.shape[1] - self.prediction_length:]
