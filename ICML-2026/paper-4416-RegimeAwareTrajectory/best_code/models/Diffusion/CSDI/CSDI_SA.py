# csdi_lightning.py
"""
CSDI PyTorch Lightning Module
Adapted from CSDI.py to work with PyTorch Lightning framework
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchtyping import TensorType
from typeguard import typechecked

import sys
sys.path.extend(['./', '../', '../../'])

# Import original CSDI components
from models.Diffusion.CSDI.CSDI_SA_base import (
    DiffusionEmbedding,
    diff_CSDI,
    ResidualBlock,
    Conv1d_with_init,
    get_torch_trans,
)
from utils.metrics import CRPS_ensemble


class CSDILightning(pl.LightningModule):
    """
    CSDI (Conditional Score-based Diffusion model for Imputation) 
    adapted for PyTorch Lightning framework.
    
    This module wraps the original CSDI components and provides
    a Lightning-compatible interface for training and inference.
    """
    
    def __init__(
        self,
        target_dim: int,
        context_length: int,
        prediction_length: int,
        # Diffusion parameters
        num_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.5,
        schedule: str = "quad",
        # Model architecture parameters
        layers: int = 4,
        channels: int = 64,
        nheads: int = 8,
        diffusion_embedding_dim: int = 128,
        timeemb: int = 128,
        featureemb: int = 16,
        # Training parameters
        optimizer_params: dict = None,
        is_unconditional: bool = False,
        target_strategy: str = "random",
        max_epochs: int = 200,
        # Conditions
        use_conditions: bool = False,
        # Visualization
        plot_n_samples: int = 10,
        plot_every_n_epochs: int = 10,
        save_predictions_to_file_path: str = None,
        # Prediction
        pred_n_samples: int = 10,
        num_samples: int = 1,  # samples per prediction during inference
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Dimensions
        self.target_dim = target_dim if target_dim > 1 else 1
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.max_epochs = max_epochs
        
        # Diffusion parameters
        self.num_steps = num_steps
        self.is_unconditional = is_unconditional
        self.target_strategy = target_strategy
        
        # Embedding dimensions
        self.emb_time_dim = timeemb
        self.emb_feature_dim = featureemb
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1  # for conditional mask
        
        # Conditions
        self.use_conditions = use_conditions
        if self.use_conditions:
            self.emb_total_dim += 2  # traj_pattern and period
        
        # Build embeddings
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, 
            embedding_dim=self.emb_feature_dim
        )
        
        # Build diffusion model
        config_diff = {
            "layers": layers,
            "channels": channels,
            "nheads": nheads,
            "diffusion_embedding_dim": diffusion_embedding_dim,
            "num_steps": num_steps,
            "side_dim": self.emb_total_dim,
        }
        
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)
        
        # Beta schedule for diffusion
        if schedule == "quad":
            self.beta = np.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_steps
            ) ** 2
        elif schedule == "linear":
            self.beta = np.linspace(beta_start, beta_end, num_steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        
        # Register as buffers for device management
        self.register_buffer(
            "alpha_torch", 
            torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1)
        )
        self.register_buffer(
            "beta_torch",
            torch.tensor(self.beta).float()
        )
        self.register_buffer(
            "alpha_hat_torch",
            torch.tensor(self.alpha_hat).float()
        )
        
        # Optimizer parameters
        self.optimizer_params = optimizer_params or {"lr": 1e-3, "weight_decay": 1e-6}
        
        # Visualization and logging
        self.plot_n_samples = plot_n_samples
        self.plot_every_n_epochs = plot_every_n_epochs
        self.save_predictions_to_file_path = save_predictions_to_file_path
        
        # Prediction parameters
        self.pred_n_samples = pred_n_samples
        self.num_samples = num_samples
        
        # Visualization data cache
        self._val_vis_data = None
        self._test_vis_data = None
        
        # Test metrics storage
        self.test_metrics = {
            'past': [], 'true': [], 'pred': [], 
            'ucty': [], 'n_pred': []
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        
        # Learning rate scheduler
        p1 = int(0.75 * self.max_epochs) if self.max_epochs else 150
        p2 = int(0.9 * self.max_epochs) if self.max_epochs else 180
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[p1, p2], gamma=0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

    @typechecked
    def _transform_data(self, data: list) -> dict:
        """Transform input data list to dictionary format."""
        past_target, future_target, conditions = data
        return {
            'x': past_target,
            'x_target': future_target,
            'conditions': conditions
        }

    def time_embedding(self, pos: torch.Tensor, d_model: int = 128) -> torch.Tensor:
        """Generate time embeddings for positional encoding."""
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model, device=self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, 
            torch.arange(0, d_model, 2, device=self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask: torch.Tensor) -> torch.Tensor:
        """Generate random mask for training."""
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(
        self, 
        observed_mask: torch.Tensor, 
        for_pattern_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Generate historical mask for training."""
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)
        
        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        
        return cond_mask

    def get_side_info(
        self, 
        observed_tp: torch.Tensor, 
        cond_mask: torch.Tensor,
        conditions: dict = None
    ) -> torch.Tensor:
        """Build side information tensor for the diffusion model."""
        B, K, L = cond_mask.shape
        
        # Time embedding
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B, L, emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        
        # Feature embedding
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim, device=self.device)
        )  # (K, emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B, L, K, *)
        side_info = side_info.permute(0, 3, 2, 1)  # (B, *, K, L)
        
        # Conditional mask
        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)  # (B, 1, K, L)
            side_info = torch.cat([side_info, side_mask], dim=1)
        
        # Additional conditions (traj_pattern, period)
        if self.use_conditions and conditions is not None:
            traj_pattern = conditions.get('traj_pattern', None)
            period = conditions.get('period', None)
            
            if traj_pattern is not None:
                # traj_pattern: [B, C] -> [B, 1, K, L]
                traj_cond = traj_pattern.unsqueeze(-1).unsqueeze(-1)
                traj_cond = traj_cond.expand(-1, -1, K, L)
                side_info = torch.cat([side_info, traj_cond], dim=1)
            
            if period is not None:
                # period: [B, C] -> [B, 1, K, L]
                period_cond = period.unsqueeze(-1).unsqueeze(-1)
                period_cond = period_cond.expand(-1, -1, K, L)
                side_info = torch.cat([side_info, period_cond], dim=1)
        
        return side_info

    def set_input_to_diffmodel(
        self, 
        noisy_data: torch.Tensor, 
        observed_data: torch.Tensor, 
        cond_mask: torch.Tensor
    ) -> torch.Tensor:
        """Prepare input for the diffusion model."""
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)  # (B, 1, K, L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B, 2, K, L)
        return total_input

    def calc_loss(
        self, 
        observed_data: torch.Tensor, 
        cond_mask: torch.Tensor, 
        observed_mask: torch.Tensor, 
        side_info: torch.Tensor, 
        is_train: int = 1, 
        set_t: int = -1
    ) -> torch.Tensor:
        """Calculate diffusion loss."""
        B, K, L = observed_data.shape
        
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B], device=self.device)
        
        current_alpha = self.alpha_torch[t]  # (B, 1, 1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t)  # (B, K, L)
        
        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        
        return loss

    def calc_loss_valid(
        self, 
        observed_data: torch.Tensor, 
        cond_mask: torch.Tensor, 
        observed_mask: torch.Tensor, 
        side_info: torch.Tensor, 
        is_train: int = 0
    ) -> torch.Tensor:
        """Calculate validation loss over all timesteps."""
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, 
                side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def impute(
        self, 
        observed_data: torch.Tensor, 
        cond_mask: torch.Tensor, 
        side_info: torch.Tensor, 
        n_samples: int
    ) -> torch.Tensor:
        """Generate imputed samples using reverse diffusion."""
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L, device=self.device)
        
        for i in range(n_samples):
            # Generate noisy observation for unconditional model
            if self.is_unconditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat_torch[t] ** 0.5) * noisy_obs + self.beta_torch[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)
            
            current_sample = torch.randn_like(observed_data)
            
            # Reverse diffusion process
            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                
                predicted = self.diffmodel(
                    diff_input, side_info, 
                    torch.tensor([t], device=self.device)
                )
                
                coeff1 = 1 / self.alpha_hat_torch[t] ** 0.5
                coeff2 = (1 - self.alpha_hat_torch[t]) / (1 - self.alpha_torch[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)
                
                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha_torch[t - 1]) / (1.0 - self.alpha_torch[t]) * self.beta_torch[t]
                    ) ** 0.5
                    current_sample += sigma * noise
            
            imputed_samples[:, i] = current_sample.detach()
        
        return imputed_samples

    @typechecked
    def _extract_features(
        self, 
        data: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Extract and process features from input data.
        
        Returns:
            observed_data: (B, K, L) - target data
            observed_mask: (B, K, L) - observation mask
            observed_tp: (B, L) - time points
            gt_mask: (B, K, L) - ground truth mask for evaluation
            for_pattern_mask: (B, K, L) - pattern mask for training
            conditions: dict - additional conditions
        """
        x = data['x']  # (B, L, C) past target
        x_target = data['x_target']  # (B, L_pred, C) future target
        conditions = data.get('conditions', None)
        
        B, L_ctx, C = x.shape
        L_pred = x_target.shape[1]
        
        # Concatenate context and prediction for full sequence
        # For forecasting: we observe context, predict future
        observed_data = torch.cat([x, x_target], dim=1)  # (B, L_ctx + L_pred, C)
        observed_data = observed_data.permute(0, 2, 1)  # (B, C, L_total)
        
        L_total = observed_data.shape[2]
        
        # Create masks
        observed_mask = torch.ones_like(observed_data)
        
        # Ground truth mask: 0 for prediction region, 1 for context
        gt_mask = torch.ones_like(observed_data)
        gt_mask[:, :, -L_pred:] = 0  # Mask out future for imputation
        
        # Time points
        observed_tp = torch.arange(L_total, device=self.device).unsqueeze(0).expand(B, -1).float()
        
        # Pattern mask (same as observed_mask for forecasting)
        for_pattern_mask = observed_mask.clone()
        
        return observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, conditions

    @typechecked
    def training_step(self, data: list, idx: int) -> dict:
        """Training step."""
        assert self.training is True
        
        data_dict = self._transform_data(data)
        observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, conditions = self._extract_features(data_dict)
        
        # Generate training mask
        if self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)
        
        side_info = self.get_side_info(observed_tp, cond_mask, conditions)
        loss = self.calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train=1)
        
        self.log(
            "train/loss",
            loss,
            batch_size=observed_data.shape[0],
            logger=True,
            prog_bar=True
        )
        return {"loss": loss}

    def val_test_step(self, data: list, idx: int, prefix: str = 'val'):
        """Validation and test step."""
        assert self.training is False
        
        data_dict = self._transform_data(data)
        x = data_dict['x']  # past target
        x_target = data_dict['x_target']  # future target
        conditions = data_dict.get('conditions', None)
        
        observed_data, observed_mask, observed_tp, gt_mask, _, conditions = self._extract_features(data_dict)
        
        # Use gt_mask as cond_mask for evaluation
        cond_mask = gt_mask
        side_info = self.get_side_info(observed_tp, cond_mask, conditions)
        
        # Generate predictions
        with torch.no_grad():
            samples = self.impute(observed_data, cond_mask, side_info, self.pred_n_samples)
            # samples: (B, n_samples, K, L_total)
            
            # Extract prediction region
            pred_samples = samples[:, :, :, -self.prediction_length:]  # (B, n_samples, K, L_pred)
            pred_samples = pred_samples.permute(0, 1, 3, 2)  # (B, n_samples, L_pred, K)
        
        # Calculate metrics
        mean_pred = pred_samples.mean(dim=1)  # (B, L_pred, K)
        std_pred = pred_samples.std(dim=1)  # uncertainty
        
        # CRPS calculation
        crps = CRPS_ensemble(pred_samples, x_target)
        
        # MSE and MAE
        loss = F.mse_loss(mean_pred, x_target)
        mae = F.l1_loss(mean_pred, x_target)
        
        # Save visualization data
        if idx == 0:
            vis_data = {'input': x, 'target': x_target, 'pred': mean_pred}
            if prefix == 'val':
                self._val_vis_data = vis_data
            elif prefix == 'test':
                self._test_vis_data = vis_data
        
        # Logging
        self.log(f"{prefix}/loss", loss, on_step=False, batch_size=x.shape[0], on_epoch=True, logger=True, prog_bar=True)
        self.log(f"{prefix}/mse", loss, on_step=False, batch_size=x.shape[0], on_epoch=True, logger=True)
        self.log(f"{prefix}/mae", mae, on_step=False, batch_size=x.shape[0], on_epoch=True, logger=True)
        self.log(f"{prefix}/CRPS", crps, on_step=False, batch_size=x.shape[0], on_epoch=True, logger=True)
        
        # Save test predictions
        if prefix == 'test':
            self.test_metrics['past'].append(x.cpu().detach().numpy())
            self.test_metrics['true'].append(x_target.cpu().detach().numpy())
            self.test_metrics['pred'].append(mean_pred.cpu().detach().numpy())
            self.test_metrics['ucty'].append(std_pred.cpu().detach().numpy())
            self.test_metrics['n_pred'].append(pred_samples.cpu().detach().numpy())
        
        return {f"{prefix}/loss": loss}

    @typechecked
    def validation_step(self, data: list, idx: int):
        return self.val_test_step(data, idx, prefix='val')

    @typechecked
    def test_step(self, data: list, idx: int):
        return self.val_test_step(data, idx, prefix='test')

    def plot_trajectory_predictions(
        self, 
        x_input: torch.Tensor, 
        x_target: torch.Tensor, 
        predictions: torch.Tensor, 
        uncertainties: torch.Tensor = None, 
        prefix: str = 'test'
    ):
        """Plot trajectory predictions for visualization."""
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
                title=f'{prefix.capitalize()} Predictions (CSDI)',
            )
            
            # Log to wandb if available
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
        """Callback at end of validation epoch."""
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
        """Callback at end of test epoch."""
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
        past_target: TensorType[float, "batch", "length", "num_series"],
        past_observed_values: TensorType[float, "batch", "length", "num_series"] = None,
        mean: TensorType[float, "batch", 1, "num_series"] = None,
        conditions: dict = None,
    ) -> TensorType[float, "batch", "num_samples", "prediction_length", "num_series"]:
        """
        Forward pass for inference/prediction.
        
        Args:
            past_target: Past observations (B, L, C)
            past_observed_values: Observation mask (optional)
            mean: Mean for normalization (optional)
            conditions: Additional conditions dict
            
        Returns:
            Predictions of shape (B, num_samples, prediction_length, num_series)
        """
        B, L, C = past_target.shape
        
        if past_observed_values is None:
            past_observed_values = torch.ones_like(past_target)
        
        # Create placeholder for future
        future_placeholder = torch.zeros(
            (B, self.prediction_length, C), 
            device=self.device, 
            dtype=past_target.dtype
        )
        
        data_dict = {
            'x': past_target,
            'x_target': future_placeholder,
            'conditions': conditions,
        }
        
        observed_data, observed_mask, observed_tp, gt_mask, _, conditions = self._extract_features(data_dict)
        
        cond_mask = gt_mask
        side_info = self.get_side_info(observed_tp, cond_mask, conditions)
        
        # Generate samples
        samples = self.impute(observed_data, cond_mask, side_info, self.num_samples)
        # samples: (B, num_samples, K, L_total)
        
        # Extract prediction region and rearrange
        pred_samples = samples[:, :, :, -self.prediction_length:]  # (B, num_samples, K, L_pred)
        pred_samples = pred_samples.permute(0, 1, 3, 2)  # (B, num_samples, L_pred, K)
        
        return pred_samples


# Utility function to create model from config
def create_csdi_model(config: dict) -> CSDILightning:
    """
    Create CSDI Lightning model from configuration dictionary.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        CSDILightning model instance
    """
    return CSDILightning(
        target_dim=config.get('target_dim', 1),
        context_length=config.get('context_length', 96),
        prediction_length=config.get('prediction_length', 24),
        num_steps=config.get('num_steps', 50),
        beta_start=config.get('beta_start', 0.0001),
        beta_end=config.get('beta_end', 0.5),
        schedule=config.get('schedule', 'quad'),
        layers=config.get('layers', 4),
        channels=config.get('channels', 64),
        nheads=config.get('nheads', 8),
        diffusion_embedding_dim=config.get('diffusion_embedding_dim', 128),
        timeemb=config.get('timeemb', 128),
        featureemb=config.get('featureemb', 16),
        optimizer_params=config.get('optimizer_params', {"lr": 1e-3, "weight_decay": 1e-6}),
        is_unconditional=config.get('is_unconditional', False),
        target_strategy=config.get('target_strategy', 'random'),
        use_conditions=config.get('use_conditions', False),
        plot_n_samples=config.get('plot_n_samples', 10),
        plot_every_n_epochs=config.get('plot_every_n_epochs', 10),
        save_predictions_to_file_path=config.get('save_predictions_to_file_path', None),
        pred_n_samples=config.get('pred_n_samples', 100),
        num_samples=config.get('num_samples', 1),
    )