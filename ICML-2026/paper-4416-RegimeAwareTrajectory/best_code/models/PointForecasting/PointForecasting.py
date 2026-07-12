"""
Universal Training Framework based on BaseModule
================================================
A flexible PyTorch Lightning training framework that can be used to train any model.

Usage:
    1. Inherit from UniversalTrainingModule
    2. Override _build_model() to return your model
    3. Override _compute_loss() if you need custom loss computation
    4. Configure your datamodule and trainer
"""

from typing import Optional, Dict, Any, Callable, List, Tuple, Union
import torch
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import sys
import math
sys.path.extend(['./', '../', '../../'])
from models._base.base_module import BaseModule
from models.PointForecasting._point_base import _PointBaseModel



# ============================================================================
# Base Training Module
# ============================================================================

class UniversalTrainingModule(BaseModule, _PointBaseModel):
    """
    A universal training module that can train any PyTorch model.
    
    Features:
    - Flexible model architecture (override _build_model)
    - Configurable loss functions
    - Multiple optimizer support (Adam, AdamW, SGD, etc.)
    - Learning rate scheduling
    - Automatic mixed precision (AMP) support
    - Validation visualization
    - Test metrics logging and saving
    
    Args:
        model_config: Dictionary containing model configuration
        learning_rate: Initial learning rate
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
        scheduler_type: Type of scheduler ('step', 'cosine', 'plateau', None)
        loss_type: Type of loss ('mse', 'mae', 'huber', 'custom')
        weight_decay: Weight decay for optimizer
        log_val_plots: Whether to log validation plots
        plot_n_samples: Number of samples to plot
        plot_every_n_epochs: Plot frequency
        save_predictions_to_file_path: Path to save test predictions
    """
    
    def __init__(
        self,
        args, 
        loss_type: str = 'mse',
        # Training
        lr: float = 1e-4,
        min_lr: float = 2e-6,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 1000,
        max_epochs: int = 100000,
        # Visualization params
        plot_n_samples: int = 4,
        plot_every_n_epochs: int = 5,
        save_predictions_to_file_path: Optional[str] = None,
        # Additional params
        **kwargs,
    ):
        super().__init__(log_val_plots=True, 
                         plot_n_samples=plot_n_samples, 
                         plot_every_n_epochs=plot_every_n_epochs, 
                         save_predictions_to_file_path=save_predictions_to_file_path)
        self.save_hyperparameters()
        
        # Core config
        self.args = args
        self.loss_type = loss_type

        # Training params
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        
        # Build model and loss
        self.model = self._build_model()
        self.criterion = self._build_criterion()
    
    # ========================================================================
    # Methods to Override
    # ========================================================================
    
    def _build_model(self) -> nn.Module:
        """
        Build and return your model here.
        Override this method to create your custom model.
        
        Returns:
            nn.Module: Your PyTorch model
            
        Example:
            def _build_model(self):
                return MyCustomModel(
                    input_dim=self.model_config['input_dim'],
                    hidden_dim=self.model_config['hidden_dim'],
                    output_dim=self.model_config['output_dim'],
                )
        """
        return self.model_dict[self.args.model].Model(self.args).float()
    
    def _extract_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract inputs, targets, and metadata from a batch.
        Override this method if your data format is different.
        
        Args:
            batch: A batch from your dataloader
            
        Returns:
            Tuple of (inputs, targets, metadata_dict)
        """
        if isinstance(batch, dict):
            batch_x = batch['x']
            batch_y = batch['y']
            batch_x_mark = batch['x_mark'] if 'x_mark' in batch else None
            batch_y_mark = batch['y_mark'] if 'y_mark' in batch else None
            conditions = batch['conditions'] if 'conditions' in batch else None

            if self.args.wo_mark:
                batch_x_mark, batch_y_mark = None, None
            return batch_x, batch_y, batch_x_mark, batch_y_mark
        else:
            raise ValueError(f"Unsupported batch format: {type(batch)}")
    
    def _forward(self, x: torch.Tensor, x_mark: torch.Tensor, dec_inp: torch.Tensor, y_mark: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Override this method if your model has a complex forward signature.
        
        Args:
            x: Input tensor
            metadata: Additional metadata from batch
            
        Returns:
            Model output tensor
        """
        return self.model(x, x_mark, dec_inp, y_mark)
    
    def _compute_loss(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor,
        metadata: Dict[str, Any] = None
    ) -> torch.Tensor:
        """
        Compute the loss.
        Override this method for custom loss computation.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            metadata: Additional metadata
            
        Returns:
            Loss tensor
        """
        return self.criterion(outputs, targets)
    
    def _compute_metrics(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor,
        prefix: str = 'train'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute additional metrics.
        Override this method to add custom metrics.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            prefix: Metric prefix ('train', 'val', 'test')
            
        Returns:
            Dictionary of metric name -> value
        """
        with torch.no_grad():
            mse = nn.functional.mse_loss(outputs, targets)
            mae = nn.functional.l1_loss(outputs, targets)
            return {
                f'{prefix}/mse': mse,
                f'{prefix}/mae': mae,
            }
    
    # ========================================================================
    # Core Training Logic
    # ========================================================================
    
    def _build_criterion(self) -> nn.Module:
        """Build the loss criterion based on loss_type."""
        loss_map = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'l1': nn.L1Loss(),
            'huber': nn.HuberLoss(),
            'smooth_l1': nn.SmoothL1Loss(),
            'cross_entropy': nn.CrossEntropyLoss(),
            'bce': nn.BCELoss(),
            'bce_with_logits': nn.BCEWithLogitsLoss(),
        }
        if self.loss_type in loss_map:
            return loss_map[self.loss_type]
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. "
                           f"Available: {list(loss_map.keys())} or override _compute_loss()")
    
    def training_step(self, batch, batch_idx):
        """Standard training step."""
        x, y, x_mark, y_mark = self._extract_batch(batch)
        # decoder input
        dec_inp = torch.zeros_like(y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float().to(self.device)
        y_mark_dec = torch.cat([x_mark[:, -self.args.label_len:, :], y_mark[:, -self.args.pred_len:, :]], dim=1)
        
        # Forward pass
        outputs = self.model(x, x_mark, dec_inp, y_mark_dec)
        outputs = outputs[:, -self.args.pred_len:, :]

        # Compute loss
        loss = self._compute_loss(outputs, y)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Compute and log additional metrics
        metrics = self._compute_metrics(outputs, y, prefix='train')
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True)
        
        return loss
    
    def val_test_step(self, batch, batch_idx, prefix: str):
        """Shared validation/test step."""
        x, y, x_mark, y_mark = self._extract_batch(batch)
        # decoder input
        dec_inp = torch.zeros_like(y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float().to(self.device)
        y_mark_dec = torch.cat([x_mark[:, -self.args.label_len:, :], y_mark[:, -self.args.pred_len:, :]], dim=1)
        
        # Forward pass
        outputs = self._forward(x, x_mark, dec_inp, y_mark_dec)
        outputs = outputs[:, -self.args.pred_len:, :]
        
        # Compute loss
        loss = self._compute_loss(outputs, y)
        
        # Log metrics
        self.log(f'{prefix}/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Compute and log additional metrics
        metrics = self._compute_metrics(outputs, y, prefix=prefix)
        for name, value in metrics.items():
            self.log(name, value, on_step=False, on_epoch=True)
        
        # Store data for visualization
        if prefix == 'val' and batch_idx == 0:
            self._val_vis_data = {
                'input': x.detach().cpu(),
                'target': y.detach().cpu(),
                'pred': outputs.detach().cpu(),
            }
        
        # Store test predictions
        if prefix == 'test':
            self.test_metrics['past'].append(x.detach().cpu().numpy())
            self.test_metrics['pred'].append(outputs.detach().cpu().numpy())
            self.test_metrics['true'].append(y.detach().cpu().numpy())
            
            if batch_idx == 0:
                self._test_vis_data = {
                    'input': x.detach().cpu(),
                    'target': y.detach().cpu(),
                    'pred': outputs.detach().cpu(),
                }
        
        return loss
    
    # ========================================================================
    # Optimizer & Scheduler
    # ========================================================================
    
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

    # ========================================================================
    # Epoch End Hooks
    # ========================================================================
