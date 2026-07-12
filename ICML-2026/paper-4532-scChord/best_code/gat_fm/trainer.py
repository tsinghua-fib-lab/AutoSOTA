"""
Training utilities for GAT-FM.

This module provides:
1. GATFM_Wrapper: Training wrapper with Flow Matching
2. Trainer: Full training loop with logging and checkpointing
3. Evaluation metrics (PCC, RMSE, etc.)

Architecture follows the design in GAT-Diffusion-Protein-Prediction.md.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from .model import GATFM, DITFM
from .graph import ProteinGraph
from .sampling import ConditionalFlowMatcher, FlowMatchingSampler, GuidedFlowMatchingSampler, create_sample_mask

@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model
    model_type: str = 'gat_fm'  # 'gat_fm' or 'dit_fm'
    hidden_size: int = 256
    depth: int = 6
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    
    # Flow Matching
    sigma: float = 0.0  # OT-CFM noise scale
    
    # Sampling evaluation during training
    sample_mask_ratio: float = 0.2  # Fraction of proteins to mask for sampling evaluation
    sample_num_steps: int = 50  # Number of ODE steps for sampling
    sample_solver: str = 'euler'  # ODE solver for sampling
    sample_num_batches: int = 10  # Number of batches to sample during training evaluation
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1
    save_interval: int = 10
    
    # Paths
    output_dir: str = 'outputs'
    run_name: str = 'gat_fm_run'

class GATFMWrapper(nn.Module):
    """
    Training wrapper for GAT-FM with Flow Matching.
    
    Implements the forward pass for training:
    1. Sample noise x0
    2. Get flow interpolation (t, x_t, target_v)
    3. Predict velocity v(x_t, t, conditions)
    4. Compute masked MSE loss
    """
    
    def __init__(
        self,
        backbone: Union[GATFM, DITFM],
        sigma: float = 0.0,
    ):
        """
        Args:
            backbone: GAT-FM or DITFM model
            sigma: Noise scale for OT-CFM (0 = deterministic)
        """
        super().__init__()
        self.backbone = backbone
        self.flow_matcher = ConditionalFlowMatcher(sigma=sigma)
    
    def forward(
        self,
        x1: torch.Tensor,
        edge_index: torch.Tensor,
        cond_rna: torch.Tensor,
        cond_dataset: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Training forward pass.
        
        Args:
            x1: Target protein expression (B, P) - normalized
            edge_index: Graph edges (2, E)
            cond_rna: RNA embeddings (B, 512)
            cond_dataset: Dataset IDs (B,)
            mask: Protein mask (B, P), 1 = observed
            
        Returns:
            loss: Scalar loss
            metrics: Dict with additional metrics
        """
        # Sample noise
        x0 = torch.randn_like(x1)
        
        # Get flow matching samples
        t, x_t, target_v = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        
        # Predict velocity
        pred_v = self.backbone(
            x_t=x_t,
            t=t,
            edge_index=edge_index,
            cond_rna=cond_rna,
            cond_dataset=cond_dataset,
            mask=mask,
        )
        
        # Compute masked loss
        loss = self.flow_matcher.compute_loss(pred_v, target_v, mask)
        
        # Additional metrics
        with torch.no_grad():
            metrics = {
                'mean_t': t.mean(),
            }
        
        return loss, metrics

class Trainer:
    """
    Full training loop for GAT-FM.
    
    Handles:
    - Training loop with gradient accumulation
    - Validation and evaluation
    - Checkpointing and logging
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: GATFMWrapper,
        protein_graph: ProteinGraph,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        device: torch.device = None,
    ):
        """
        Args:
            model: GATFMWrapper
            protein_graph: ProteinGraph for graph structure
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration
            device: Training device
        """
        self.config = config or TrainingConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.protein_graph = protein_graph.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Setup scheduler with warmup
        warmup_steps = self.config.warmup_epochs * len(train_loader)
        total_steps = self.config.epochs * len(train_loader)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir) / self.config.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        metrics_sum = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            x1 = batch['protein_expr'].to(self.device)
            mask = batch['protein_mask'].to(self.device)
            rna_embed = batch['rna_embed'].to(self.device)
            dataset_id = batch['dataset_id'].to(self.device)
            
            # Get edge index for batched graph
            # The model treats each protein as a node and performs internal batching.
            # Use the base graph here and let the model build batched edges.
            edge_index = self.protein_graph.base_edge_index
            
            # Forward pass
            loss, metrics = self.model(
                x1=x1,
                edge_index=edge_index,
                cond_rna=rna_embed,
                cond_dataset=dataset_id,
                mask=mask,
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            batch_size = x1.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v.item() * batch_size
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'lr': self.scheduler.get_last_lr()[0],
                })
            
            self.global_step += 1
        
        # Compute epoch metrics
        epoch_metrics = {
            'loss': total_loss / total_samples,
        }
        for k, v in metrics_sum.items():
            epoch_metrics[k] = v / total_samples
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        metrics_sum = {}
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            x1 = batch['protein_expr'].to(self.device)
            mask = batch['protein_mask'].to(self.device)
            rna_embed = batch['rna_embed'].to(self.device)
            dataset_id = batch['dataset_id'].to(self.device)
            
            # Validation uses the same base graph; batching is handled in-model.
            edge_index = self.protein_graph.base_edge_index
            
            loss, metrics = self.model(
                x1=x1,
                edge_index=edge_index,
                cond_rna=rna_embed,
                cond_dataset=dataset_id,
                mask=mask,
            )
            
            batch_size = x1.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v.item() * batch_size
        
        val_metrics = {
            'val_loss': total_loss / total_samples,
        }
        for k, v in metrics_sum.items():
            val_metrics[f'val_{k}'] = v / total_samples
        
        return val_metrics
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.output_dir / 'best.ckpt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
    
    @torch.no_grad()
    def sample_and_evaluate(self, num_batches: int = None) -> Dict[str, float]:
            """
            Optimized Block-wise Sampling:
            Since protein coverage is identical across cells, we compute masks for the whole batch at once.
            This reduces inference calls from (B * K) to just (K).
            """
            if self.val_loader is None:
                return {}
            
            self.model.eval()
            
            sampler = GuidedFlowMatchingSampler(
                model=self.model.backbone,
                num_steps=self.config.sample_num_steps,
                solver=self.config.sample_solver,
            )
            
            epoch_preds = []
            epoch_targets = []
            epoch_eval_masks = []
            
            for batch in tqdm(self.val_loader, desc='Batch-Parallel Sampling Eval'):
                x1 = batch['protein_expr'].to(self.device)   # (B, P)
                mask = batch['protein_mask'].to(self.device) # (B, P)
                rna_embed = batch['rna_embed'].to(self.device)
                dataset_id = batch['dataset_id'].to(self.device)
                edge_index = self.protein_graph.base_edge_index.to(self.device)
                
                B, P = x1.shape
                
                # Collect predictions and evaluation masks for this batch.
                batch_preds_collector = torch.zeros_like(x1)
                batch_eval_mask_collector = torch.zeros_like(mask)
                
                # ============================================================
                # Optimization: perform shared blocking when protein coverage is
                # consistent within a batch.
                # ============================================================
                
                # 1) Build observed indices from the first row.
                # Assumption: cells in this batch share the same valid proteins.
                first_row_mask = mask[0] > 0
                obs_indices = torch.where(first_row_mask)[0].cpu().numpy()
                num_obs = len(obs_indices)
                
                if num_obs > 0:
                    # 2) Compute block partitioning.
                    block_size = max(1, int(round(self.config.sample_mask_ratio * num_obs)))
                    n_blocks = int(np.ceil(num_obs / block_size))
                    
                    # 3) Shuffle and split observed proteins into blocks.
                    shuffled_obs = np.copy(obs_indices)
                    np.random.shuffle(shuffled_obs)
                    # array_split ensures all indices are assigned.
                    block_list = np.array_split(shuffled_obs, n_blocks)
                    
                    # 4) Iterate over blocks (K loops instead of B*K).
                    for block in block_list:
                        # block contains protein indices predicted in this pass.
                        
                        # Build a batch-level mask:
                        # all cells use current block as prediction targets.
                        batch_sample_mask = torch.zeros_like(mask)
                        batch_sample_mask[:, block] = 1 
                        
                        # Parallel sampling for all cells in this batch.
                        pred = sampler.sample(
                            x1_true=x1,
                            sample_mask=batch_sample_mask,
                            edge_index=edge_index,
                            cond_rna=rna_embed,
                            cond_dataset=dataset_id,
                            full_mask=mask,
                            device=self.device,
                        )
                        
                            # 5) Vectorized write-back of current block predictions.
                        batch_preds_collector[:, block] = pred[:, block]
                        batch_eval_mask_collector[:, block] = 1.0

                # ============================================================
                        # End optimization block
                # ============================================================

                epoch_preds.append(batch_preds_collector.cpu().numpy())
                epoch_targets.append(x1.cpu().numpy())
                epoch_eval_masks.append(batch_eval_mask_collector.cpu().numpy())
            
            if not epoch_preds:
                return {}
            
            # Concatenate batch outputs and compute metrics.
            preds = np.concatenate(epoch_preds, axis=0)
            targets = np.concatenate(epoch_targets, axis=0)
            sample_masks = np.concatenate(epoch_eval_masks, axis=0)
            
            pcc_protein, _ = compute_pcc_protein(preds, targets, sample_masks, standardize=True)
            pcc_cell, _ = compute_pcc_cell(preds, targets, sample_masks, standardize=True)
            rmse = compute_rmse_standardized(preds, targets, sample_masks, standardize=True)
            cmd_cell = compute_cmd_cell(preds, targets, sample_masks, standardize=True)
            cmd_protein = compute_cmd_protein(preds, targets, sample_masks, standardize=True)
            
            metrics = {
                'sample_pcc_protein': pcc_protein,
                'sample_pcc_cell': pcc_cell,
                'sample_rmse': rmse,
                'sample_cmd_cell': cmd_cell,
                'sample_cmd_protein': cmd_protein,
            }
            
            return metrics
    def train(self):
        """Full training loop."""
        print(f"Training on {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        # Track best sample metrics
        self.best_sample_pcc = -float('inf')
        self.sample_history = []
        
        for epoch in range(1, self.config.epochs + 1):
            # Training
            train_metrics = self.train_epoch(epoch)
            self.train_history.append(train_metrics)
            
            # Validation
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate()
                self.val_history.append(val_metrics)
                
                # Check for best model
                val_loss = val_metrics.get('val_loss', float('inf'))
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                # Sampling evaluation: after warmup and when val_loss is new best
                sample_metrics = {}
                if is_best and epoch > self.config.warmup_epochs:
                    print(f"  Val loss improved! Running sampling evaluation...")
                    sample_metrics = self.sample_and_evaluate()
                    self.sample_history.append({
                        'epoch': epoch,
                        **sample_metrics
                    })
                    
                    # Track best sample PCC
                    sample_pcc = sample_metrics.get('sample_pcc_protein', -float('inf'))
                    if sample_pcc > self.best_sample_pcc:
                        self.best_sample_pcc = sample_pcc
                
                # Logging
                log_str = f"Epoch {epoch}: "
                log_str += " | ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
                log_str += " | " + " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                if sample_metrics:
                    log_str += " | " + " | ".join(f"{k}: {v:.4f}" for k, v in sample_metrics.items())
                print(log_str)
                
                # Save checkpoint
                if epoch % self.config.save_interval == 0:
                    self.save_checkpoint(f'epoch_{epoch}.ckpt', is_best=is_best)
            else:
                log_str = f"Epoch {epoch}: "
                log_str += " | ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
                print(log_str)
        
        # Save final checkpoint
        self.save_checkpoint('final.ckpt')
        
        # Print best sample metrics
        if self.sample_history:
            print(f"\nBest sampling metrics:")
            print(f"  Best Sample PCC (protein): {self.best_sample_pcc:.4f}")
        
        return self.train_history, self.val_history

# =============================================================================
# Evaluation Metrics
# =============================================================================

def standardize_for_evaluation(
    data: np.ndarray,
    mask: np.ndarray = None,
) -> np.ndarray:
    """
    Standardize data for fair evaluation.
    
    Following the evaluation protocol in ComputePCC&CMD&RMSE.ipynb:
    - Per-protein z-score normalization (zero mean, unit variance)
    
    Args:
        data: Array of shape (N, P) where N is samples, P is proteins
        mask: Optional mask (N, P), 1 = observed
        
    Returns:
        Standardized data with same shape
    """
    data = data.copy()
    
    for i in range(data.shape[1]):
        col = data[:, i]
        if mask is not None:
            m = mask[:, i] > 0
            if m.sum() > 1:
                mean = col[m].mean()
                std = col[m].std()
                if std > 1e-8:
                    data[:, i] = (col - mean) / std
                else:
                    data[:, i] = col - mean
        else:
            mean = col.mean()
            std = col.std()
            if std > 1e-8:
                data[:, i] = (col - mean) / std
            else:
                data[:, i] = col - mean
    
    return data

def compute_pcc_protein(
    pred: np.ndarray, 
    target: np.ndarray, 
    mask: np.ndarray = None,
    standardize: bool = True,
) -> Tuple[float, List[float]]:
    """
    Compute protein-wise Pearson Correlation Coefficient.
    
    For each protein, compute PCC across all cells.
    Following ComputePCC&CMD&RMSE.ipynb protocol.
    
    Args:
        pred: Predictions (N, P) where N is cells, P is proteins
        target: Ground truth (N, P)
        mask: Observation mask (N, P), 1 = observed
        standardize: Whether to standardize data before computing PCC
        
    Returns:
        mean_pcc: Average PCC across proteins
        protein_pccs: List of PCC for each protein
    """
    if standardize:
        pred = standardize_for_evaluation(pred, mask)
        target = standardize_for_evaluation(target, mask)
    
    protein_pccs = []
    for i in range(pred.shape[1]):
        p, t = pred[:, i], target[:, i]
        if mask is not None:
            m = mask[:, i] > 0
            p, t = p[m], t[m]
        if len(p) > 1:
            std_p, std_t = np.std(p), np.std(t)
            if std_p > 1e-8 and std_t > 1e-8:
                pcc = np.corrcoef(p, t)[0, 1]
                if not np.isnan(pcc):
                    protein_pccs.append(pcc)
    
    mean_pcc = np.mean(protein_pccs) if protein_pccs else 0.0
    return mean_pcc, protein_pccs

def compute_pcc_cell(
    pred: np.ndarray, 
    target: np.ndarray, 
    mask: np.ndarray = None,
    standardize: bool = True,
) -> Tuple[float, List[float]]:
    """
    Compute cell-wise Pearson Correlation Coefficient.
    
    For each cell, compute PCC across all proteins.
    
    Args:
        pred: Predictions (N, P) where N is cells, P is proteins
        target: Ground truth (N, P)
        mask: Observation mask (N, P), 1 = observed
        standardize: Whether to standardize data before computing PCC
        
    Returns:
        mean_pcc: Average PCC across cells
        cell_pccs: List of PCC for each cell
    """
    if standardize:
        pred = standardize_for_evaluation(pred, mask)
        target = standardize_for_evaluation(target, mask)
    
    cell_pccs = []
    for i in range(pred.shape[0]):
        p, t = pred[i, :], target[i, :]
        if mask is not None:
            m = mask[i, :] > 0
            p, t = p[m], t[m]
        if len(p) > 1:
            std_p, std_t = np.std(p), np.std(t)
            if std_p > 1e-8 and std_t > 1e-8:
                pcc = np.corrcoef(p, t)[0, 1]
                if not np.isnan(pcc):
                    cell_pccs.append(pcc)
    
    mean_pcc = np.mean(cell_pccs) if cell_pccs else 0.0
    return mean_pcc, cell_pccs

def compute_rmse_standardized(
    pred: np.ndarray, 
    target: np.ndarray, 
    mask: np.ndarray = None,
    standardize: bool = True,
) -> float:
    """
    Compute RMSE on standardized data.
    
    Following the protocol in ComputePCC&CMD&RMSE.ipynb.
    
    Args:
        pred: Predictions (N, P)
        target: Ground truth (N, P)
        mask: Observation mask (N, P), 1 = observed
        standardize: Whether to standardize data before computing RMSE
        
    Returns:
        RMSE value
    """
    if standardize:
        pred = standardize_for_evaluation(pred, mask)
        target = standardize_for_evaluation(target, mask)
    
    diff = pred - target
    if mask is not None:
        diff = diff * mask
        return np.sqrt((diff ** 2).sum() / mask.sum())
    return np.sqrt((diff ** 2).mean())

def cmd_dist(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute Centered Mean Distance (CMD) between two correlation matrices.
    
    CMD = 1 - (A*B) / (||A||_F * ||B||_F)
    
    Following the protocol in ComputePCC&CMD&RMSE.ipynb.
    
    Args:
        A: First correlation matrix
        B: Second correlation matrix (same shape as A)
        
    Returns:
        CMD distance (lower is better, 0 = identical)
    """
    a = np.multiply(A, B).sum()
    b = np.linalg.norm(A, 'fro') * np.linalg.norm(B, 'fro')
    return 1.0 - a / (b + 1e-8)

def compute_cmd_cell(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray = None,
    standardize: bool = True,
) -> float:
    """
    Compute cell-wise CMD (Centered Mean Distance).
    
    Computes CMD between cell-cell correlation matrices of predictions and targets.
    Following ComputePCC&CMD&RMSE.ipynb protocol.
    
    Args:
        pred: Predictions (N, P) where N is cells, P is proteins
        target: Ground truth (N, P)
        mask: Observation mask (N, P), 1 = observed
        standardize: Whether to standardize data before computing correlations
        
    Returns:
        CMD value (lower is better)
    """
    if standardize:
        pred = standardize_for_evaluation(pred, mask)
        target = standardize_for_evaluation(target, mask)
    
    # Convert to DataFrame for easier correlation computation
    # Use mask to filter valid observations
    if mask is not None:
        # Only use rows with sufficient observations
        valid_rows = mask.sum(axis=1) > 1  # Need at least 2 proteins per cell
        pred_filtered = pred[valid_rows, :]
        target_filtered = target[valid_rows, :]
    else:
        pred_filtered = pred
        target_filtered = target
    
    if pred_filtered.shape[0] < 2:
        return 1.0  # Cannot compute correlation with < 2 samples
    
    # Compute cell-cell correlation matrices (transpose to get cell correlations)
    pred_df = pd.DataFrame(pred_filtered.T)  # (P, N) -> cells are columns
    target_df = pd.DataFrame(target_filtered.T)
    
    pred_corr = pred_df.corr()
    target_corr = target_df.corr()
    
    # Drop NaN rows/columns
    pred_corr = pred_corr.dropna(how='all', axis=1).dropna(how='all')
    target_corr = target_corr.dropna(how='all', axis=1).dropna(how='all')
    
    # Find intersection of valid indices
    common_idx = pred_corr.index.intersection(target_corr.index)
    common_cols = pred_corr.columns.intersection(target_corr.columns)
    
    if len(common_idx) == 0 or len(common_cols) == 0:
        return 1.0
    
    pred_corr_aligned = pred_corr.loc[common_idx, common_cols]
    target_corr_aligned = target_corr.loc[common_idx, common_cols]
    
    # Compute CMD (following notebook: CMD_dist(A.values.T, B.values))
    cmd = cmd_dist(pred_corr_aligned.values.T, target_corr_aligned.values)
    return cmd

def compute_cmd_protein(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray = None,
    standardize: bool = True,
) -> float:
    """
    Compute protein-wise CMD (Centered Mean Distance).
    
    Computes CMD between protein-protein correlation matrices of predictions and targets.
    Following ComputePCC&CMD&RMSE.ipynb protocol.
    
    Args:
        pred: Predictions (N, P) where N is cells, P is proteins
        target: Ground truth (N, P)
        mask: Observation mask (N, P), 1 = observed
        standardize: Whether to standardize data before computing correlations
        
    Returns:
        CMD value (lower is better)
    """
    if standardize:
        pred = standardize_for_evaluation(pred, mask)
        target = standardize_for_evaluation(target, mask)
    
    # Convert to DataFrame for easier correlation computation
    # Use mask to filter valid observations
    if mask is not None:
        # Only use columns (proteins) with sufficient observations
        valid_cols = mask.sum(axis=0) > 1  # Need at least 2 cells per protein
        pred_filtered = pred[:, valid_cols]
        target_filtered = target[:, valid_cols]
    else:
        pred_filtered = pred
        target_filtered = target
    
    if pred_filtered.shape[1] < 2:
        return 1.0  # Cannot compute correlation with < 2 proteins
    
    # Compute protein-protein correlation matrices
    pred_df = pd.DataFrame(pred_filtered)  # (N, P)
    target_df = pd.DataFrame(target_filtered)
    
    pred_corr = pred_df.corr()
    target_corr = target_df.corr()
    
    # Drop NaN rows/columns
    pred_corr = pred_corr.dropna(how='all', axis=1).dropna(how='all')
    target_corr = target_corr.dropna(how='all', axis=1).dropna(how='all')
    
    # Find intersection of valid indices/columns
    common_idx = pred_corr.index.intersection(target_corr.index)
    common_cols = pred_corr.columns.intersection(target_corr.columns)
    
    if len(common_idx) == 0 or len(common_cols) == 0:
        return 1.0
    
    pred_corr_aligned = pred_corr.loc[common_idx, common_cols]
    target_corr_aligned = target_corr.loc[common_idx, common_cols]
    
    # Compute CMD (following notebook: CMD_dist(A.values.T, B.values))
    cmd = cmd_dist(pred_corr_aligned.values.T, target_corr_aligned.values)
    return cmd

def compute_pcc(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Compute Pearson Correlation Coefficient.
    
    Args:
        pred: Predictions (N, P) or (P,)
        target: Ground truth (N, P) or (P,)
        mask: Observation mask, optional
        
    Returns:
        Average PCC across proteins
    """
    if mask is not None:
        pred = pred * mask
        target = target * mask
    
    if pred.ndim == 1:
        return np.corrcoef(pred, target)[0, 1]
    
    pccs = []
    for i in range(pred.shape[1]):
        p, t = pred[:, i], target[:, i]
        if mask is not None:
            m = mask[:, i] > 0
            p, t = p[m], t[m]
        if len(p) > 1 and np.std(p) > 0 and np.std(t) > 0:
            pccs.append(np.corrcoef(p, t)[0, 1])
    
    return np.mean(pccs) if pccs else 0.0

def compute_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Compute Root Mean Square Error.
    
    Args:
        pred: Predictions
        target: Ground truth
        mask: Observation mask, optional
        
    Returns:
        RMSE
    """
    diff = pred - target
    if mask is not None:
        diff = diff * mask
        return np.sqrt((diff ** 2).sum() / mask.sum())
    return np.sqrt((diff ** 2).mean())

def compute_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        pred: Predictions
        target: Ground truth
        mask: Observation mask, optional
        
    Returns:
        MAE
    """
    diff = np.abs(pred - target)
    if mask is not None:
        diff = diff * mask
        return diff.sum() / mask.sum()
    return diff.mean()

@torch.no_grad()
def evaluate_model(
    model: GATFM,
    dataloader: DataLoader,
    protein_graph: ProteinGraph,
    sampler: FlowMatchingSampler,
    device: torch.device,
    norm_stats = None,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: GAT-FM model
        dataloader: Data loader
        protein_graph: Protein graph
        sampler: Flow matching sampler
        device: Device
        norm_stats: Normalization statistics for denormalization
        
    Returns:
        Dict of evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_masks = []
    
    for batch in tqdm(dataloader, desc='Evaluating'):
        x1 = batch['protein_expr'].to(device)
        mask = batch['protein_mask'].to(device)
        rna_embed = batch['rna_embed'].to(device)
        dataset_id = batch['dataset_id'].to(device)
        
        edge_index = protein_graph.base_edge_index.to(device)
        
        # Generate predictions
        pred = sampler.sample(
            shape=x1.shape,
            edge_index=edge_index,
            cond_rna=rna_embed,
            cond_dataset=dataset_id,
            mask=mask,
            device=device,
        )
        
        all_preds.append(pred.cpu().numpy())
        all_targets.append(x1.cpu().numpy())
        all_masks.append(mask.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    
    # Denormalize if stats provided
    if norm_stats is not None:
        from .data import denormalize_protein_expression
        preds = denormalize_protein_expression(preds, norm_stats)
        targets = denormalize_protein_expression(targets, norm_stats)
    
    # Compute metrics
    metrics = {
        'pcc': compute_pcc(preds, targets, masks),
        'rmse': compute_rmse(preds, targets, masks),
        'mae': compute_mae(preds, targets, masks),
    }
    
    return metrics
