#!/usr/bin/env python3
"""Trainer for SelfIE with trained adapters."""

from pathlib import Path
from typing import Optional, List
from datetime import datetime
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import Subset, DataLoader
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import Config
from training.data import create_dataloaders
from training.model import SelfIEModel
from training.utils import set_seed

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not available, logging disabled")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("WARNING: mlflow not available, logging disabled")


class Trainer:
    """Trainer for SelfIE bias vector model."""
    
    def __init__(self, config: Config, cache_dir: Optional[str] = None):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Configuration object
            cache_dir: Optional HuggingFace cache directory
        """
        self.config = config
        self.cache_dir = cache_dir
        
        # Set seeds
        set_seed(config.seed)
        print(f"Set random seed: {config.seed}")
        
        # Initialize MLflow or WandB and get run name
        self.mlflow_run = None
        self.wandb_run = None
        
        if config.logging.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
            # Get run name from wandb (it's guaranteed to exist after init)
            if wandb.run is not None:
                self.run_name = wandb.run.name
            else:
                # Fallback in case of unexpected wandb behavior
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.run_name = f"run_{timestamp}"
        else:
            # Generate a fallback run name using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"run_{timestamp}"
        
        # Create dataloaders
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        self.train_loader, self.val_loader, model_dim, self.train_dataset_counts, self.val_dataset_counts = create_dataloaders(
            labels_file=config.data.labels_file,
            batch_size=config.data.batch_size,
            train_dataset_ratios=config.data.train_dataset_ratios,
            val_dataset_ratios=config.data.val_dataset_ratios,
            epoch=0,
            shuffle=config.data.shuffle,
            num_workers=config.data.num_workers,
            seed=config.seed,
            eos_token=config.data.eos_token,
            strip_labels=config.data.strip_labels,
            debug=config.data.debug_dataset_mixing,
        )
        
        # Subsample validation set if requested
        if self.val_loader and config.training.val_fraction < 1.0:
            original_val_size = len(self.val_loader.dataset)
            subset_size = int(original_val_size * config.training.val_fraction)
            
            # Use fixed seed for consistent subsampling across runs
            rng = np.random.RandomState(config.seed)
            subset_indices = rng.choice(original_val_size, size=subset_size, replace=False)
            subset_indices = sorted(subset_indices)  # Sort for reproducibility
            
            # Create subset dataset
            subset_dataset = Subset(self.val_loader.dataset, subset_indices)
            
            # Create new DataLoader with the subset
            self.val_loader = DataLoader(
                subset_dataset,
                batch_size=config.data.batch_size,
                shuffle=False,  # Don't shuffle validation
                num_workers=config.data.num_workers,
            )
            
            print(f"✓ Subsampled validation set: {original_val_size} → {subset_size} examples ({config.training.val_fraction:.1%})")
        
        # Sanity check: warn if validation compute will dominate training
        self._check_validation_compute_ratio()
        
        # Create model
        print("\n" + "="*60)
        print("CREATING MODEL")
        print("="*60)
        self.model = SelfIEModel(config, model_dim, cache_dir=cache_dir)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.current_batch_in_epoch = 0  # Track position within current epoch
        self.best_val_loss = float("inf")
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume from checkpoint if specified
        if config.training.resume_from_checkpoint:
            self._load_checkpoint(config.training.resume_from_checkpoint)
        
        print("\n" + "="*60)
        print("TRAINER INITIALIZED")
        print("="*60)
        print(f"Training examples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation examples: {len(self.val_loader.dataset)}")
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
    
    def _check_validation_compute_ratio(self):
        """
        Sanity check to prevent accidentally wasting compute on validation.
        
        Estimates the ratio of validation forward passes to training forward passes.
        If validation would consume more than 50% of total compute, raises an error.
        
        This correctly accounts for max_steps when calculating the ratio, so it compares
        the validation batches you'll actually run to the training batches you'll actually
        process (limited by max_steps if set).
        
        This catches common mistakes like forgetting to set val_fraction on large datasets.
        """
        if not self.val_loader:
            return  # No validation, nothing to check
        
        config = self.config
        
        # Training examples and batches per epoch
        train_examples_per_epoch = len(self.train_loader.dataset)
        train_batches_per_epoch = len(self.train_loader)
        
        # Validation examples and batches per run (already subsampled if val_fraction < 1)
        val_examples_per_run = len(self.val_loader.dataset)
        val_batches_per_run = len(self.val_loader)
        
        # Number of optimizer steps per epoch (or until max_steps)
        steps_per_epoch = train_batches_per_epoch // config.training.gradient_accumulation_steps
        
        # If max_steps is set and less than a full epoch, use that instead
        if config.training.max_steps is not None and config.training.max_steps < steps_per_epoch:
            steps_to_check = config.training.max_steps
            print(f"⚠️  max_steps ({config.training.max_steps}) < steps_per_epoch ({steps_per_epoch})")
            print(f"   Validation compute check will use max_steps for calculation")
        else:
            steps_to_check = steps_per_epoch
        
        # Number of validation runs 
        val_runs_per_epoch = steps_to_check / config.training.validation_every_n_steps
        
        # Total validation batches across all runs
        total_val_batches = val_batches_per_run * val_runs_per_epoch
        
        # Total training batches that will actually be processed
        # (might be less than train_batches_per_epoch if using max_steps)
        train_batches_to_process = steps_to_check * config.training.gradient_accumulation_steps
        
        # Ratio of validation to training compute (at the forward pass level)
        # Both train and val do one forward pass per batch, so we compare batch counts
        val_to_train_ratio = total_val_batches / train_batches_to_process
        
        # 50% threshold - if validation is more than 1/2 of training compute, something's probably wrong
        THRESHOLD = 0.5
        
        # Compute what the user actually observes: batches between validation runs
        train_batches_between_val_runs = config.training.validation_every_n_steps * config.training.gradient_accumulation_steps
        val_to_train_ratio_per_run = val_batches_per_run / train_batches_between_val_runs if train_batches_between_val_runs > 0 else 0
        
        if val_to_train_ratio > THRESHOLD:
            error_msg = (
                f"\n{'='*60}\n"
                f"⚠️  VALIDATION COMPUTE SANITY CHECK FAILED\n"
                f"{'='*60}\n"
                f"\n"
                f"Validation would consume {val_to_train_ratio:.1%} of compute (threshold: {THRESHOLD:.0%})\n"
                f"\n"
                f"Details:\n"
                f"  • Training examples in dataset: {train_examples_per_epoch:,}\n"
                f"  • Training batches in dataset: {train_batches_per_epoch:,}\n"
                f"  • Gradient accumulation steps: {config.training.gradient_accumulation_steps}\n"
                f"  • Steps per full epoch: {steps_per_epoch:,}\n"
                f"  • Steps to check (limited by max_steps): {steps_to_check:,}\n"
                f"  • Training batches to process: {train_batches_to_process:,}\n"
                f"  • Validation runs: {val_runs_per_epoch:.1f}\n"
                f"  • Val examples per run: {val_examples_per_run:,}\n"
                f"  • Val batches per run: {val_batches_per_run:,}\n"
                f"  • Total val batches: {total_val_batches:,.0f}\n"
                f"\n"
                f"Per validation run:\n"
                f"  • Training batches between val runs: {train_batches_between_val_runs:,}\n"
                f"  • Validation batches per run: {val_batches_per_run:,}\n"
                f"  • Ratio (per run): {val_to_train_ratio_per_run:.1%}\n"
                f"\n"
                f"This usually means you forgot to set val_fraction for a large dataset.\n"
                f"\n"
                f"Fix options:\n"
                f"  1. Add 'val_fraction: 0.1' (or similar) to your config\n"
                f"  2. Increase 'validation_every_n_steps' to validate less often\n"
                f"  3. If this is intentional, you'll need to modify the threshold in trainer.py\n"
                f"{'='*60}"
            )
            raise ValueError(error_msg)
        else:
            print(f"✓ Validation compute check passed:")
            print(f"    Overall: {val_to_train_ratio:.1%} of training compute (threshold: {THRESHOLD:.0%})")
            print(f"    Train batches to process: {train_batches_to_process:,} (steps: {steps_to_check:,}, grad_accum: {config.training.gradient_accumulation_steps})")
            print(f"    Val batches per run: {val_batches_per_run:,}, Validation runs: {val_runs_per_epoch:.1f}, Total val batches: {total_val_batches:.0f}")
            print(f"    Per validation run: {val_batches_per_run:,} val batches / {train_batches_between_val_runs:,} train batches ({val_to_train_ratio_per_run:.1%})")
    
    def _init_mlflow(self):
        """Initialize MLflow logging."""
        # Set tracking URI (ensure it has protocol)
        tracking_uri = self.config.logging.mlflow_tracking_uri
        if not tracking_uri.startswith(("http://", "https://")):
            tracking_uri = f"https://{tracking_uri}"
        mlflow.set_tracking_uri(tracking_uri)
        
        # Get or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.config.logging.mlflow_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.config.logging.mlflow_experiment_name)
                print(f"✓ Created MLflow experiment: {self.config.logging.mlflow_experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                print(f"✓ Using existing MLflow experiment: {self.config.logging.mlflow_experiment_name}")
        except Exception as e:
            print(f"Warning: Failed to get/create experiment: {e}")
            # Fallback: use default experiment
            experiment_id = "0"
        
        # Generate unique run name
        # Priority: wandb_run_name (if set) > experiment_name + timestamp > timestamp only
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.logging.wandb_run_name:
            # If explicit name provided, append timestamp to make it unique
            run_name = f"{self.config.logging.wandb_run_name}_{timestamp}"
        elif self.config.experiment_name:
            # Use experiment_name with timestamp
            run_name = f"{self.config.experiment_name}_{timestamp}"
        else:
            # Fallback: use timestamp with generic prefix
            run_name = f"run_{timestamp}"
        
        # Start run with unique name
        self.mlflow_run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
        
        # Set run color for visual differentiation in MLflow UI
        # Each run gets a unique color based on run ID
        import hashlib
        color_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d3", "#c7c7c7", "#dbdb8d", "#9edae5"
        ]
        # Use run ID to get unique color for each run
        run_id = mlflow.active_run().info.run_id
        hash_val = int(hashlib.md5(run_id.encode()).hexdigest(), 16)
        run_color = color_palette[hash_val % len(color_palette)]
        mlflow.set_tag("mlflow.runColor", run_color)
        
        # Log config as parameters (flatten nested dict)
        config_dict = self.config.to_dict()
        flattened_config = self._flatten_config(config_dict)
        mlflow.log_params(flattened_config)
        
        print(f"✓ Initialized MLflow run: {mlflow.active_run().info.run_id}")
    
    def _flatten_config(self, config_dict: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten nested config dict for MLflow parameter logging."""
        items = []
        for k, v in config_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            else:
                # Skip None values (MLflow doesn't accept None for parameters)
                if v is None:
                    continue
                # Convert non-string values to strings for MLflow
                if not isinstance(v, (str, int, float, bool)):
                    v = str(v)
                items.append((new_key, v))
        return dict(items)
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging (deprecated)."""
        wandb_config = {
            "project": self.config.logging.wandb_project,
            "entity": self.config.logging.wandb_entity,
            "name": self.config.logging.wandb_run_name,
            "config": self.config.to_dict(),
        }
        
        # Remove None values
        wandb_config = {k: v for k, v in wandb_config.items() if v is not None}
        
        self.wandb_run = wandb.init(**wandb_config)
        print(f"✓ Initialized WandB run: {wandb.run.name}")
    
    def _setup_optimizer(self):
        """Setup optimizer with parameter-specific learning rates."""
        # Get learning rates (use defaults if not specified)
        base_lr = self.config.training.learning_rate
        scale_lr = self.config.training.scale_learning_rate or base_lr
        direction_lr = self.config.training.direction_learning_rate or base_lr
        bias_lr = self.config.training.bias_learning_rate or base_lr
        
        # Separate parameter groups based on projection type
        param_groups = []
        
        # Collect parameters by type
        scale_params = []
        direction_params = []
        bias_params = []
        other_params = []
        
        for name, param in self.model.projection.named_parameters():
            if "scale_direction" in name:
                # Separate scale_direction from other scale parameters
                direction_params.append(param)
            elif "log_scale" in name or "base_log_scale" in name:
                # Scalar scale parameters
                scale_params.append(param)
            elif "bias" in name:
                bias_params.append(param)
            else:
                other_params.append(param)
        
        # Add parameter groups
        if scale_params:
            param_groups.append({
                "params": scale_params,
                "lr": scale_lr,
                "weight_decay": 0.0,  # No weight decay on scale
            })
        
        if direction_params:
            param_groups.append({
                "params": direction_params,
                "lr": direction_lr,
                "weight_decay": self.config.training.weight_decay,
            })
        
        if bias_params:
            param_groups.append({
                "params": bias_params,
                "lr": bias_lr,
                "weight_decay": self.config.training.weight_decay,
            })
        
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": base_lr,
                "weight_decay": self.config.training.weight_decay,
            })
        
        # Create optimizer based on config
        optimizer_type = self.config.training.optimizer_type.lower()
        if optimizer_type == "sgd_momentum":
            optimizer = optim.SGD(param_groups, momentum=self.config.training.momentum)
            optimizer_name = f"SGD (momentum={self.config.training.momentum})"
        elif optimizer_type == "adamw":
            optimizer = optim.AdamW(param_groups)
            optimizer_name = "AdamW"
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}. Supported types: 'adamw', 'sgd_momentum'")
        
        print(f"\n✓ Optimizer configured: {optimizer_name}")
        if scale_params:
            print(f"    Scale parameters (log_scale, base_log_scale): LR = {scale_lr}")
        if direction_params:
            print(f"    Direction parameters (scale_direction): LR = {direction_lr}")
        if bias_params:
            print(f"    Bias parameters: LR = {bias_lr}")
        if other_params:
            print(f"    Other parameters (U, V, weight, etc.): LR = {base_lr}")
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        # Calculate total steps
        steps_per_epoch = len(self.train_loader) // self.config.training.gradient_accumulation_steps
        if self.config.training.max_steps is not None:
            total_steps = self.config.training.max_steps
        else:
            total_steps = steps_per_epoch * self.config.training.num_epochs

        if self.config.training.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.config.training.warmup_steps,
                eta_min=0,
            )
        elif self.config.training.scheduler_type == "linear":
            scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps - self.config.training.warmup_steps,
            )
        else:  # constant
            scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=1.0,
                total_iters=total_steps,
            )
        
        # Wrap with warmup
        if self.config.training.warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=self.config.training.warmup_steps,
            )
            self.warmup_scheduler = warmup_scheduler
            self.main_scheduler = scheduler
            scheduler = warmup_scheduler
        else:
            self.warmup_scheduler = None
            self.main_scheduler = scheduler
        
        print(f"✓ Scheduler: {self.config.training.scheduler_type}")
        print(f"    Warmup steps: {self.config.training.warmup_steps}")
        print(f"    Total steps: {total_steps}")
        
        return scheduler
    
    def _get_current_scheduler(self):
        """Get the appropriate scheduler based on current step."""
        if self.warmup_scheduler and self.global_step < self.config.training.warmup_steps:
            return self.warmup_scheduler
        else:
            return self.main_scheduler
    
    def _save_checkpoint(self, suffix: str = ""):
        """Save checkpoint with all training state."""
        checkpoint_name = f"{self.run_name}_step_{self.global_step}{suffix}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "current_batch_in_epoch": self.current_batch_in_epoch,
            "projection_state": self.model.projection.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
            "best_val_loss": self.best_val_loss,
            
            # Metadata for external loading (added for inference compatibility)
            "model_dim": self.model.model_dim,
            "checkpoint_format_version": 1,
            "projection_num_params": self.model.projection.num_parameters(),
        }
        
        if self.warmup_scheduler:
            checkpoint["warmup_scheduler_state"] = self.warmup_scheduler.state_dict()
            checkpoint["main_scheduler_state"] = self.main_scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved checkpoint: {checkpoint_path}")
        
        # Upload to MLflow as artifact
        if self.mlflow_run:
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
        # Also support WandB for backward compatibility
        elif self.wandb_run:
            artifact = wandb.Artifact(
                name=f"checkpoint-{wandb.run.name}-step-{self.global_step}",
                type="model",
            )
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Note: New metadata fields (model_dim, checkpoint_format_version, etc.) are for
        # external inference tools and don't need to be loaded here for training resumption
        
        # Restore training state
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.current_batch_in_epoch = checkpoint.get("current_batch_in_epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
        # Restore model parameters
        self.model.projection.load_state_dict(checkpoint["projection_state"])
        
        # Restore optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        if "warmup_scheduler_state" in checkpoint and self.warmup_scheduler:
            self.warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state"])
            self.main_scheduler.load_state_dict(checkpoint["main_scheduler_state"])
        
        print(f"✓ Resumed from step {self.global_step}, epoch {self.current_epoch}, batch {self.current_batch_in_epoch}")
    
    def _log_metrics(self, metrics: dict, prefix: str = "train"):
        """Log metrics to console and MLflow/WandB."""
        # Console logging
        log_str = f"[{prefix}] Step {self.global_step}: "
        log_str += ", ".join([
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        ])
        print(log_str)
        
        # MLflow logging (preferred)
        if self.mlflow_run:
            mlflow_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            mlflow.log_metrics(mlflow_metrics, step=self.global_step)
        # WandB logging (deprecated, backward compatibility)
        elif self.wandb_run:
            wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            wandb_metrics["step"] = self.global_step
            wandb.log(wandb_metrics)
    
    def _log_singular_values(self):
        """Log singular values of mapping matrix for nontrivial projections."""
        if not (self.mlflow_run or self.wandb_run):
            return
        
        # Check if projection has singular value computation
        if not hasattr(self.model.projection, 'get_singular_values'):
            return
        
        try:
            # Get singular values
            singular_values = self.model.projection.get_singular_values()
            sv_numpy = singular_values.cpu().numpy()
            
            # Log statistics (MLflow doesn't have histograms, so log aggregated stats)
            stats = {
                "train/singular_value_max": float(sv_numpy[0]) if len(sv_numpy) > 0 else 0.0,
                "train/singular_value_mean": float(sv_numpy.mean()),
                "train/singular_value_min": float(sv_numpy[-1]) if len(sv_numpy) > 0 else 0.0,
                "train/singular_value_std": float(sv_numpy.std()),
            }
            
            # Log top 10 singular values individually (if we have that many)
            for i in range(min(10, len(sv_numpy))):
                stats[f"train/singular_value_top_{i+1}"] = float(sv_numpy[i])
            
            # MLflow logging
            if self.mlflow_run:
                mlflow.log_metrics(stats, step=self.global_step)
                # Also log histogram data as JSON artifact
                import json
                import tempfile
                import os
                temp_file = None
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        temp_file = f.name
                        json.dump({"singular_values": sv_numpy.tolist()}, f)
                    mlflow.log_artifact(temp_file, artifact_path="singular_values")
                finally:
                    # Clean up temporary file
                    if temp_file and os.path.exists(temp_file):
                        os.remove(temp_file)
            
            # WandB logging (deprecated, backward compatibility)
            if self.wandb_run:
                wandb.log({
                    "train/singular_values_histogram": wandb.Histogram(sv_numpy),
                    "step": self.global_step,
                })
                wandb.log({**stats, "step": self.global_step})
        
        except Exception as e:
            # Don't let logging errors crash training
            print(f"Warning: Failed to log singular values: {e}")
            print("Training will continue...")
    
    def _log_low_rank_metrics(self):
        """Log low-rank component metrics for scalar_affine_plus_low_rank projection."""
        if not (self.mlflow_run or self.wandb_run):
            return
        
        # Only log for scalar_affine_plus_low_rank projection type
        if self.config.projection.type != "scalar_affine_plus_low_rank":
            return
        
        # Check if projection has low-rank ratio computation
        if not hasattr(self.model.projection, 'get_low_rank_to_diagonal_ratio'):
            return
        
        try:
            # Get the ratio ||UV^T||_F / ||diag(s)||
            ratio = self.model.projection.get_low_rank_to_diagonal_ratio()
            
            # MLflow logging
            if self.mlflow_run:
                mlflow.log_metric("train/low_rank_to_diagonal_ratio", ratio, step=self.global_step)
            
            # WandB logging (deprecated, backward compatibility)
            if self.wandb_run:
                wandb.log({
                    "train/low_rank_to_diagonal_ratio": ratio,
                    "step": self.global_step,
                })
        
        except Exception as e:
            # Don't let logging errors crash training
            print(f"Warning: Failed to log low-rank metrics: {e}")
            print("Training will continue...")
    
    def _refresh_train_loader(self, epoch: int):
        """Recreate train loader with new subsampling for the epoch."""
        if not self.config.data.train_dataset_ratios or not self.config.data.resample_each_epoch:
            return  # Nothing to do
        
        print(f"\nRefreshing train loader for epoch {epoch}...")
        
        # Recreate just the train loader
        train_loader, _, _, self.train_dataset_counts, _ = create_dataloaders(
            labels_file=self.config.data.labels_file,
            batch_size=self.config.data.batch_size,
            train_dataset_ratios=self.config.data.train_dataset_ratios,
            val_dataset_ratios=self.config.data.val_dataset_ratios,
            epoch=epoch,  # Different epoch = different subsample
            shuffle=self.config.data.shuffle,
            num_workers=self.config.data.num_workers,
            seed=self.config.seed,
            eos_token=self.config.data.eos_token,
            strip_labels=self.config.data.strip_labels,
            debug=self.config.data.debug_dataset_mixing,
        )
        
        self.train_loader = train_loader
    
    def _log_sample_generations(
        self,
        vectors: torch.Tensor,
        true_labels: List[str],
        vector_indices: torch.Tensor,
        dataset_names: List[str],
    ):
        """
        Log sample generations to MLflow/WandB, or print to console if neither is available.
        
        Args:
            vectors: Input vectors to generate from
            true_labels: Ground truth labels
            vector_indices: Vector indices for reference
            dataset_names: Dataset names for each vector
        """
        if self.config.logging.log_sample_generations <= 0:
            return
        
        try:
            # Generate descriptions (reduce max_new_tokens to save memory)
            self.model.model.eval()
            with torch.no_grad():
                generated_descriptions = self.model.generate_descriptions(
                    vectors,
                    max_new_tokens=20  # Reduced from 30 to save memory during generation
                )
            self.model.model.train()
            
            # Prepare data for logging
            samples = []
            for i, (generated, true_label, vec_idx, dataset_name) in enumerate(
                zip(generated_descriptions, true_labels, vector_indices, dataset_names)
            ):
                # Remove the closing quote and EOS token from true label for display
                display_label = true_label.replace('"<|eot_id|>', '').replace('"<end_of_turn>', '')
                
                samples.append({
                    "dataset_name": dataset_name,
                    "vector_index": int(vec_idx.item()),
                    "true_label": display_label,
                    "generated_description": generated,
                    "step": self.global_step,
                })
            
            # MLflow logging (save as JSON artifact)
            if self.mlflow_run:
                try:
                    import json
                    import tempfile
                    import os
                    temp_file = None
                    try:
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                            temp_file = f.name
                            json.dump({
                                "step": self.global_step,
                                "samples": samples
                            }, f, indent=2)
                        mlflow.log_artifact(temp_file, artifact_path="sample_generations")
                        print(f"✓ Logged {len(samples)} sample generations to MLflow")
                    finally:
                        # Clean up temporary file
                        if temp_file and os.path.exists(temp_file):
                            os.remove(temp_file)
                except Exception as e:
                    print(f"Warning: Failed to log sample generations to MLflow: {e}")
            
            # WandB logging
            if self.wandb_run:
                try:
                    columns = [
                        "Dataset",
                        "Vector Index",
                        "True Label",
                        "Generated Description",
                        "Step",
                    ]
                    table = wandb.Table(columns=columns)
                    
                    for sample in samples:
                        table.add_data(
                            sample["dataset_name"],
                            sample["vector_index"],
                            sample["true_label"],
                            sample["generated_description"],
                            sample["step"],
                        )
                    
                    wandb.log({
                        "sample_generations": table,
                        "step": self.global_step,
                    })
                    print(f"✓ Logged {len(samples)} sample generations to WandB")
                except Exception as e:
                    print(f"Warning: Failed to log sample generations to WandB: {e}")
            
            # Console logging (when no logging backend is available, or always for visibility)
            if not (self.mlflow_run or self.wandb_run):
                print(f"\n{'='*60}")
                print(f"SAMPLE GENERATIONS (Step {self.global_step})")
                print(f"{'='*60}")
                for sample in samples:
                    print(f"\n[{sample['dataset_name']}] Vector #{sample['vector_index']}")
                    print(f"  True:      {sample['true_label']}")
                    print(f"  Generated: {sample['generated_description']}")
                print(f"{'='*60}\n")
        
        except Exception as e:
            # Don't let generation errors crash training
            print(f"Warning: Failed to generate sample descriptions: {e}")
            print("Training will continue...")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.model.train()
        
        accumulated_loss = 0.0
        accumulation_count = 0
        
        for batch_idx, (vectors, labels, indices, dataset_names) in enumerate(self.train_loader):
            # Skip batches if resuming from checkpoint mid-epoch
            if batch_idx < self.current_batch_in_epoch:
                continue
            
            # Debug: log batch composition
            if self.config.data.debug_dataset_mixing:
                from collections import Counter
                dataset_counts_in_batch = Counter(dataset_names)
                batch_comp_str = ", ".join([f"{name}: {count}" for name, count in sorted(dataset_counts_in_batch.items())])
                print(f"Batch {batch_idx}: {batch_comp_str}")
            
            # Forward pass and compute loss
            loss, stats = self.model.compute_loss(
                vectors,
                labels,
                label_smoothing=self.config.training.label_smoothing,
                max_loss=self.config.training.max_loss,
            )
            
            # Scale loss by accumulation steps for gradient averaging
            scaled_loss = loss / self.config.training.gradient_accumulation_steps

            # Backward pass
            scaled_loss.backward()

            # Accumulate unscaled loss for logging
            accumulated_loss += loss.item()
            accumulation_count += 1
            
            # Optimizer step after accumulation
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm,
                )
                
                # Capture gradient norms BEFORE optimizer step (gradients are zeroed after step)
                grad_norms = {}
                if self.global_step % 10 == 0:
                    for name, param in self.model.projection.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_norm = torch.norm(param.grad).item()
                            # Use short name for logging
                            short_name = name.replace("log_scale", "grad_scale").replace("bias", "grad_bias")
                            grad_norms[short_name] = grad_norm
                            # Warn if gradient is very small (potential vanishing gradient)
                            if grad_norm < 1e-4:
                                print(f"  ⚠ WARNING: {name} gradient norm is very small: {grad_norm:.6e}")
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Scheduler step
                current_scheduler = self._get_current_scheduler()
                current_scheduler.step()
                
                self.global_step += 1
                self.current_batch_in_epoch = batch_idx + 1
                
                # Log metrics
                if self.global_step % self.config.logging.log_every_n_steps == 0:
                    metrics = {
                        "loss": accumulated_loss / accumulation_count,
                        **stats,
                    }
                    
                    # Add learning rates
                    for i, group in enumerate(self.optimizer.param_groups):
                        metrics[f"lr_group{i}"] = group["lr"]
                    
                    # Add gradient norms (captured before optimizer step)
                    metrics.update(grad_norms)
                    
                    self._log_metrics(metrics, prefix="train")
                    
                    # Log singular values for nontrivial mappings (less frequently due to cost)
                    if (self.config.logging.log_singular_values_every_n_steps > 0 and 
                        self.global_step % self.config.logging.log_singular_values_every_n_steps == 0):
                        self._log_singular_values()
                    
                    # Log low-rank metrics for scalar_affine_plus_low_rank projection
                    self._log_low_rank_metrics()
                    
                    accumulated_loss = 0.0
                    accumulation_count = 0
                
                # Validation
                if self.val_loader and self.global_step % self.config.training.validation_every_n_steps == 0:
                    val_metrics = self.validate()
                    self._log_metrics(val_metrics, prefix="val")
                    
                    # Track best model
                    if val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]
                        self._save_checkpoint(suffix="_best")
                    
                    self.model.model.train()
                
                # Sample generation logging
                if (self.config.logging.log_sample_generations > 0 and 
                    self.global_step % self.config.logging.log_generations_every_n_steps == 0):
                    # Use samples from current batch
                    num_samples = min(
                        self.config.logging.log_sample_generations,
                        vectors.shape[0]
                    )
                    sample_vectors = vectors[:num_samples]
                    sample_labels = labels[:num_samples]
                    sample_indices = indices[:num_samples]
                    sample_dataset_names = dataset_names[:num_samples]
                    
                    self._log_sample_generations(
                        sample_vectors,
                        sample_labels,
                        sample_indices,
                        sample_dataset_names,
                    )
                
                # Checkpointing
                if self.global_step % self.config.training.checkpoint_every_n_steps == 0:
                    self._save_checkpoint()
                
                # Check if we've reached max_steps
                if self.config.training.max_steps is not None and self.global_step >= self.config.training.max_steps:
                    print(f"\n✓ Reached max_steps ({self.config.training.max_steps}), stopping training")
                    break
        
        # Reset batch counter when epoch completes
        self.current_batch_in_epoch = 0
        self.current_epoch += 1
    
    def validate(self):
        """Run validation."""
        if not self.val_loader:
            return {}
        
        self.model.model.eval()
        
        total_loss = 0.0
        total_batches = 0
        all_stats = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", unit="batch")
            for vectors, labels, indices, dataset_names in pbar:
                loss, stats = self.model.compute_loss(
                    vectors,
                    labels,
                    label_smoothing=self.config.training.label_smoothing,
                    max_loss=self.config.training.max_loss,
                )
                
                total_loss += loss.item()
                total_batches += 1
                all_stats.append(stats)
                
                # Update progress bar with current average loss
                avg_loss = total_loss / total_batches
                pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
        
        # Aggregate statistics
        avg_metrics = {"loss": total_loss / total_batches}
        
        # Average all numeric stats
        stat_keys = all_stats[0].keys()
        for key in stat_keys:
            if isinstance(all_stats[0][key], (int, float)):
                avg_metrics[key] = np.mean([s[key] for s in all_stats])
        
        return avg_metrics
    
    def evaluate_only(self):
        """
        Run a single pass on the validation set without any training.
        Useful for evaluating models with 0 trainable parameters (e.g., identity mapping).
        """
        print("\n" + "="*60)
        print("EVALUATION MODE (NO TRAINING)")
        print("="*60)
        
        if not self.val_loader:
            print("ERROR: No validation set available for evaluation")
            if self.mlflow_run:
                mlflow.end_run()
            elif self.wandb_run:
                wandb.finish()
            return
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Validation examples: {len(self.val_loader.dataset)}")
        
        # Run validation
        print("\nRunning validation...")
        val_metrics = self.validate()
        
        # Log metrics
        self._log_metrics(val_metrics, prefix="eval")
        
        # Also print to console in a nice format
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for key, value in val_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        print("="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        
        if self.mlflow_run:
            mlflow.end_run()
        elif self.wandb_run:
            wandb.finish()
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Epochs: {self.config.training.num_epochs}")
        print(f"Starting from epoch {self.current_epoch}, step {self.global_step}")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{self.config.training.num_epochs}")
            print(f"{'='*60}")
            
            # Refresh train loader if using dataset ratios with resampling
            self._refresh_train_loader(epoch)
            
            self.train_epoch()
        
        # Final checkpoint
        if self.config.training.save_final_checkpoint:
            self._save_checkpoint(suffix="_final")
        
        # Final validation
        if self.val_loader:
            print("\nRunning final validation...")
            val_metrics = self.validate()
            self._log_metrics(val_metrics, prefix="val_final")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        if self.mlflow_run:
            mlflow.end_run()
        elif self.wandb_run:
            wandb.finish()
