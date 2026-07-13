"""Base trainer classes and configuration.

This module defines the core abstractions for training:
- TrainerConfig: Configuration dataclass for training parameters
- BaseTrainer: Abstract base class for all trainers
- TrainingState: Tracks training progress and checkpointing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from utils.timing import HierarchicalTimer


class _OptunaPruned(Exception):
    """Internal exception for Optuna pruning."""
    pass


@dataclass
class TrainerConfig:
    """Configuration for a trainer.

    This standardizes training parameters across all methods.
    """

    # Training hyperparameters
    epochs: int = 1000
    lr: float = 1e-4
    batch_size: int = 256
    train_size: float = 0.8
    max_patience: Optional[int] = 20  # None = no early stopping, return last model

    # Data processing
    rescale: str = "z_score"

    # Checkpointing
    save: bool = True
    load: bool = False

    # Method-specific config (passed through)
    model_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainerConfig":
        """Create config from dictionary, extracting known fields."""
        known_fields = {
            "epochs", "lr", "batch_size", "train_size",
            "max_patience", "rescale", "save", "load"
        }
        known = {k: v for k, v in d.items() if k in known_fields}
        model_config = {k: v for k, v in d.items() if k not in known_fields}
        return cls(**known, model_config=model_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "train_size": self.train_size,
            "max_patience": self.max_patience,
            "rescale": self.rescale,
            "save": self.save,
            "load": self.load,
            "model_config": self.model_config,
        }


@dataclass
class TrainingState:
    """Tracks training progress and state."""

    epoch: int = 0
    best_val_loss: float = float("inf")
    best_train_loss: float = float("inf")
    patience_counter: int = 0
    best_state_dict: Optional[Dict[str, Any]] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    loss_history: Dict[str, List[float]] = field(default_factory=lambda: {"train": [], "val": []})

    def update(self, train_loss: float, val_loss: float) -> bool:
        """Update state with new losses. Returns True if improved."""
        self.epoch += 1
        self.loss_history["train"].append(train_loss)
        self.loss_history["val"].append(val_loss)
        improved = val_loss < self.best_val_loss

        if improved:
            self.best_val_loss = val_loss
            self.best_train_loss = train_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return improved

    def should_stop(self, max_patience: Optional[int]) -> bool:
        """Check if early stopping should trigger."""
        if max_patience is None:
            return False  # No early stopping
        return self.patience_counter >= max_patience

    def mark_completed(self):
        """Mark training as completed."""
        self.completed_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "best_train_loss": self.best_train_loss,
            "patience_counter": self.patience_counter,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "loss_history": self.loss_history,
        }


class BaseTrainer(ABC):
    """Abstract base class for all trainers.

    All trainers must implement:
    - build_model(): Construct the model architecture
    - train_step(): Single training step
    - validate_step(): Single validation step
    - create_posterior(): Wrap trained model in a Posterior

    The training loop and checkpointing are handled by the base class.

    For Optuna HPO integration:
    - Pass trial parameter to train() for pruning support
    - Use get_best_val_loss() after training to get objective value
    """

    # Class attributes for registration
    name: str = "base"
    category: str = "base"  # "simulation", "calibration", or "baseline"

    def __init__(
        self,
        config: TrainerConfig,
        model_path: Path,
        task_config: Mapping[str, Any],
        logger: Optional[Any] = None,
    ):
        """Initialize trainer.

        Args:
            config: Training configuration
            model_path: Path for saving/loading models
            task_config: Task-specific configuration
            logger: Optional logger for metrics
        """
        self.config = config
        self.model_path = Path(model_path)
        self.task_config = task_config
        self.logger = logger
        self.timer = HierarchicalTimer()
        self.state = TrainingState()
        self._optuna_trial = None  # For HPO integration

    def get_best_val_loss(self) -> float:
        """Get the best validation loss from training.

        Returns:
            Best validation loss achieved during training.
            Returns inf if training hasn't been run.
        """
        return self.state.best_val_loss

    def get_training_state(self) -> Dict[str, Any]:
        """Get the full training state as a dictionary.

        Returns:
            Dictionary with training metrics and state.
        """
        return self.state.to_dict()

    def _print_training_config(self):
        """Print training configuration summary."""
        print(f"  Rescaling: {self.config.rescale}")
        print(f"  Epochs: {self.config.epochs}, LR: {self.config.lr}, Batch size: {self.config.batch_size}")
        if self.config.max_patience is None:
            print(f"  Early stopping: disabled (returning last epoch model)")
        else:
            print(f"  Early stopping: max_patience={self.config.max_patience}")

    @abstractmethod
    def build_model(self, **kwargs) -> nn.Module:
        """Build and return the model architecture."""
        pass

    def train_step(
        self,
        model: nn.Module,
        batch: Tuple[Tensor, ...],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> float:
        """Execute single training step. Returns loss value."""
        raise NotImplementedError

    def validate_step(
        self,
        model: nn.Module,
        batch: Tuple[Tensor, ...],
        device: torch.device,
    ) -> float:
        """Execute single validation step. Returns loss value."""
        raise NotImplementedError

    @abstractmethod
    def create_posterior(self, model: nn.Module) -> Any:
        """Wrap trained model in appropriate Posterior class."""
        pass

    def setup_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Setup optimizer. Override for custom optimizers."""
        return torch.optim.Adam(model.parameters(), lr=self.config.lr)

    def setup_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler. Override for custom schedulers."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.config.epochs
        )

    def prepare_data(
        self,
        data: Tuple[Tensor, ...],
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Prepare data loaders. Override for custom data handling."""
        from utils.misc import train_val_split

        if len(data) == 2:
            return train_val_split(
                data[0], data[1],
                self.config.train_size,
                self.config.batch_size
            )
        elif len(data) == 3:
            from utils.misc import train_val_split_n
            return train_val_split_n(
                data[0], data[1], data[2],
                self.config.train_size,
                self.config.batch_size
            )
        else:
            raise ValueError(f"Unsupported data tuple length: {len(data)}")

    def train(
        self,
        data: Tuple[Tensor, ...],
        device: torch.device,
        logname: str = "",
        trial: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Main training loop.

        Args:
            data: Training data tuple
            device: Compute device
            logname: Name for logging
            trial: Optional Optuna trial for HPO pruning support
            **kwargs: Additional arguments passed to build_model

        Returns:
            Trained posterior

        Note:
            For Optuna HPO, pass a trial object. The trainer will report
            intermediate values and check for pruning. After training,
            use get_best_val_loss() to get the objective value.
        """
        self._optuna_trial = trial
        from copy import deepcopy
        from tqdm.auto import trange

        # Build model
        model = self.build_model(**kwargs)
        model.to(device)

        # Check for existing checkpoint
        if self.config.load:
            loaded = self._try_load_checkpoint(model, device)
            if loaded:
                return self.create_posterior(model)

        # Setup training
        optimizer = self.setup_optimizer(model)
        scheduler = self.setup_scheduler(optimizer)
        train_loader, val_loader = self.prepare_data(data)

        self._print_training_config()

        # Training loop
        try:
            with self.timer.time_operation("training", self.name):
                for epoch in (pbar := trange(self.config.epochs, desc=f"Training {self.name}")):
                    # Training phase
                    model.train()
                    train_losses = []
                    for batch in train_loader:
                        loss = self.train_step(model, batch, optimizer, device)
                        train_losses.append(loss)

                    # Validation phase
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for batch in val_loader:
                            loss = self.validate_step(model, batch, device)
                            val_losses.append(loss)

                    # Update scheduler
                    if scheduler is not None:
                        scheduler.step()

                    # Compute epoch losses
                    train_loss = sum(train_losses) / len(train_losses)
                    val_loss = sum(val_losses) / len(val_losses)

                    # Update state
                    improved = self.state.update(train_loss, val_loss)
                    if improved:
                        self.state.best_state_dict = deepcopy(model.state_dict())

                    # Progress bar
                    pbar.set_postfix(train=train_loss, val=val_loss)

                    # Logging
                    if self.logger is not None:
                        self.logger.log_metrics(
                            {"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch},
                            log_name=logname,
                        )

                    # Early stopping
                    if self.state.should_stop(self.config.max_patience):
                        pbar.write(f"\tEarly stopping at epoch {epoch + 1}")
                        pbar.write(f"\tBest validation loss: {self.state.best_val_loss:.4f}")
                        break

                    # Optuna pruning support
                    if self._optuna_trial is not None:
                        self._optuna_trial.report(val_loss, epoch)
                        if self._optuna_trial.should_prune():
                            pbar.write(f"\tOptuna pruned at epoch {epoch + 1}")
                            raise _OptunaPruned()

        except _OptunaPruned:
            # Re-raise as Optuna's TrialPruned
            import optuna
            raise optuna.TrialPruned()

        # Restore best model (unless max_patience=None, which means return last model)
        if self.config.max_patience is not None and self.state.best_state_dict is not None:
            model.load_state_dict(self.state.best_state_dict, strict=False)

        self.state.mark_completed()
        model.cpu()

        # Save checkpoint
        if self.config.save:
            self._save_checkpoint(model, logname)

        return self.create_posterior(model)

    def _save_checkpoint(self, model: nn.Module, logname: str):
        """Save model checkpoint."""
        self.model_path.mkdir(parents=True, exist_ok=True)

        # Save state dict
        save_name = f"{logname}.pth" if logname else f"{self.name}.pth"
        torch.save(model.state_dict(), self.model_path / save_name)

        # Save full model for reconstruction
        import pickle
        pkl_name = f"{logname}.pkl" if logname else f"{self.name}.pkl"
        with open(self.model_path / pkl_name, "wb") as f:
            pickle.dump(model, f)

    def _try_load_checkpoint(self, model: nn.Module, device: torch.device) -> bool:
        """Try to load existing checkpoint. Returns True if successful."""
        checkpoint_path = self.model_path / f"{self.name}.pth"
        if not checkpoint_path.exists():
            return False

        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print(f"Loaded checkpoint from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False

    def save_timing(self, path: Path):
        """Save timing information."""
        self.timer.save(path)

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary for logging."""
        return {
            "name": self.name,
            "category": self.category,
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
            "timing": self.timer.get_summary() if hasattr(self.timer, "get_summary") else {},
        }
