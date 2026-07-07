import torch
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrainerCheckpoint:
    epoch: int
    state_dict: Dict[str, Any]
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'state_dict': self.state_dict,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'config': self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainerCheckpoint':
        missing = []
        for field in ['epoch', 'state_dict', 'optimizer', 'scheduler', 'config']:
            if field not in data:
                missing.append(field)

        if missing:
            raise KeyError(f"TrainerCheckpoint.from_dict: missing keys {missing}")

        return cls(
            epoch=data['epoch'],
            state_dict=data['state_dict'],
            optimizer=data['optimizer'],
            scheduler=data['scheduler'],
            config=data['config']
        )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    config: Dict[str, Any],
    filepath: str,
    is_best: bool = False
):
    """

    Args:
        model:
        optimizer:
        scheduler:
        epoch:
        config:  object.__dict__
        filepath: save file
        is_best:
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        ckpt = TrainerCheckpoint(
            epoch=epoch,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            config=config
        )

        torch.save(ckpt.to_dict(), filepath)
        logger.info(f"Checkpoint saved to {filepath}")

        if is_best:
            best_path = Path(filepath).parent / "model_best.pth"
            torch.save(ckpt.to_dict(), best_path)
            logger.info(f"Best model saved to {best_path}")

    except Exception as e:
        logger.error(f"Failed to save checkpoint to {filepath}: {str(e)}")
        raise


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    map_location: Optional[str] = None,
    return_config: bool = False,
) -> int | tuple[int, dict[str, Any]]:
    """

    Args:
        filepath: checkpoint file path
        model: model object
        optimizer:
        scheduler:
        map_location: 'cpu' or 'cuda'
        return_config:

    Returns:
        int: resume epoch (start_epoch)

    Raises:
        FileNotFoundError: File not found
        KeyError: the state_dicts are missing.
        Exception: other error
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    try:
        if map_location is None:
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint_dict = torch.load(filepath, map_location=map_location, weights_only=False)
        logger.info(f"Checkpoint loaded from {filepath}")

        ckpt = TrainerCheckpoint.from_dict(checkpoint_dict)

        model.load_state_dict(ckpt.state_dict)

        if optimizer is not None:
            try:
                optimizer.load_state_dict(ckpt.optimizer)
                logger.debug("Optimizer state restored.")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")

        if scheduler is not None:
            try:
                scheduler.load_state_dict(ckpt.scheduler)
                logger.debug("Scheduler state restored.")
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")

        logger.info(f"Successfully resumed from epoch {ckpt.epoch}")
        if return_config:
            return ckpt.epoch, ckpt.config
        else:
            return ckpt.epoch

    except Exception as e:
        logger.error(f"Failed to load checkpoint from {filepath}: {str(e)}")
        raise


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """

    Args:
        checkpoint_dir:

    Returns:
        latest checkpoint path, or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    files = []
    for f in checkpoint_dir.iterdir():
        if f.suffix == '.pth' and f.name != 'model_best.pth' and 'pretrain_epoch' in f.name:
            try:
                epoch = int(f.stem.split('_')[-1])  # e.g., pretrain_epoch_10.pth
                files.append((epoch, f))
            except (ValueError, IndexError):
                continue

    if not files:
        return None

    latest_file = sorted(files, key=lambda x: x[0])[-1][1]
    return str(latest_file)


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int = 5):
    """

    Args:
        checkpoint_dir:
        keep_last:
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    files = []
    for f in checkpoint_dir.iterdir():
        if f.suffix == '.pth' and f.name != 'model_best.pth' and 'pretrain_epoch' in f.name:
            try:
                epoch = int(f.stem.split('_')[-1])
                files.append((epoch, f))
            except:
                continue

    if len(files) <= keep_last:
        return

    files.sort(key=lambda x: x[0])
    for _, f in files[:-keep_last]:
        f.unlink()
        logger.info(f"Removed old checkpoint: {f}")


class EarlyStopping:
    """
    Early stopping to terminate training when validation metric stops improving.

    Example:
        early_stopper = EarlyStopping(
            patience=5,
            mode='max',
            delta=0.0,
            checkpoint_dir='./checkpoints'
        )

        for epoch in range(n_epochs):
            # ... training ...
            test_loss, test_acc = evaluate(model, test_loader)
            should_stop = early_stopper.step(test_acc, model, optimizer, scheduler, epoch, config)
            if should_stop:
                break
    """

    def __init__(
        self,
        patience: int = 7,
        mode: str = 'min',
        delta: float = 0.0,
        checkpoint_dir: str = './checkpoints',
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped.
            mode: One of 'min' or 'max'. In 'min' mode, lower metric is better; in 'max', higher is better.
            delta: Minimum change in the monitored quantity to qualify as an improvement.
            checkpoint_dir: Directory to save the best model checkpoint.
            verbose: If True, prints message for new best model.
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.checkpoint_dir = Path(checkpoint_dir)
        self.verbose = verbose

        self.best_score = None
        self.best_epoch = None
        self.counter = 0
        self.early_stop = False
        self.is_better = None

        # Set comparison function based on mode
        if mode == 'min':
            self.is_better = lambda new, best: new < best - delta
            self.best_score = float('inf')
        elif mode == 'max':
            self.is_better = lambda new, best: new > best + delta
            self.best_score = float('-inf')
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def step(
        self,
        metric: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
        config: dict
    ) -> bool:
        """
        Call this at the end of each epoch.

        Args:
            metric: Current validation/test metric (e.g., accuracy, F1, loss).
            model: Model to save if metric improves.
            optimizer: Optimizer state to save.
            scheduler: Scheduler state to save.
            epoch: Current epoch index.
            config: Training config dict.

        Returns:
            bool: True if early stop signal is triggered, else False.
        """
        if self.early_stop:
            return True  # Already triggered

        if self.is_better(metric, self.best_score):
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(model, optimizer, scheduler, epoch, config)
            if self.verbose:
                print(f"EarlyStopping: New best model at epoch {epoch} with metric {metric:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} (best: {self.best_score:.6f} at epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training at epoch {epoch}. Best was {self.best_score:.6f} at epoch {self.best_epoch}")

        return self.early_stop

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
        config: dict
    ):
        """Save the best model checkpoint using your existing save_checkpoint function."""
        filepath = self.checkpoint_dir / "model_best.pth"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config,
            filepath=str(filepath),
            is_best=True  # This will also save to model_best.pth
        )

    def get_best_score(self) -> float:
        """Return the best metric score observed so far."""
        return self.best_score

    def get_best_epoch(self) -> Optional[int]:
        """Return the epoch at which the best metric was observed."""
        return self.best_epoch