"""Checkpoint save/resume helpers for MASS training.

The functions here write model, optimizer, scheduler, scaler, EMA, and config
state into the run checkpoint directory and restore them for resumed training
or standalone evaluation.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from pathlib import Path

from .distributed import is_master, get_rank, get_local_rank


def get_checkpoint_path(config: Dict[str, Any], tag: str = "latest") -> str:
    """
    Get path to checkpoint file.
    
    Args:
        config: Configuration dictionary
        tag: Checkpoint tag (e.g., "latest", "best")
        
    Returns:
        Path to checkpoint file
    """
    output_dir = config['run']['output_dir']
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, f"{tag}.pth")
    return checkpoint_path


def save_checkpoint(
    state: Dict[str, Any],
    config: Dict[str, Any],
    tag: str = "latest",
    is_best: bool = False
) -> None:
    """
    Save checkpoint to file.
    
    Args:
        state: State dictionary with model, optimizer, etc.
        config: Configuration dictionary
        tag: Checkpoint tag
        is_best: Whether this is the best model so far
    """
    if not is_master():
        return
    
    output_dir = config['run']['output_dir']
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{tag}.pth")
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save(state, best_path)

    logging.info(f"Checkpoint saved: {checkpoint_path}")


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Unwrap model from DDP wrapper.
    
    Args:
        model: Model, possibly wrapped in DDP
        
    Returns:
        Unwrapped model
    """
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


def prepare_checkpoint_state(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[nn.Module] = None,
    epoch: Optional[int] = None,
    iteration: Optional[int] = None,
    best_metric: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    additional_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prepare state dictionary for checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        ema_model: EMA model to save
        epoch: Current epoch
        iteration: Current iteration
        best_metric: Best metric value so far
        config: Configuration dictionary
        scaler: Gradient scaler for mixed precision training
        scheduler: Learning rate scheduler to save
        additional_state: Additional state to save
        
    Returns:
        Checkpoint state dictionary
    """
    model_state = unwrap_model(model).state_dict()

    state = {
        'model': model_state,
        'epoch': epoch,
        'iteration': iteration,
    }
    
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    
    if ema_model is not None:
        ema_state = unwrap_model(ema_model).state_dict()
        state['ema_model'] = ema_state
    
    if best_metric is not None:
        state['best_metric'] = best_metric
    
    if config is not None:
        config_to_save = config.copy()

        # Store both the full run config and a portable base output_dir so
        # evaluation scripts can be launched from a copied checkpoint folder.
        if 'output_dir' not in config_to_save:
            if 'run' in config_to_save and 'output_dir' in config_to_save['run']:
                config_to_save['output_dir'] = os.path.dirname(config_to_save['run']['output_dir'])
            else:
                config_to_save['output_dir'] = 'runs'
        
        if 'run' in config_to_save:
            run_config = config_to_save['run']
            if 'output_dir' in run_config and os.path.isabs(run_config['output_dir']):
                run_config['output_dir_abs'] = run_config['output_dir']
                try:
                    run_config['output_dir_rel'] = os.path.relpath(run_config['output_dir'])
                except ValueError:
                    pass
        
        state['config'] = config_to_save
    
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    
    if additional_state is not None:
        state.update(additional_state)
    
    return state

def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[nn.Module] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
    config_only: bool = False
) -> Dict[str, Any]:
    """
    Load checkpoint from file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        ema_model: EMA model to load weights into
        scaler: Gradient scaler for mixed precision training
        scheduler: Learning rate scheduler to load state into
        map_location: Device to map tensors to
        strict: Whether to strictly enforce that the keys in state_dict match the keys in model
        config_only: If True, only load and return the checkpoint without loading weights
        
    Returns:
        Loaded checkpoint state
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if map_location is None and not config_only:
        map_location = f'cuda:{get_local_rank()}' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(checkpoint_path, map_location=map_location if not config_only else 'cpu')

    if config_only:
        if 'config' not in checkpoint:
            raise KeyError(f"Checkpoint {checkpoint_path} does not contain configuration. "
                          "Please provide the training config explicitly.")
        
        logging.info(f"Loaded configuration from checkpoint: {checkpoint_path}")
        if 'epoch' in checkpoint:
            logging.info(f"Checkpoint is from epoch {checkpoint['epoch']}")
        
        return checkpoint
    
    # Load only the states requested by the caller; this keeps evaluation and
    # resume paths using the same checkpoint format.
    if model is not None and 'model' in checkpoint:
        unwrap_model(model).load_state_dict(checkpoint['model'], strict=strict)

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if ema_model is not None and 'ema_model' in checkpoint:
        unwrap_model(ema_model).load_state_dict(checkpoint['ema_model'], strict=strict)

    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        logging.info("Loaded scheduler state from checkpoint")
    
    return checkpoint

def resume_from_checkpoint(
    config: Dict[str, Any],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[nn.Module] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    resume_path: Optional[str] = None
) -> Tuple[int, int, float]:
    """
    Resume training from checkpoint.
    
    Args:
        config: Configuration dictionary
        model: Model to load weights into
        optimizer: Optimizer to load state into
        ema_model: EMA model to load weights into
        scaler: Gradient scaler for mixed precision training
        scheduler: Learning rate scheduler to load state into
        resume_path: Path to checkpoint file, if None, uses config['run']['resume']
        
    Returns:
        Tuple of (epoch, iteration, best_metric)
    """
    if resume_path is None:
        resume_path = config['run'].get('resume')
    
    if resume_path is None:
        return 0, 0, float('-inf')

    checkpoint = load_checkpoint(
        resume_path,
        model,
        optimizer,
        ema_model,
        scaler,
        scheduler,
    )
    
    epoch = checkpoint.get('epoch', 0)
    iteration = checkpoint.get('iteration', 0)
    best_metric = checkpoint.get('best_metric', float('-inf'))
    
    logging.info(f"Resumed from checkpoint: {resume_path} (epoch {epoch})")
    
    # Older checkpoints may not contain scheduler state.
    if scheduler is not None and 'scheduler' not in checkpoint:
        logging.warning("No scheduler state in checkpoint, fast-forwarding scheduler steps")
        # Some schedulers depend on the historical step count.
        if hasattr(scheduler, '_step_count'):
            scheduler._step_count = iteration
        elif hasattr(scheduler, 'last_epoch'):
            scheduler.last_epoch = epoch - 1
        else:
            for _ in range(iteration):
                scheduler.step()
    
    return epoch, iteration, best_metric


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[str]:
    """
    Find the latest checkpoint in a directory based on epoch number.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    
    latest_path = os.path.join(checkpoint_dir, "latest.pth")
    if os.path.isfile(latest_path):
        return latest_path
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pth")]
    
    if not checkpoint_files:
        return None
    
    checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoint_files[-1])
