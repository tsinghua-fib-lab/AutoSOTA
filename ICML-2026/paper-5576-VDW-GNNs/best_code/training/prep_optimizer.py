import torch
import math
from typing import Tuple, Type, Dict, Any
from config.optimizer_config import OptimizerConfig
from config.scheduler_config import SchedulerConfig
from config.train_config import TrainingConfig


def prepare_optimizer(
    config: OptimizerConfig,
) -> Tuple[Type, Dict[str, Any]]:
    """
    Prepare optimizer class and parameters based on configuration.
    
    Args:
        config: OptimizerConfig object containing optimizer parameters
        
    Returns:
        Tuple of (optimizer_class, optimizer_kwargs)
    """
    # Initialize optimizer kwargs with common parameters
    optimizer_kwargs = {
        'lr': config.learn_rate
    }
    
    # Add optimizer-specific parameters
    if config.optimizer_key == 'AdamW':
        from torch.optim import AdamW
        optimizer_class = AdamW
        optimizer_kwargs['weight_decay'] = config.weight_decay
    elif config.optimizer_key == 'Adam':
        from torch.optim import Adam
        optimizer_class = Adam
        optimizer_kwargs['weight_decay'] = config.weight_decay
    else:
        raise ValueError(f"Optimizer '{config.optimizer_key}' not supported")
    
    return optimizer_class, optimizer_kwargs


def prepare_scheduler(
    config: SchedulerConfig,
    validate_every: int
) -> Tuple[Type, Dict[str, Any]]:
    """
    Prepare scheduler class and parameters based on configuration.
    
    Args:
        config: SchedulerConfig object containing scheduler parameters
        validate_every: How often validation is performed (used to adjust patience)
        
    Returns:
        Tuple of (scheduler_class, scheduler_kwargs)
    """
    if config.scheduler_key == 'ReduceLROnPlateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler_class = ReduceLROnPlateau
        scheduler_kwargs = config.scheduler_kwargs.copy()
        # Interpret config.patience in EPOCHS and convert to validation steps
        # that the PyTorch scheduler expects. Ensure at least 1 validation step.
        try:
            patience_epochs = int(config.patience)
            val_every = int(max(1, validate_every))
            patience_val_steps = max(1, math.ceil(patience_epochs / val_every))
            scheduler_kwargs['patience'] = patience_val_steps
        except Exception:
            # Fallback: leave provided patience unmodified
            pass

    elif config.scheduler_key == 'CosineAnnealingLR':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler_class = CosineAnnealingLR
        scheduler_kwargs = config.scheduler_kwargs.copy()

        # CosineAnnealingLR works per training epoch; no scaling needed.
        # Ensure required keys exist
        if 'T_max' not in scheduler_kwargs:
            scheduler_kwargs['T_max'] = 1000
        if 'eta_min' not in scheduler_kwargs:
            scheduler_kwargs['eta_min'] = 0.0001
        else: # ensure scientific notation is handled
            scheduler_kwargs['eta_min'] = float(config.eta_min)
    else:
        raise ValueError(f"Scheduler '{config.scheduler_key}' not supported")
    
    return scheduler_class, scheduler_kwargs
