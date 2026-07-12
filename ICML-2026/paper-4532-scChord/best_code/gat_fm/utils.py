"""
Utility functions for GAT-FM.

General utilities for:
- Logging and visualization
- Device management
- Reproducibility
- Configuration management
"""

import os
import random
import json
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate device.
    
    Args:
        gpu_id: Specific GPU ID, or None for auto-detect
        
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            return torch.device(f'cuda:{gpu_id}')
        return torch.device('cuda')
    return torch.device('cpu')


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_shape: tuple = None) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Optional input shape for forward pass
        
    Returns:
        Summary string
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"Model: {model.__class__.__name__}")
    lines.append("=" * 60)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        lines.append(f"{name}: {list(param.shape)} = {param_count:,}")
    
    lines.append("=" * 60)
    lines.append(f"Total parameters: {total_params:,}")
    lines.append(f"Trainable parameters: {trainable_params:,}")
    lines.append(f"Non-trainable parameters: {total_params - trainable_params:,}")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def save_config(config: Any, path: Union[str, Path], format: str = 'yaml'):
    """
    Save configuration to file.
    
    Args:
        config: Configuration object (dataclass or dict)
        path: Output path
        format: 'yaml' or 'json'
    """
    path = Path(path)
    
    if hasattr(config, '__dataclass_fields__'):
        config_dict = asdict(config)
    else:
        config_dict = dict(config)
    
    with open(path, 'w') as f:
        if format == 'yaml':
            yaml.dump(config_dict, f, default_flow_style=False)
        else:
            json.dump(config_dict, f, indent=2)


def load_config(path: Union[str, Path]) -> Dict:
    """
    Load configuration from file.
    
    Args:
        path: Configuration file path
        
    Returns:
        Configuration dict
    """
    path = Path(path)
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            return json.load(f)


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking training metrics.
    """
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric
            
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Logger:
    """
    Simple logger for training metrics.
    
    Logs to console and optionally to file.
    """
    
    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        name: str = 'train',
    ):
        """
        Args:
            log_dir: Directory for log files
            name: Logger name
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self.name = name
        self.history: List[Dict] = []
        
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / f'{name}.log'
    
    def log(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics.
        
        Args:
            metrics: Dict of metric name -> value
            step: Current step/epoch
        """
        entry = {'step': step} if step is not None else {}
        entry.update(metrics)
        self.history.append(entry)
        
        # Console output
        msg = f"[{self.name}]"
        if step is not None:
            msg += f" Step {step}:"
        msg += " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        print(msg)
        
        # File output
        if self.log_dir:
            with open(self.log_file, 'a') as f:
                f.write(msg + '\n')
    
    def save_history(self, path: Optional[Union[str, Path]] = None):
        """Save training history to JSON."""
        if path is None:
            path = self.log_dir / f'{self.name}_history.json'
        
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def move_to_device(data: Any, device: torch.device) -> Any:
    """
    Recursively move data to device.
    
    Args:
        data: Tensor, dict, list, or tuple
        device: Target device
        
    Returns:
        Data on device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(v, device) for v in data)
    else:
        return data


def gradient_norm(model: nn.Module) -> float:
    """
    Compute the gradient norm for a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def exponential_moving_average(
    model: nn.Module,
    ema_model: nn.Module,
    decay: float = 0.9999,
):
    """
    Update EMA model weights.
    
    Args:
        model: Source model
        ema_model: EMA target model
        decay: EMA decay rate
    """
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def create_ema_model(model: nn.Module) -> nn.Module:
    """
    Create an EMA copy of a model.
    
    Args:
        model: Source model
        
    Returns:
        EMA model (deep copy)
    """
    import copy
    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad = False
    return ema_model
