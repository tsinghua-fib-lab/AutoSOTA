# config/scheduler_config.py
"""
This file contains the configuration for the learning rate scheduler.
It is used to specify the scheduler, the scheduler parameters,
and the scheduler-specific parameters.

Example yaml file:
scheduler:
  # Scheduler selection
  scheduler_key: 'ReduceLROnPlateau'  # or 'CosineAnnealingLR' or 'OneCycleLR'

  # ReduceLROnPlateau specific parameters
  mode: 'min'  # or 'max'
  factor: 0.2
  patience: 50
  threshold: 0.0001
  threshold_mode: 'rel'  # or 'abs'
  min_lr: 0.00001

  # CosineAnnealingLR specific parameters
  T_max: 100
  eta_min: 1e-5

  # OneCycleLR specific parameters
  max_lr: 1e-2
  total_steps: 100
  pct_start: 0.3
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal, Union

@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler initialization and behavior."""
    
    # Scheduler selection
    scheduler_key: Optional[Literal['ReduceLROnPlateau', 'CosineAnnealingLR', 'OneCycleLR']] = None
    # When True, detect plateaus using the same patience as the scheduler and,
    # upon plateau, reload the best model weights and reduce the learning rate by
    # the same factor as the scheduler. This avoids reducing LR on a potentially
    # overfit state and instead explores around the best generalizing weights.
    reload_best_on_plateau: bool = False
    # Maximum number of plateau restarts (reload-best + LR cut) to perform.
    plateau_max_restarts: int = 2
    
    # ReduceLROnPlateau specific parameters
    mode: Optional[Literal['min', 'max']] = 'min'
    factor: float = 0.2
    patience: int = 50
    threshold: float = 0.0001
    threshold_mode: Literal['abs', 'rel'] = 'rel'
    min_lr: float = 0.00001
    
    # CosineAnnealingLR specific parameters
    T_max: int = 1000
    eta_min: float = 1e-4
    
    # OneCycleLR specific parameters
    max_lr: float = 1e-2
    total_steps: int = 100
    pct_start: float = 0.3
    
    def __post_init__(self):
        """Convert parameters to scheduler kwargs."""
        self.scheduler_kwargs = {}
        
        # Add scheduler-specific parameters
        if self.scheduler_key == 'ReduceLROnPlateau':
            self.scheduler_kwargs.update({
                'mode': self.mode,
                'factor': self.factor,
                'patience': self.patience,
                'threshold': self.threshold,
                'threshold_mode': self.threshold_mode,
                'min_lr': self.min_lr
            })
        elif self.scheduler_key == 'CosineAnnealingLR':
            self.scheduler_kwargs.update({
                'T_max': self.T_max,
                'eta_min': self.eta_min
            })
        elif self.scheduler_key == 'OneCycleLR':
            self.scheduler_kwargs.update({
                'max_lr': self.max_lr,
                'total_steps': self.total_steps,
                'pct_start': self.pct_start
            })

        # Cast numeric strings to int/float
        def _maybe_numeric(val, to_int=False):
            if isinstance(val, str):
                try:
                    return int(val) if to_int else float(val)
                except ValueError:
                    return val
            return val

        self.T_max = _maybe_numeric(self.T_max, to_int=True)
        self.eta_min = _maybe_numeric(self.eta_min) 