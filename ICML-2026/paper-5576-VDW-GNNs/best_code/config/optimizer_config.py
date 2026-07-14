# config/optimizer_config.py
"""
This file contains the configuration for the optimizer.
It is used to specify the optimizer, the learning rate,
and the optimizer-specific parameters.

Example yaml file:
optimizer:
  optimizer_key: 'AdamW'
  learn_rate: 1e-2
  weight_decay: 1e-5
  # Optional optimizer-specific parameters
  betas: [0.9, 0.999]  # for Adam/AdamW
  eps: 1e-8           # for Adam/AdamW
  momentum: 0.9       # for SGD
  nesterov: true      # for SGD
"""

from dataclasses import dataclass
from typing import Optional, Literal, Tuple

@dataclass
class OptimizerConfig:
    """Configuration for optimizer initialization and behavior."""
    
    # Optimizer selection
    optimizer_key: Literal['Adam', 'AdamW'] = 'AdamW'
    
    # Learning rate parameters
    learn_rate: float = 1e-2
    weight_decay: Optional[float] = 1e-5
    
    # Optimizer-specific parameters with defaults
    betas: Tuple[float, float] = (0.9, 0.999)  # for Adam/AdamW
    eps: float = 1e-8                         # for Adam/AdamW
    
    def __post_init__(self):
        """Convert parameters to optimizer kwargs."""
        self.optimizer_kwargs = {}
        
        # Add parameters based on optimizer type
        if self.optimizer_key in ['Adam', 'AdamW']:
            self.optimizer_kwargs.update({
                'betas': self.betas,
                'eps': self.eps
            })

        # Cast numeric strings to floats for robustness (handles scientific notation passed as strings)
        def _maybe_float(x):
            return float(x) if isinstance(x, str) and x.replace('.', '', 1).replace('e', '', 1).replace('-', '', 1).isdigit() else x

        self.learn_rate = _maybe_float(self.learn_rate)
        if self.weight_decay is not None:
            self.weight_decay = _maybe_float(self.weight_decay)
        self.eps = _maybe_float(self.eps)

