"""
Registry initialization for training-related components.

Importing this module registers common optimizers, schedulers, and loss
criteria with the MASS registry system. Entry points call
``verify_registry_integrity`` early so configuration errors fail fast.
"""

import torch
from torch.optim import (
    SGD, 
    Adam, 
    AdamW, 
    RMSprop
)
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau
)

# Import registry functions
from utils.registry import (
    register_optimizer, 
    register_scheduler,
    register_criterion,
    Registry
)

# Import custom optimizer
from training.lamb_optimizer import Lamb

# Import loss functions (already registered in their module)
from metrics.losses import (
    BinaryDiceLoss,
    BinaryCrossEntropyLoss,
)

# Initialize registry categories if they don't exist
if 'optimizer' not in Registry._registry:
    Registry._registry['optimizer'] = {}
    
if 'scheduler' not in Registry._registry:
    Registry._registry['scheduler'] = {}
    
if 'criterion' not in Registry._registry:
    Registry._registry['criterion'] = {}

# Register optimizers - each one should only be registered once
register_optimizer()(SGD)
register_optimizer()(Adam)
register_optimizer()(AdamW)
register_optimizer()(RMSprop)

# Register schedulers
register_scheduler()(StepLR)
register_scheduler()(MultiStepLR)
register_scheduler()(ExponentialLR)
register_scheduler()(CosineAnnealingLR)
register_scheduler()(OneCycleLR)
register_scheduler()(ReduceLROnPlateau)

# BinaryDiceLoss and BinaryCrossEntropyLoss are already registered in metrics/losses.py
# So we don't register them again here to avoid duplicate registration

# Define a helper function to verify registry integrity
def verify_registry_integrity():
    """
    Verify that all required components are properly registered.
    Raises an error if any component is missing.
    """
    # Check optimizer registry
    required_optimizers = ['SGD', 'Adam', 'AdamW', 'RMSprop', 'Lamb']
    for opt in required_optimizers:
        if opt not in Registry._registry.get('optimizer', {}):
            raise RuntimeError(f"Required optimizer '{opt}' not found in registry")
    
    # Check scheduler registry
    required_schedulers = ['StepLR', 'MultiStepLR', 'ExponentialLR', 
                          'CosineAnnealingLR', 'OneCycleLR', 'ReduceLROnPlateau']
    for sched in required_schedulers:
        if sched not in Registry._registry.get('scheduler', {}):
            raise RuntimeError(f"Required scheduler '{sched}' not found in registry")
    
    # Check criterion registry
    required_criteria = ['BinaryDiceLoss', 'BinaryCrossEntropyLoss']
    for crit in required_criteria:
        if crit not in Registry._registry.get('criterion', {}):
            raise RuntimeError(f"Required criterion '{crit}' not found in registry")
    
    return True
