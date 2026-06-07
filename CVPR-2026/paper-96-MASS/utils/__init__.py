"""Shared utility package for MASS.

Importing this package exposes registry, distributed training, and checkpoint
helpers used by the training and evaluation entry points.
"""

from . import registry
from . import distributed
from . import checkpoint

# Export important functions
from .registry import (
    register_model,
    register_dataset,
    register_optimizer,
    register_scheduler,
    register_criterion,
    get_model,
    get_dataset,
    get_optimizer,
    get_scheduler,
    get_criterion,
)
from .distributed import (
    setup_distributed,
    cleanup_distributed,
    is_master,
    get_rank,
    get_world_size,
    is_distributed,
    set_seed,
)
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    resume_from_checkpoint,
)

__all__ = [
    'registry',
    'distributed',
    'checkpoint',
    'register_model',
    'register_dataset',
    'register_optimizer',
    'register_scheduler',
    'register_criterion',
    'get_model',
    'get_dataset',
    'get_optimizer',
    'get_scheduler',
    'get_criterion',
    'setup_distributed',
    'cleanup_distributed',
    'is_master',
    'get_rank',
    'get_world_size',
    'is_distributed',
    'set_seed',
    'save_checkpoint',
    'load_checkpoint',
    'resume_from_checkpoint',
]
