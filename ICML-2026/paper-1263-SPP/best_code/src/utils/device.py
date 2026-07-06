"""
Device management module
Handles GPU/CPU device selection and random seed setup
"""

import torch
import random
import numpy as np
from typing import Optional


def get_device(device_id: Optional[int] = None, use_cuda: bool = True) -> torch.device:
    """
    Get the compute device (GPU or CPU)

    Args:
        device_id: GPU device ID (auto-selected if None)
                  Note: if CUDA_VISIBLE_DEVICES is set, device_id should be None or 0
        use_cuda: Whether to use CUDA

    Returns:
        torch.device object
    """
    if use_cuda and torch.cuda.is_available():
        # Check whether CUDA_VISIBLE_DEVICES is set
        import os
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)

        if cuda_visible is not None:
            # If CUDA_VISIBLE_DEVICES is set, only device 0 can be used
            # because PyTorch renumbers the visible GPUs as 0, 1, 2...
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            # Parse CUDA_VISIBLE_DEVICES to display the actual physical GPU
            visible_gpus = cuda_visible.split(',')
            physical_gpu = visible_gpus[0].strip() if len(visible_gpus) > 0 else cuda_visible
            print(f"Using GPU: {device} (physical GPU {physical_gpu}, CUDA_VISIBLE_DEVICES={cuda_visible})")
        elif device_id is not None:
            # Explicitly set the current CUDA device to ensure all operations run on the correct device
            torch.cuda.set_device(device_id)
            device = torch.device(f'cuda:{device_id}')
            print(f"Using GPU: {device} (physical GPU {device_id})")
        else:
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
            print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU")

    return device


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds to ensure reproducible experiments

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure determinism (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False