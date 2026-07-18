"""
Memory management utilities for handling large-scale distance computations.
"""

import numpy as np
import psutil
import torch
from typing import Optional
from loguru import logger


def estimate_matrix_memory_gb(n: int, m: int = None, dtype=np.float32) -> float:
    """
    Estimate memory required for an n×m matrix in GB.

    Args:
        n: Number of rows
        m: Number of columns (default: n for square matrix)
        dtype: Data type (default: float32)

    Returns:
        Memory in GB
    """
    if m is None:
        m = n

    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = n * m * bytes_per_element
    return total_bytes / (1024**3)


def get_available_memory_gb(use_gpu: bool = False, device: Optional[torch.device] = None) -> float:
    """
    Get available memory in GB.

    Args:
        use_gpu: Whether to check GPU memory
        device: GPU device to check (if use_gpu=True)

    Returns:
        Available memory in GB
    """
    if use_gpu and torch.cuda.is_available():
        try:
            # Handle device specification
            if device is None:
                device_id = 0
            elif device.type == 'cuda':
                # If device has an index, use it; otherwise default to 0
                device_id = device.index if device.index is not None else 0
            else:
                # Not a CUDA device, fall back to CPU
                use_gpu = False

            if use_gpu:
                torch.cuda.synchronize(device_id)
                free_memory, total_memory = torch.cuda.mem_get_info(device_id)
                return free_memory / (1024**3)
        except Exception as e:
            logger.warning(f"Failed to get GPU memory: {e}, falling back to CPU memory")
            use_gpu = False

    if not use_gpu:
        # Get available RAM
        memory = psutil.virtual_memory()
        return memory.available / (1024**3)


def compute_max_refs(n_unique: int, available_ram_gb: Optional[float] = None,
                     ram_fraction: float = 0.3) -> int:
    """
    Compute maximum number of reference points that fit in RAM for precomputed distance matrices.

    The precomputed distance matrices have shape 2 * (n_unique, n_ref) float32.
    We want: 2 * n_unique * max_refs * 4 bytes <= available_ram * ram_fraction

    Args:
        n_unique: Number of unique points (max of emb1_unique, emb2_unique)
        available_ram_gb: Available RAM in GB (auto-detected if None)
        ram_fraction: Fraction of RAM to allocate for distance matrices

    Returns:
        Maximum number of references
    """
    if available_ram_gb is None:
        available_ram_gb = get_available_memory_gb(use_gpu=False)

    usable_bytes = available_ram_gb * ram_fraction * (1024**3)
    # 2 matrices * n_unique * max_refs * 4 bytes
    max_refs = int(usable_bytes / (2 * n_unique * 4))
    max_refs = max(50, max_refs)  # Keep at least 50 references
    return max_refs
