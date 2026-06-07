"""Distributed training utilities.

This module centralizes process-group setup, rank/world-size helpers,
deterministic seeding, tensor reduction, and object gathering for single-GPU
and distributed MASS runs.
"""

import os
import random
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Tuple, List, Dict, Any, Optional, Union


def setup_distributed(config: Dict[str, Any]) -> None:
    """
    Set up distributed training environment.
    
    Args:
        config: Configuration dictionary with distributed settings
    """
    seed = config.get('seed', None)
    if seed is not None:
        set_seed(seed)
            
    dist_config = config.get('distributed', {})
    world_size = dist_config.get('world_size', 1)
    
    if world_size <= 1:
        return
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = dist_config.get('master_addr', 'localhost')
    
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(dist_config.get('master_port', 29500))

    node_rank = dist_config.get('node_rank', 0)
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    gpus_per_node = dist_config.get('gpus_per_node', 1)
    global_rank = node_rank * gpus_per_node + local_rank
    world_size = dist_config.get('world_size', 1)

    
    dist_backend = dist_config.get('backend', 'nccl')
    dist_url = dist_config.get('dist_url', 'env://')
    
    dist.init_process_group(
        backend=dist_backend,
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank
    )
    

    
    # Synchronize all processes
    dist.barrier()
    logging.info("Distributed process group initialized")
    

def cleanup_distributed() -> None:
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def is_dist_avail_and_initialized() -> bool:
    """Check if distributed training is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get the number of processes in the distributed group."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the global rank of the current process in the distributed group."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_local_rank():
    """Get local rank from environment variable."""
    return int(os.environ.get("LOCAL_RANK", 0))

def is_master() -> bool:
    """Check if this is the master process (rank 0)."""
    return get_rank() == 0


def is_distributed() -> bool:
    """Check if we are in distributed mode."""
    return get_world_size() > 1


def use_ddp_model(model: nn.Module) -> nn.Module:
    """
    Wrap model in DistributedDataParallel if in distributed mode.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model wrapped in DistributedDataParallel if distributed, otherwise unchanged
    """
    if is_distributed():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))

        # Wrap in DistributedDataParallel
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=local_rank,
            find_unused_parameters=True
        )
    
    return model


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce tensor across all processes in distributed training.
    
    Args:
        tensor: Tensor to reduce
        average: Whether to average or sum the tensor
        
    Returns:
        Reduced tensor
    """
    if not is_distributed():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    
    if average:
        rt /= get_world_size()
    
    return rt


def all_gather_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of gathered tensors, one per process
    """
    if not is_distributed():
        return [tensor]
    
    world_size = get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    dist.all_gather(gathered_tensors, tensor)
    
    return gathered_tensors


def all_gather_object(obj: Any) -> List[Any]:
    """
    Gather arbitrary Python objects from all processes.
    
    Args:
        obj: Object to gather
        
    Returns:
        List of gathered objects, one per process
    """
    if not is_distributed():
        return [obj]
    
    world_size = get_world_size()
    object_list = [None] * world_size
    
    dist.all_gather_object(object_list, obj)
    
    return object_list
