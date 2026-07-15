"""
Memory optimization utilities for APPFL components.

This module provides reusable functions for common memory optimization patterns
used throughout the APPFL codebase to reduce memory usage and improve garbage collection.
"""

import gc
import torch
import io
from typing import Dict, Any, Optional, Union, OrderedDict
from contextlib import contextmanager


def clone_state_dict_optimized(
    state_dict: Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor]],
    include_buffers: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient cloning of model state dict using tensor.clone().detach()
    instead of copy.deepcopy().

    Args:
        state_dict: The state dictionary to clone
        include_buffers: Whether to include buffer tensors (default: True)

    Returns:
        Cloned state dictionary
    """
    result = {}
    with torch.no_grad():
        for name, tensor in state_dict.items():
            if tensor is not None:
                result[name] = tensor.clone().detach()
    gc.collect()
    return result


def extract_model_state_optimized(
    model: torch.nn.Module, include_buffers: bool = True, cpu_transfer: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient extraction of model state using tensor cloning.

    Args:
        model: PyTorch model
        include_buffers: Whether to include named buffers (default: True)
        cpu_transfer: Whether to transfer tensors to CPU (default: False)

    Returns:
        Model state dictionary
    """
    state = {}
    with torch.no_grad():
        # Extract parameters
        for name, param in model.named_parameters():
            tensor = param.clone().detach()
            if cpu_transfer and tensor.device.type != "cpu":
                tensor = tensor.cpu()
            state[name] = tensor

        # Extract buffers if requested
        if include_buffers:
            for name, buffer in model.named_buffers():
                if buffer is not None:
                    tensor = buffer.clone().detach()
                    if cpu_transfer and tensor.device.type != "cpu":
                        tensor = tensor.cpu()
                    state[name] = tensor

    gc.collect()
    return state


def safe_inplace_operation(
    tensor: torch.Tensor,
    operation: str,
    operand: Union[torch.Tensor, float, int],
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """
    Safely perform in-place operations with dtype checking to avoid errors
    with integer tensors.

    Args:
        tensor: Target tensor for operation
        operation: Operation type ('add', 'sub', 'mul', 'div')
        operand: Operand for the operation
        alpha: Optional scaling factor for add/sub operations

    Returns:
        Result tensor (may be the same tensor if in-place was used)
    """
    # Check if tensor supports in-place operations (avoid integer tensors)
    if tensor.dtype in [torch.int32, torch.int64, torch.long]:
        # For integer tensors, use regular operations
        if operation == "add":
            if alpha is not None:
                return tensor + alpha * operand
            else:
                return tensor + operand
        elif operation == "sub":
            if alpha is not None:
                return tensor - alpha * operand
            else:
                return tensor - operand
        elif operation == "mul":
            return tensor * operand
        elif operation == "div":
            return torch.div(tensor, operand).type(tensor.dtype)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    else:
        # For float tensors, use in-place operations
        if operation == "add":
            if alpha is not None:
                tensor.add_(operand, alpha=alpha)
            else:
                tensor.add_(operand)
        elif operation == "sub":
            if alpha is not None:
                tensor.sub_(operand, alpha=alpha)
            else:
                tensor.sub_(operand)
        elif operation == "mul":
            tensor.mul_(operand)
        elif operation == "div":
            tensor.div_(operand)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        return tensor


def efficient_tensor_aggregation(
    tensors: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
    cleanup_intermediate: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient tensor aggregation with dtype-aware operations.

    Args:
        tensors: Dictionary of tensors to aggregate
        weights: Optional weights for weighted averaging
        cleanup_intermediate: Whether to clean up intermediate results

    Returns:
        Aggregated tensors
    """
    if not tensors:
        return {}

    # Initialize result with zeros
    result = {}
    tensor_names = set()
    for tensor_dict in tensors.values():
        tensor_names.update(tensor_dict.keys())

    with torch.no_grad():
        for name in tensor_names:
            # Find a tensor to use as template for this parameter
            template_tensor = None
            for tensor_dict in tensors.values():
                if name in tensor_dict:
                    template_tensor = tensor_dict[name]
                    break

            if template_tensor is not None:
                result[name] = torch.zeros_like(template_tensor)

        for client_id, tensor_dict in tensors.items():
            weight = (
                1.0 / len(tensors) if weights is None else weights.get(client_id, 0.0)
            )

            for name, tensor in tensor_dict.items():
                if name in result:
                    # Use safe in-place operations
                    weighted_tensor = tensor * weight
                    result[name] = safe_inplace_operation(
                        result[name], "add", weighted_tensor
                    )

                    if cleanup_intermediate:
                        del weighted_tensor

        if cleanup_intermediate:
            gc.collect()

    return result


@contextmanager
def memory_efficient_model_io(optimize_memory: bool = True, force_cpu: bool = True):
    """
    Context manager for memory-efficient model serialization/deserialization.

    Args:
        optimize_memory: Whether to use memory optimizations
        force_cpu: Whether to force CPU loading

    Yields:
        BytesIO buffer for model operations
    """
    if optimize_memory:
        buffer = io.BytesIO()
        try:
            yield buffer
        finally:
            # Explicit cleanup
            if hasattr(buffer, "close"):
                buffer.close()
            del buffer
            gc.collect()
    else:
        # Standard BytesIO without explicit cleanup
        buffer = io.BytesIO()
        yield buffer


def efficient_bytearray_concatenation(
    data_chunks: list, optimize_memory: bool = True
) -> bytes:
    """
    Memory-efficient concatenation using bytearray instead of bytes.

    Args:
        data_chunks: List of byte chunks to concatenate
        optimize_memory: Whether to use memory optimizations

    Returns:
        Concatenated bytes
    """
    if optimize_memory:
        # Use bytearray for more efficient concatenation
        result = bytearray()
        for chunk in data_chunks:
            result.extend(chunk)
        return bytes(result)
    else:
        # Standard bytes concatenation
        return b"".join(data_chunks)


def optimize_memory_cleanup(
    *objects, force_gc: bool = True, clear_cuda_cache: bool = False
) -> None:
    """
    Comprehensive memory cleanup utility.

    Args:
        *objects: Objects to delete
        force_gc: Whether to force garbage collection
        clear_cuda_cache: Whether to clear CUDA cache
    """
    # Delete objects
    for obj in objects:
        if obj is not None:
            del obj

    # Force garbage collection
    if force_gc:
        gc.collect()

    # Clear CUDA cache if requested and available
    if clear_cuda_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_tensor_memory_info(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Get memory usage information for a tensor.

    Args:
        tensor: PyTorch tensor

    Returns:
        Dictionary with memory information
    """
    return {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": tuple(tensor.shape),
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "memory_bytes": tensor.numel() * tensor.element_size(),
        "memory_mb": (tensor.numel() * tensor.element_size()) / (1024 * 1024),
    }


def log_optimization_status(
    component_name: str, optimize_memory: bool, logger=None
) -> None:
    """
    Log memory optimization status for a component.

    Args:
        component_name: Name of the component
        optimize_memory: Whether optimization is enabled
        logger: Optional logger object
    """
    status = "ENABLED" if optimize_memory else "DISABLED (original behavior)"
    message = f"{component_name}: Memory optimization {status}"

    if logger and hasattr(logger, "info"):
        logger.info(message)
    else:
        print(message)


# Backwards compatibility aliases
memory_efficient_clone = clone_state_dict_optimized
memory_efficient_extract = extract_model_state_optimized
safe_tensor_operation = safe_inplace_operation


def split_state_dict_by_size(
    state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"],
    max_chunk_size: int = 1 * 1024 * 1024 * 1024,  # 1GB default
) -> list:
    """
    Split a state_dict into chunks based on memory size.

    Parameters are sorted by name and grouped into chunks where each chunk
    is at most max_chunk_size bytes. If a single parameter exceeds max_chunk_size,
    it becomes its own chunk.

    Args:
        state_dict: Model state dictionary to split
        max_chunk_size: Maximum size per chunk in bytes (default: 1GB)

    Returns:
        List of tuples: [(chunk_idx, chunk_dict, chunk_keys), ...]
        - chunk_idx: Index of this chunk (0-based)
        - chunk_dict: Dictionary containing the parameters for this chunk
        - chunk_keys: List of parameter names in this chunk
    """
    # Sort parameters by name for consistent ordering
    sorted_keys = sorted(state_dict.keys())

    chunks = []
    current_chunk = {}
    current_chunk_keys = []
    current_size = 0

    for key in sorted_keys:
        tensor = state_dict[key]
        tensor_size = tensor.numel() * tensor.element_size()

        # If this single tensor exceeds max_chunk_size, put it in its own chunk
        if tensor_size > max_chunk_size:
            # First, save current chunk if not empty
            if current_chunk:
                chunks.append((len(chunks), current_chunk, current_chunk_keys))
                current_chunk = {}
                current_chunk_keys = []
                current_size = 0

            # Add large tensor as its own chunk
            chunks.append((len(chunks), {key: tensor}, [key]))
            continue

        # If adding this tensor would exceed max_chunk_size, start new chunk
        if current_size + tensor_size > max_chunk_size and current_chunk:
            chunks.append((len(chunks), current_chunk, current_chunk_keys))
            current_chunk = {}
            current_chunk_keys = []
            current_size = 0

        # Add tensor to current chunk
        current_chunk[key] = tensor
        current_chunk_keys.append(key)
        current_size += tensor_size

    # Add remaining chunk if not empty
    if current_chunk:
        chunks.append((len(chunks), current_chunk, current_chunk_keys))

    return chunks


def merge_state_dict_chunks(chunks: list) -> Dict[str, torch.Tensor]:
    """
    Merge state_dict chunks back into a single state_dict.

    Args:
        chunks: List of chunk dictionaries or list of (chunk_idx, chunk_dict, chunk_keys) tuples

    Returns:
        Merged state dictionary
    """
    merged = {}

    for chunk_item in chunks:
        # Handle both formats: just dict or (idx, dict, keys) tuple
        if isinstance(chunk_item, tuple):
            _, chunk_dict, _ = chunk_item
        else:
            chunk_dict = chunk_item

        merged.update(chunk_dict)

    return merged


def get_state_dict_memory_info(
    state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"],
) -> Dict[str, Any]:
    """
    Get detailed memory information about a state_dict.

    Args:
        state_dict: Model state dictionary

    Returns:
        Dictionary with memory statistics
    """
    total_params = 0
    total_bytes = 0
    param_info = {}

    for name, tensor in state_dict.items():
        numel = tensor.numel()
        elem_size = tensor.element_size()
        tensor_bytes = numel * elem_size

        total_params += numel
        total_bytes += tensor_bytes

        param_info[name] = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": numel,
            "bytes": tensor_bytes,
            "mb": tensor_bytes / (1024 * 1024),
        }

    return {
        "total_parameters": total_params,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "total_gb": total_bytes / (1024 * 1024 * 1024),
        "num_tensors": len(state_dict),
        "parameter_details": param_info,
    }
