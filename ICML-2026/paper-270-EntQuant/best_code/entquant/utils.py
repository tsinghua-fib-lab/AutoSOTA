"""Shared utilities for device management, caching, and module manipulation."""

import gc
import logging
from fnmatch import fnmatch

import optimum.quanto
import torch
from torch import nn

logger = logging.getLogger(__name__)

DeviceMap = dict[str, str | torch.device] | str | None


def clear_cache(use_cuda: bool = True, use_gc: bool = True) -> None:
    """Clear Python garbage collector and CUDA memory caches.

    Args:
        use_cuda: Whether to synchronize and clear CUDA caches.
        use_gc: Whether to run Python garbage collection.
    """
    if use_gc:
        gc.collect()
    if use_cuda:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    logger.debug("Cache cleared.")


def get_device(device: str | torch.device | int) -> torch.device:
    """Normalize a device specification to a ``torch.device``.

    Args:
        device: Device as string, int (CUDA ordinal), or torch.device.

    Returns:
        Canonical torch.device instance.
    """
    return device if isinstance(device, torch.device) else torch.device(device)


def get_matching_module_names(
    model: nn.Module,
    include: str | list[str] | None = None,
    exclude: str | list[str] | None = None,
    layer_types: tuple[type[nn.Module]] | list[type[nn.Module]] | None = None,
) -> list[str]:
    """Get module names matching include/exclude patterns (fnmatch style).

    Args:
        model: The model to search.
        include: Pattern(s) that names must match (None = all).
        exclude: Pattern(s) that names must not match (None = none).
        layer_types: Only include modules of these types.

    Returns:
        List of matching fully-qualified module names.
    """
    if include is not None:
        include = [include] if isinstance(include, str) else include
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude
    module_names = []
    for name, module in model.named_modules():
        if include is not None and not any(fnmatch(name, pattern) for pattern in include):
            continue
        if exclude is not None and any(fnmatch(name, pattern) for pattern in exclude):
            continue
        if layer_types is not None and not any(isinstance(module, t) for t in layer_types):
            continue
        module_names.append(name)
    return module_names


def str_to_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Convert a string to a ``torch.dtype`` (no-op if already a dtype).

    Args:
        dtype: Dtype name (e.g. ``"bfloat16"``) or torch.dtype.

    Returns:
        Corresponding torch.dtype.

    Raises:
        ValueError: If the string doesn't match a known dtype.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if hasattr(torch, dtype):
        return getattr(torch, dtype)
    raise ValueError(f"Unknown dtype: {dtype}")


def str_to_qtype(qtype: str | optimum.quanto.qtype) -> optimum.quanto.qtype:
    """Convert a string to an ``optimum.quanto.qtype`` (no-op if already).

    Args:
        qtype: Qtype name (e.g. ``"qfloat8"``) or qtype object.

    Returns:
        Corresponding qtype instance.

    Raises:
        ValueError: If the string doesn't match a known qtype.
    """
    if isinstance(qtype, optimum.quanto.qtype):
        return qtype
    if hasattr(optimum.quanto, qtype):
        return getattr(optimum.quanto, qtype)
    raise ValueError(f"Unknown qtype: {qtype}")


def get_memory_stats(label: str = "") -> tuple[float, float]:
    """Log GPU memory stats and return totals (allocated, reserved) in GiB.

    Args:
        label: Label prefix for log messages.

    Returns:
        ``(total_allocated_gib, total_reserved_gib)`` across all GPUs.
        Returns ``(0.0, 0.0)`` if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0

    total_a, total_r = 0.0, 0.0
    n_devices = torch.cuda.device_count()

    for i in range(n_devices):
        a = torch.cuda.memory_allocated(i) / 1024**3
        r = torch.cuda.memory_reserved(i) / 1024**3
        logger.debug(f"[{label:45}] GPU {i}: Alloc: {a:.3f} GiB | Rsrvd: {r:.3f} GiB | Gap: {r - a:.3f} GiB")
        total_a += a
        total_r += r

    if n_devices > 1:
        logger.debug(
            f"[{label:45}] Total: "
            f"Alloc: {total_a:.3f} GiB | Rsrvd: {total_r:.3f} GiB | "
            f"Gap: {total_r - total_a:.3f} GiB"
        )

    return total_a, total_r


def set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a submodule in the model by its dotted name.

    Args:
        model: Root module.
        name: Dotted path to the submodule (e.g. ``"layer.0.attn"``).
        new_module: Replacement module.
    """
    *parent_path, attr = name.split(".")
    parent = model.get_submodule(".".join(parent_path)) if parent_path else model
    setattr(parent, attr, new_module)
