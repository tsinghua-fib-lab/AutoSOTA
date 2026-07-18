#
# Copyright (C) 2024, Flex-3DGS
# Utility functions for safe tensor checkpoint saving.
#
# Fixes the "view-of-shared-storage" problem where torch.save serializes
# the entire underlying storage of a tensor view, causing massive file bloat.
#

import os
import sys
import tempfile

import torch


def _compact_for_save(tensor):
    """Return a compact, self-contained copy of *tensor* suitable for torch.save.

    If the tensor is a view (non-contiguous, or its storage is larger than its
    logical data), this creates a fresh contiguous clone that owns its own
    storage.  Otherwise, the original tensor is returned unchanged to avoid
    unnecessary copies.

    Always detaches from the autograd graph — checkpoint data never needs
    gradient history.
    """
    if tensor is None:
        return None

    t = tensor.detach()

    logical_bytes = t.nelement() * t.element_size()
    storage_bytes = t.untyped_storage().nbytes()

    # Clone if the tensor is non-contiguous or its storage is oversized
    if not t.is_contiguous() or storage_bytes > logical_bytes:
        t = t.contiguous().clone()

    # Move to CPU if on GPU (checkpoint files should be device-agnostic)
    if t.is_cuda:
        t = t.cpu()

    return t


def safe_torch_save(obj, path):
    """Atomically save *obj* to *path* using a temporary file + os.replace.

    This prevents corrupted checkpoint files when the process is killed or
    disk space runs out mid-write: either the old file remains intact or the
    new file is fully written.
    """
    parent_dir = os.path.dirname(path) or "."
    os.makedirs(parent_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=parent_dir, suffix=".pt.tmp")
    try:
        os.close(fd)
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up the temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_tensor_compact(tensor, path):
    """Compact a tensor and atomically save it to *path*.

    Combines ``_compact_for_save`` and ``safe_torch_save`` for convenience.
    """
    safe_torch_save(_compact_for_save(tensor), path)


def debug_tensor_storage(name, tensor, file=None):
    """Print diagnostic information about a tensor's memory layout.

    Useful for identifying tensors that are views of a larger shared storage,
    which would cause torch.save to write the entire storage.

    Args:
        name: A human-readable label for the tensor.
        tensor: The tensor to inspect.
        file: Output stream (default: sys.stderr).
    """
    if file is None:
        file = sys.stderr

    if tensor is None:
        print(f"[TENSOR DEBUG] {name}: None", file=file)
        return

    logical_bytes = tensor.nelement() * tensor.element_size()
    storage_bytes = tensor.untyped_storage().nbytes()
    ratio = storage_bytes / logical_bytes if logical_bytes > 0 else float("inf")
    is_view = storage_bytes > logical_bytes or not tensor.is_contiguous()

    print(
        f"[TENSOR DEBUG] {name}:\n"
        f"  shape          = {tuple(tensor.shape)}\n"
        f"  dtype          = {tensor.dtype}\n"
        f"  device         = {tensor.device}\n"
        f"  stride         = {tensor.stride()}\n"
        f"  is_contiguous  = {tensor.is_contiguous()}\n"
        f"  storage_offset = {tensor.storage_offset()}\n"
        f"  logical_bytes  = {logical_bytes:,} ({logical_bytes / (1024**3):.3f} GiB)\n"
        f"  storage_bytes  = {storage_bytes:,} ({storage_bytes / (1024**3):.3f} GiB)\n"
        f"  bloat_ratio    = {ratio:.1f}x\n"
        f"  is_view        = {is_view}\n"
        f"  needs_compact  = {is_view}",
        file=file,
    )
