"""BlockCompressor: per-block ANS compression and on-the-fly decompression.

Manages incremental per-block compression during streaming and
on-the-fly decompression via forward pre-hooks during inference.
"""

import logging
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Callable

import torch
from torch import nn, Tensor

from ..quantization.tensor import get_tensor_data
from ..quantization.utils import entropy
from .backends import CompressionBackend, nvCOMPBackend

__all__ = ["BlockCompressor", "CompressedBlock"]

logger = logging.getLogger(__name__)


@dataclass
class CompressedBlock:
    """Metadata and module reference for a single compressed block."""

    module: nn.Module
    original_size: int
    compressed_size: int
    entropy: float
    # Relative name (e.g. "self_attn.q_proj.weight") ->
    # {shape, dtype, size, byte_offset}
    weights: dict[str, dict[str, Any]] = field(default_factory=dict)


class _DecompressionBuffer(dict):
    """Device-indexed buffer that lazily allocates on first access.

    Each device gets a single buffer sized to the largest compressed block
    that will be decompressed on that device. The buffer is reused across
    all blocks - no per-block allocation during inference.
    """

    def __init__(self, allocation_fn: Callable[[torch.device], Tensor]):
        super().__init__()
        self.allocation_fn = allocation_fn

    def __missing__(self, key: torch.device) -> Tensor:
        res = self.allocation_fn(key)
        self[key] = res
        return res


class BlockCompressor:
    """Per-block ANS compression and on-the-fly decompression.

    Blocks are added incrementally via compress_block() during streaming.
    After compression, register_block() attaches weight pointers and a
    decompression forward pre-hook per block.

    Decompression uses: single nvCOMP ANS kernel per block, shared
    decompression buffer, zero-copy weight reconstruction via
    tensor.set_(), CUDA event-based sync.

    Args:
        backend: Compression backend instance. Defaults to nvCOMPBackend.
        dtype_compressed: Dtype for compressed byte representation.
    """

    def __init__(
        self,
        backend: CompressionBackend | None = None,
        dtype_compressed: torch.dtype = torch.uint8,
    ):
        self.backend = backend if backend is not None else nvCOMPBackend()
        self.dtype_compressed = dtype_compressed

        # Populated incrementally by compress_block()
        self.blocks: dict[str, CompressedBlock] = {}

        self._max_buffer_size_per_device: dict[torch.device, int] = {}
        self._decompression_buffer = _DecompressionBuffer(self._allocate_decompression_buffer)
        self._hook_handles: dict[str, Any] = {}

    def set_buffer_sizes(self, max_sizes: dict[torch.device, int]) -> None:
        """Pre-set max decompression buffer sizes per device.

        Must be called before register_block() so the buffer is
        large enough for the largest block on each device.
        """
        self._max_buffer_size_per_device = max_sizes

    def update_buffer_sizes(
        self,
        model: nn.Module,
        block_device_map: dict[str, torch.device],
        weight_dtype: torch.dtype,
    ) -> None:
        """Estimate and set max decompression buffer sizes per device.

        Iterates blocks in the meta model to compute the total
        weight bytes per block, then takes the max per device.

        Args:
            model: Meta-device model skeleton.
            block_device_map: Block name -> target device mapping.
            weight_dtype: Torch dtype of weights for element size.
        """
        elem_size = torch.tensor([], dtype=weight_dtype).element_size()

        max_sizes: dict[torch.device, int] = {}
        for block_name, device in block_device_map.items():
            block = model.get_submodule(block_name)
            block_bytes = sum(
                m.weight.numel() * elem_size for _, m in block.named_modules() if isinstance(m, nn.Linear)
            )
            max_sizes[device] = max(max_sizes.get(device, 0), block_bytes)

        self.set_buffer_sizes(max_sizes)

    def compress_block(
        self,
        block_module: nn.Module,
        block_name: str,
        include: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
    ) -> None:
        """Compress Linear weights in a block.

        Extracts raw bytes via get_tensor_data(), concatenates,
        compresses via backend, and stores compressed blob as a
        registered buffer on block_module. The compressed blob stays
        on the same device as the source weights.

        Args:
            block_module: The block module containing Linear layers.
            block_name: Full dotted name of the block in the model.
            include: Optional fnmatch pattern(s) - only matching
                modules are compressed. Matched against full module
                name (e.g. "model.layers.0.self_attn.q_proj").
            exclude: Optional fnmatch pattern(s) - matching modules
                are skipped. Matched against full module name.
        """
        weights_all: list[Tensor] = []
        weight_meta: dict[str, dict[str, Any]] = {}
        byte_offset = 0

        for name, module in block_module.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            full_name = f"{block_name}.{name}" if name else block_name

            # Apply include/exclude filtering
            if include is not None:
                patterns = [include] if isinstance(include, str) else include
                if not any(fnmatch(full_name, p) for p in patterns):
                    continue
            if exclude is not None:
                patterns = [exclude] if isinstance(exclude, str) else exclude
                if any(fnmatch(full_name, p) for p in patterns):
                    continue

            rel_name = f"{name}.weight" if name else "weight"
            weight_data = get_tensor_data(module.weight)
            size = weight_data.numel() * weight_data.element_size()
            weight_meta[rel_name] = {
                "shape": weight_data.shape,
                "dtype": weight_data.dtype,
                "size": size,
                "byte_offset": byte_offset,
            }
            weights_all.append(weight_data.flatten())
            byte_offset += size

        if not weights_all:
            return

        # Concatenate and compress
        weights_bytes = torch.cat(weights_all).view(self.dtype_compressed).contiguous()
        original_size = weights_bytes.numel() * weights_bytes.element_size()
        block_entropy = float(entropy(weights_bytes))

        compressed = self.backend.compress(weights_bytes)
        compressed_size = compressed.numel() * compressed.element_size()

        del weights_all, weights_bytes

        # Store compressed data as registered buffer on the block module
        block_module.register_buffer("_compressed_weights", compressed)

        self.blocks[block_name] = CompressedBlock(
            module=block_module,
            original_size=original_size,
            compressed_size=compressed_size,
            entropy=block_entropy,
            weights=weight_meta,
        )

        # Update per-device max buffer size
        device = compressed.device
        prev = self._max_buffer_size_per_device.get(device, 0)
        self._max_buffer_size_per_device[device] = max(prev, original_size)

        ratio = original_size / compressed_size if compressed_size > 0 else 0
        logger.info(
            f"Compressed {block_name}: "
            f"{original_size / (1024**2):.2f} -> "
            f"{compressed_size / (1024**2):.2f} MiB ({ratio:.2f}x)"
        )

    def register_block(self, block_name: str) -> None:
        """Setup weight pointers and decompression hook for one block.

        Points weight tensors into the shared decompression buffer via
        tensor.set_() and registers a forward pre-hook that triggers
        decompression before the block's forward pass.

        Must be called after compress_block() for this block, and after
        set_buffer_sizes() has been called with the correct max sizes.
        """
        self._setup_weight_pointers(block_name)

        block = self.blocks[block_name]

        def make_hook(name: str):
            def hook(module, inp):
                self.decompress(name)

            return hook

        handle = block.module.register_forward_pre_hook(make_hook(block_name))
        self._hook_handles[block_name] = handle
        logger.debug(f"Registered decompression hook for {block_name}")

    def decompress(self, block_name: str) -> None:
        """Decompress a single block's weights into the decompression buffer.

        Called by forward pre-hooks during inference. The backend uses
        CUDA events so the PyTorch compute stream waits without
        blocking CPU.
        """
        block = self.blocks[block_name]
        compressed = block.module.get_buffer("_compressed_weights")
        self.backend.decompress(
            block_name,
            compressed,
            self._decompression_buffer[compressed.device],
        )

    def decompress_model(self) -> None:
        """Decompress all blocks and remove hooks.

        Reverts to uncompressed state: decompresses each block,
        clones weight data out of the shared buffer into standalone
        tensors, removes hooks and compressed buffers.
        """
        for block_name, block in self.blocks.items():
            self.decompress(block_name)

            # Clone buffer slices into standalone tensors
            all_weights = dict(block.module.named_parameters()) | dict(block.module.named_buffers())
            for rel_name in block.weights:
                weight_data = get_tensor_data(all_weights[rel_name])
                weight_data.set_(weight_data.clone())

            # Remove _compressed_weights registered buffer
            delattr(block.module, "_compressed_weights")

        for handle in self._hook_handles.values():
            handle.remove()
        self._hook_handles.clear()
        self.blocks.clear()
        self._decompression_buffer.clear()

    def _allocate_decompression_buffer(self, device: torch.device) -> Tensor:
        """Allocate decompression buffer for a device."""
        max_size = self._max_buffer_size_per_device.get(device, 0)
        if max_size <= 0:
            raise RuntimeError(f"No buffer size set for {device}. Call set_buffer_sizes() before register_block().")
        logger.debug(f"Allocating decompression buffer: {device}, {max_size / 1024**2:.1f} MiB")
        return torch.empty(max_size, dtype=self.dtype_compressed, device=device)

    def _setup_weight_pointers(self, block_name: str) -> None:
        """Point weight tensors into the decompression buffer.

        After decompression fills the buffer, weights reference buffer
        slices via tensor.set_() - zero-copy reconstruction.
        """
        block = self.blocks[block_name]
        all_weights = dict(block.module.named_parameters()) | dict(block.module.named_buffers())
        for rel_name, meta in block.weights.items():
            weight_data = get_tensor_data(all_weights[rel_name])
            buffer = self._decompression_buffer[weight_data.device]
            offset = meta["byte_offset"]
            buffer_slice = buffer[offset : offset + meta["size"]]
            weight_data.set_(buffer_slice.view(meta["dtype"]).view(meta["shape"]))
