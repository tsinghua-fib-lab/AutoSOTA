"""Abstract backend interface and concrete implementations for compression."""

import logging
from abc import ABC, abstractmethod

import torch

from ..utils import clear_cache

logger = logging.getLogger(__name__)


class CompressionBackend(ABC):
    """Abstract interface for compression backends.

    A backend handles the low-level compression and decompression operations.
    Implementations may use different algorithms (nvcomp, zstd, etc.) or hardware.
    """

    @abstractmethod
    def compress(self, inp: torch.Tensor) -> torch.Tensor:
        """Compress input tensor.

        Args:
            inp: Contiguous byte tensor on CUDA device.

        Returns:
            Compressed byte tensor on same device.
        """
        pass

    @abstractmethod
    def decompress(self, key: str, compressed: torch.Tensor, output_buffer: torch.Tensor) -> None:
        """Decompress data into output buffer.

        Args:
            key: Unique identifier for caching decompression config.
            compressed: Compressed byte tensor on CUDA device.
            output_buffer: Pre-allocated output buffer on same device.
        """
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear any cached state (e.g., decompression configs)."""
        pass


class DummyBackend(CompressionBackend):
    """Identity backend for testing (no actual compression)."""

    def compress(self, inp: torch.Tensor) -> torch.Tensor:
        return inp.detach().clone()

    def decompress(self, key: str, compressed: torch.Tensor, output_buffer: torch.Tensor) -> None:
        output_buffer[: compressed.numel()].copy_(compressed)

    def clear_cache(self) -> None:
        clear_cache()


class nvCOMPBackend(CompressionBackend):
    """Compression backend using nvcomp ANS algorithm.

    Wraps the CUDA C++ implementation in _compression_cuda.CompressionBackend.
    Compiles CUDA extension on first instantiation using JIT compilation.
    """

    def __init__(self, chunk_size: int = 2**18):
        """Initialize nvcomp backend.

        Args:
            chunk_size: Chunk size for compression (default 256KB).
        """
        CudaBackend = self._load_cuda_backend()
        self._backend = CudaBackend(chunk_size)
        self._chunk_size = chunk_size

    @staticmethod
    def _load_cuda_backend():
        """JIT compile and load the CUDA backend (uses cached build)."""
        logger.debug("Loading nvCOMP CUDA backend, compiling if necessary")

        import os
        from pathlib import Path

        from torch.utils.cpp_extension import load

        backend_dir = Path(__file__).parent / "backend"
        nvcomp_root = os.environ.get("NVCOMP_ROOT")
        if not nvcomp_root:
            raise RuntimeError(
                "NVCOMP_ROOT not set. Set NVCOMP_ROOT environment variable to your nvcomp installation path"
            )
        nvcomp_lib_dir = str(Path(nvcomp_root) / "lib")

        logger.debug(f"Loading CUDA backend from {backend_dir}")
        logger.debug(f"Using nvCOMP library from {nvcomp_lib_dir}")

        compression_cuda = load(
            name="_compression_cuda",
            sources=[str(backend_dir / "compression_bindings.cu")],
            extra_include_paths=[str(backend_dir)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            ],
            extra_ldflags=[
                f"-L{nvcomp_lib_dir}",
                "-lnvcomp",
                f"-Wl,-rpath,{nvcomp_lib_dir}",
            ],
            verbose=False,
        )

        logger.debug("nvCOMP CUDA backend loaded successfully")
        return compression_cuda.CompressionBackend

    def compress(self, inp: torch.Tensor) -> torch.Tensor:
        """Compress a byte tensor using nvcomp ANS."""
        return self._backend.compress(inp)

    def decompress(self, key: str, compressed: torch.Tensor, output_buffer: torch.Tensor) -> None:
        """Decompress into pre-allocated buffer using cached config."""
        self._backend.decompress(key, compressed, output_buffer)

    def clear_cache(self) -> None:
        """Clear cached decompression configurations."""
        self._backend.clear_cache()

    def synchronize(self) -> None:
        """Wait for all pending nvcomp CUDA operations."""
        self._backend.synchronize()

    @property
    def chunk_size(self) -> int:
        """ANS chunk size in bytes."""
        return self._chunk_size
