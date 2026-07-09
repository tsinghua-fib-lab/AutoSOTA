"""Configuration for distributed sequence parallel inference."""

from dataclasses import dataclass, field
from typing import Optional, List
import torch
import torch.distributed as dist


@dataclass
class DistributedConfig:
    """
    Configuration for sequence parallel inference with Ring Attention.

    Attributes:
        enabled: Whether distributed mode is enabled
        rank: Current process rank
        world_size: Total number of processes
        process_group: PyTorch distributed process group
        local_seq_start: Global start position for this rank's sequence partition
        local_seq_end: Global end position for this rank's sequence partition
        recompute_k: Number of positions to recompute (optional)
        recompute_ratio: Ratio of positions to recompute (default 0.15)
    """

    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    process_group: Optional[dist.ProcessGroup] = None

    # Sequence partitioning
    local_seq_start: int = 0  # Global start position for this rank
    local_seq_end: int = 0  # Global end position for this rank

    # Recompute configuration
    recompute_k: Optional[int] = None
    recompute_ratio: float = 0.15

    # Ring attention settings
    ring_impl: str = "zigzag"  # "basic", "zigzag", or "llama3"

    @classmethod
    def from_env(cls, recompute_ratio: float = 0.15) -> "DistributedConfig":
        """
        Initialize configuration from torchrun environment.

        This automatically detects distributed settings from environment
        variables set by torchrun/torch.distributed.launch.

        Args:
            recompute_ratio: Ratio of positions to recompute

        Returns:
            DistributedConfig instance configured for the current process
        """
        if not dist.is_initialized():
            return cls(enabled=False, recompute_ratio=recompute_ratio)

        return cls(
            enabled=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            process_group=dist.group.WORLD,
            recompute_ratio=recompute_ratio,
        )

    @classmethod
    def initialize_distributed(
        cls,
        backend: str = "nccl",
        recompute_ratio: float = 0.15,
    ) -> "DistributedConfig":
        """
        Initialize distributed process group and return configuration.

        Args:
            backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
            recompute_ratio: Ratio of positions to recompute

        Returns:
            DistributedConfig instance
        """
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

        return cls.from_env(recompute_ratio=recompute_ratio)

    def set_sequence_partition(self, total_seq_len: int) -> None:
        """
        Calculate and set the sequence partition for this rank.

        Args:
            total_seq_len: Total sequence length across all ranks
        """
        if not self.enabled or self.world_size == 1:
            self.local_seq_start = 0
            self.local_seq_end = total_seq_len
            return

        chunk_size = total_seq_len // self.world_size
        remainder = total_seq_len % self.world_size

        # Distribute remainder tokens to first `remainder` ranks
        if self.rank < remainder:
            self.local_seq_start = self.rank * (chunk_size + 1)
            self.local_seq_end = self.local_seq_start + chunk_size + 1
        else:
            self.local_seq_start = remainder * (chunk_size + 1) + (self.rank - remainder) * chunk_size
            self.local_seq_end = self.local_seq_start + chunk_size

    @property
    def local_seq_len(self) -> int:
        """Get the local sequence length for this rank."""
        return self.local_seq_end - self.local_seq_start

    def get_num_recompute_positions(self, seq_len: Optional[int] = None) -> int:
        """
        Calculate number of positions to recompute.

        Args:
            seq_len: Sequence length (uses local_seq_len if not provided)

        Returns:
            Number of positions to recompute
        """
        if seq_len is None:
            seq_len = self.local_seq_len

        if self.recompute_k is not None:
            return min(self.recompute_k, seq_len)
        return max(1, int(seq_len * self.recompute_ratio))

    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.enabled and self.process_group is not None:
            dist.barrier(group=self.process_group)

    def is_main_rank(self) -> bool:
        """Check if this is the main (rank 0) process."""
        return self.rank == 0

    def __repr__(self) -> str:
        if not self.enabled:
            return "DistributedConfig(enabled=False)"
        return (
            f"DistributedConfig("
            f"rank={self.rank}, "
            f"world_size={self.world_size}, "
            f"local_seq=[{self.local_seq_start}:{self.local_seq_end}], "
            f"recompute_ratio={self.recompute_ratio})"
        )
