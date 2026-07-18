"""
Storage package for billion-scale Gaussian Splatting with SSD offloading.

This package provides a complete tiered storage system:
- LogStorageManager: Log-structured SSD storage with patch files
- TieredCacheManager: RAM cache with LRU eviction and dirty tracking
- AsyncPipeline: Multi-threaded pipeline for async training
- GaussianBlock: Data structures for Gaussian blocks
"""

from .log_storage_manager import LogStorageManager, BlockLocation
from .tiered_cache_manager import TieredCacheManager
from .gaussian_block import GaussianBlock, BlockScheduleInfo, FrustumCuller, compute_morton_code, compute_block_bounds, get_morton_code_torch, get_hierarchical_morton_code_torch, compute_block_bounds_cpu
from .async_pipeline import AsyncPipeline, TSPScheduler
from .block_reader import (
    BlockLayout,
    BlockReader,
    UnifiedParamsBlockReader,
    TieredCacheBlockReader,
    parse_block_row_components,
    resolve_block_reader_backend,
)
from .streaming_ply_init import read_binary_ply_header, streaming_ply_to_ssd_base

__all__ = [
    'LogStorageManager',
    'BlockLocation',
    'TieredCacheManager',
    'GaussianBlock',
    'BlockScheduleInfo',
    'FrustumCuller',
    'compute_morton_code',
    'compute_block_bounds',
    'get_morton_code_torch',
    'get_hierarchical_morton_code_torch',
    'compute_block_bounds_cpu',
    'AsyncPipeline',
    'TSPScheduler',
    'BlockLayout',
    'BlockReader',
    'UnifiedParamsBlockReader',
    'TieredCacheBlockReader',
    'parse_block_row_components',
    'resolve_block_reader_backend',
    'read_binary_ply_header',
    'streaming_ply_to_ssd_base',
]
