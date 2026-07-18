"""
Double-Buffered GPU Working Set with N+1 Prefetch

This module implements pipeline parallelism for the Pure SSD/Tide path:
- Double buffer: Two GPU working sets that alternate each iteration
- N+1 prefetch: While GPU computes iteration N, load data for N+1 in background

Architecture:
    Iteration N:
        GPU Compute Stream:  [Forward/Backward on Buffer A]
        GPU Prefetch Stream: [Load N+1 data into Buffer B]
    
    Iteration N+1:
        GPU Compute Stream:  [Forward/Backward on Buffer B]  
        GPU Prefetch Stream: [Load N+2 data into Buffer A]

This eliminates the CPU→GPU transfer latency from the critical path.
"""

import torch
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from storage.schedule_utils import get_current_and_next_camera_batches


def _build_block_starts(
    block_to_local_slice: Dict[int, slice],
    num_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a compact block-start lookup table.

    Returns a tensor of shape (num_blocks,) where entry *k* holds the
    local-index offset of block *k* inside the working set, or -1 if
    the block is not loaded.  Size: O(num_blocks) ≈ O(N/B), typically
    a few hundred KB even for billion-scale scenes — versus the O(N)
    tensor it replaces (~8 GB at 1 B Gaussians).
    """
    block_starts = torch.full((num_blocks,), -1, dtype=torch.long, device=device)
    for block_id, s in block_to_local_slice.items():
        if 0 <= block_id < num_blocks:
            block_starts[block_id] = s.start
    return block_starts


def _global_ids_to_local(
    global_ids: torch.Tensor,
    block_starts: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Map global Gaussian IDs to working-set-local indices.

    Uses the compact *block_starts* table (size num_blocks) instead of
    an N-sized lookup tensor.  Returns -1 for IDs whose block is not
    loaded in the current working set.
    """
    num_blocks = block_starts.shape[0]
    block_ids = torch.div(global_ids, block_size, rounding_mode='floor')
    within_block = global_ids - block_ids * block_size
    safe_block_ids = block_ids.clamp(0, num_blocks - 1)
    starts = block_starts[safe_block_ids]
    invalid = (block_ids != safe_block_ids) | (starts < 0)
    local_ids = starts + within_block
    local_ids[invalid] = -1
    return local_ids


@dataclass
class GPUBuffer:
    """A single GPU buffer holding all Gaussian parameters"""
    xyz: Optional[torch.Tensor] = None           # (M, 3)
    scaling: Optional[torch.Tensor] = None       # (M, 3)
    rotation: Optional[torch.Tensor] = None      # (M, 4)
    opacity: Optional[torch.Tensor] = None       # (M, 1)
    features_dc: Optional[torch.Tensor] = None   # (M, 3)
    features_rest: Optional[torch.Tensor] = None # (M, 45)
    
    # Mapping info
    local_to_global_idx: Optional[torch.Tensor] = None
    _block_starts: Optional[torch.Tensor] = None
    loaded_blocks: List[int] = None
    num_gaussians: int = 0
    
    # Filters for each camera in the batch (local indices)
    filters_local: List[torch.Tensor] = None
    block_to_local_slice: Dict[int, slice] = None
    resident_blocks: List[int] = None
    streamed_blocks: List[int] = None
    evicted_blocks: List[int] = None
    
    def __post_init__(self):
        if self.loaded_blocks is None:
            self.loaded_blocks = []
        if self.filters_local is None:
            self.filters_local = []
        if self.block_to_local_slice is None:
            self.block_to_local_slice = {}
        if self.resident_blocks is None:
            self.resident_blocks = []
        if self.streamed_blocks is None:
            self.streamed_blocks = []
        if self.evicted_blocks is None:
            self.evicted_blocks = []
    
    def is_empty(self) -> bool:
        return self.xyz is None
    
    def clear(self):
        """Release GPU memory"""
        self.xyz = None
        self.scaling = None
        self.rotation = None
        self.opacity = None
        self.features_dc = None
        self.features_rest = None
        self.local_to_global_idx = None
        self._block_starts = None
        self.loaded_blocks = []
        self.filters_local = []
        self.block_to_local_slice = {}
        self.resident_blocks = []
        self.streamed_blocks = []
        self.evicted_blocks = []
        self.num_gaussians = 0


class DoubleBufferGPUWorkingSet:
    """
    Double-buffered GPU working set manager with N+1 prefetch.
    
    Key features:
    1. Two GPU buffers that alternate each iteration
    2. Background prefetch thread loads next iteration's data
    3. Non-blocking swap between buffers
    4. Seamless integration with existing training loop
    """
    
    def __init__(
        self,
        num_total_gaussians: int,
        block_size: int,
        device: str = 'cuda',
        verbose: bool = True,
    ):
        """
        Initialize double-buffered GPU working set.
        
        Args:
            num_total_gaussians: Total number of Gaussians in scene
            block_size: Number of Gaussians per block
            device: GPU device
            verbose: Print routine buffer lifecycle messages
        """
        self.num_total = num_total_gaussians
        self.block_size = block_size
        self.device = torch.device(device)
        self.num_blocks = (num_total_gaussians + block_size - 1) // block_size
        self.verbose = bool(verbose)
        
        # ====================================================================
        # Double buffer: Two GPU working sets
        # ====================================================================
        self.buffer_a = GPUBuffer()
        self.buffer_b = GPUBuffer()
        
        # Which buffer is currently active (being used for compute)
        self.active_buffer_idx = 0  # 0 = A, 1 = B
        
        # ====================================================================
        # Prefetch management
        # ====================================================================
        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        self.prefetch_complete_event = torch.cuda.Event()
        self.prefetch_in_progress = False
        self.prefetch_iteration = -1
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'swaps': 0,
            'prefetch_hits': 0,
            'prefetch_misses': 0,
            'total_prefetch_time_ms': 0.0,
        }
        
        self._log("[DoubleBufferGPU] Initialized with 2 GPU buffers")
        self._log(f"[DoubleBufferGPU] Total Gaussians: {num_total_gaussians:,}")
        self._log(f"[DoubleBufferGPU] Block size: {block_size}, Blocks: {self.num_blocks}")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
    
    @property
    def active_buffer(self) -> GPUBuffer:
        """Get currently active buffer (for compute)"""
        return self.buffer_a if self.active_buffer_idx == 0 else self.buffer_b
    
    @property
    def loading_buffer(self) -> GPUBuffer:
        """Get buffer being loaded (for prefetch)"""
        return self.buffer_b if self.active_buffer_idx == 0 else self.buffer_a
    
    # ========================================================================
    # Compatibility properties - expose active buffer's tensors directly
    # ========================================================================
    @property
    def gpu_xyz(self) -> Optional[torch.Tensor]:
        return self.active_buffer.xyz
    
    @property
    def gpu_scaling(self) -> Optional[torch.Tensor]:
        return self.active_buffer.scaling
    
    @property
    def gpu_rotation(self) -> Optional[torch.Tensor]:
        return self.active_buffer.rotation
    
    @property
    def gpu_opacity(self) -> Optional[torch.Tensor]:
        return self.active_buffer.opacity
    
    @property
    def gpu_features_dc(self) -> Optional[torch.Tensor]:
        return self.active_buffer.features_dc
    
    @property
    def gpu_features_rest(self) -> Optional[torch.Tensor]:
        return self.active_buffer.features_rest
    
    @property
    def local_to_global_idx(self) -> Optional[torch.Tensor]:
        return self.active_buffer.local_to_global_idx
    
    def global_to_local(self, global_ids: torch.Tensor) -> torch.Tensor:
        """Map global Gaussian IDs to working-set-local indices.

        Uses a compact O(num_blocks) lookup instead of an O(N) tensor,
        saving ~8 GB VRAM at 1 B Gaussians.  Returns -1 for IDs whose
        block is not in the current working set.
        """
        buf = self.active_buffer
        if buf._block_starts is None:
            raise RuntimeError(
                '[DoubleBufferGPU] global_to_local called but _block_starts '
                'has not been built yet (buffer is empty or not materialized)'
            )
        return _global_ids_to_local(global_ids, buf._block_starts, self.block_size)

    @property
    def loaded_blocks(self) -> List[int]:
        return self.active_buffer.loaded_blocks
    
    @property
    def filters_local(self) -> List[torch.Tensor]:
        return self.active_buffer.filters_local
    
    def swap_buffers(self):
        """
        Swap active and loading buffers.
        
        Called at the start of each iteration AFTER prefetch is complete.
        """
        with self.lock:
            # Wait for any ongoing prefetch to complete
            if self.prefetch_in_progress:
                self.prefetch_complete_event.synchronize()
                self.prefetch_in_progress = False
            
            # Swap
            self.active_buffer_idx = 1 - self.active_buffer_idx
            self.stats['swaps'] += 1
    
    def start_prefetch(
        self,
        iteration: int,
        visible_block_ids: List[int],
        filters_global: Optional[List[torch.Tensor]] = None,
        ram_cache: Optional[Dict[str, torch.Tensor]] = None,
        unified_params: Optional[torch.Tensor] = None,
        block_cache: Optional[Dict[int, torch.Tensor]] = None,
        resident_block_ids: Optional[List[int]] = None,
        evicted_block_ids: Optional[List[int]] = None,
        allow_resident_copy: bool = True,
        block_reader: 'Optional[object]' = None,
    ):
        """
        Start async prefetch of next iteration's data into loading buffer.

        This is non-blocking! Returns immediately while prefetch happens in background.

        Args:
            iteration: The iteration this data is for (N+1)
            visible_block_ids: Block IDs needed for iteration N+1
            filters_global: Optional global Gaussian indices for each camera
            ram_cache: [Fallback] Optional full RAM cache containing all parameters
            unified_params: [Fallback] Optional _unified_params tensor (if using unified layout)
            block_cache: [Fallback] Optional dict {block_id: tensor(block_size, 59)}
            resident_block_ids: Optional blocks to keep resident from the active buffer (Omega)
            evicted_block_ids: Optional blocks evicted from the current working set (Delta-)
            allow_resident_copy: Whether to reuse resident blocks from the active buffer
            block_reader: Preferred source for block reads.  Must implement the
                ``BlockReader`` protocol from ``storage.block_reader``.  When
                provided, it supersedes the three fallback data-source arguments.
        """
        with self.lock:
            if self.prefetch_in_progress:
                self.prefetch_complete_event.synchronize()

            self.prefetch_in_progress = True
            self.prefetch_iteration = iteration

        with torch.cuda.stream(self.prefetch_stream):
            start_time = time.time()

            target_buffer = self.loading_buffer
            source_buffer = self.active_buffer
            target_buffer.clear()

            sorted_visible_blocks = sorted(set(int(block_id) for block_id in visible_block_ids))
            resident_set = set(int(block_id) for block_id in (resident_block_ids or []))
            evicted_set = set(int(block_id) for block_id in (evicted_block_ids or []))
            filters_global = filters_global or []

            if len(sorted_visible_blocks) == 0:
                target_buffer.evicted_blocks = sorted(evicted_set)
                self.prefetch_complete_event.record(self.prefetch_stream)
                return

            block_lengths: Dict[int, int] = {}
            all_gaussian_ids_list: List[int] = []
            num_visible = 0
            for block_id in sorted_visible_blocks:
                start_idx = block_id * self.block_size
                end_idx = min(start_idx + self.block_size, self.num_total)
                block_len = end_idx - start_idx
                block_lengths[block_id] = block_len
                num_visible += block_len
                all_gaussian_ids_list.extend(range(start_idx, end_idx))

            target_buffer.xyz = torch.empty(num_visible, 3, device=self.device, dtype=torch.float32)
            target_buffer.scaling = torch.empty(num_visible, 3, device=self.device, dtype=torch.float32)
            target_buffer.rotation = torch.empty(num_visible, 4, device=self.device, dtype=torch.float32)
            target_buffer.opacity = torch.empty(num_visible, 1, device=self.device, dtype=torch.float32)
            target_buffer.features_dc = torch.empty(num_visible, 3, device=self.device, dtype=torch.float32)
            target_buffer.features_rest = torch.empty(num_visible, 45, device=self.device, dtype=torch.float32)

            # Batch-resolve cold blocks via BlockReader up front (paper-correct path).
            # Skip blocks that will be copied from the active buffer (resident in Omega).
            reader_blocks: Optional[Dict[int, torch.Tensor]] = None
            reader_layout = None
            if block_reader is not None:
                from storage.block_reader import BlockLayout
                reader_layout = block_reader.layout
                reader_cold_ids = [
                    block_id for block_id in sorted_visible_blocks
                    if not (
                        allow_resident_copy
                        and block_id in resident_set
                        and source_buffer is not None
                        and source_buffer.block_to_local_slice is not None
                        and block_id in source_buffer.block_to_local_slice
                        and not source_buffer.is_empty()
                    )
                ]
                reader_blocks = block_reader.read_blocks(reader_cold_ids)

            offset = 0
            resident_blocks_used: List[int] = []
            streamed_blocks_used: List[int] = []

            for block_id in sorted_visible_blocks:
                block_len = block_lengths[block_id]
                start_idx = block_id * self.block_size
                end_idx = min(start_idx + self.block_size, self.num_total)
                target_slice = slice(offset, offset + block_len)
                target_buffer.block_to_local_slice[block_id] = target_slice

                can_copy_resident = (
                    allow_resident_copy
                    and block_id in resident_set
                    and source_buffer is not None
                    and source_buffer.block_to_local_slice is not None
                    and block_id in source_buffer.block_to_local_slice
                    and not source_buffer.is_empty()
                )

                if can_copy_resident:
                    source_slice = source_buffer.block_to_local_slice[block_id]
                    target_buffer.xyz[target_slice] = source_buffer.xyz[source_slice].detach()
                    target_buffer.scaling[target_slice] = source_buffer.scaling[source_slice].detach()
                    target_buffer.rotation[target_slice] = source_buffer.rotation[source_slice].detach()
                    target_buffer.opacity[target_slice] = source_buffer.opacity[source_slice].detach()
                    target_buffer.features_dc[target_slice] = source_buffer.features_dc[source_slice].detach()
                    target_buffer.features_rest[target_slice] = source_buffer.features_rest[source_slice].detach()
                    resident_blocks_used.append(block_id)
                elif reader_blocks is not None and block_id in reader_blocks:
                    from storage.block_reader import BlockLayout  # local import keeps fallback paths untouched
                    block_tensor = reader_blocks[block_id][:block_len]
                    block_gpu = block_tensor.to(self.device, non_blocking=True)
                    if reader_layout == BlockLayout.UNIFIED:
                        target_buffer.xyz[target_slice] = block_gpu[:, 0:3]
                        target_buffer.opacity[target_slice] = block_gpu[:, 3:4]
                        target_buffer.scaling[target_slice] = block_gpu[:, 4:7]
                        target_buffer.rotation[target_slice] = block_gpu[:, 7:11]
                        target_buffer.features_dc[target_slice] = block_gpu[:, 11:14]
                        target_buffer.features_rest[target_slice] = block_gpu[:, 14:59]
                    else:  # BlockLayout.CACHE
                        target_buffer.xyz[target_slice] = block_gpu[:, 0:3]
                        target_buffer.scaling[target_slice] = block_gpu[:, 3:6]
                        target_buffer.rotation[target_slice] = block_gpu[:, 6:10]
                        target_buffer.opacity[target_slice] = block_gpu[:, 10:11]
                        target_buffer.features_dc[target_slice] = block_gpu[:, 11:14]
                        target_buffer.features_rest[target_slice] = block_gpu[:, 14:59]
                    streamed_blocks_used.append(block_id)
                elif unified_params is not None:
                    block_params = unified_params[start_idx:end_idx]
                    target_buffer.xyz[target_slice] = block_params[:, 0:3].to(self.device, non_blocking=True)
                    target_buffer.opacity[target_slice] = block_params[:, 3:4].to(self.device, non_blocking=True)
                    target_buffer.scaling[target_slice] = block_params[:, 4:7].to(self.device, non_blocking=True)
                    target_buffer.rotation[target_slice] = block_params[:, 7:11].to(self.device, non_blocking=True)
                    target_buffer.features_dc[target_slice] = block_params[:, 11:14].to(self.device, non_blocking=True)
                    target_buffer.features_rest[target_slice] = block_params[:, 14:59].to(self.device, non_blocking=True)
                    streamed_blocks_used.append(block_id)
                elif block_cache is not None:
                    if block_id not in block_cache:
                        raise KeyError(f"Block {block_id} missing from block_cache during double-buffer prefetch")
                    block_tensor = block_cache[block_id][:block_len]
                    block_gpu = block_tensor.to(self.device, non_blocking=True)
                    target_buffer.xyz[target_slice] = block_gpu[:, 0:3]
                    target_buffer.scaling[target_slice] = block_gpu[:, 3:6]
                    target_buffer.rotation[target_slice] = block_gpu[:, 6:10]
                    target_buffer.opacity[target_slice] = block_gpu[:, 10:11]
                    target_buffer.features_dc[target_slice] = block_gpu[:, 11:14]
                    target_buffer.features_rest[target_slice] = block_gpu[:, 14:59]
                    streamed_blocks_used.append(block_id)
                else:
                    if ram_cache is None:
                        raise ValueError('Either block_reader, unified_params, block_cache, or ram_cache must be provided')
                    block_ids_cpu = torch.arange(start_idx, end_idx, dtype=torch.long)
                    target_buffer.xyz[target_slice] = ram_cache['xyz'][block_ids_cpu].to(self.device, non_blocking=True)
                    target_buffer.scaling[target_slice] = ram_cache['scaling'][block_ids_cpu].to(self.device, non_blocking=True)
                    target_buffer.rotation[target_slice] = ram_cache['rotation'][block_ids_cpu].to(self.device, non_blocking=True)
                    target_buffer.opacity[target_slice] = ram_cache['opacity'][block_ids_cpu].to(self.device, non_blocking=True)

                    if 'features_dc' in ram_cache:
                        target_buffer.features_dc[target_slice] = ram_cache['features_dc'][block_ids_cpu].to(self.device, non_blocking=True)
                        target_buffer.features_rest[target_slice] = ram_cache['features_rest'][block_ids_cpu].to(self.device, non_blocking=True)
                    elif 'sh_cache' in ram_cache:
                        sh_params = ram_cache['sh_cache'][block_ids_cpu]
                        target_buffer.features_dc[target_slice] = sh_params[:, :3].to(self.device, non_blocking=True)
                        target_buffer.features_rest[target_slice] = sh_params[:, 3:48].to(self.device, non_blocking=True)
                    else:
                        raise KeyError('ram_cache must contain either features_dc/features_rest or sh_cache')
                    streamed_blocks_used.append(block_id)

                offset += block_len

            all_gaussian_ids = torch.tensor(all_gaussian_ids_list, dtype=torch.long)
            target_buffer.local_to_global_idx = all_gaussian_ids.to(self.device)
            target_buffer.loaded_blocks = sorted_visible_blocks
            target_buffer.num_gaussians = num_visible
            target_buffer.resident_blocks = resident_blocks_used
            target_buffer.streamed_blocks = streamed_blocks_used
            target_buffer.evicted_blocks = sorted(evicted_set)

            target_buffer._block_starts = _build_block_starts(
                target_buffer.block_to_local_slice, self.num_blocks, self.device,
            )

            target_buffer.filters_local = []
            for filter_global in filters_global:
                if filter_global is not None and len(filter_global) > 0:
                    filter_local = _global_ids_to_local(
                        filter_global.to(self.device),
                        target_buffer._block_starts,
                        self.block_size,
                    )
                    valid_mask = filter_local >= 0
                    filter_local = filter_local[valid_mask]
                    target_buffer.filters_local.append(filter_local)
                else:
                    target_buffer.filters_local.append(torch.tensor([], dtype=torch.long, device=self.device))

            self.prefetch_complete_event.record(self.prefetch_stream)

            elapsed_ms = (time.time() - start_time) * 1000
            self.stats['total_prefetch_time_ms'] += elapsed_ms

    def refresh_blocks_from_block_cache(
        self,
        block_cache: Dict[int, torch.Tensor],
        block_ids: List[int],
        target: str = 'loading',
    ) -> int:
        """
        Refresh selected resident blocks in-place from updated CPU block tensors.

        Paper mode uses the CPU cache as the inter-iteration handoff after
        writeback, so Omega blocks kept in the next buffer are refreshed after
        the CPU cache receives updated block tensors.
        """
        target_buffer = self.loading_buffer if target == 'loading' else self.active_buffer
        if target_buffer.is_empty() or not block_ids:
            return 0

        refreshed = 0
        with torch.cuda.stream(self.prefetch_stream):
            for block_id in block_ids:
                if block_id not in target_buffer.block_to_local_slice:
                    continue
                if block_id not in block_cache:
                    continue

                target_slice = target_buffer.block_to_local_slice[block_id]
                block_len = target_slice.stop - target_slice.start
                if block_len <= 0:
                    continue

                block_tensor = block_cache[block_id]
                if block_tensor is None or block_tensor.numel() == 0:
                    continue

                refresh_len = min(block_len, int(block_tensor.shape[0]))
                if refresh_len <= 0:
                    continue

                refresh_slice = slice(target_slice.start, target_slice.start + refresh_len)
                block_gpu = block_tensor[:refresh_len].to(self.device, non_blocking=True)
                target_buffer.xyz[refresh_slice] = block_gpu[:, 0:3]
                target_buffer.scaling[refresh_slice] = block_gpu[:, 3:6]
                target_buffer.rotation[refresh_slice] = block_gpu[:, 6:10]
                target_buffer.opacity[refresh_slice] = block_gpu[:, 10:11]
                target_buffer.features_dc[refresh_slice] = block_gpu[:, 11:14]
                target_buffer.features_rest[refresh_slice] = block_gpu[:, 14:59]
                refreshed += 1

            if refreshed > 0:
                self.prefetch_complete_event.record(self.prefetch_stream)

        return refreshed

    def wait_for_prefetch(self, iteration: int) -> bool:
        """
        Wait for prefetch to complete (if needed).
        
        Args:
            iteration: Expected iteration number
            
        Returns:
            True if prefetch was ready, False if had to wait
        """
        with self.lock:
            if not self.prefetch_in_progress:
                self.stats['prefetch_misses'] += 1
                return False
            
            if self.prefetch_iteration != iteration:
                self.stats['prefetch_misses'] += 1
                return False
        
        # Check if already complete (non-blocking query)
        if self.prefetch_complete_event.query():
            self.stats['prefetch_hits'] += 1
            with self.lock:
                self.prefetch_in_progress = False
            return True
        
        # Have to wait
        self.prefetch_complete_event.synchronize()
        self.stats['prefetch_hits'] += 1
        with self.lock:
            self.prefetch_in_progress = False
        return True
    
    def load_visible_blocks_sync(
        self,
        visible_block_ids: List[int],
        filters_global: List[torch.Tensor],
        unified_params: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Synchronous load (fallback when prefetch not available).
        
        Loads directly into active buffer.
        """
        target_buffer = self.active_buffer
        target_buffer.clear()
        
        if len(visible_block_ids) == 0:
            raise ValueError("No visible blocks to load!")
        
        # Collect all Gaussian IDs
        all_gaussian_ids = []
        for block_id in sorted(visible_block_ids):
            start_idx = block_id * self.block_size
            end_idx = min(start_idx + self.block_size, self.num_total)
            all_gaussian_ids.extend(range(start_idx, end_idx))
        
        all_gaussian_ids = torch.tensor(all_gaussian_ids, dtype=torch.long)
        num_visible = len(all_gaussian_ids)
        
        # Load from unified_params
        visible_params = unified_params[all_gaussian_ids]
        
        target_buffer.xyz = visible_params[:, 0:3].to(self.device)
        target_buffer.opacity = visible_params[:, 3:4].to(self.device)
        target_buffer.scaling = visible_params[:, 4:7].to(self.device)
        target_buffer.rotation = visible_params[:, 7:11].to(self.device)
        target_buffer.features_dc = visible_params[:, 11:14].to(self.device)
        target_buffer.features_rest = visible_params[:, 14:59].to(self.device)
        
        # Store mapping
        target_buffer.local_to_global_idx = all_gaussian_ids.to(self.device)
        target_buffer.loaded_blocks = sorted(visible_block_ids)
        target_buffer.num_gaussians = num_visible
        
        # Build compact block_starts mapping — O(num_blocks) instead of O(N)
        block_to_slice: Dict[int, slice] = {}
        offset = 0
        for block_id in sorted(visible_block_ids):
            start_idx = block_id * self.block_size
            end_idx = min(start_idx + self.block_size, self.num_total)
            block_len = end_idx - start_idx
            block_to_slice[block_id] = slice(offset, offset + block_len)
            offset += block_len
        target_buffer.block_to_local_slice = block_to_slice
        target_buffer._block_starts = _build_block_starts(
            block_to_slice, self.num_blocks, self.device,
        )
        
        # Convert filters
        target_buffer.filters_local = []
        for filter_global in filters_global:
            if filter_global is not None and len(filter_global) > 0:
                filter_local = _global_ids_to_local(
                    filter_global.to(self.device),
                    target_buffer._block_starts,
                    self.block_size,
                )
                valid_mask = filter_local >= 0
                filter_local = filter_local[valid_mask]
                target_buffer.filters_local.append(filter_local)
            else:
                target_buffer.filters_local.append(torch.tensor([], dtype=torch.long, device=self.device))
        
        return {
            'xyz': target_buffer.xyz,
            'scaling': target_buffer.scaling,
            'rotation': target_buffer.rotation,
            'opacity': target_buffer.opacity,
            'features_dc': target_buffer.features_dc,
            'features_rest': target_buffer.features_rest,
        }
    
    def update_ram_cache(
        self,
        geometry_cache: Optional[Dict[str, torch.Tensor]] = None,
        sh_cache: Optional[torch.Tensor] = None,
        unified_params: Optional[torch.Tensor] = None
    ):
        """
        Writeback updated GPU parameters to RAM cache.
        
        Args:
            geometry_cache: Dict with geometry tensors (optional)
            sh_cache: SH parameter cache (optional)
            unified_params: Unified params tensor (optional, preferred for SSD offload)
        """
        buf = self.active_buffer
        if buf.is_empty():
            return
        
        global_ids = buf.local_to_global_idx.cpu()
        
        if unified_params is not None:
            # Update unified_params directly
            # Layout: xyz(3)|opacity(1)|scaling(3)|rotation(4)|dc(3)|rest(45)
            unified_params[global_ids, 0:3] = buf.xyz.detach().cpu()
            unified_params[global_ids, 3:4] = buf.opacity.detach().cpu()
            unified_params[global_ids, 4:7] = buf.scaling.detach().cpu()
            unified_params[global_ids, 7:11] = buf.rotation.detach().cpu()
            unified_params[global_ids, 11:14] = buf.features_dc.detach().cpu()
            unified_params[global_ids, 14:59] = buf.features_rest.detach().cpu()
        else:
            # Update separate caches
            if geometry_cache is not None:
                geometry_cache['xyz'][global_ids] = buf.xyz.detach().cpu()
                geometry_cache['scaling'][global_ids] = buf.scaling.detach().cpu()
                geometry_cache['rotation'][global_ids] = buf.rotation.detach().cpu()
                geometry_cache['opacity'][global_ids] = buf.opacity.detach().cpu()
            
            if sh_cache is not None:
                sh_data = torch.cat([buf.features_dc, buf.features_rest], dim=1).detach().cpu()
                sh_cache[global_ids] = sh_data
    
    def get_stats(self) -> Dict[str, any]:
        """Get performance statistics"""
        stats = self.stats.copy()
        
        # Calculate hit rate
        total = stats['prefetch_hits'] + stats['prefetch_misses']
        if total > 0:
            stats['hit_rate'] = stats['prefetch_hits'] / total
        else:
            stats['hit_rate'] = 0.0
        
        # Calculate average prefetch time
        if stats['prefetch_hits'] > 0:
            stats['avg_prefetch_time_ms'] = stats['total_prefetch_time_ms'] / stats['prefetch_hits']
        else:
            stats['avg_prefetch_time_ms'] = 0.0
        
        # Current buffer stats
        stats['active_buffer'] = 'A' if self.active_buffer_idx == 0 else 'B'
        stats['active_gaussians'] = self.active_buffer.num_gaussians
        stats['active_blocks'] = len(self.active_buffer.loaded_blocks)
        
        return stats
    
    def clear(self):
        """Clear both buffers"""
        self.buffer_a.clear()
        self.buffer_b.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        self.clear()


# ============================================================================
# Helper function for N+1 prefetch scheduling
# ============================================================================

def get_next_iteration_blocks(
    storage_adapter,
    training_schedule: List[int],
    iteration: int,
    batch_size: int,
    schedule_ordering: str = "trajectory",
) -> Tuple[List[int], List[int]]:
    """
    Get visible blocks for the next scheduled training iteration.

    Args:
        storage_adapter: Storage adapter with frustum culling
        training_schedule: Canonical TSP camera order used by training
        iteration: Current logical iteration number
        batch_size: Number of cameras in the current batch

    Returns:
        (next_camera_ids, next_visible_blocks)
    """
    _, next_batch = get_current_and_next_camera_batches(
        training_schedule=training_schedule,
        iteration=iteration,
        batch_size=batch_size,
        schedule_ordering=schedule_ordering,
    )

    visible_blocks = set()
    for cam_id in next_batch.batch_indices:
        blocks = storage_adapter.get_visible_blocks(cam_id)
        visible_blocks.update(blocks)

    return next_batch.batch_indices, sorted(list(visible_blocks))
