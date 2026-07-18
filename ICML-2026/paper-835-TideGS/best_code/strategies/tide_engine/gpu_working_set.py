"""
GPU Working Set Manager for the Pure SSD/Tide path.

This module manages dynamic GPU memory allocation for visible Gaussian blocks.
In the TideGS out-of-core path, all parameters are stored as block records
and only the resident blocks are materialized on GPU per iteration.

Architecture:
- SSD Storage: Cold storage for all Gaussians (59 dims: xyz+scale+rot+opacity+SH)
- CPU RAM: Warm cache with LRU eviction (pinned memory)
- GPU VRAM: Hot working set (resident blocks)

Key features:
- Unified 59-float block layout: xyz, scaling, rotation, opacity, features_dc, features_rest
- Dynamic allocation: Only visible blocks occupy GPU memory
- Block-wise transfer: Efficient CPU→GPU transfers via pinned memory
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class GPUWorkingSet:
    """
    Manages GPU working set for pure SSD/Tide training.
    
    This manages all per-Gaussian parameters in the current resident set:
    - Geometry: xyz(3), scaling(3), rotation(4), opacity(1) = 11 dims
    - SH: features_dc(3), features_rest(45) = 48 dims
    - Total: 59 dims per Gaussian
    
    Responsibilities:
    1. Load visible blocks from RAM cache to GPU (all 59 dims)
    2. Maintain GPU tensors for current iteration only
    3. Writeback updated parameters to RAM cache after optimizer step
    
    Block-level resident retention:
    - Track blocks from previous iteration
    - Retain overlapping blocks on GPU when the backend permits it
    - Only load new cold blocks from RAM
    """
    
    def __init__(
        self,
        num_total_gaussians: int,
        block_size: int,
        device: str = 'cuda',
        verbose: bool = False,
    ):
        """
        Initialize GPU working set manager.
        
        Args:
            num_total_gaussians: Total number of Gaussians in the scene
            block_size: Number of Gaussians per block
            device: Target device (usually 'cuda')
            verbose: Print routine working-set lifecycle messages
        """
        self.num_total = num_total_gaussians
        self.block_size = block_size
        self.device = torch.device(device)
        self.num_blocks = (num_total_gaussians + block_size - 1) // block_size
        self.verbose = bool(verbose)
        
        # ====================================================================
        # The GPU working set holds all parameter types for resident blocks.
        # ====================================================================
        # Geometry parameters on GPU (loaded block-wise)
        self.gpu_xyz: Optional[torch.Tensor] = None          # (M, 3)
        self.gpu_scaling: Optional[torch.Tensor] = None      # (M, 3)
        self.gpu_rotation: Optional[torch.Tensor] = None     # (M, 4)
        self.gpu_opacity: Optional[torch.Tensor] = None      # (M, 1)
        
        # SH features on GPU (loaded block-wise)
        self.gpu_features_dc: Optional[torch.Tensor] = None   # (M, 3)
        self.gpu_features_rest: Optional[torch.Tensor] = None # (M, 45)
        
        # ====================================================================
        # [RESIDENT OVERLAP TRACKING] Track previous iteration's blocks.
        # Historical stat keys still use "hotspot" for compatibility.
        # ====================================================================
        self.previous_blocks: List[int] = []  # Blocks from last iteration
        self.previous_block_data: Dict[int, Dict[str, torch.Tensor]] = {}  # block_id -> {params}
        self.hotspot_stats = {
            'total_iterations': 0,
            'total_blocks_loaded': 0,
            'total_blocks_retained': 0,
            'total_blocks_cold': 0,
        }
        
        # Mapping: block_id -> slice in GPU tensor
        self.block_to_gpu_slice: Dict[int, slice] = {}
        
        # Currently loaded block IDs
        self.loaded_blocks: List[int] = []
        
        # Global index mapping (GPU local index -> global Gaussian ID)
        self.local_to_global_idx: Optional[torch.Tensor] = None

        # Compact block-start lookup: O(num_blocks) instead of O(N)
        self._block_starts: Optional[torch.Tensor] = None
        
        self._log(f"[GPUWorkingSet] Initialized for {num_total_gaussians:,} Gaussians")
        self._log(f"[GPUWorkingSet] Block size: {block_size:,}, Total blocks: {self.num_blocks}")
        self._log("[GPUWorkingSet] Managing all parameters (geometry + SH = 59 dims)")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def refresh_topology(self, num_total_gaussians: int):
        """Refresh total-count metadata after densification/pruning changes topology."""
        num_total_gaussians = int(num_total_gaussians)
        if num_total_gaussians == self.num_total:
            return
        self.num_total = num_total_gaussians
        self.num_blocks = (self.num_total + self.block_size - 1) // self.block_size
        self.clear()
        self.previous_blocks.clear()
        self.previous_block_data.clear()
        self._log(
            f"[GPUWorkingSet] Topology refreshed: num_total={self.num_total:,}, total_blocks={self.num_blocks}"
        )

    def _rebuild_block_starts(self) -> None:
        """Rebuild the compact block-start lookup from block_to_gpu_slice."""
        self._block_starts = torch.full(
            (self.num_blocks,), -1, dtype=torch.long, device=self.device,
        )
        for block_id, s in self.block_to_gpu_slice.items():
            if 0 <= block_id < self.num_blocks:
                self._block_starts[block_id] = s.start

    def global_to_local(self, global_ids: torch.Tensor) -> torch.Tensor:
        """Map global Gaussian IDs to working-set-local indices.

        Uses the compact O(num_blocks) block_starts lookup instead of
        an O(N) tensor — saving ~8 GB VRAM at 1 B Gaussians.
        Returns -1 for IDs not in the current working set.
        """
        if self._block_starts is None:
            self._rebuild_block_starts()
        num_blocks = self._block_starts.shape[0]
        block_ids = torch.div(global_ids, self.block_size, rounding_mode='floor')
        within_block = global_ids - block_ids * self.block_size
        safe_block_ids = block_ids.clamp(0, num_blocks - 1)
        starts = self._block_starts[safe_block_ids]
        invalid = (block_ids != safe_block_ids) | (starts < 0)
        local_ids = starts + within_block
        local_ids[invalid] = -1
        return local_ids

    def load_visible_blocks(
        self,
        visible_block_ids: List[int],
        geometry_cache: Dict[str, torch.Tensor],
        sh_cache: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Load visible blocks from RAM cache to GPU working set.
        
        This loads all parameters, not just SH:
        - Geometry: xyz, scaling, rotation, opacity (from separate tensors)
        - SH: features_dc, features_rest (from concatenated cache)
        
        Args:
            visible_block_ids: List of block IDs to load
            geometry_cache: Dict with keys ['xyz', 'scaling', 'rotation', 'opacity']
                           Each value is a CPU tensor (N, dim)
            sh_cache: Full SH parameter cache in RAM (N, 48) with DC + Rest
            
        Returns:
            Dict of GPU tensors: {
                'xyz': (M, 3),
                'scaling': (M, 3),
                'rotation': (M, 4),
                'opacity': (M, 1),
                'features_dc': (M, 3),
                'features_rest': (M, 45)
            }
        """
        if len(visible_block_ids) == 0:
            raise ValueError("No visible blocks to load!")

        # Topology may change after densification/pruning; trust current source tensors.
        inferred_num_total = min(
            int(geometry_cache['xyz'].shape[0]),
            int(geometry_cache['scaling'].shape[0]),
            int(geometry_cache['rotation'].shape[0]),
            int(geometry_cache['opacity'].shape[0]),
            int(sh_cache.shape[0]),
        )
        self.refresh_topology(inferred_num_total)
        
        # Clear previous working set
        self.clear()
        
        # Collect Gaussian IDs for all visible blocks
        all_gaussian_ids = []
        for block_id in sorted(visible_block_ids):
            start_idx = block_id * self.block_size
            end_idx = min(start_idx + self.block_size, self.num_total)
            all_gaussian_ids.extend(range(start_idx, end_idx))
        
        all_gaussian_ids = np.array(all_gaussian_ids, dtype=np.int64)
        num_visible = len(all_gaussian_ids)
        
        # ====================================================================
        # Extract visible parameters from RAM caches (CPU)
        # ====================================================================
        # Geometry parameters (4 separate tensors)
        xyz_cpu = geometry_cache['xyz'][all_gaussian_ids]          # (M, 3)
        scaling_cpu = geometry_cache['scaling'][all_gaussian_ids]  # (M, 3)
        rotation_cpu = geometry_cache['rotation'][all_gaussian_ids]  # (M, 4)
        opacity_cpu = geometry_cache['opacity'][all_gaussian_ids]  # (M, 1)
        
        # SH parameters (concatenated: DC + Rest)
        sh_params_cpu = sh_cache[all_gaussian_ids]  # (M, 48)
        features_dc_cpu = sh_params_cpu[:, :3]       # (M, 3)
        features_rest_cpu = sh_params_cpu[:, 3:48]   # (M, 45)
        
        # ====================================================================
        # Transfer to GPU (non-blocking since caches are pinned)
        # ====================================================================
        self.gpu_xyz = xyz_cpu.to(self.device, non_blocking=True)
        self.gpu_scaling = scaling_cpu.to(self.device, non_blocking=True)
        self.gpu_rotation = rotation_cpu.to(self.device, non_blocking=True)
        self.gpu_opacity = opacity_cpu.to(self.device, non_blocking=True)
        
        self.gpu_features_dc = features_dc_cpu.to(self.device, non_blocking=True)
        self.gpu_features_rest = features_rest_cpu.to(self.device, non_blocking=True)
        
        # Store mapping
        self.loaded_blocks = sorted(set(int(block_id) for block_id in visible_block_ids if 0 <= int(block_id) < self.num_blocks))
        self.local_to_global_idx = torch.from_numpy(all_gaussian_ids).to(self.device)
        
        # Build block-to-slice mapping
        offset = 0
        for block_id in self.loaded_blocks:
            start_idx = block_id * self.block_size
            end_idx = min(start_idx + self.block_size, self.num_total)
            block_len = end_idx - start_idx
            self.block_to_gpu_slice[block_id] = slice(offset, offset + block_len)
            offset += block_len
        
        # Calculate memory footprint
        # xyz(3) + scaling(3) + rotation(4) + opacity(1) + DC(3) + Rest(45) = 59 floats
        memory_mb = num_visible * 59 * 4 / (1024**2)
        
        self._log(
            f"[GPUWorkingSet] Loaded {len(visible_block_ids)} blocks "
            f"({num_visible:,} Gaussians, {memory_mb:.2f} MB on GPU)"
        )
        self._log("[GPUWorkingSet] All parameters loaded: geometry(11 dims) + SH(48 dims)")
        
        return {
            'xyz': self.gpu_xyz,
            'scaling': self.gpu_scaling,
            'rotation': self.gpu_rotation,
            'opacity': self.gpu_opacity,
            'features_dc': self.gpu_features_dc,
            'features_rest': self.gpu_features_rest
        }
    
    def load_visible_blocks_with_retention(
        self,
        visible_block_ids: List[int],
        active_blocks_ram: Optional[Dict[int, torch.Tensor]] = None,
        enable_retention: bool = True,
        unified_params: 'torch.Tensor | None' = None,
        block_reader: 'Optional[object]' = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        """
        Load visible blocks with resident-block overlap reuse.
        
        When safe, retain overlapping blocks from the previous iteration on GPU
        and only load cold blocks from RAM.
        
        Args:
            visible_block_ids: List of block IDs needed for current iteration
            active_blocks_ram: [Fallback] Dict {block_id: tensor(block_size, 59)} from the
                TieredCache prefetch.  Only used when ``block_reader`` is ``None``.
            enable_retention: If True, reuse resident overlap; if False, load all from RAM
            unified_params: [Fallback] Optional _unified_params tensor (N, 59) on CPU pinned
                memory.  Only used when ``block_reader`` is ``None``.
                Layout: xyz(3)|opacity(1)|scaling(3)|rotation(4)|dc(3)|rest(45)
            block_reader: Preferred source for cold-block reads.  Must implement the
                ``BlockReader`` protocol from ``storage.block_reader`` and expose a
                ``layout`` attribute (``BlockLayout.UNIFIED`` or ``BlockLayout.CACHE``).
                When provided, it supersedes both older parameters.
            
        Returns:
            Tuple of:
            - Dict of GPU tensors: {xyz, scaling, rotation, opacity, features_dc, features_rest}
            - Stats dict with existing names: {hotspot_count, cold_count, total_count, hit_rate}
        """
        # Topology may change after densification/pruning; trust whichever source
        # reports the current total count.
        if block_reader is not None:
            self.refresh_topology(int(block_reader.total_gaussians))
        elif unified_params is not None:
            self.refresh_topology(int(unified_params.shape[0]))

        visible_set = set(int(block_id) for block_id in visible_block_ids if 0 <= int(block_id) < self.num_blocks)
        previous_set = set(self.previous_blocks)
        
        # ====================================================================
        # [ERROR HANDLING] Check for empty visible blocks
        # ====================================================================
        if len(visible_block_ids) == 0:
            # Return empty tensors
            print("[GPUWorkingSet WARNING] visible_block_ids is empty! Returning empty working set.")
            self.gpu_xyz = torch.empty(0, 3, device=self.device, dtype=torch.float32)
            self.gpu_scaling = torch.empty(0, 3, device=self.device, dtype=torch.float32)
            self.gpu_rotation = torch.empty(0, 4, device=self.device, dtype=torch.float32)
            self.gpu_opacity = torch.empty(0, 1, device=self.device, dtype=torch.float32)
            self.gpu_features_dc = torch.empty(0, 3, device=self.device, dtype=torch.float32)
            self.gpu_features_rest = torch.empty(0, 45, device=self.device, dtype=torch.float32)
            
            self.loaded_blocks = []
            self.local_to_global_idx = torch.empty(0, dtype=torch.long, device=self.device)
            self.previous_blocks = []
            self.block_to_gpu_slice.clear()
            
            stats = {
                'hotspot_count': 0,
                'cold_count': 0,
                'total_count': 0,
                'hit_rate': 0.0,
                'memory_mb': 0.0,
                'num_gaussians': 0,
            }
            
            return {
                'xyz': self.gpu_xyz,
                'scaling': self.gpu_scaling,
                'rotation': self.gpu_rotation,
                'opacity': self.gpu_opacity,
                'features_dc': self.gpu_features_dc,
                'features_rest': self.gpu_features_rest
            }, stats
        
        # ====================================================================
        # Compute resident overlap and cold (new) blocks.
        # ====================================================================
        if enable_retention and len(previous_set) > 0:
            hotspot_blocks = list(visible_set & previous_set)  # Blocks resident from last iter
            cold_blocks = list(visible_set - previous_set)     # New blocks to load from RAM
        else:
            # First iteration or retention disabled: load everything
            hotspot_blocks = []
            cold_blocks = list(visible_set)
        
        hotspot_count = len(hotspot_blocks)
        cold_count = len(cold_blocks)
        total_count = len(visible_block_ids)
        hit_rate = hotspot_count / total_count if total_count > 0 else 0.0
        
        # Update statistics
        self.hotspot_stats['total_iterations'] += 1
        self.hotspot_stats['total_blocks_loaded'] += total_count
        self.hotspot_stats['total_blocks_retained'] += hotspot_count
        self.hotspot_stats['total_blocks_cold'] += cold_count
        
        # ====================================================================
        # STEP 1: Prepare direct-reuse data references
        # ====================================================================
        # Direct-reuse path: GPU tensors contain correct post-update values,
        #   so references can be reused without an extra CPU->GPU copy.
        # BlockReader-backed release path: the CPU cache is the inter-iteration
        #   handoff point after writeback, so re-read from the CPU-side source.
        #
        # We treat every BlockReader-backed flow as CPU-cache-backed for direct
        # GPU reuse purposes: the canonical CPU tensor (either a fallback
        # unified_params slice or a TieredCache entry) is the source of truth
        # between iterations.
        ssd_mode = (block_reader is not None) or (unified_params is not None)
        can_use_gpu_hotspots = (
            not ssd_mode
            and hotspot_count > 0
            and self.gpu_xyz is not None
        )
        
        # Save old state references (no clone!) for zero-copy resident reuse.
        if can_use_gpu_hotspots:
            old_block_to_gpu_slice = dict(self.block_to_gpu_slice)
            old_gpu_xyz = self.gpu_xyz
            old_gpu_scaling = self.gpu_scaling
            old_gpu_rotation = self.gpu_rotation
            old_gpu_opacity = self.gpu_opacity
            old_gpu_features_dc = self.gpu_features_dc
            old_gpu_features_rest = self.gpu_features_rest
        else:
            old_block_to_gpu_slice = {}
        
        # ====================================================================
        # STEP 2: Clear old working set mapping (data preserved via refs above)
        # ====================================================================
        self.block_to_gpu_slice.clear()
        
        # ====================================================================
        # STEP 3: Build new tensors with correct ordering
        # ====================================================================
        # Sort visible blocks for deterministic ordering
        sorted_visible = sorted(visible_block_ids)
        
        # Compute total Gaussians in new working set
        new_local_to_global = []
        for block_id in sorted_visible:
            start_idx = block_id * self.block_size
            end_idx = min(start_idx + self.block_size, self.num_total)
            new_local_to_global.extend(range(start_idx, end_idx))
        
        num_new_gaussians = len(new_local_to_global)
        
        # Allocate new GPU tensors
        new_xyz = torch.empty(num_new_gaussians, 3, device=self.device, dtype=torch.float32)
        new_scaling = torch.empty(num_new_gaussians, 3, device=self.device, dtype=torch.float32)
        new_rotation = torch.empty(num_new_gaussians, 4, device=self.device, dtype=torch.float32)
        new_opacity = torch.empty(num_new_gaussians, 1, device=self.device, dtype=torch.float32)
        new_features_dc = torch.empty(num_new_gaussians, 3, device=self.device, dtype=torch.float32)
        new_features_rest = torch.empty(num_new_gaussians, 45, device=self.device, dtype=torch.float32)
        
        # ====================================================================
        # STEP 4: Fill tensors - resident overlap from GPU, cold from CPU source
        # ====================================================================
        # Batch-resolve all cold blocks up front via the BlockReader (if provided).
        # This lets TieredCacheBlockReader do one prefetch() call per iteration
        # instead of incurring cache_lock acquisition per block.
        cold_blocks_source: Optional[Dict[int, torch.Tensor]] = None
        cold_source_layout = None
        if block_reader is not None:
            from storage.block_reader import BlockLayout
            cold_source_layout = block_reader.layout
            cold_ids_for_reader = [
                bid for bid in sorted_visible
                if not (can_use_gpu_hotspots and bid in old_block_to_gpu_slice)
            ]
            cold_blocks_source = block_reader.read_blocks(cold_ids_for_reader)

        offset = 0
        for block_id in sorted_visible:
            start_idx = block_id * self.block_size
            end_idx = min(start_idx + self.block_size, self.num_total)
            block_len = end_idx - start_idx
            s = slice(offset, offset + block_len)
            
            if can_use_gpu_hotspots and block_id in old_block_to_gpu_slice:
                # ============================================================
                # DIRECT OVERLAP REUSE: Copy from retained GPU data.
                # Post-update values are correct -> saves CPU->GPU bandwidth.
                # Zero-copy: reference old tensor slice, no intermediate clone
                # ============================================================
                old_s = old_block_to_gpu_slice[block_id]
                new_xyz[s] = old_gpu_xyz[old_s]
                new_scaling[s] = old_gpu_scaling[old_s]
                new_rotation[s] = old_gpu_rotation[old_s]
                new_opacity[s] = old_gpu_opacity[old_s]
                new_features_dc[s] = old_gpu_features_dc[old_s]
                new_features_rest[s] = old_gpu_features_rest[old_s]
            elif cold_blocks_source is not None and block_id in cold_blocks_source:
                # ============================================================
                # COLD via BlockReader: layout determined by reader backend.
                # UnifiedParamsBlockReader  -> xyz | opacity | scale | rot | dc | rest
                # TieredCacheBlockReader    -> xyz | scale   | rot   | opacity | dc | rest
                # ============================================================
                block_tensor = cold_blocks_source[block_id][:block_len]
                src_gpu = block_tensor.to(self.device, non_blocking=True)
                from storage.block_reader import BlockLayout  # local import avoids cold-path import cost
                if cold_source_layout == BlockLayout.UNIFIED:
                    new_xyz[s] = src_gpu[:, 0:3]
                    new_opacity[s] = src_gpu[:, 3:4]
                    new_scaling[s] = src_gpu[:, 4:7]
                    new_rotation[s] = src_gpu[:, 7:11]
                    new_features_dc[s] = src_gpu[:, 11:14]
                    new_features_rest[s] = src_gpu[:, 14:59]
                else:  # BlockLayout.CACHE
                    new_xyz[s] = src_gpu[:, 0:3]
                    new_scaling[s] = src_gpu[:, 3:6]
                    new_rotation[s] = src_gpu[:, 6:10]
                    new_opacity[s] = src_gpu[:, 10:11]
                    new_features_dc[s] = src_gpu[:, 11:14]
                    new_features_rest[s] = src_gpu[:, 14:59]
            elif unified_params is not None:
                # [Fallback] Read directly from unified_params (CPU pinned)
                # Layout: xyz(3)|opacity(1)|scaling(3)|rotation(4)|dc(3)|rest(45)
                src = unified_params.data[start_idx:end_idx]
                src_gpu = src.to(self.device, non_blocking=True)
                new_xyz[s] = src_gpu[:, 0:3]
                new_opacity[s] = src_gpu[:, 3:4]
                new_scaling[s] = src_gpu[:, 4:7]
                new_rotation[s] = src_gpu[:, 7:11]
                new_features_dc[s] = src_gpu[:, 11:14]
                new_features_rest[s] = src_gpu[:, 14:59]
            elif active_blocks_ram is not None and block_id in active_blocks_ram:
                # [Fallback] From TieredCache-style dict; cache layout.
                block_tensor = active_blocks_ram[block_id]  # (block_size, 59) CPU
                block_gpu = block_tensor[:block_len].to(self.device, non_blocking=True)
                new_xyz[s] = block_gpu[:, 0:3]
                new_scaling[s] = block_gpu[:, 3:6]
                new_rotation[s] = block_gpu[:, 6:10]
                new_opacity[s] = block_gpu[:, 10:11]
                new_features_dc[s] = block_gpu[:, 11:14]
                new_features_rest[s] = block_gpu[:, 14:59]
            else:
                # Block unavailable from every source — fill with zeros.
                new_xyz[s].zero_()
                new_scaling[s].zero_()
                new_rotation[s].zero_()
                new_opacity[s].zero_()
                new_features_dc[s].zero_()
                new_features_rest[s].zero_()
                print(f"[WARNING] Block {block_id} not reachable via any source (reader/unified_params/active_blocks_ram)")
            
            # Record slice mapping
            self.block_to_gpu_slice[block_id] = s
            offset += block_len
        
        # ====================================================================
        # STEP 4.5: Ensure DMA transfers complete before GPU kernels read data
        # ====================================================================
        # The .to(device, non_blocking=True) calls above issue H2D DMA on the
        # *default* CUDA stream.  All subsequent compute kernels (calculate_filters,
        # gsplat render) also run on the default stream, so CUDA stream ordering
        # already guarantees they will not read stale data — no explicit sync needed.
        #
        # Earlier CPU-side optimizer experiments could race with DMA reads from
        # pinned pages. The release path uses GPUResidentAdam and staged
        # writeback, so the CPU unified table is not updated concurrently here.
        #
        # torch.cuda.synchronize() is therefore NOT needed for correctness and
        # would serialize the CPU/GPU pipeline (~0.5-1 ms per call, adds up).
        # We keep a gated version for defensive debugging: set the env var
        #     FLEX3DGS_SYNC_DMA=1
        # to force a device-wide sync after every DMA batch.
        import os
        if os.environ.get('FLEX3DGS_SYNC_DMA', '0') == '1':
            torch.cuda.synchronize()
        
        # ====================================================================
        # STEP 5: Update manager state
        # ====================================================================
        self.gpu_xyz = new_xyz
        self.gpu_scaling = new_scaling
        self.gpu_rotation = new_rotation
        self.gpu_opacity = new_opacity
        self.gpu_features_dc = new_features_dc
        self.gpu_features_rest = new_features_rest
        
        self.loaded_blocks = sorted_visible
        self.local_to_global_idx = torch.tensor(new_local_to_global, dtype=torch.long, device=self.device)
        self._rebuild_block_starts()
        
        # Record for next iteration's retention
        self.previous_blocks = sorted_visible.copy()
        
        # Free old tensor references (allow GC of previous iteration's GPU data)
        if can_use_gpu_hotspots:
            del old_gpu_xyz, old_gpu_scaling, old_gpu_rotation, old_gpu_opacity
            del old_gpu_features_dc, old_gpu_features_rest
            del old_block_to_gpu_slice
        
        # Calculate memory footprint
        memory_mb = num_new_gaussians * 59 * 4 / (1024**2)
        
        # Return stats for logging
        # In the BlockReader release path, hotspot_count reflects locality for
        # monitoring even when data is re-read from the CPU cache.  In the
        # direct-reuse path, overlapping blocks use retained GPU data and save
        # CPU->GPU bandwidth.
        data_reused = hotspot_count if can_use_gpu_hotspots else 0
        
        stats = {
            'hotspot_count': hotspot_count,
            'cold_count': cold_count,
            'total_count': total_count,
            'hit_rate': hit_rate,
            'memory_mb': memory_mb,
            'num_gaussians': num_new_gaussians,
            'data_reused_count': data_reused,
        }
        
        return {
            'xyz': self.gpu_xyz,
            'scaling': self.gpu_scaling,
            'rotation': self.gpu_rotation,
            'opacity': self.gpu_opacity,
            'features_dc': self.gpu_features_dc,
            'features_rest': self.gpu_features_rest
        }, stats
    
    def get_retention_stats(self) -> Dict[str, float]:
        """Get cumulative resident-overlap retention statistics."""
        total_loaded = self.hotspot_stats['total_blocks_loaded']
        total_retained = self.hotspot_stats['total_blocks_retained']
        total_cold = self.hotspot_stats['total_blocks_cold']
        
        avg_hit_rate = total_retained / total_loaded if total_loaded > 0 else 0.0
        
        return {
            'total_iterations': self.hotspot_stats['total_iterations'],
            'avg_hit_rate': avg_hit_rate,
            'total_blocks_loaded': total_loaded,
            'total_blocks_retained': total_retained,
            'total_blocks_cold': total_cold,
            'bandwidth_savings_ratio': avg_hit_rate,  # Approx savings
        }
    
    def update_ram_cache(
        self,
        geometry_cache: Dict[str, torch.Tensor],
        sh_cache: torch.Tensor
    ):
        """
        Update RAM cache with modified GPU parameters (after optimizer step).
        
        This updates all parameter types:
        - Geometry: xyz, scaling, rotation, opacity
        - SH: features_dc, features_rest
        
        Args:
            geometry_cache: Dict with keys ['xyz', 'scaling', 'rotation', 'opacity']
                           Each value is a CPU tensor (N, dim) - will be updated in-place
            sh_cache: Full SH parameter cache in RAM (N, 48) - will be updated in-place
        """
        if self.gpu_xyz is None:
            print("[GPUWorkingSet] Warning: No parameters to update (empty working set)")
            return
        
        # Get global indices for scatter operation
        global_ids = self.local_to_global_idx.cpu().numpy()
        
        # ====================================================================
        # Update geometry parameters: GPU → CPU RAM
        # ====================================================================
        geometry_cache['xyz'][global_ids] = self.gpu_xyz.detach().cpu()
        geometry_cache['scaling'][global_ids] = self.gpu_scaling.detach().cpu()
        geometry_cache['rotation'][global_ids] = self.gpu_rotation.detach().cpu()
        geometry_cache['opacity'][global_ids] = self.gpu_opacity.detach().cpu()
        
        # ====================================================================
        # Update SH parameters: GPU → CPU RAM
        # ====================================================================
        # Concatenate DC and Rest
        gpu_sh_params = torch.cat([self.gpu_features_dc, self.gpu_features_rest], dim=1)  # (M, 48)
        cpu_sh_params = gpu_sh_params.detach().cpu()
        sh_cache[global_ids] = cpu_sh_params
        
        self._log(f"[GPUWorkingSet] Updated RAM cache for {len(global_ids):,} Gaussians")
        self._log("[GPUWorkingSet] Updated all parameters: geometry(11 dims) + SH(48 dims)")
    
    def get_working_set_size(self) -> Dict[str, float]:
        """Get current GPU memory usage statistics."""
        stats = {
            "num_gaussians": 0,
            "memory_mb": 0.0,
            "num_blocks": len(self.loaded_blocks)
        }
        
        if self.gpu_xyz is not None:
            num_gaussians = self.gpu_xyz.shape[0]
            # xyz(3) + scaling(3) + rotation(4) + opacity(1) + DC(3) + Rest(45) = 59 floats
            memory_mb = num_gaussians * 59 * 4 / (1024**2)
            
            stats["num_gaussians"] = num_gaussians
            stats["memory_mb"] = memory_mb
        
        return stats
    
    def clear(self):
        """Clear GPU working set and release ALL memory.
        
        Note: During training iterations, prefer prepare_for_retention() which
        preserves GPU data for inter-batch resident-overlap reuse and avoids expensive
        torch.cuda.empty_cache() CUDA synchronization.
        """
        self.gpu_xyz = None
        self.gpu_scaling = None
        self.gpu_rotation = None
        self.gpu_opacity = None
        self.gpu_features_dc = None
        self.gpu_features_rest = None
        self.block_to_gpu_slice.clear()
        self.loaded_blocks.clear()
        self.local_to_global_idx = None
        self._block_starts = None
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def prepare_for_retention(self):
        """Preserve GPU working set for next iteration's resident-overlap reuse.
        
        Unlike clear(), this keeps GPU tensors and block mappings alive so that
        the next call to load_visible_blocks_with_retention() can:
        - Reuse overlapping block data directly from GPU when that path is valid,
          saving CPU->GPU bandwidth for overlapping blocks
        - Skip torch.cuda.empty_cache() CUDA sync overhead
          and log accurate retention statistics for monitoring spatial locality
        
        Detaches from autograd graph to allow computation graph garbage collection.
        Does NOT call torch.cuda.empty_cache() to avoid CUDA synchronization overhead.
        
        The retained GPU memory (~2-4 GB working set) is automatically freed when
        the next iteration's load_visible_blocks_with_retention() reassigns tensors.
        
        Interface compatible: all public methods (load_visible_blocks_with_retention,
        get_retention_stats, get_working_set_size, clear) continue to work correctly.
        """
        if self.gpu_xyz is not None:
            # Detach from autograd to release computation graph (grads, optimizer states)
            # while keeping the data tensor alive for resident-overlap reuse.
            self.gpu_xyz = self.gpu_xyz.detach()
            self.gpu_scaling = self.gpu_scaling.detach()
            self.gpu_rotation = self.gpu_rotation.detach()
            self.gpu_opacity = self.gpu_opacity.detach()
            self.gpu_features_dc = self.gpu_features_dc.detach()
            self.gpu_features_rest = self.gpu_features_rest.detach()
        # Keep block_to_gpu_slice, loaded_blocks, local_to_global_idx intact
        # previous_blocks is already set by load_visible_blocks_with_retention()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.clear()
