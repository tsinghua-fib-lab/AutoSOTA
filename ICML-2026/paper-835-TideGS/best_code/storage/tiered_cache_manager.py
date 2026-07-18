"""
Tiered Cache Manager with RAM buffering and LRU eviction.

Manages the CPU RAM layer between GPU and SSD:
- LRU cache for frequently accessed blocks
- Dirty tracking for modified blocks
- Asynchronous GPU->RAM sync
- Intelligent prefetching and eviction
"""

import threading
import time
from collections import OrderedDict
from queue import Queue, Empty, Full
from typing import Dict, List, Set, Optional, Tuple
import torch
import psutil


class TieredCacheManager:
    """
    RAM Cache & Write Buffer for tiered storage system.

    Architecture:
    - cache_data: In-RAM copy of Gaussian blocks
    - dirty_set: Tracks blocks modified by GPU (needs flush to SSD)
    - lru_queue: Maintains access order for eviction
    - async_sync: Background thread for GPU->RAM transfers
    """

    def __init__(
        self,
        storage_manager,
        max_ram_gb: float = 16.0,
        block_size: int = 4096,
        point_dim: int = 59,
        prefetch_distance: int = 5,
        eviction_threshold: float = 0.8,
        verbose: bool = True,
    ):
        """
        Initialize Tiered Cache Manager.

        Args:
            storage_manager: LogStorageManager instance
            max_ram_gb: Maximum RAM usage in GB
            block_size: Number of points per block
            point_dim: Dimension per point
            prefetch_distance: Number of future blocks to prefetch
            eviction_threshold: RAM usage ratio to trigger eviction (0-1)
        """

        # 保存底层存储引用
        # 当 RAM 缓存 Miss时，需要调用它从SSD 中读取
        # 当 RAM 缓存中有脏数据需要罗盘时，需要调用它写入 SSD
        self.storage = storage_manager
        self.max_ram_bytes = int(max_ram_gb * 1024 * 1024 * 1024)
        self.block_size = block_size
        self.point_dim = point_dim
        self.prefetch_distance = prefetch_distance
        self.eviction_threshold = eviction_threshold
        self.verbose = bool(verbose)

        # Bytes per block
        self.bytes_per_block = block_size * point_dim * 4  # float32

        # Cache storage
        # 核心缓存结构
        # 用OrderedDict 来实现 Least Recently Used （LRU）
        # - 最新访问的移动到末尾
        # - 内存满时，弹出最前面的（最久未使用的）
        self.cache_data: OrderedDict[int, torch.Tensor] = OrderedDict()
        # 记录脏块的id，如果id在这个set中，那么说明它在内存中的版本比SSD中的版本更新
        # 驱逐前必须先flush，写回SSD
        self.dirty_set: Set[int] = set()
        self.block_versions: Dict[int, int] = {}
        self.cache_lock = threading.RLock()
        
        # [FIX] Flushing buffer to prevent race condition
        # 正在写入 SSD 的 dirty blocks 会暂存在这里，防止 prefetch 读到旧数据
        # Key: block_id, Value: (tensor, timestamp)
        self.flushing_buffer: Dict[int, Tuple[torch.Tensor, float, int]] = {}
        self.flushing_lock = threading.RLock()

        # Async sync queue: (block_id, gpu_tensor, cuda_stream)
        # 生产者消费者模式，用于异步将GPU上的更新同步回RAM
        # 队列元素: (block_id, gpu_tensor, cuda_stream)
        # maxsize = 100, 限制队列长度，防止GPU跑太快导致内存堆积
        self.sync_queue: Queue = Queue(maxsize=100)
        self.sync_thread = None
        self.sync_running = False

        # Async SSD writeback queue (dirty RAM -> SSD patch log)
        self.flush_queue: Queue = Queue(maxsize=128)
        self.flush_thread = None
        self.flush_running = False

        # Async SSD -> RAM future prefetch queue.  Urgent prefetch() remains
        # synchronous fallback; this queue warms Delta+ blocks off the main path.
        self.future_prefetch_queue: Queue = Queue(maxsize=64)
        self.future_prefetch_thread = None
        self.future_prefetch_running = False
        self.future_pending: Set[int] = set()
        self.future_lock = threading.RLock()

        # Prefetch tracking， 防止重复预取同一个block
        self.prefetch_history: Set[int] = set()

        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'flushes': 0,
            'prefetches': 0,
            'syncs': 0,
            'async_flush_requests': 0,
            'flush_queue_drops': 0,
            'async_flush_jobs': 0,
            'async_flush_blocks': 0,
            'async_flush_time': 0.0,
            'sync_flush_jobs': 0,
            'sync_flush_blocks': 0,
            'sync_flush_time': 0.0,
            'urgent_prefetch_calls': 0,
            'urgent_prefetch_blocks': 0,
            'urgent_prefetch_miss_blocks': 0,
            'urgent_prefetch_time': 0.0,
            'future_prefetch_submitted': 0,
            'future_prefetch_skipped': 0,
            'future_prefetch_dropped': 0,
            'future_prefetch_jobs': 0,
            'future_prefetch_blocks': 0,
            'future_prefetch_time': 0.0,
            'future_prefetch_errors': 0,
        }

        # Start async worker threads
        self._start_sync_thread()
        self._start_flush_thread()
        self._start_future_prefetch_thread()

    def _start_sync_thread(self):
        """Start background thread for async GPU->RAM sync."""
        # 启动后台线程，用于异步将 GPU 数据 同步到 CPU RAM
        self.sync_running = True
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True) # 这里创建了线程，指定目标函数，并且daemon=True表示线程会随着主程序退出而退出
        self.sync_thread.start()
        self._log("[TieredCache] Async sync thread started")

    def _start_flush_thread(self):
        """Start background thread for async RAM->SSD dirty writeback."""
        self.flush_running = True
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()
        self._log("[TieredCache] Async flush thread started")

    def _start_future_prefetch_thread(self):
        """Start background thread for SSD->RAM future prefetch."""
        self.future_prefetch_running = True
        self.future_prefetch_thread = threading.Thread(
            target=self._future_prefetch_worker,
            daemon=True,
        )
        self.future_prefetch_thread.start()
        self._log("[TieredCache] Async future prefetch thread started")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _flush_dirty_blocks(
        self,
        dirty_blocks: Dict[int, torch.Tensor],
        block_versions: Optional[Dict[int, int]] = None,
        mode: str = 'async',
    ):
        """Flush a batch of dirty blocks to SSD and clear flushing metadata."""
        if not dirty_blocks:
            return

        block_versions = block_versions or {}
        t0 = time.time()
        try:
            self.storage.write_patch(dirty_blocks)
            elapsed = time.time() - t0

            with self.flushing_lock:
                for block_id in dirty_blocks.keys():
                    flushed_version = block_versions.get(block_id)
                    current_entry = self.flushing_buffer.get(block_id)
                    if current_entry is None:
                        continue
                    _, _, buffered_version = current_entry
                    if flushed_version is None or buffered_version <= flushed_version:
                        self.flushing_buffer.pop(block_id, None)

            with self.cache_lock:
                for block_id in dirty_blocks.keys():
                    flushed_version = block_versions.get(block_id)
                    if flushed_version is None:
                        self.dirty_set.discard(block_id)
                        continue
                    if self.block_versions.get(block_id, 0) == flushed_version:
                        self.dirty_set.discard(block_id)

            self.stats['flushes'] += 1
            if mode == 'async':
                self.stats['async_flush_jobs'] += 1
                self.stats['async_flush_blocks'] += len(dirty_blocks)
                self.stats['async_flush_time'] += elapsed
            else:
                self.stats['sync_flush_jobs'] += 1
                self.stats['sync_flush_blocks'] += len(dirty_blocks)
                self.stats['sync_flush_time'] += elapsed
        except Exception as e:
            print(f"[TieredCache] ERROR during SSD write: {e}")
            with self.cache_lock:
                for block_id, tensor in dirty_blocks.items():
                    if block_id not in self.cache_data:
                        self.cache_data[block_id] = tensor
                        self.cache_data.move_to_end(block_id)
                    self.dirty_set.add(block_id)
            with self.flushing_lock:
                for block_id in dirty_blocks.keys():
                    self.flushing_buffer.pop(block_id, None)
            raise

    def _enqueue_dirty_flush(
        self,
        dirty_blocks: Dict[int, torch.Tensor],
        block_versions: Optional[Dict[int, int]] = None,
        reason: str = 'background',
    ) -> bool:
        """Schedule dirty RAM blocks for background SSD writeback."""
        if not dirty_blocks:
            return True

        try:
            self.flush_queue.put_nowait((reason, dirty_blocks, block_versions or {}))
            self.stats['async_flush_requests'] += 1
            return True
        except Full:
            self.stats['flush_queue_drops'] += 1
            print(f"[TieredCache] Flush queue full; falling back to synchronous write for {len(dirty_blocks)} blocks")
            return False

    def _flush_worker(self):
        """Background worker for async RAM->SSD dirty block writeback."""
        while self.flush_running or not self.flush_queue.empty():
            try:
                reason, dirty_blocks, block_versions = self.flush_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                self._flush_dirty_blocks(dirty_blocks, block_versions=block_versions, mode='async')
            except Exception as e:
                print(f"[TieredCache] Flush worker error ({reason}): {e}")
            finally:
                self.flush_queue.task_done()

    def _future_prefetch_worker(self):
        """Background worker for SSD->RAM future block prefetch."""
        while self.future_prefetch_running or not self.future_prefetch_queue.empty():
            try:
                block_ids = self.future_prefetch_queue.get(timeout=0.1)
            except Empty:
                continue

            block_ids = [int(block_id) for block_id in block_ids]
            loaded_count = 0
            t0 = time.time()
            try:
                to_load = []
                with self.cache_lock:
                    for block_id in block_ids:
                        if block_id in self.cache_data:
                            self.cache_data.move_to_end(block_id)
                            continue

                        with self.flushing_lock:
                            flushing_entry = self.flushing_buffer.get(block_id)
                        if flushing_entry is not None:
                            tensor, _, _ = flushing_entry
                            self.cache_data[block_id] = tensor
                            self.cache_data.move_to_end(block_id)
                            continue

                        to_load.append(block_id)

                if to_load:
                    loaded = self.storage.read_blocks(to_load)
                    with self.cache_lock:
                        for block_id, tensor in loaded.items():
                            with self.flushing_lock:
                                if block_id in self.flushing_buffer:
                                    continue
                            if block_id not in self.cache_data:
                                self.cache_data[block_id] = tensor
                                self.cache_data.move_to_end(block_id)
                                loaded_count += 1

                elapsed = time.time() - t0
                self.stats['future_prefetch_jobs'] += 1
                self.stats['future_prefetch_blocks'] += int(loaded_count)
                self.stats['future_prefetch_time'] += elapsed
                if loaded_count:
                    self.stats['prefetches'] += int(loaded_count)
                self._maybe_evict()
            except Exception as e:
                self.stats['future_prefetch_errors'] += 1
                print(f"[TieredCache] Future prefetch worker error: {e}")
            finally:
                with self.future_lock:
                    for block_id in block_ids:
                        self.future_pending.discard(block_id)
                self.future_prefetch_queue.task_done()


    def _sync_worker(self):
        """Background worker for async D2H transfers."""
        while self.sync_running:
            try:
                # Get sync request (timeout to allow shutdown), 生产者/主线程是sync_from_gpu
                item = self.sync_queue.get(timeout=0.1) # 这是一个阻塞操作，如果队列空，线程会在这里暂停 (挂起)，不消耗cpu，一旦sync_from_gpu中有数据，线程会被唤醒，继续执行. 每0.1秒检查一次sync_running标志，确保能shutdown
                block_id, gpu_tensor, cuda_stream = item

                # Wait for CUDA stream to finish
                if cuda_stream is not None:
                    cuda_stream.synchronize()

                # Copy to CPU (pinned memory for faster transfer)
                cpu_tensor = gpu_tensor.cpu().clone()

                # Update cache
                with self.cache_lock:
                    self.cache_data[block_id] = cpu_tensor
                    self.cache_data.move_to_end(block_id)  # Mark as recently used
                    self.block_versions[block_id] = self.block_versions.get(block_id, 0) + 1
                    self.dirty_set.add(block_id) # 标记为脏，也就是修改过，未来需要写回SSD

                self.stats['syncs'] += 1

                # Check if eviction is needed
                self._maybe_evict()

            except Empty:
                continue
            except Exception as e:
                print(f"[TieredCache] Sync worker error: {e}")

    def sync_from_gpu( # TODO: 这段代码认为 Adam 在GPU上进行，GPU更新完了，把结果传回CPU，但是我们的想法是：如果是 CPU 优化，流程会是：GPU 算梯度 -> 梯度传回 CPU -> CPU 做 Adam 更新
        self,
        block_ids: List[int],
        gpu_tensors: List[torch.Tensor],
        cuda_stream: Optional[torch.cuda.Stream] = None
    ):
        """
        Asynchronously sync updated blocks from GPU to RAM.

        Called after optimizer.step() to persist gradient updates.

        Args:
            block_ids: List of block IDs
            gpu_tensors: List of corresponding GPU tensors
            cuda_stream: CUDA stream for async copy (optional)
        """
        if len(block_ids) != len(gpu_tensors):
            raise ValueError("block_ids and gpu_tensors must have same length")

        for block_id, gpu_tensor in zip(block_ids, gpu_tensors):
            # Enqueue async sync (non-blocking)
            self.sync_queue.put((block_id, gpu_tensor, cuda_stream))

    def _is_valid_block_tensor(
        self,
        block_id: int,
        tensor: Optional[torch.Tensor],
        context: str,
    ) -> bool:
        if tensor is None or tensor.numel() == 0:
            print(f"[TieredCache] WARNING: skip empty block {block_id} during {context}")
            return False
        if tensor.dim() != 2 or tensor.shape[1] != self.point_dim:
            raise ValueError(
                f"Invalid block {block_id} shape {tuple(tensor.shape)} during {context}; "
                f"expected (*, {self.point_dim})"
            )
        if tensor.shape[0] > self.block_size:
            raise ValueError(
                f"Invalid block {block_id} rows {tensor.shape[0]} during {context}; "
                f"expected at most {self.block_size}"
            )
        return True

    def upsert_dirty_blocks(self, block_tensors: Dict[int, torch.Tensor], clone: bool = False) -> int:
        """Refresh RAM cache entries from CPU tensors and mark them dirty.

        This path is used by the paper-mode bridge after GPU resident updates:
        modified blocks are staged in the warm RAM cache before background SSD
        writeback is scheduled.
        """
        if not block_tensors:
            return 0

        staged = 0
        with self.cache_lock:
            for block_id, tensor in block_tensors.items():
                cached_tensor = tensor.clone() if clone else tensor
                if not self._is_valid_block_tensor(block_id, cached_tensor, 'upsert_dirty_blocks'):
                    continue
                self.cache_data[block_id] = cached_tensor
                self.cache_data.move_to_end(block_id)
                self.block_versions[block_id] = self.block_versions.get(block_id, 0) + 1
                self.dirty_set.add(block_id)
                staged += 1

        self._maybe_evict()
        return staged

    def submit_dirty_blocks(self, block_ids: List[int], clone: bool = True, reason: str = 'paper_writeback') -> int:
        """Submit dirty RAM blocks for asynchronous SSD writeback."""
        if not block_ids:
            return 0

        dirty_blocks: Dict[int, torch.Tensor] = {}
        dirty_versions: Dict[int, int] = {}

        with self.cache_lock:
            for block_id in block_ids:
                if block_id not in self.cache_data or block_id not in self.dirty_set:
                    continue
                tensor = self.cache_data[block_id]
                if not self._is_valid_block_tensor(block_id, tensor, 'submit_dirty_blocks'):
                    self.dirty_set.discard(block_id)
                    continue
                dirty_blocks[block_id] = tensor.clone() if clone else tensor
                dirty_versions[block_id] = self.block_versions.get(block_id, 0)

        if not dirty_blocks:
            return 0

        with self.flushing_lock:
            current_time = time.time()
            for block_id, tensor in dirty_blocks.items():
                version = dirty_versions.get(block_id, 0)
                existing = self.flushing_buffer.get(block_id)
                if existing is None or existing[2] <= version:
                    self.flushing_buffer[block_id] = (tensor, current_time, version)

        enqueued = self._enqueue_dirty_flush(dirty_blocks, block_versions=dirty_versions, reason=reason)
        if not enqueued:
            self._flush_dirty_blocks(dirty_blocks, block_versions=dirty_versions, mode='sync')

        return len(dirty_blocks)

    def prefetch(
        self,
        needed_block_ids: List[int],
        future_block_ids: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Prefetch blocks from cache or SSD.

        Args:
            needed_block_ids: Blocks needed immediately
            future_block_ids: Blocks that will be needed soon (for prefetching)

        Returns:
            Dictionary mapping block_id -> tensor (CPU)
        """
        result = {}
        cache_hits = []
        cache_misses = []
        self.stats['urgent_prefetch_calls'] += 1

        # Check cache for needed blocks
        # [FIX] Check order: cache_data -> flushing_buffer -> SSD
        with self.cache_lock:
            for block_id in needed_block_ids:
                if block_id in self.cache_data:
                    result[block_id] = self.cache_data[block_id]
                    self.cache_data.move_to_end(block_id)  # LRU update
                    cache_hits.append(block_id)
                else:
                    cache_misses.append(block_id)
        
        # [FIX] Check flushing buffer for evicted-but-not-yet-written blocks
        still_missing = []
        with self.flushing_lock:
            for block_id in cache_misses:
                if block_id in self.flushing_buffer:
                    tensor, timestamp, _ = self.flushing_buffer[block_id]
                    result[block_id] = tensor
                    cache_hits.append(block_id)  # Count as hit (avoided reading stale data)
                    print(f"[TieredCache] ⚠️ Race avoided: block {block_id} found in flushing_buffer")
                else:
                    still_missing.append(block_id)
        
        # Update cache_misses to only include blocks not in flushing_buffer
        cache_misses = still_missing

        # Load misses from SSD
        if cache_misses:
            t0 = time.time()
            loaded = self.storage.read_blocks(cache_misses)
            elapsed = time.time() - t0

            with self.cache_lock:
                for block_id, tensor in loaded.items():
                    self.cache_data[block_id] = tensor
                    self.cache_data.move_to_end(block_id)
                    result[block_id] = tensor
            self.stats['urgent_prefetch_blocks'] += len(cache_misses)
            self.stats['urgent_prefetch_miss_blocks'] += len(cache_misses)
            self.stats['urgent_prefetch_time'] += elapsed

        # Update stats
        self.stats['cache_hits'] += len(cache_hits)
        self.stats['cache_misses'] += len(cache_misses)

        # Background prefetch for future blocks
        # if future_block_ids:
        #     self._background_prefetch(future_block_ids)

        # Trigger eviction if needed
        self._maybe_evict()

        return result

    def prefetch_future(self, block_ids: List[int]) -> int:
        """Prefetch future blocks in background without blocking urgent fetches."""
        to_prefetch = []
        skipped = 0

        with self.cache_lock:
            for block_id in sorted(set(int(block_id) for block_id in block_ids)):
                if block_id in self.cache_data:
                    self.cache_data.move_to_end(block_id)
                    skipped += 1
                    continue

                with self.flushing_lock:
                    flushing_entry = self.flushing_buffer.get(block_id)
                if flushing_entry is not None:
                    tensor, _, _ = flushing_entry
                    self.cache_data[block_id] = tensor
                    self.cache_data.move_to_end(block_id)
                    skipped += 1
                    continue

                with self.future_lock:
                    if block_id in self.future_pending:
                        skipped += 1
                        continue
                    self.future_pending.add(block_id)
                to_prefetch.append(block_id)

        self.stats['future_prefetch_skipped'] += int(skipped)

        if not to_prefetch:
            return 0

        try:
            self.future_prefetch_queue.put_nowait(to_prefetch)
            self.stats['future_prefetch_submitted'] += len(to_prefetch)
            return len(to_prefetch)
        except Full:
            with self.future_lock:
                for block_id in to_prefetch:
                    self.future_pending.discard(block_id)
            self.stats['future_prefetch_dropped'] += len(to_prefetch)
            print(f"[TieredCache] Future prefetch queue full; dropped {len(to_prefetch)} blocks")
            return 0

    def _maybe_evict(self):
        """Check RAM usage and trigger eviction if needed."""
        current_usage = self._get_ram_usage()
        threshold_bytes = self.max_ram_bytes * self.eviction_threshold

        if current_usage > threshold_bytes:
            # Calculate how much to evict (evict 20% of cache)
            target_evict = int(len(self.cache_data) * 0.2)
            self.evict_and_flush(num_blocks=target_evict)

    def _get_ram_usage(self) -> int:
        """Get current RAM usage by cache."""
        with self.cache_lock:
            return len(self.cache_data) * self.bytes_per_block

    def evict_and_flush(self, num_blocks: Optional[int] = None):
        """
        Evict blocks from RAM cache using LRU policy.

        Critical logic (RACE CONDITION FIXED):
        - Clean blocks: Just remove from cache
        - Dirty blocks: Move to flushing_buffer -> Flush to SSD -> Remove from flushing_buffer
        
        This prevents race condition where:
        1. evict removes block from cache
        2. prefetch reads stale data from SSD (write not finished)
        3. prefetch overwrites cache with stale data

        Args:
            num_blocks: Number of blocks to evict (default: evict 20% of cache)
        """
        if num_blocks is None:
            num_blocks = max(1, int(len(self.cache_data) * 0.2))

        victims = []
        dirty_victims = {}
        dirty_versions = {}

        with self.cache_lock:
            # Select victims from LRU tail
            for _ in range(min(num_blocks, len(self.cache_data))):
                if not self.cache_data:
                    break

                # Pop from front (oldest)
                block_id, tensor = self.cache_data.popitem(last=False)
                victims.append(block_id)

                # Check if dirty
                if block_id in self.dirty_set:
                    if not self._is_valid_block_tensor(block_id, tensor, 'evict_and_flush'):
                        self.dirty_set.discard(block_id)
                        continue
                    dirty_victims[block_id] = tensor
                    dirty_versions[block_id] = self.block_versions.get(block_id, 0)
        
        # [FIX] Move dirty victims to flushing buffer BEFORE releasing lock
        # This ensures prefetch can find them during SSD write
        if dirty_victims:
            with self.flushing_lock:
                current_time = time.time()
                for block_id, tensor in dirty_victims.items():
                    version = dirty_versions.get(block_id, self.block_versions.get(block_id, 0))
                    self.flushing_buffer[block_id] = (tensor, current_time, version)

        # Flush dirty blocks to SSD on the background writeback worker.
        if dirty_victims:
            enqueued = self._enqueue_dirty_flush(dirty_victims, block_versions=dirty_versions, reason='evict')
            if not enqueued:
                self._flush_dirty_blocks(dirty_victims, block_versions=dirty_versions, mode='sync')

        self.stats['evictions'] += len(victims)

        if victims:
            print(f"[TieredCache] Evicted {len(victims)} blocks ({len(dirty_victims)} dirty)")

    def flush_all_dirty(self):
        """Flush all dirty blocks to SSD (e.g., at end of epoch)."""
        with self.cache_lock:
            dirty_blocks = {
                block_id: self.cache_data[block_id]
                for block_id in self.dirty_set
                if block_id in self.cache_data
            }
            dirty_versions = {
                block_id: self.block_versions.get(block_id, 0)
                for block_id in dirty_blocks.keys()
            }

        if dirty_blocks:
            print(f"[TieredCache] Flushing {len(dirty_blocks)} dirty blocks to SSD")

            with self.flushing_lock:
                current_time = time.time()
                for block_id, tensor in dirty_blocks.items():
                    version = dirty_versions.get(block_id, self.block_versions.get(block_id, 0))
                    self.flushing_buffer[block_id] = (tensor, current_time, version)

            self._flush_dirty_blocks(dirty_blocks, block_versions=dirty_versions, mode='sync')

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.cache_lock:
            cache_size = len(self.cache_data)
            dirty_count = len(self.dirty_set)
            ram_usage_mb = self._get_ram_usage() / 1024 / 1024
        
        with self.flushing_lock:
            flushing_count = len(self.flushing_buffer)

        flush_queue_size = self.flush_queue.qsize() if hasattr(self, 'flush_queue') else 0
        future_queue_size = self.future_prefetch_queue.qsize() if hasattr(self, 'future_prefetch_queue') else 0
        with self.future_lock:
            future_pending_count = len(self.future_pending)

        hit_rate = 0.0
        total_accesses = self.stats['cache_hits'] + self.stats['cache_misses']
        if total_accesses > 0:
            hit_rate = self.stats['cache_hits'] / total_accesses

        return {
            **self.stats,
            'cache_size': cache_size,
            'dirty_blocks': dirty_count,
            'flushing_blocks': flushing_count,  # [NEW]
            'flush_queue_size': flush_queue_size,
            'future_queue_size': future_queue_size,
            'future_pending_blocks': future_pending_count,
            'future_prefetch_queue_size': future_queue_size,
            'future_prefetch_pending': future_pending_count,
            'ram_usage_mb': ram_usage_mb,
            'hit_rate': hit_rate,
            'max_ram_mb': self.max_ram_bytes / 1024 / 1024
        }

    def shutdown(self):
        """Shutdown cache manager and flush all dirty blocks."""
        self._log("[TieredCache] Shutting down...")

        self.sync_running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5.0)

        if self.flush_thread:
            self.flush_queue.join()

        if self.future_prefetch_thread:
            self.future_prefetch_queue.join()

        self.flush_all_dirty()

        self.future_prefetch_running = False
        if self.future_prefetch_thread:
            self.future_prefetch_thread.join(timeout=5.0)

        self.flush_running = False
        if self.flush_thread:
            self.flush_thread.join(timeout=5.0)

        max_wait = 10.0  # seconds
        start_time = time.time()
        while True:
            with self.flushing_lock:
                if not self.flushing_buffer:
                    break
                remaining = len(self.flushing_buffer)

            if time.time() - start_time > max_wait:
                print(f"[TieredCache] WARNING: {remaining} blocks still in flushing_buffer after {max_wait}s")
                break

            time.sleep(0.1)

        stats = self.get_stats()
        print(
            "[TieredCache] Shutdown summary: "
            f"hits={stats.get('cache_hits', 0)} "
            f"misses={stats.get('cache_misses', 0)} "
            f"hit_rate={stats.get('hit_rate', 0.0):.3f} "
            f"dirty={stats.get('dirty_blocks', 0)} "
            f"sync_flush_blocks={stats.get('sync_flush_blocks', 0)} "
            f"ram_mb={stats.get('ram_usage_mb', 0.0):.1f}"
        )
