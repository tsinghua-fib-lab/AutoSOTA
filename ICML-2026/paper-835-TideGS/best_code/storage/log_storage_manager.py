"""
Log-Structured Storage Manager for Billion-scale Gaussian Splatting.

Implements append-only patch-based storage to optimize for SSD characteristics:
- Sequential writes for performance and longevity
- Metadata index for efficient block lookup
- Patch file management with compaction support
"""

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch


@dataclass
class BlockLocation:
    """Metadata for locating a block in storage."""
    file_id: int  # 0 = base_file, 1+ = patch files
    offset: int   # Byte offset in the file
    size: int     # Size in bytes
    version: int  # Version number for tracking updates


class LogStorageManager:
    """
    Log-Structured Storage Manager with append-only semantics.

    Architecture:
    - base_file.bin: Initial complete Gaussian dataset
    - patch_XXX.bin: Update logs (append-only)
    - Metadata index maps block_id -> (file, offset, size)
    """

    def __init__(
        self,
        storage_dir: str,
        block_size: int,
        num_blocks: int,
        point_dim: int = 59,  # 3(xyz) + 3(scale) + 4(rot) + 3(opacity) + 48(SH)
        dtype: torch.dtype = torch.float32,
        verbose: bool = True,
    ):
        """
        Initialize Log-Structured Storage Manager.

        Args:
            storage_dir: Directory for base and patch files
            block_size: Number of Gaussian points per block
            num_blocks: Total number of blocks
            point_dim: Dimension per Gaussian point (default 59)
            dtype: Data type for storage
            verbose: Print routine storage lifecycle messages
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.block_size = block_size
        self.num_blocks = num_blocks
        self.point_dim = point_dim
        self.dtype = dtype
        self.verbose = bool(verbose)
        self.bytes_per_point = point_dim * torch.finfo(dtype).bits // 8
        self.bytes_per_block = block_size * self.bytes_per_point

        # Metadata index: block_id -> BlockLocation
        # 用于快速查找任意 Block 当前存储在文件的哪个位置
        self.index: Dict[int, BlockLocation] = {}
        # 线程锁: 防止多线程同时修改索引导致竞争条件
        self.index_lock = threading.Lock()

        # File handles (lazy opened)
        # file_handles: 缓存已打开的文件对象，避免重复打开关闭
        self.file_handles: Dict[int, object] = {}  # file_id -> open file handle
        # 维护 file_id 到磁盘物理路径的映射
        self.file_paths: Dict[int, Path] = {0: self.storage_dir / "base_file.bin"}

        # Patch file counter
        self.next_patch_id = 1
        # 计数器锁：确保多线程并发场景 patch 时ID不会冲突
        self.patch_counter_lock = threading.Lock()

        # Statistics
        self.stats = {
            'reads': 0,
            'writes': 0,
            'patches_created': 0,
            'compactions': 0
        }

        # 执行初始化逻辑， 建立初始的文件索引映射
        self._initialize_index()

    def _initialize_index(self):
        """Initialize metadata index pointing all blocks to base file."""
        base_file = self.file_paths[0]

        if base_file.exists():
            # Load existing base file
            for block_id in range(self.num_blocks):
                self.index[block_id] = BlockLocation(
                    file_id=0,
                    offset=block_id * self.bytes_per_block,
                    size=self.bytes_per_block,
                    version=0
                )
            if self.verbose:
                print(f"[LogStorage] Loaded base file with {self.num_blocks} blocks")
        else:
            # Create empty base file
            if self.verbose:
                print(f"[LogStorage] Creating new base file at {base_file}")
            with open(base_file, 'wb') as f:
                # Pre-allocate space
                # 将文件指针移动到文件最后一个字节的位置
                f.seek(self.num_blocks * self.bytes_per_block - 1)
                # 写入一个空字节，操作系统会立刻为文件分配这个大的逻辑大小，不需要将中间都填满0
                f.write(b'\0')

            # 初始化索引，将所有 block 指向这个新创建的空文件
            for block_id in range(self.num_blocks):
                self.index[block_id] = BlockLocation(
                    file_id=0,
                    offset=block_id * self.bytes_per_block,
                    size=self.bytes_per_block,
                    version=0
                )

    def read_blocks(self, block_ids: List[int]) -> Dict[int, torch.Tensor]:
        """
        Read multiple blocks from storage.
        从存储中批量读取多个 blocks
        Groups reads by file to minimize seeking and maximize sequential access.
        核心优化：按文件分组，按offset排序，最小化seeking时间，最大化顺序访问
        Args:
            block_ids: List of block IDs to read

        Returns:
            Dictionary mapping block_id -> tensor data
        """
        if not block_ids:
            return {}

        result = {}

        # Group blocks by file for efficient batched reads
        # 建立分组字典： k是文件id，v是(block_id, Blocklocation)列表
        file_groups: Dict[int, List[Tuple[int, BlockLocation]]] = {}

        # 读取索引时加锁，防止读取过程中索引被后台写线程修改
        with self.index_lock:
            for block_id in block_ids:
                if block_id not in self.index:
                    raise ValueError(f"Block {block_id} not in index")

                location = self.index[block_id]
                if location.file_id not in file_groups:
                    file_groups[location.file_id] = []
                file_groups[location.file_id].append((block_id, location))

        # Read from each file
        for file_id, blocks in file_groups.items():
            file_path = self.file_paths[file_id]

            # 核心优化: Sort by offset for sequential reading
            # 这样 f.seek() 总是向后移动，不会来回跳跃，极大提升了SSD 的吞吐量
            blocks.sort(key=lambda x: x[1].offset)

            with open(file_path, 'rb') as f:
                for block_id, location in blocks:
                    if location.size <= 0:
                        print(
                            f"[LogStorage] WARNING: block {block_id} points to an empty patch record; "
                            "falling back to base file"
                        )
                        result[block_id] = self._read_base_block(block_id)
                        continue

                    # 移动指针到block的起始位置
                    f.seek(location.offset)
                    # 读取指定大小的二进制数据
                    data_bytes = f.read(location.size)

                    # Convert to tensor (copy to make it writable)
                    np_array = np.frombuffer(data_bytes, dtype=np.float32).copy() 
                    curr_num_points = np_array.size // self.point_dim
                    if curr_num_points <= 0:
                        print(
                            f"[LogStorage] WARNING: block {block_id} read as empty from {file_path}; "
                            "falling back to base file"
                        )
                        tensor = self._read_base_block(block_id)
                    else:
                        tensor = torch.from_numpy(np_array).reshape(curr_num_points, self.point_dim)
                    result[block_id] = tensor

            self.stats['reads'] += len(blocks)

        return result

    def _read_base_block(self, block_id: int) -> torch.Tensor:
        """Read a block directly from the immutable base segment."""
        base_path = self.file_paths[0]
        offset = int(block_id) * self.bytes_per_block
        with open(base_path, 'rb') as f:
            f.seek(offset)
            data_bytes = f.read(self.bytes_per_block)
        np_array = np.frombuffer(data_bytes, dtype=np.float32).copy()
        curr_num_points = np_array.size // self.point_dim
        if curr_num_points <= 0:
            raise RuntimeError(f"Base segment returned empty data for block {block_id}")
        return torch.from_numpy(np_array).reshape(curr_num_points, self.point_dim)

    def write_patch(self, block_dict: Dict[int, torch.Tensor]) -> int:
        """
        Write updated blocks to a new patch file (append-only).

        Args:
            block_dict: Dictionary mapping block_id -> updated tensor data

        Returns:
            patch_id: ID of the created patch file
        """
        if not block_dict:
            return -1

        valid_blocks = {}
        for block_id, tensor in block_dict.items():
            if tensor is None or tensor.numel() == 0:
                print(f"[LogStorage] WARNING: skip empty dirty block {block_id}")
                continue
            if tensor.dim() != 2 or tensor.shape[1] != self.point_dim:
                raise ValueError(
                    f"Invalid dirty block {block_id} shape {tuple(tensor.shape)}; "
                    f"expected (*, {self.point_dim})"
                )
            if tensor.shape[0] > self.block_size:
                raise ValueError(
                    f"Invalid dirty block {block_id} rows {tensor.shape[0]}; "
                    f"expected at most {self.block_size}"
                )
            valid_blocks[int(block_id)] = tensor

        if not valid_blocks:
            print("[LogStorage] Skipped patch creation: no non-empty dirty blocks")
            return -1

        # Allocate new patch file ID
        with self.patch_counter_lock:
            patch_id = self.next_patch_id
            self.next_patch_id += 1

        # Create patch file
        timestamp = int(time.time() * 1000000)
        patch_file = self.storage_dir / f"patch_{patch_id:06d}_{timestamp}.bin"
        self.file_paths[patch_id] = patch_file

        # Write blocks sequentially
        current_offset = 0
        updated_locations = {}

        with open(patch_file, 'wb') as f:
            for block_id in sorted(valid_blocks.keys()):  # Sort for determinism
                tensor = valid_blocks[block_id]

                # Convert to bytes
                if tensor.is_cuda:
                    tensor = tensor.cpu()

                data_bytes = tensor.numpy().astype(np.float32).tobytes()

                # Write to file
                f.write(data_bytes)

                # Record new location
                with self.index_lock:
                    old_version = self.index[block_id].version if block_id in self.index else 0
                    updated_locations[block_id] = BlockLocation(
                        file_id=patch_id,
                        offset=current_offset,
                        size=len(data_bytes),
                        version=old_version + 1
                    )

                # 指针后移
                current_offset += len(data_bytes)

        # Update index atomically
        # 只有文件全部写完落盘后，才会更新内存索引，确保不会读到写了一半的文件
        with self.index_lock:
            for block_id, location in updated_locations.items():
                self.index[block_id] = location

        self.stats['writes'] += len(valid_blocks)
        self.stats['patches_created'] += 1

        print(f"[LogStorage] Created patch {patch_id} with {len(valid_blocks)} blocks ({current_offset / 1024 / 1024:.2f} MB)")

        return patch_id

    def export_index_manifest(
        self,
        manifest_path: str | Path,
        patches_dir: str | Path,
        copy_patch_files: bool = True,
    ) -> Dict:
        """Persist the current log-structured index for incremental checkpoints.

        The immutable base file is referenced by absolute path. Patch files can be
        copied into the checkpoint so resume does not depend on the original run's
        SSD cache directory.
        """
        manifest_path = Path(manifest_path)
        patches_dir = Path(patches_dir)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        patches_dir.mkdir(parents=True, exist_ok=True)

        with self.index_lock:
            index_snapshot = dict(self.index)
            file_paths_snapshot = dict(self.file_paths)
        with self.patch_counter_lock:
            next_patch_id = int(self.next_patch_id)

        referenced_file_ids = {0}
        referenced_file_ids.update(int(location.file_id) for location in index_snapshot.values())
        files = {}
        copied_patch_files = 0
        copied_patch_bytes = 0
        for file_id, source_path in sorted(file_paths_snapshot.items()):
            file_id = int(file_id)
            if file_id not in referenced_file_ids:
                continue
            source_path = Path(source_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Log storage file missing for file_id={file_id}: {source_path}")

            if file_id == 0 or not copy_patch_files:
                checkpoint_path = source_path.resolve()
                copied = False
            else:
                checkpoint_path = patches_dir / source_path.name
                if source_path.resolve() != checkpoint_path.resolve():
                    shutil.copy2(source_path, checkpoint_path)
                checkpoint_path = checkpoint_path.resolve()
                copied = True
                copied_patch_files += 1
                copied_patch_bytes += int(checkpoint_path.stat().st_size)

            files[str(file_id)] = {
                "path": str(checkpoint_path),
                "role": "base" if file_id == 0 else "patch",
                "copied": bool(copied),
                "size": int(checkpoint_path.stat().st_size),
            }

        index_payload = {
            str(block_id): {
                "file_id": int(location.file_id),
                "offset": int(location.offset),
                "size": int(location.size),
                "version": int(location.version),
            }
            for block_id, location in sorted(index_snapshot.items())
        }
        manifest = {
            "version": 1,
            "storage_type": "log_structured_blocks",
            "block_size": int(self.block_size),
            "num_blocks": int(self.num_blocks),
            "point_dim": int(self.point_dim),
            "dtype": "float32",
            "bytes_per_block": int(self.bytes_per_block),
            "next_patch_id": int(next_patch_id),
            "files": files,
            "index": index_payload,
            "copied_patch_files": int(copied_patch_files),
            "copied_patch_bytes": int(copied_patch_bytes),
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return manifest

    def load_index_manifest(self, manifest_path: str | Path) -> Dict:
        """Load a previously persisted log-structured index."""
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        if int(manifest.get("block_size", -1)) != int(self.block_size):
            raise ValueError(
                f"Index block_size mismatch: manifest={manifest.get('block_size')} runtime={self.block_size}"
            )
        if int(manifest.get("num_blocks", -1)) != int(self.num_blocks):
            raise ValueError(
                f"Index num_blocks mismatch: manifest={manifest.get('num_blocks')} runtime={self.num_blocks}"
            )
        if int(manifest.get("point_dim", -1)) != int(self.point_dim):
            raise ValueError(
                f"Index point_dim mismatch: manifest={manifest.get('point_dim')} runtime={self.point_dim}"
            )

        loaded_file_paths: Dict[int, Path] = {}
        for file_id_raw, file_info in manifest.get("files", {}).items():
            file_id = int(file_id_raw)
            path = Path(file_info["path"])
            if not path.exists():
                raise FileNotFoundError(f"Index file_id={file_id} path does not exist: {path}")
            loaded_file_paths[file_id] = path

        loaded_index: Dict[int, BlockLocation] = {}
        for block_id_raw, location in manifest.get("index", {}).items():
            block_id = int(block_id_raw)
            file_id = int(location["file_id"])
            if file_id not in loaded_file_paths:
                raise KeyError(f"Block {block_id} references missing file_id={file_id}")
            loaded_index[block_id] = BlockLocation(
                file_id=file_id,
                offset=int(location["offset"]),
                size=int(location["size"]),
                version=int(location["version"]),
            )

        if len(loaded_index) != int(self.num_blocks):
            raise ValueError(
                f"Index block count mismatch: manifest={len(loaded_index)} runtime={self.num_blocks}"
            )

        self.close()
        with self.index_lock:
            self.file_paths = loaded_file_paths
            self.index = loaded_index
        with self.patch_counter_lock:
            max_file_id = max(loaded_file_paths.keys()) if loaded_file_paths else 0
            self.next_patch_id = max(int(manifest.get("next_patch_id", 1)), max_file_id + 1)
        print(
            f"[LogStorage] Loaded index manifest {manifest_path} "
            f"files={len(self.file_paths)} next_patch_id={self.next_patch_id}"
        )
        return manifest

    # def compact(self, min_patches: int = 10) -> bool:
    #     """
    #     Compact patch files back into base file (background maintenance).

    #     This is a placeholder for future optimization. In production:
    #     1. Identify blocks spread across many patches
    #     2. Rewrite them to a new base file
    #     3. Update index and delete old patches

    #     Args:
    #         min_patches: Minimum number of patches before compaction triggers

    #     Returns:
    #         True if compaction was performed
    #     """
    #     num_patches = len(self.file_paths) - 1  # Exclude base file

    #     if num_patches < min_patches:
    #         return False

    #     print(f"[LogStorage] Compaction triggered ({num_patches} patches)")

    #     # TODO: Implement compaction logic
    #     # 1. Create new base file
    #     # 2. Read all blocks in order
    #     # 3. Write to new base sequentially
    #     # 4. Update index
    #     # 5. Delete old patches

    #     self.stats['compactions'] += 1
    #     return True

    def compact(self, min_patches: int = 10) -> bool:
        """
        执行压缩：将所有零散的 Patch 和 Base 文件合并为一个新的 Base 文件。
        """
        # 1. 检查是否需要压缩
        # 这里的 file_paths 包含了 base(id=0) 和 patches。减1是去掉 base。
        current_patch_files = [pid for pid in self.file_paths if pid != 0]
        if len(current_patch_files) < min_patches:
            return False

        print(f"[LogStorage] Compaction triggered. Merging {len(current_patch_files)} patches...")
        start_time = time.time()

        # 2. 准备新文件路径
        timestamp = int(time.time() * 1000000)
        new_base_path = self.storage_dir / f"base_compacted_{timestamp}.bin"
        
        # 3. 执行合并 (核心逻辑)
        # 我们需要读取所有 Block 的最新数据，按顺序写入新文件
        current_offset = 0
        new_index_map = {} # 暂存新的索引信息

        try:
            with open(new_base_path, 'wb') as f_out:
                # 按顺序遍历所有 Block ID
                # 这样可以保证新文件是完美的顺序存储，读取速度最快
                # 注意：这里我们分批读取以节省内存
                chunk_size = 1024 # 每次处理 1024 个 Block
                
                for chunk_start in range(0, self.num_blocks, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, self.num_blocks)
                    block_ids = list(range(chunk_start, chunk_end))
                    
                    # 复用现有的 read_blocks 方法读取最新数据
                    # 这会自动去各个 Patch 文件里找最新的版本
                    blocks_data = self.read_blocks(block_ids)
                    
                    # 按顺序写入新文件
                    for block_id in block_ids:
                        if block_id not in blocks_data:
                            # 理论上不应该发生，除非初始化有问题
                            # 如果是空块，写入全0
                            data_bytes = b'\0' * self.bytes_per_block
                        else:
                            tensor = blocks_data[block_id]
                            data_bytes = tensor.numpy().astype(np.float32).tobytes()
                        
                        f_out.write(data_bytes)
                        
                        # 记录新位置
                        new_index_map[block_id] = BlockLocation(
                            file_id=0, # 重置为 Base ID (我们稍后会把这个新文件映射为 ID 0)
                            offset=current_offset,
                            size=len(data_bytes),
                            version=self.index[block_id].version + 1 # 版本延续
                        )
                        current_offset += len(data_bytes)
                        
            # 确保数据落盘
            f_out.flush()
            os.fsync(f_out.fileno())

        except Exception as e:
            print(f"[LogStorage] Compaction failed: {e}")
            if new_base_path.exists():
                new_base_path.unlink() # 删除失败的临时文件
            return False

        # 4. 原子性切换 (Atomic Switch)
        # 这是最危险的一步，需要加锁，防止此时有其他线程在读写
        with self.index_lock:
            with self.patch_counter_lock:
                # A. 关闭所有旧文件句柄
                self.close() 
                
                # B. 删除旧文件 (清理磁盘空间!)
                # 注意：保留原 base_file.bin 的路径名用于重命名，或者直接用新名字
                old_files = list(self.file_paths.values())
                
                # C. 更新索引
                self.index = new_index_map
                
                # D. 重置文件路径映射
                # 将新文件设为 ID 0 (Base File)
                self.file_paths = {0: new_base_path}
                self.file_handles = {} # 清空句柄缓存
                
                # E. 删除物理磁盘上的旧文件
                for p in old_files:
                    try:
                        if p.exists() and p != new_base_path:
                            p.unlink()
                    except OSError as e:
                        print(f"[LogStorage] Warning: Failed to delete {p}: {e}")

        duration = time.time() - start_time
        final_size_mb = current_offset / 1024 / 1024
        print(f"[LogStorage] Compaction finished in {duration:.2f}s.")
        print(f"[LogStorage] Merged into new base file ({final_size_mb:.2f} MB). Old patches deleted.")
        
        self.stats['compactions'] += 1
        return True

    def get_stats(self) -> Dict:
        """Get storage statistics."""
        total_files = len(self.file_paths)
        total_size = sum(p.stat().st_size for p in self.file_paths.values() if p.exists())

        return {
            **self.stats,
            'total_files': total_files,
            'total_size_mb': total_size / 1024 / 1024,
            'num_patches': total_files - 1
        }

    def close(self):
        """Close all file handles."""
        for fh in self.file_handles.values():
            if hasattr(fh, 'close'):
                fh.close()
        self.file_handles.clear()
        try:
            stats = self.get_stats()
        except Exception as exc:
            stats = {"error": str(exc)}
        if "error" in stats:
            print(f"[LogStorage] Closed with stats error: {stats['error']}")
        else:
            print(
                "[LogStorage] Closed summary: "
                f"reads={stats.get('reads', 0)} "
                f"writes={stats.get('writes', 0)} "
                f"patches={stats.get('patches_created', 0)} "
                f"files={stats.get('total_files', 0)} "
                f"size_mb={stats.get('total_size_mb', 0.0):.1f}"
            )
