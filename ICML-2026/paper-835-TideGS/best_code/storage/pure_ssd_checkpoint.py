"""Block-wise checkpoints for the pure SSD paper path.

The default checkpoint format is incremental: it stores the log-structured SSD
index plus the patch files currently referenced by that index, and keeps the
immutable base file as a path reference.  A full snapshot writer is retained as
an explicit debug/fallback path.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


PARAM_DIM = 59
CHECKPOINT_MANIFEST = "pure_ssd_checkpoint.json"


def _log(message: str, log_file=None) -> None:
    print(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _resolve_path(path_value: str, checkpoint_dir: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((checkpoint_dir / path).resolve())


def is_pure_ssd_checkpoint(path: str | Path) -> bool:
    """Return True when ``path`` is a pure SSD checkpoint directory."""
    if not path:
        return False
    return (Path(path) / CHECKPOINT_MANIFEST).is_file()


def load_pure_ssd_checkpoint_manifest(path: str | Path) -> Dict[str, Any]:
    """Load and normalize a pure SSD checkpoint manifest."""
    checkpoint_dir = Path(path)
    manifest_path = checkpoint_dir / CHECKPOINT_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pure SSD checkpoint manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["checkpoint_dir"] = str(checkpoint_dir.resolve())
    checkpoint_type = str(manifest.get("checkpoint_type", "pure_ssd_snapshot"))
    manifest["checkpoint_type"] = checkpoint_type

    if checkpoint_type == "pure_ssd_incremental":
        delta_dir = manifest.get("delta_dir", "ssd_delta")
        manifest["delta_dir"] = _resolve_path(delta_dir, checkpoint_dir)
        manifest["storage_index"] = _resolve_path(
            manifest.get("storage_index", "ssd_delta/storage_index.json"),
            checkpoint_dir,
        )
        manifest["base_file"] = _resolve_path(manifest["base_file"], checkpoint_dir)
        manifest["block_bounds"] = _resolve_path(
            manifest.get("block_bounds", "ssd_delta/block_bounds.npy"),
            checkpoint_dir,
        )
    else:
        snapshot_dir = manifest.get("snapshot_dir", "ssd_snapshot")
        manifest["snapshot_dir"] = _resolve_path(snapshot_dir, checkpoint_dir)
        manifest["base_file"] = _resolve_path(
            manifest.get("base_file", "ssd_snapshot/base_file.bin"),
            checkpoint_dir,
        )
        manifest["block_bounds"] = _resolve_path(
            manifest.get("block_bounds", "ssd_snapshot/block_bounds.npy"),
            checkpoint_dir,
        )
    manifest["training_state"] = _resolve_path(
        manifest.get("training_state", "training_state.pth"),
        checkpoint_dir,
    )

    for key in ("total_points", "num_blocks", "block_size", "param_dim", "next_iteration"):
        if key not in manifest:
            raise KeyError(f"Pure SSD checkpoint manifest missing required key: {key}")

    return manifest


def _rows_for_block(block_id: int, total_points: int, block_size: int) -> int:
    start = int(block_id) * int(block_size)
    return max(0, min(int(block_size), int(total_points) - start))


def _wait_for_pending_writeback(cache, log_file=None) -> None:
    """Drain pending RAM/SSD writeback queues before snapshotting storage."""
    if cache is None:
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    flush_queue = getattr(cache, "flush_queue", None)
    if flush_queue is not None:
        flush_queue.join()

    if hasattr(cache, "flush_all_dirty"):
        cache.flush_all_dirty()

    if flush_queue is not None:
        flush_queue.join()

    # ``flush_all_dirty`` is synchronous, but background eviction may briefly
    # leave entries in flushing_buffer.  Wait a short bounded interval so the
    # snapshot reads the post-flush storage index.
    deadline = time.time() + 30.0
    while time.time() < deadline:
        flushing = 0
        flushing_lock = getattr(cache, "flushing_lock", None)
        flushing_buffer = getattr(cache, "flushing_buffer", None)
        if flushing_buffer is None:
            break
        if flushing_lock is not None:
            with flushing_lock:
                flushing = len(flushing_buffer)
        else:
            flushing = len(flushing_buffer)
        if flushing == 0:
            break
        time.sleep(0.05)
    else:
        _log("[PURE SSD CHECKPOINT] WARNING: timed out waiting for flushing_buffer", log_file)


def _manifest_base(storage_adapter, gaussians) -> Dict[str, Any]:
    manifest = dict(getattr(storage_adapter, "streaming_init_manifest", None) or {})
    manifest.setdefault("source_ply", getattr(gaussians, "_streaming_init_ply_path", ""))
    manifest.setdefault("scale_mode", "debug_fast_init_scales")
    return manifest


def write_pure_ssd_snapshot_checkpoint(
    *,
    storage_adapter,
    gaussians,
    checkpoint_dir: str | Path,
    iteration: int,
    next_iteration: int,
    args,
    chunk_blocks: int = 256,
    log_file=None,
) -> Dict[str, Any]:
    """Write a block-wise pure SSD snapshot checkpoint.

    Reads the latest block state through ``LogStorageManager.read_blocks`` and
    writes a fresh contiguous ``base_file.bin``.  At most ``chunk_blocks`` are
    resident in CPU memory at once.
    """
    if storage_adapter is None or getattr(storage_adapter, "storage", None) is None:
        raise RuntimeError("Pure SSD checkpoint requires an initialized storage_adapter")
    if getattr(gaussians, "_unified_params", None) is not None:
        raise RuntimeError("Pure SSD checkpoint refuses to serialize a full _unified_params table")

    chunk_blocks = int(chunk_blocks)
    if chunk_blocks <= 0:
        raise ValueError("pure_ssd_checkpoint_chunk_blocks must be positive")

    storage = storage_adapter.storage
    total_points = int(getattr(storage_adapter, "num_points"))
    block_size = int(getattr(storage_adapter, "block_size"))
    num_blocks = int(getattr(storage_adapter, "num_blocks"))
    param_dim = int(getattr(storage, "point_dim", PARAM_DIM))
    if param_dim != PARAM_DIM:
        raise ValueError(f"Pure SSD checkpoint expected param_dim={PARAM_DIM}, got {param_dim}")

    checkpoint_dir = Path(checkpoint_dir)
    snapshot_dir = checkpoint_dir / "ssd_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    base_file = snapshot_dir / "base_file.bin"
    block_bounds_file = snapshot_dir / "block_bounds.npy"
    training_state_file = checkpoint_dir / "training_state.pth"
    manifest_file = checkpoint_dir / CHECKPOINT_MANIFEST

    _log(
        f"[PURE SSD CHECKPOINT] Flushing dirty blocks before snapshot iter={iteration}",
        log_file,
    )
    _wait_for_pending_writeback(getattr(storage_adapter, "cache", None), log_file=log_file)

    _log(
        f"[PURE SSD CHECKPOINT] Writing SSD snapshot: blocks={num_blocks:,} "
        f"chunk_blocks={chunk_blocks} target={base_file}",
        log_file,
    )
    with open(base_file, "wb") as f:
        for block_start in range(0, num_blocks, chunk_blocks):
            block_end = min(block_start + chunk_blocks, num_blocks)
            block_ids = list(range(block_start, block_end))
            blocks = storage.read_blocks(block_ids)
            for block_id in block_ids:
                expected_rows = _rows_for_block(block_id, total_points, block_size)
                tensor = blocks.get(block_id)
                if tensor is None:
                    raise RuntimeError(f"Missing block {block_id} while snapshotting checkpoint")
                if tensor.is_cuda:
                    tensor = tensor.detach().cpu()
                else:
                    tensor = tensor.detach()
                if tensor.dim() != 2 or tensor.shape[1] != param_dim:
                    raise RuntimeError(
                        f"Invalid block {block_id} shape {tuple(tensor.shape)}; "
                        f"expected (*, {param_dim})"
                    )
                if tensor.shape[0] < expected_rows:
                    raise RuntimeError(
                        f"Block {block_id} has {tensor.shape[0]} rows, expected {expected_rows}"
                    )
                if tensor.shape[0] > expected_rows:
                    tensor = tensor[:expected_rows]
                array = np.ascontiguousarray(tensor.numpy(), dtype=np.float32)
                f.write(array.tobytes())

    expected_size = total_points * param_dim * np.dtype(np.float32).itemsize
    actual_size = base_file.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Pure SSD snapshot size mismatch: got {actual_size}, expected {expected_size}"
        )

    block_bounds = getattr(storage_adapter, "block_bounds", None)
    if block_bounds is None:
        block_bounds = np.zeros((num_blocks, 6), dtype=np.float32)
    block_bounds = np.asarray(block_bounds, dtype=np.float32)
    np.save(block_bounds_file, block_bounds)

    base_manifest = _manifest_base(storage_adapter, gaussians)
    scene_min = getattr(storage_adapter, "scene_min", base_manifest.get("scene_min", [0.0, 0.0, 0.0]))
    scene_max = getattr(storage_adapter, "scene_max", base_manifest.get("scene_max", [0.0, 0.0, 0.0]))
    manifest = {
        **base_manifest,
        "checkpoint_version": 1,
        "checkpoint_type": "pure_ssd_snapshot",
        "iteration": int(iteration),
        "checkpoint_iter": int(iteration),
        "next_iteration": int(next_iteration),
        "active_sh_degree": int(getattr(gaussians, "active_sh_degree", 0)),
        "optimizer_state_mode": "cold_start",
        "total_points": total_points,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "param_dim": param_dim,
        "base_file": str(base_file.resolve()),
        "block_bounds": str(block_bounds_file.resolve()),
        "snapshot_dir": str(snapshot_dir.resolve()),
        "training_state": str(training_state_file.resolve()),
        "scene_min": np.asarray(scene_min, dtype=np.float32).tolist(),
        "scene_max": np.asarray(scene_max, dtype=np.float32).tolist(),
        "args": {
            "ssd_execution_mode": getattr(args, "ssd_execution_mode", None),
            "paper_optimizer_backend": getattr(args, "paper_optimizer_backend", None),
            "paper_optimizer_state_mode": getattr(args, "paper_optimizer_state_mode", None),
            "paper_block_reader_backend": getattr(args, "paper_block_reader_backend", None),
            "paper_resident_capacity_blocks": getattr(args, "paper_resident_capacity_blocks", None),
        },
    }

    torch.save(
        {
            "next_iteration": int(next_iteration),
            "active_sh_degree": int(getattr(gaussians, "active_sh_degree", 0)),
            "optimizer_state_mode": "cold_start",
            "adam_moments_saved": False,
        },
        training_state_file,
    )

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, indent=2, sort_keys=True)

    _log(
        f"[PURE SSD CHECKPOINT] Snapshot complete: {actual_size / (1024 ** 3):.2f}GB "
        f"manifest={manifest_file}",
        log_file,
    )
    return manifest


def write_pure_ssd_incremental_checkpoint(
    *,
    storage_adapter,
    gaussians,
    checkpoint_dir: str | Path,
    iteration: int,
    next_iteration: int,
    args,
    log_file=None,
) -> Dict[str, Any]:
    """Write an incremental pure SSD checkpoint.

    This persists the current log-structured storage index and the patch files
    referenced by it. The immutable base segment is referenced by path instead
    of copied, so 1B checkpoints scale with dirty patch volume rather than the
    full 245GB base.
    """
    if storage_adapter is None or getattr(storage_adapter, "storage", None) is None:
        raise RuntimeError("Pure SSD checkpoint requires an initialized storage_adapter")
    if getattr(gaussians, "_unified_params", None) is not None:
        raise RuntimeError("Pure SSD checkpoint refuses to serialize a full _unified_params table")

    storage = storage_adapter.storage
    total_points = int(getattr(storage_adapter, "num_points"))
    block_size = int(getattr(storage_adapter, "block_size"))
    num_blocks = int(getattr(storage_adapter, "num_blocks"))
    param_dim = int(getattr(storage, "point_dim", PARAM_DIM))
    if param_dim != PARAM_DIM:
        raise ValueError(f"Pure SSD checkpoint expected param_dim={PARAM_DIM}, got {param_dim}")

    checkpoint_dir = Path(checkpoint_dir)
    delta_dir = checkpoint_dir / "ssd_delta"
    patches_dir = delta_dir / "patches"
    delta_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)
    storage_index_file = delta_dir / "storage_index.json"
    block_bounds_file = delta_dir / "block_bounds.npy"
    training_state_file = checkpoint_dir / "training_state.pth"
    manifest_file = checkpoint_dir / CHECKPOINT_MANIFEST

    _log(
        f"[PURE SSD CHECKPOINT] Flushing dirty blocks before incremental iter={iteration}",
        log_file,
    )
    _wait_for_pending_writeback(getattr(storage_adapter, "cache", None), log_file=log_file)

    index_manifest = storage.export_index_manifest(
        manifest_path=storage_index_file,
        patches_dir=patches_dir,
        copy_patch_files=True,
    )

    base_file = Path(index_manifest["files"]["0"]["path"])
    expected_size = total_points * param_dim * np.dtype(np.float32).itemsize
    actual_size = base_file.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Pure SSD incremental base size mismatch: got {actual_size}, expected {expected_size}"
        )

    block_bounds = getattr(storage_adapter, "block_bounds", None)
    if block_bounds is None:
        block_bounds = np.zeros((num_blocks, 6), dtype=np.float32)
    block_bounds = np.asarray(block_bounds, dtype=np.float32)
    np.save(block_bounds_file, block_bounds)

    copied_patch_files = int(index_manifest.get("copied_patch_files", 0))
    copied_patch_bytes = int(index_manifest.get("copied_patch_bytes", 0))
    base_manifest = _manifest_base(storage_adapter, gaussians)
    scene_min = getattr(storage_adapter, "scene_min", base_manifest.get("scene_min", [0.0, 0.0, 0.0]))
    scene_max = getattr(storage_adapter, "scene_max", base_manifest.get("scene_max", [0.0, 0.0, 0.0]))
    manifest = {
        **base_manifest,
        "checkpoint_version": 1,
        "checkpoint_type": "pure_ssd_incremental",
        "iteration": int(iteration),
        "checkpoint_iter": int(iteration),
        "next_iteration": int(next_iteration),
        "active_sh_degree": int(getattr(gaussians, "active_sh_degree", 0)),
        "optimizer_state_mode": "cold_start",
        "total_points": total_points,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "param_dim": param_dim,
        "base_file": str(base_file.resolve()),
        "block_bounds": str(block_bounds_file.resolve()),
        "delta_dir": str(delta_dir.resolve()),
        "storage_index": str(storage_index_file.resolve()),
        "patches_dir": str(patches_dir.resolve()),
        "patch_files_copied": copied_patch_files,
        "patch_bytes_copied": copied_patch_bytes,
        "training_state": str(training_state_file.resolve()),
        "scene_min": np.asarray(scene_min, dtype=np.float32).tolist(),
        "scene_max": np.asarray(scene_max, dtype=np.float32).tolist(),
        "args": {
            "ssd_execution_mode": getattr(args, "ssd_execution_mode", None),
            "paper_optimizer_backend": getattr(args, "paper_optimizer_backend", None),
            "paper_optimizer_state_mode": getattr(args, "paper_optimizer_state_mode", None),
            "paper_block_reader_backend": getattr(args, "paper_block_reader_backend", None),
            "paper_resident_capacity_blocks": getattr(args, "paper_resident_capacity_blocks", None),
            "pure_ssd_checkpoint_mode": getattr(args, "pure_ssd_checkpoint_mode", None),
        },
    }

    torch.save(
        {
            "next_iteration": int(next_iteration),
            "active_sh_degree": int(getattr(gaussians, "active_sh_degree", 0)),
            "optimizer_state_mode": "cold_start",
            "adam_moments_saved": False,
            "storage_index": str(storage_index_file.resolve()),
        },
        training_state_file,
    )

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, indent=2, sort_keys=True)

    _log(
        f"[PURE SSD CHECKPOINT] Incremental complete: patches={copied_patch_files} "
        f"size={copied_patch_bytes / (1024 ** 3):.2f}GB manifest={manifest_file}",
        log_file,
    )
    return manifest
