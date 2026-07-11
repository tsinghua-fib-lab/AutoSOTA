from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


def stream_blocks_from_npy_files(
    npy_files: Sequence[str | Path],
    dt: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield summed spike blocks from one or more [T,H,W] .npy files."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if len(npy_files) == 0:
        raise ValueError("npy_files must not be empty")

    block_sum = None
    block_id = 0
    count = 0
    height = None
    width = None

    for npy_file in npy_files:
        arr = np.load(npy_file, mmap_mode="r")
        if arr.ndim != 3:
            raise ValueError(f"{npy_file} must have shape [T,H,W], got {arr.shape}")

        frames, h, w = arr.shape
        if block_sum is None:
            height, width = h, w
            block_sum = np.zeros((height, width), dtype=np.float32)
        elif h != height or w != width:
            raise ValueError(
                f"inconsistent frame size in {npy_file}: expected {(height, width)}, got {(h, w)}"
            )

        for frame_idx in range(frames):
            block_sum += (np.asarray(arr[frame_idx]) > 0).astype(np.float32)
            count += 1
            if count == dt:
                yield block_id, block_sum.copy()
                block_id += 1
                block_sum.fill(0.0)
                count = 0


def stream_blocks_from_dat(
    path: str | Path,
    dt: int,
    width: int = 400,
    height: int = 250,
    padding_bits: int = 16,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield summed spike blocks from the packed .dat format used by the raw SAD files."""
    if dt <= 0:
        raise ValueError("dt must be positive")

    bytes_per_row = (width + padding_bits) // 8
    bytes_per_frame = bytes_per_row * height
    data = Path(path).read_bytes()
    total_frames = len(data) // bytes_per_frame
    total_blocks = total_frames // dt

    offset = 0
    block_sum = np.zeros((height, width), dtype=np.float32)
    count = 0
    block_id = 0

    for _ in range(total_frames):
        frame_bytes = np.frombuffer(
            data[offset : offset + bytes_per_frame],
            dtype=np.uint8,
        ).reshape((bytes_per_row, height), order="F")
        offset += bytes_per_frame

        useful = frame_bytes[: width // 8, :]
        flat_bytes = useful.reshape(-1, order="F")
        bits = np.unpackbits(flat_bytes).reshape(-1, 8)[:, ::-1].reshape(-1)
        frame = bits.reshape((width, height), order="F").T.astype(np.float32)

        block_sum += frame
        count += 1
        if count == dt:
            yield block_id, block_sum.copy()
            block_id += 1
            if block_id >= total_blocks:
                break
            block_sum.fill(0.0)
            count = 0


def materialize_blocks(blocks: Iterable[tuple[int, np.ndarray]]) -> list[np.ndarray]:
    return [block for _, block in blocks]
