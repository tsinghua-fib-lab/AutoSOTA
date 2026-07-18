#!/usr/bin/env python3
"""Stream a pure SSD checkpoint snapshot to a 3DGS-compatible PLY file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from storage.log_storage_manager import LogStorageManager  # noqa: E402
from storage.pure_ssd_checkpoint import load_pure_ssd_checkpoint_manifest  # noqa: E402


SSD_PARAM_DIM = 59
PLY_ATTR_DIM = 62


def ply_attribute_names() -> List[str]:
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names.extend(f"f_dc_{i}" for i in range(3))
    names.extend(f"f_rest_{i}" for i in range(45))
    names.append("opacity")
    names.extend(f"scale_{i}" for i in range(3))
    names.extend(f"rot_{i}" for i in range(4))
    return names


def write_binary_little_endian_ply_header(handle, vertex_count: int, attributes: Iterable[str]) -> None:
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {int(vertex_count)}",
    ]
    lines.extend(f"property float {name}" for name in attributes)
    lines.append("end_header")
    handle.write(("\n".join(lines) + "\n").encode("ascii"))


def ssd_rows_to_ply_rows(ssd_rows: np.ndarray) -> np.ndarray:
    """Convert SSD layout to standard 3DGS PLY attribute layout.

    SSD layout:
        xyz(3) | scale(3) | rot(4) | opacity(1) | f_dc(3) | f_rest(45)
    PLY layout:
        xyz(3) | normals(3) | f_dc(3) | f_rest(45) | opacity(1) | scale(3) | rot(4)
    """
    if ssd_rows.ndim != 2 or ssd_rows.shape[1] != SSD_PARAM_DIM:
        raise ValueError(f"Expected SSD rows with shape (N, {SSD_PARAM_DIM}), got {ssd_rows.shape}")

    ply_rows = np.zeros((ssd_rows.shape[0], PLY_ATTR_DIM), dtype=np.float32)
    ply_rows[:, 0:3] = ssd_rows[:, 0:3]
    ply_rows[:, 6:9] = ssd_rows[:, 11:14]
    ply_rows[:, 9:54] = ssd_rows[:, 14:59]
    ply_rows[:, 54:55] = ssd_rows[:, 10:11]
    ply_rows[:, 55:58] = ssd_rows[:, 3:6]
    ply_rows[:, 58:62] = ssd_rows[:, 6:10]
    return ply_rows


def resolve_checkpoint_dir(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        if checkpoint_path.name != "pure_ssd_checkpoint.json":
            raise ValueError(f"Expected pure_ssd_checkpoint.json, got {checkpoint_path}")
        return checkpoint_path.parent
    return checkpoint_path


def export_checkpoint_to_ply(
    checkpoint: str | Path,
    output: str | Path,
    chunk_points: int = 1_000_000,
    max_points: int = -1,
) -> Dict[str, int | str]:
    checkpoint_dir = resolve_checkpoint_dir(checkpoint)
    manifest = load_pure_ssd_checkpoint_manifest(checkpoint_dir)

    total_points = int(manifest["total_points"])
    param_dim = int(manifest.get("param_dim", SSD_PARAM_DIM))
    if param_dim != SSD_PARAM_DIM:
        raise ValueError(f"Expected param_dim={SSD_PARAM_DIM}, got {param_dim}")

    export_points = total_points if int(max_points) < 0 else min(total_points, int(max_points))
    chunk_points = int(chunk_points)
    if chunk_points <= 0:
        raise ValueError("chunk_points must be positive")

    base_file = Path(manifest["base_file"])
    expected_size = total_points * SSD_PARAM_DIM * np.dtype(np.float32).itemsize
    actual_size = base_file.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Checkpoint base_file size mismatch: got {actual_size}, expected {expected_size}"
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    attrs = ply_attribute_names()
    if len(attrs) != PLY_ATTR_DIM:
        raise AssertionError(f"PLY attribute count mismatch: {len(attrs)} != {PLY_ATTR_DIM}")

    print(
        "[PURE SSD EXPORT] Exporting checkpoint snapshot to PLY: "
        f"points={export_points:,}/{total_points:,} chunk_points={chunk_points:,}"
    )
    storage_index = manifest.get("storage_index", "")
    print(f"[PURE SSD EXPORT] source={base_file}")
    if storage_index:
        print(f"[PURE SSD EXPORT] storage_index={storage_index}")
    print(f"[PURE SSD EXPORT] output={output}")

    bytes_per_row = SSD_PARAM_DIM * np.dtype(np.float32).itemsize
    written = 0
    if storage_index:
        storage = LogStorageManager(
            storage_dir=str(base_file.parent),
            block_size=int(manifest["block_size"]),
            num_blocks=int(manifest["num_blocks"]),
            point_dim=SSD_PARAM_DIM,
        )
        storage.load_index_manifest(storage_index)
        with open(output, "wb") as dst:
            write_binary_little_endian_ply_header(dst, export_points, attrs)
            block_id = 0
            while written < export_points:
                blocks = storage.read_blocks([block_id])
                ssd_rows = blocks[block_id].detach().cpu().numpy().astype(np.float32, copy=False)
                rows_this_chunk = min(int(ssd_rows.shape[0]), export_points - written)
                ply_rows = ssd_rows_to_ply_rows(ssd_rows[:rows_this_chunk])
                dst.write(np.ascontiguousarray(ply_rows, dtype="<f4").tobytes())
                written += rows_this_chunk
                block_id += 1
                if written == export_points or written % max(chunk_points * 10, 1) == 0:
                    print(f"[PURE SSD EXPORT] wrote {written:,}/{export_points:,} vertices")
        storage.close()
    else:
        with open(base_file, "rb") as src, open(output, "wb") as dst:
            write_binary_little_endian_ply_header(dst, export_points, attrs)
            while written < export_points:
                rows_this_chunk = min(chunk_points, export_points - written)
                raw = src.read(rows_this_chunk * bytes_per_row)
                if len(raw) != rows_this_chunk * bytes_per_row:
                    raise RuntimeError(
                        f"Unexpected EOF while reading base_file at row {written}: "
                        f"got {len(raw)} bytes, expected {rows_this_chunk * bytes_per_row}"
                    )
                ssd_rows = np.frombuffer(raw, dtype="<f4").reshape(rows_this_chunk, SSD_PARAM_DIM)
                ply_rows = ssd_rows_to_ply_rows(ssd_rows)
                dst.write(np.ascontiguousarray(ply_rows, dtype="<f4").tobytes())
                written += rows_this_chunk
                if written == export_points or written % max(chunk_points * 10, 1) == 0:
                    print(f"[PURE SSD EXPORT] wrote {written:,}/{export_points:,} vertices")

    print(
        f"[PURE SSD EXPORT] done: output_size={output.stat().st_size / (1024 ** 3):.2f}GB"
    )
    return {
        "output": str(output),
        "written_points": int(written),
        "total_points": int(total_points),
        "output_size": int(output.stat().st_size),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream a pure SSD checkpoint snapshot base_file.bin to 3DGS PLY."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint directory or pure_ssd_checkpoint.json path.",
    )
    parser.add_argument("--output", required=True, help="Output .ply path.")
    parser.add_argument(
        "--chunk_points",
        type=int,
        default=1_000_000,
        help="Number of points to convert per chunk.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=-1,
        help="Optional debug limit; -1 exports all points.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    export_checkpoint_to_ply(
        checkpoint=args.checkpoint,
        output=args.output,
        chunk_points=args.chunk_points,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()
