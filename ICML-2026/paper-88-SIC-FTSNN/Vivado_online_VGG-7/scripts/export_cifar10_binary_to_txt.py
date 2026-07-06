#!/usr/bin/env python3
"""
Export CIFAR-10 binary batches into the plain-text format expected by the VHDL testbench.

Each output line is:
  <label> <p0> <p1> ... <p3071>

The pixel order matches the official CIFAR-10 binary format:
  1024 red bytes, then 1024 green bytes, then 1024 blue bytes.
"""

from __future__ import annotations

import argparse
import os
import tarfile
import tempfile
from pathlib import Path


RECORD_BYTES = 1 + 3072


def maybe_extract_tar(tar_path: Path, out_root: Path) -> Path:
    if out_root.exists() and (out_root / "data_batch_1.bin").exists():
        return out_root

    out_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=out_root.parent)
    extracted = out_root.parent / "cifar-10-batches-bin"
    if not extracted.exists():
        raise FileNotFoundError("Expected extracted directory 'cifar-10-batches-bin' not found.")
    return extracted


def iter_records(bin_file: Path):
    data = bin_file.read_bytes()
    if len(data) % RECORD_BYTES != 0:
        raise ValueError(f"{bin_file} has invalid length {len(data)}.")
    for off in range(0, len(data), RECORD_BYTES):
        rec = data[off : off + RECORD_BYTES]
        label = rec[0]
        pixels = rec[1:]
        yield label, pixels


def write_txt(out_path: Path, source_files: list[Path], limit: int | None) -> int:
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for src in source_files:
            for label, pixels in iter_records(src):
                row = [str(label)] + [str(v) for v in pixels]
                f.write(" ".join(row))
                f.write("\n")
                count += 1
                if limit is not None and count >= limit:
                    return count
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary-root", type=Path, default=None,
                    help="Path to extracted cifar-10-batches-bin directory.")
    ap.add_argument("--tar", type=Path, default=None,
                    help="Optional path to cifar-10-binary.tar.gz. If given, it is extracted.")
    ap.add_argument("--outdir", type=Path, default=Path("../data"),
                    help="Output directory for cifar10_train.txt and cifar10_test.txt.")
    ap.add_argument("--train-samples", type=int, default=1024,
                    help="Number of training samples to export. Use 50000 for the full set.")
    ap.add_argument("--test-samples", type=int, default=256,
                    help="Number of test samples to export. Use 10000 for the full set.")
    args = ap.parse_args()

    if args.binary_root is None and args.tar is None:
        raise SystemExit("Provide either --binary-root or --tar.")

    if args.binary_root is not None:
        binary_root = args.binary_root
    else:
        parent = args.outdir.resolve().parent
        binary_root = maybe_extract_tar(args.tar.resolve(), parent / "cifar-10-batches-bin")

    if not binary_root.exists():
        raise FileNotFoundError(f"Binary root not found: {binary_root}")

    train_files = [binary_root / f"data_batch_{i}.bin" for i in range(1, 6)]
    test_files = [binary_root / "test_batch.bin"]
    for p in train_files + test_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing CIFAR-10 binary file: {p}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    train_out = args.outdir / "cifar10_train.txt"
    test_out = args.outdir / "cifar10_test.txt"

    n_train = write_txt(train_out, train_files, args.train_samples)
    n_test = write_txt(test_out, test_files, args.test_samples)

    print(f"Wrote {n_train} samples to {train_out}")
    print(f"Wrote {n_test} samples to {test_out}")


if __name__ == "__main__":
    main()
