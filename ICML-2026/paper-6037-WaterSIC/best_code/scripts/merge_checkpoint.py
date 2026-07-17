"""Merge N tensor-parallel checkpoint shards into a single consolidated checkpoint.

Reverses the operation of shard_checkpoint.py. Useful for re-sharding
(e.g., 8 shards -> merge -> re-shard to 4).

Usage:
    python -m scripts.merge_checkpoint \
        --ckpt_dir $QUANT_BUCKET/Llama-3-70B-fixed \
        --output_dir $QUANT_BUCKET/Llama-3-70B-merged

ColumnParallel layers (wq, wk, wv, w1, w3, output): cat rows (dim 0)
RowParallel layers (wo, w2): cat columns (dim 1)
ParallelEmbedding (tok_embeddings): cat embedding dim (dim 1)
Other (norm weights, rope freqs): take from shard 0 (replicated)
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch

from scripts.shard_checkpoint import (
    _is_column_parallel,
    _is_parallel_embedding,
    _is_row_parallel,
)


def merge_checkpoint(ckpt_dir: str | Path, output_dir: str | Path):
    ckpt_dir = Path(ckpt_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pth_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pth"))
    if len(pth_files) < 2:
        raise RuntimeError(
            f"Expected >= 2 .pth files in {ckpt_dir}, found {len(pth_files)}. "
            f"Nothing to merge."
        )

    nshards = len(pth_files)
    print(f"Merging {nshards} shards from {ckpt_dir}")

    # Load all shards
    shards = []
    for f in pth_files:
        print(f"Loading {f}...")
        shards.append(torch.load(ckpt_dir / f, map_location="cpu", weights_only=True))

    # Merge
    merged = {}
    for key in shards[0].keys():
        if _is_column_parallel(key):
            merged[key] = torch.cat([s[key] for s in shards], dim=0)
            print(f"  [col-merge] {key}: {nshards}x{list(shards[0][key].shape)} -> {list(merged[key].shape)}")
        elif _is_row_parallel(key):
            merged[key] = torch.cat([s[key] for s in shards], dim=1)
            print(f"  [row-merge] {key}: {nshards}x{list(shards[0][key].shape)} -> {list(merged[key].shape)}")
        elif _is_parallel_embedding(key):
            merged[key] = torch.cat([s[key] for s in shards], dim=1)
            print(f"  [emb-merge] {key}: {nshards}x{list(shards[0][key].shape)} -> {list(merged[key].shape)}")
        else:
            merged[key] = shards[0][key]
            print(f"  [replicate] {key}: {list(merged[key].shape)}")

    # Save
    out_path = output_dir / "consolidated.00.pth"
    print(f"Saving {out_path} ({len(merged)} keys)...")
    torch.save(merged, out_path)

    # Copy params.json and tokenizer.model
    for fname in ["params.json", "tokenizer.model"]:
        src = ckpt_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"Copied {fname}")

    print(f"\nDone. Merged {nshards} shards into {out_path}")


def main():
    p = argparse.ArgumentParser(description="Merge N checkpoint shards into one")
    p.add_argument("--ckpt_dir", required=True, help="Directory with consolidated.{00..N}.pth")
    p.add_argument("--output_dir", required=True, help="Output directory for merged checkpoint")
    args = p.parse_args()

    merge_checkpoint(args.ckpt_dir, args.output_dir)


if __name__ == "__main__":
    main()
