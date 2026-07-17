"""Shard a single-GPU checkpoint into N tensor-parallel shards.

This is needed to test multi-GPU quantization on models that have only one
checkpoint shard (e.g., 3.2-1B). The output directory will contain N .pth files
compatible with `parallel/start.py` which expects len(checkpoints) == WORLD_SIZE.

Usage:
    python -m scripts.shard_checkpoint \
        --ckpt_dir $QUANT_BUCKET/Llama-3.2-1B \
        --output_dir $QUANT_BUCKET/Llama-3.2-1B-4shard \
        --nshards 4

ColumnParallel layers (wq, wk, wv, w1, w3, output): split rows (dim 0)
RowParallel layers (wo, w2): split columns (dim 1)
ParallelEmbedding (tok_embeddings): split embedding dim (dim 1)
Other (norm weights, rope freqs): replicated to all shards
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch


# ColumnParallel: split dim 0 (output rows)
# RowParallel: split dim 1 (input columns)
# ParallelEmbedding: split dim 1 (embedding dimension)
COLUMN_PARALLEL_KEYS = {
    "attention.wq.weight",
    "attention.wk.weight",
    "attention.wv.weight",
    "feed_forward.w1.weight",
    "feed_forward.w3.weight",
}
ROW_PARALLEL_KEYS = {
    "attention.wo.weight",
    "feed_forward.w2.weight",
}
# output layer is ColumnParallel: split dim 0
TOP_COLUMN_PARALLEL = {
    "output.weight",
}
# tok_embeddings uses ParallelEmbedding: splits along embedding dim (dim 1)
PARALLEL_EMBEDDING_KEYS = {
    "tok_embeddings.weight",
}


def _is_column_parallel(key: str) -> bool:
    for suffix in COLUMN_PARALLEL_KEYS:
        if key.endswith(suffix):
            return True
    return key in TOP_COLUMN_PARALLEL


def _is_row_parallel(key: str) -> bool:
    for suffix in ROW_PARALLEL_KEYS:
        if key.endswith(suffix):
            return True
    return False


def _is_parallel_embedding(key: str) -> bool:
    return key in PARALLEL_EMBEDDING_KEYS


def shard_checkpoint(ckpt_dir: str | Path, output_dir: str | Path, nshards: int):
    ckpt_dir = Path(ckpt_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the single checkpoint
    pth_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pth"))
    if len(pth_files) != 1:
        raise RuntimeError(
            f"Expected exactly 1 .pth file in {ckpt_dir}, found {len(pth_files)}: {pth_files}. "
            f"This script is for sharding a single-GPU checkpoint."
        )

    print(f"Loading {ckpt_dir / pth_files[0]}...")
    state_dict = torch.load(ckpt_dir / pth_files[0], map_location="cpu", weights_only=True)

    # Build per-shard state dicts
    shards = [{} for _ in range(nshards)]
    for key, tensor in state_dict.items():
        if _is_column_parallel(key):
            # Split rows (dim 0)
            chunks = tensor.chunk(nshards, dim=0)
            for i, chunk in enumerate(chunks):
                shards[i][key] = chunk.clone()
            print(f"  [col-split] {key}: {list(tensor.shape)} -> {nshards}x{list(chunks[0].shape)}")
        elif _is_row_parallel(key):
            # Split columns (dim 1)
            chunks = tensor.chunk(nshards, dim=1)
            for i, chunk in enumerate(chunks):
                shards[i][key] = chunk.clone()
            print(f"  [row-split] {key}: {list(tensor.shape)} -> {nshards}x{list(chunks[0].shape)}")
        elif _is_parallel_embedding(key):
            # ParallelEmbedding: split along embedding dim (dim 1)
            chunks = tensor.chunk(nshards, dim=1)
            for i, chunk in enumerate(chunks):
                shards[i][key] = chunk.clone()
            print(f"  [emb-split] {key}: {list(tensor.shape)} -> {nshards}x{list(chunks[0].shape)}")
        else:
            # Replicate (norms, rope, etc.)
            for i in range(nshards):
                shards[i][key] = tensor.clone()
            print(f"  [replicate] {key}: {list(tensor.shape)}")

    # Save shards
    for i in range(nshards):
        out_path = output_dir / f"consolidated.{i:02d}.pth"
        print(f"Saving {out_path} ({len(shards[i])} keys)...")
        torch.save(shards[i], out_path)

    # Copy params.json and tokenizer.model
    for fname in ["params.json", "tokenizer.model"]:
        src = ckpt_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"Copied {fname}")

    print(f"\nDone. Sharded {pth_files[0]} into {nshards} shards in {output_dir}")
    print(f"Usage: torchrun --nproc-per-node={nshards} -m scripts.run_pipeline_job --model ... ")


def main():
    p = argparse.ArgumentParser(description="Shard a single-GPU checkpoint for multi-GPU testing")
    p.add_argument("--ckpt_dir", required=True, help="Directory with single consolidated.00.pth")
    p.add_argument("--output_dir", required=True, help="Output directory for sharded checkpoints")
    p.add_argument("--nshards", type=int, required=True, help="Number of shards (= number of GPUs)")
    args = p.parse_args()

    shard_checkpoint(args.ckpt_dir, args.output_dir, args.nshards)


if __name__ == "__main__":
    main()
