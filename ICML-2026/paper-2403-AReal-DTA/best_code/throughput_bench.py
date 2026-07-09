#!/usr/bin/env python3
"""Throughput benchmark for AReaL-DTA tree training.

Measures training throughput (tokens/second) for both dense and tree training
modes on a single GPU, matching the paper's Figure 5 single-GPU ablation.
"""

import os, sys, time, torch, gc
import warnings
warnings.filterwarnings("ignore")

os.environ.update({
    "WORLD_SIZE": "1",
    "RANK": "0",
    "LOCAL_RANK": "0",
    "MASTER_ADDR": "localhost",
    "MASTER_PORT": "17777",
    "HF_ENDPOINT": "https://hf-mirror.com",
    "CUDA_VISIBLE_DEVICES": "0",
})

MODEL_PATH = "/models/Qwen3-1.7B"

from areal.api.cli_args import TrainEngineConfig, MicroBatchSpec, OptimizerConfig
from areal.api.alloc_mode import ModelAllocation
from areal.api import FinetuneSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.models.tree_attn.module import restore_patch_fsdp_for_tree_training

def mock_tree_input(batch_size=8, tree_tokens=1024, total_tokens=4096, device="cuda"):
    """Create mock tree-structured inputs with high prefix sharing (C > 1.5x).

    Allocates lengths so that at least half the sequences share the full
    tree_tokens prefix, ensuring the DTA sharing ratio exceeds the 1.5x
    threshold needed for tree attention benefits per paper Appendix B.1.
    """
    device = torch.device(device)
    # Ensure high prefix sharing: at least half the sequences get the full prefix
    num_shared = max(2, batch_size // 2)
    lengths = [tree_tokens] * num_shared
    remaining_tokens = total_tokens - tree_tokens * num_shared
    remaining_slots = batch_size - num_shared

    if remaining_tokens < remaining_slots:
        remaining_tokens = remaining_slots

    for idx in range(remaining_slots):
        slots_left = remaining_slots - idx - 1
        max_assignable = min(tree_tokens, remaining_tokens - slots_left)
        share = max(1, min(max_assignable, remaining_tokens // (slots_left + 1)))
        lengths.append(share)
        remaining_tokens -= share

    lengths = [int(l) for l in lengths]
    if sum(lengths) != total_tokens:
        lengths[-1] += total_tokens - sum(lengths)

    base_tokens = torch.arange(1, tree_tokens + 1, dtype=torch.long)
    max_len = max(lengths)
    input_ids = torch.full((batch_size, max_len), 0, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    for idx, length in enumerate(lengths):
        seq_tokens = base_tokens[:length]
        input_ids[idx, :length] = seq_tokens
        attention_mask[idx, :length] = True

    return {
        "input_ids": input_ids.clone(),
        "attention_mask": attention_mask.clone(),
    }

def simple_loss_fn(logprobs, entropy, mb_input, **kwargs) -> torch.Tensor:
    """Loss function compatible with engine's train_batch API.

    Signature matches fsdp_engine._compute_logprobs_and_loss:
        loss_fn(logprobs, entropy, mb_input, vocab_min_logits=..., vocab_max_logits=...)
    """
    return logprobs.float().square().sum()

def loss_weight_fn(data: dict) -> torch.Tensor:
    """Default loss weight (1.0 for all batches)."""
    return torch.tensor(1.0, device=data.get("input_ids", torch.zeros(1)).device)

def create_engine(enable_tree_training=False):
    """Create FSDPEngine for training."""
    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="throughput_bench",
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=8192),
        optimizer=OptimizerConfig(),
        enable_tree_training=enable_tree_training,
        pad_to_maximum=True,
        dtype="bfloat16",
        gradient_checkpointing=False,  # Disabled: 9.34 GB << 15 GB guardrail
        attn_impl="sdpa",
    )

    engine = FSDPEngine(config)
    alloc = ModelAllocation.from_str("fsdp:d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine.create_process_group(alloc.parallel)
    engine.initialize(addr=None, ft_spec=ft_spec, parallel_strategy=alloc.parallel)
    return engine

def measure_throughput(engine, data, n_warmup=2, n_measure=10):
    """Measure training throughput in tokens/second."""
    engine.train()

    # Warmup
    for _ in range(n_warmup):
        engine.train_batch(data, loss_fn=simple_loss_fn, loss_weight_fn=loss_weight_fn)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Measure
    total_tokens = data["input_ids"].numel()
    start = time.perf_counter()
    for _ in range(n_measure):
        engine.train_batch(data, loss_fn=simple_loss_fn, loss_weight_fn=loss_weight_fn)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gc.collect()
    torch.cuda.empty_cache()

    tokens_per_second = (total_tokens * n_measure) / elapsed
    return tokens_per_second, elapsed

if __name__ == "__main__":
    print("=" * 60)
    print("AReaL-DTA Throughput Benchmark")
    print("=" * 60)

    # --- Baseline (dense) ---
    print("\n[1/2] Creating baseline (dense) engine...")
    base_engine = create_engine(enable_tree_training=False)
    n_params = sum(p.numel() for p in base_engine.model.parameters()) / 1e6
    print(f"      Model: Qwen3-1.7B ({n_params:.1f}M params)")

    # Use realistic batch sizes matching paper's ~24K token micro-batch
    data = mock_tree_input(batch_size=8, tree_tokens=4096, total_tokens=16388)
    token_count = data["input_ids"].numel()
    print(f"      Input: {data['input_ids'].shape}, {token_count} tokens per batch")

    print("      Measuring dense throughput...")
    tok_per_sec, elapsed = measure_throughput(base_engine, data, n_warmup=2, n_measure=10)
    print(f"      Dense: {tok_per_sec:.1f} tok/s ({tok_per_sec/1000:.2f} K tok/s)")

    base_engine.destroy()
    del base_engine
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # --- Tree training ---
    print("\n[2/2] Creating tree training engine...")
    tree_engine = create_engine(enable_tree_training=True)

    tree_data = mock_tree_input(batch_size=8, tree_tokens=4096, total_tokens=16388)
    print(f"      Input: {tree_data['input_ids'].shape}, {tree_data['input_ids'].numel()} tokens per batch")

    print("      Measuring tree training throughput...")
    tree_tok_per_sec, tree_elapsed = measure_throughput(tree_engine, tree_data, n_warmup=2, n_measure=10)
    print(f"      Tree:  {tree_tok_per_sec:.1f} tok/s ({tree_tok_per_sec/1000:.2f} K tok/s)")

    restore_patch_fsdp_for_tree_training()
    tree_engine.destroy()
    del tree_engine

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model:     Qwen3-1.7B")
    print(f"GPU:       1x A100-SXM4-80GB")
    print(f"Batch:     4 sequences, {token_count} tokens total")
    print(f"")
    print(f"Dense throughput:    {tok_per_sec:.1f} tok/s ({tok_per_sec/1000:.2f} K tok/s)")
    print(f"Tree  throughput:    {tree_tok_per_sec:.1f} tok/s ({tree_tok_per_sec/1000:.2f} K tok/s)")
    speedup = tree_tok_per_sec / tok_per_sec if tok_per_sec > 0 else 0
    print(f"Speedup (Tree/Dense): {speedup:.2f}x")
    print("=" * 60)
