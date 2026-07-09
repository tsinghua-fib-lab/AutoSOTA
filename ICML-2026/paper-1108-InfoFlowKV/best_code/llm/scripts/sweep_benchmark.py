#!/usr/bin/env python
"""
Sweep benchmark for comparing ring attention vs guided recompute across
different sequence lengths and configurations.

Outputs JSON results for plotting.

Usage:
    torchrun --nproc_per_node=4 scripts/sweep_benchmark.py \
        --model ../models/Qwen3-14B/ \
        --output results/sweep_results.json
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep benchmark")
    parser.add_argument("--model", type=str, default="/path/to/Qwen3-14B")
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32768, 65536, 131072],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--recompute_ratios",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20],
        help="Recompute ratios to test",
    )
    parser.add_argument("--num_warmup", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--output", type=str, default="results/sweep_results.json")
    parser.add_argument("--heads_k_stride", type=int, default=1,
                        help="KV heads per all-gather in ring attention (1=per-head, num_kv_heads=all-at-once)")
    return parser.parse_args()


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def generate_context(tokenizer, target_length):
    """Generate synthetic context."""
    passage = """
    This is a passage about artificial intelligence and machine learning.
    Deep learning models have revolutionized many fields including natural
    language processing, computer vision, and speech recognition. Transformer
    architectures have become the foundation of modern large language models.
    """
    passage_tokens = len(tokenizer.encode(passage, add_special_tokens=False))
    num_passages = (target_length // passage_tokens) + 1
    passages = [f"\nPassage {i+1}: {passage.strip()}" for i in range(num_passages)]
    context = "".join(passages)
    tokens = tokenizer.encode(context, add_special_tokens=False)[:target_length]
    return tokenizer.decode(tokens)


def benchmark_single_gpu_prefill(model, tokenizer, context, query, device, num_iterations):
    """Single GPU full prefill - baseline for comparison."""
    context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    query_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
    full_ids = torch.cat([context_ids, query_ids], dim=1)

    timings = []
    for _ in range(num_iterations):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = model(full_ids, use_cache=True)

        torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - t0) * 1000)

    return {"mean_ms": sum(timings) / len(timings), "std_ms": torch.tensor(timings).std().item()}


def benchmark_ring_attention(model, tokenizer, context, query, device, config, num_iterations):
    """Ring attention baseline using ring-flash-attention library.

    Matches eval_longbench.py ring_attention pipeline exactly:
    substitute_hf_flash_attn (done by caller) + use_ring_attn toggle.
    """
    from ring_flash_attn import update_ring_flash_attn_params
    from ring_flash_attn.adapters.hf_adapter import use_ring_attn

    context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    query_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)
    full_ids = torch.cat([context_ids, query_ids], dim=1)

    total_len = full_ids.shape[1]
    world_size = config.world_size
    rank = config.rank

    # Pad to be divisible by world_size (required by ring-flash-attention)
    if total_len % world_size != 0:
        pad_len = world_size - (total_len % world_size)
        full_ids = torch.cat([
            full_ids,
            torch.full((1, pad_len), tokenizer.pad_token_id, device=device, dtype=full_ids.dtype)
        ], dim=1)
        padded_len = total_len + pad_len
    else:
        padded_len = total_len

    # Setup ring attention params
    position_ids = torch.arange(padded_len, device=device).unsqueeze(0)
    cu_seqlens = torch.tensor([0, padded_len], device=device, dtype=torch.int32)
    update_ring_flash_attn_params(cu_seqlens, config.process_group)

    # Chunk input across GPUs
    chunk_size = padded_len // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size
    local_ids = full_ids[:, start_idx:end_idx]
    local_position_ids = position_ids[:, start_idx:end_idx]

    timings = []
    for _ in range(num_iterations):
        if dist.is_initialized():
            dist.barrier()

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        use_ring_attn(True)
        with torch.no_grad():
            outputs = model(
                input_ids=local_ids,
                position_ids=local_position_ids,
                use_cache=False,
            )
        use_ring_attn(False)

        torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - t0) * 1000)

    return {"mean_ms": sum(timings) / len(timings), "std_ms": torch.tensor(timings).std().item()}


def benchmark_guided_recompute(model, tokenizer, context, query, device, config, recompute_ratio, num_iterations):
    """Guided recompute with sparse attention (matches eval_longbench.py pipeline)."""
    from models.parallel import (
        DistributedExtractor,
        DistributedScorer,
        RingAttentionRecomputer,
    )
    from models.parallel.recomputer import all_gather_kv
    from models.qwen.kv_cache import KVCacheExtractor

    rank = config.rank

    base_extractor = KVCacheExtractor(model, tokenizer, model_type="qwen")
    extractor = DistributedExtractor(base_extractor, config)
    scorer = DistributedScorer(model, config, method="norm", layer_indices=[22, 23, 24, 25])
    recomputer = RingAttentionRecomputer(model, config, use_ring_attention=True)

    context_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    query_ids = tokenizer(query, return_tensors="pt").input_ids.to(device)

    timings = []
    for _ in range(num_iterations):
        if dist.is_initialized():
            dist.barrier()

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        local_kv = extractor.extract_distributed(context_ids)

        # Exclude rank 0's first chunk from recomputation (matching eval_longbench.py)
        exclude_first = local_kv.local_seq_len if rank == 0 else 0
        exclude_tensor = torch.tensor([exclude_first], device=device, dtype=torch.long)
        dist.broadcast(exclude_tensor, src=0, group=config.process_group)
        exclude_first = int(exclude_tensor.item())

        local_important, global_important = scorer.score_distributed(
            local_kv, query_ids, top_ratio=recompute_ratio,
            exclude_first_tokens=exclude_first,
        )
        updated_kv = recomputer.recompute_distributed(
            local_kv, local_important, global_important
        )
        final_kv = all_gather_kv(updated_kv, config)

        torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - t0) * 1000)

    return {"mean_ms": sum(timings) / len(timings), "std_ms": torch.tensor(timings).std().item()}


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}"
    is_main = rank == 0

    if is_main:
        print("=" * 70)
        print("Sweep Benchmark: Ring Attention vs Guided Recompute")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"World size: {world_size}")
        print(f"Sequence lengths: {args.seq_lengths}")
        print(f"Recompute ratios: {args.recompute_ratios}")
        print("=" * 70)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if is_main:
        print("\nLoading model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from models.parallel.config import DistributedConfig

    # Setup ring-flash-attention library (must be done before any ring_attention benchmarks)
    ring_group = None
    if world_size > 1:
        ring_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")

        from ring_flash_attn import substitute_hf_flash_attn
        from ring_flash_attn.adapters.hf_adapter import use_ring_attn

        substitute_hf_flash_attn(ring_group, heads_k_stride=args.heads_k_stride)
        use_ring_attn(False)  # Default OFF; toggled on only during ring_attention benchmark

    query = "What is the main topic discussed in the passages above?"

    all_results = {
        "metadata": {
            "model": args.model,
            "world_size": world_size,
            "num_iterations": args.num_iterations,
            "heads_k_stride": args.heads_k_stride,
            "timestamp": datetime.now().isoformat(),
        },
        "results": []
    }

    for seq_len in args.seq_lengths:
        if is_main:
            print(f"\n{'='*70}")
            print(f"Sequence length: {seq_len}")
            print(f"{'='*70}")

        context = generate_context(tokenizer, seq_len)
        actual_len = len(tokenizer.encode(context, add_special_tokens=False))

        if is_main:
            print(f"Actual tokens: {actual_len}")

        # Single GPU baseline (only on rank 0)
        if rank == 0:
            if is_main:
                print(f"\n  Single GPU Prefill (baseline):")
                print(f"    Warming up...")

            try:
                benchmark_single_gpu_prefill(model, tokenizer, context, query, device, args.num_warmup)

                if is_main:
                    print(f"    Benchmarking...")

                result = benchmark_single_gpu_prefill(model, tokenizer, context, query, device, args.num_iterations)

                if is_main:
                    print(f"    TTFT: {result['mean_ms']:.1f}ms (+/- {result['std_ms']:.1f}ms)")

                all_results["results"].append({
                    "seq_len": seq_len,
                    "method": "single_gpu_prefill",
                    "recompute_ratio": None,
                    "ttft_ms": result["mean_ms"],
                    "std_ms": result["std_ms"],
                })
            except Exception as e:
                if is_main:
                    print(f"    Failed: {e}")

        # Synchronize before distributed benchmarks
        if dist.is_initialized():
            dist.barrier()

        # Ring attention baseline
        if world_size > 1:
            config = DistributedConfig.from_env(recompute_ratio=0.15)
            config.process_group = ring_group

            if is_main:
                print(f"\n  Ring Attention (baseline):")
                print(f"    Warming up...")

            try:
                benchmark_ring_attention(model, tokenizer, context, query, device, config, args.num_warmup)

                if is_main:
                    print(f"    Benchmarking...")

                result = benchmark_ring_attention(model, tokenizer, context, query, device, config, args.num_iterations)

                if is_main:
                    print(f"    TTFT: {result['mean_ms']:.1f}ms (+/- {result['std_ms']:.1f}ms)")

                all_results["results"].append({
                    "seq_len": seq_len,
                    "method": "ring_attention",
                    "recompute_ratio": None,
                    "ttft_ms": result["mean_ms"],
                    "std_ms": result["std_ms"],
                })
            except Exception as e:
                if is_main:
                    print(f"    Failed: {e}")

        # SP guided recompute with different ratios
        for ratio in args.recompute_ratios:
            config = DistributedConfig.from_env(recompute_ratio=ratio)
            if ring_group is not None:
                config.process_group = ring_group

            if is_main:
                print(f"\n  SP Guided Recompute (ratio={ratio}):")
                print(f"    Warming up...")

            try:
                benchmark_guided_recompute(model, tokenizer, context, query, device, config, ratio, args.num_warmup)

                if is_main:
                    print(f"    Benchmarking...")

                result = benchmark_guided_recompute(model, tokenizer, context, query, device, config, ratio, args.num_iterations)

                if is_main:
                    print(f"    TTFT: {result['mean_ms']:.1f}ms (+/- {result['std_ms']:.1f}ms)")

                all_results["results"].append({
                    "seq_len": seq_len,
                    "method": "sp_guided_recompute",
                    "recompute_ratio": ratio,
                    "ttft_ms": result["mean_ms"],
                    "std_ms": result["std_ms"],
                })
            except Exception as e:
                if is_main:
                    print(f"    Failed: {e}")

    # Save results
    if is_main:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        # Print summary table
        print(f"\n{'='*70}")
        print("Summary (speedup relative to single GPU)")
        print(f"{'='*70}")
        print(f"{'Seq Len':<10} {'Method':<25} {'Ratio':<10} {'TTFT (ms)':<12} {'vs 1-GPU':<10}")
        print("-" * 70)

        # Group by seq_len and calculate speedups relative to single GPU
        for seq_len in args.seq_lengths:
            seq_results = [r for r in all_results["results"] if r["seq_len"] == seq_len]
            single_gpu = next((r for r in seq_results if r["method"] == "single_gpu_prefill"), None)
            single_gpu_ttft = single_gpu["ttft_ms"] if single_gpu else None

            for r in seq_results:
                if single_gpu_ttft and r["method"] != "single_gpu_prefill":
                    speedup = f"{single_gpu_ttft / r['ttft_ms']:.2f}x"
                elif r["method"] == "single_gpu_prefill":
                    speedup = "1.00x"
                else:
                    speedup = "N/A"
                ratio_str = f"{r['recompute_ratio']}" if r['recompute_ratio'] else "-"
                print(f"{r['seq_len']:<10} {r['method']:<25} {ratio_str:<10} {r['ttft_ms']:<12.1f} {speedup:<10}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
