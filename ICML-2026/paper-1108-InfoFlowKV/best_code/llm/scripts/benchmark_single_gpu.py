#!/usr/bin/env python
"""
Single GPU prefill benchmark for baseline timing.

Usage:
    python scripts/benchmark_single_gpu.py \
        --model ../models/Qwen3-14B/ \
        --seq_lengths 4096 8192 16384 32768
"""

import argparse
import time
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description="Single GPU prefill benchmark")
    parser.add_argument("--model", type=str, default="/path/to/Qwen3-14B")
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 16384, 32768],
    )
    parser.add_argument("--num_warmup", type=int, default=2)
    parser.add_argument("--num_iterations", type=int, default=5)
    return parser.parse_args()


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


def benchmark_prefill(model, tokenizer, context, query, device, num_iterations):
    """Benchmark single GPU prefill."""
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

    return {
        "mean_ms": sum(timings) / len(timings),
        "std_ms": torch.tensor(timings).std().item(),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }


def main():
    args = parse_args()
    device = "cuda:0"

    print("=" * 60)
    print("Single GPU Prefill Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Warmup iterations: {args.num_warmup}")
    print(f"Benchmark iterations: {args.num_iterations}")
    print("=" * 60)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    query = "What is the main topic discussed in the passages above?"

    results = []
    print(f"\n{'Seq Len':<12} {'Prefill (ms)':<15} {'Std (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-" * 60)

    for seq_len in args.seq_lengths:
        context = generate_context(tokenizer, seq_len)
        actual_len = len(tokenizer.encode(context, add_special_tokens=False))

        # Warmup
        benchmark_prefill(model, tokenizer, context, query, device, args.num_warmup)

        # Benchmark
        result = benchmark_prefill(model, tokenizer, context, query, device, args.num_iterations)

        print(f"{actual_len:<12} {result['mean_ms']:<15.1f} {result['std_ms']:<12.1f} {result['min_ms']:<12.1f} {result['max_ms']:<12.1f}")

        results.append({
            "seq_len": actual_len,
            "mean_ms": result["mean_ms"],
            "std_ms": result["std_ms"],
        })

    print("\n" + "=" * 60)
    print("Summary (copy-paste format):")
    print("=" * 60)
    for r in results:
        print(f"{r['seq_len']}\t{r['mean_ms']:.1f}")


if __name__ == "__main__":
    main()
