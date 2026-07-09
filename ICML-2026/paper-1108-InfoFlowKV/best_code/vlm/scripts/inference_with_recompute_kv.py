#!/usr/bin/env python3
"""
KV cache recomputation for Qwen3-VL.

Usage:
    python scripts/inference_with_recompute_kv.py --config configs/blink_counting.yaml
"""

import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from models.qwen.kv_cache import (
    VLMKVCacheExtractor,
    ImportanceScorer,
    KVCacheRecomputer,
    KVCacheInference,
    RecomputeConfig,
)
from benchmarks import get_dataset
import time


def make_baseline_inference_fn(model, processor, max_new_tokens):
    """Create baseline inference function using native model.generate()."""

    def fn(sample, messages):
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()

        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.perf_counter() - start_time

        new_tokens = generated_ids[:, input_len:]
        prediction = processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

        return {
            "prediction": prediction,
            "ttft_ms": 0,  # Not measured for baseline
            "total_time_ms": total_time * 1000,
            "tokens_generated": new_tokens.shape[1],
            "context_tokens": input_len,
            "recompute_positions": 0,
        }

    return fn


def make_recompute_inference_fn(
    model, processor, extractor, scorer, recomputer, inference, config, max_new_tokens, chunk_k
):
    """Create inference function with KV cache recomputation."""
    strategy = config.strategy

    def fn(sample, messages):
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Extract KV cache
        kv_data = extractor.extract(inputs, chunk_k=chunk_k)

        # At chunk_k=0 or 1 (single chunk = full attention), skip recomputation.
        # The extracted KV cache is already correct with no cross-chunk artifacts.
        if chunk_k is None or chunk_k <= 1:
            updated_cache = kv_data.past_key_values
            recompute_count = 0
        elif strategy == "no_recompute":
            updated_cache = recomputer.recompute_noop(kv_data)
            recompute_count = 0
        elif strategy == "lego":
            updated_cache = recomputer.recompute_lego(kv_data, ratio=config.recompute_ratio)
            recompute_count = max(1, int(kv_data.seq_len * config.recompute_ratio))
        elif strategy == "cacheblend":
            updated_cache = recomputer.recompute_cacheblend(kv_data, config.recompute_ratio)
            recompute_count = int(kv_data.seq_len * config.recompute_ratio)
        else:  # guided_recompute (default)
            scores = scorer.compute(kv_data)
            recompute_indices = scorer.select_positions(scores, config, kv_data.image_ranges)
            updated_cache = recomputer.recompute(kv_data, recompute_indices)
            recompute_count = recompute_indices.numel()

        result = inference.generate(
            updated_cache, input_ids=kv_data.input_ids,
            context_len=kv_data.seq_len, max_new_tokens=max_new_tokens,
        )

        return {
            "prediction": result["text"],
            "ttft_ms": result["ttft_ms"],
            "total_time_ms": result["total_time_ms"],
            "tokens_generated": result["tokens_generated"],
            "context_tokens": kv_data.seq_len,
            "recompute_positions": recompute_count,
        }

    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Model config
    model_name = cfg["model"]
    cache_dir = cfg["cache_dir"]

    # Dataset config
    dataset_name = cfg["dataset"]
    num_samples = cfg.get("num_samples")
    dataset_dir = cfg.get("dataset_dir")
    output_dir = cfg.get("output_dir")

    # Recompute config
    recompute_ratio = cfg.get("recompute_ratio", 0.15)
    method = cfg.get("method", "norm")
    max_new_tokens = cfg.get("max_new_tokens", 128)
    chunk_k_cfg = cfg.get("chunk_k")
    strategy = cfg.get("strategy", "guided_recompute")

    # Normalize chunk_k to a list
    if chunk_k_cfg is None:
        chunk_k_list = [None]
    elif isinstance(chunk_k_cfg, list):
        chunk_k_list = chunk_k_cfg
    else:
        chunk_k_list = [chunk_k_cfg]

    print("=" * 60)
    print("KV Cache Recomputation for Qwen3-VL")
    print(f"Strategy: {strategy}")
    print(f"Chunk_k values: {chunk_k_list}")
    print("=" * 60)

    print(f"\n[1/5] Loading model: {model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto", cache_dir=cache_dir,
        local_files_only=True,
    )
    print("[2/5] Model loaded. Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    print("[3/5] Processor loaded. Initializing components...")

    # Initialize components
    extractor = VLMKVCacheExtractor(model)
    scorer = ImportanceScorer(model, method=method)
    recomputer = KVCacheRecomputer(model)
    inference = KVCacheInference(model, processor)
    config = RecomputeConfig(
        strategy=strategy,
        recompute_ratio=float(recompute_ratio),
        method=method,
    )
    print("[4/5] Components initialized. Loading dataset...")

    # Load dataset
    dataset_base = get_dataset(dataset_name, dataset_dir=dataset_dir, output_dir=output_dir)
    print(f"[5/5] Dataset loaded: {len(dataset_base)} samples")

    # Loop over all chunk_k values
    for i, chunk_k in enumerate(chunk_k_list):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(chunk_k_list)}] Running with chunk_k={chunk_k}")
        print(f"{'='*60}")

        # Reload dataset for each chunk_k run
        dataset = get_dataset(dataset_name, dataset_dir=dataset_dir, output_dir=output_dir)
        if num_samples:
            dataset.data = dataset.data.select(range(min(num_samples, len(dataset))))

        infer_fn = make_recompute_inference_fn(
            model,
            processor,
            extractor,
            scorer,
            recomputer,
            inference,
            config,
            max_new_tokens,
            chunk_k,
        )

        def sample_fn(sample):
            messages = dataset.build_messages(sample)
            return infer_fn(sample, messages)

        output = dataset.evaluate(sample_fn, num_samples=num_samples)
        output["summary"]["strategy"] = strategy
        output["summary"]["recompute_ratio"] = recompute_ratio
        output["summary"]["method"] = method
        output["summary"]["model"] = model_name
        output["summary"]["chunk_k"] = chunk_k

        chunk_label = f"chunk{chunk_k}" if chunk_k else "chunk0"
        dataset.save_results(output, f"{strategy}_{recompute_ratio}_{chunk_label}")


if __name__ == "__main__":
    main()
