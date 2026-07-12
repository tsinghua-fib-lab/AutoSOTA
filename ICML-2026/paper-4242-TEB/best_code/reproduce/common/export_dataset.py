#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Export datasets to JSONL format for use with vllm bench serve.

Usage:
    # Alpaca dataset (balanced prefill/decode)
    python pd_exp/export_dataset.py \
        --dataset alpaca \
        --model Qwen/Qwen3-8B \
        --num-samples 4000 \
        --output ./outputs/alpaca_prompts.jsonl

    # ProcessBench dataset (decode-heavy with step-by-step reasoning)
    python pd_exp/export_dataset.py \
        --dataset processbench \
        --model Qwen/Qwen3-8B \
        --num-samples 4000 \
        --processbench-split gsm8k \
        --output ./outputs/processbench_gsm8k.jsonl

    # NuminaMath-CoT dataset (very decode-heavy, 850k samples)
    # Use --min-output-len to filter for longer outputs
    python pd_exp/export_dataset.py \
        --dataset numina_math \
        --model Qwen/Qwen3-8B \
        --num-samples 4000 \
        --min-output-len 800 \
        --output ./outputs/numina_math_long.jsonl

    # LongBench-v2 dataset (prefill-heavy, long context)
    # Use --min-input-len and --max-input-len to filter input length
    python pd_exp/export_dataset.py \
        --dataset longbench \
        --model Qwen/Qwen3-8B \
        --num-samples 4000 \
        --min-input-len 1000 \
        --max-input-len 4000 \
        --output ./outputs/longbench_prefill.jsonl

Then use with vllm bench serve:
    vllm bench serve \
        --dataset-name custom \
        --dataset-path ./outputs/alpaca_prompts.jsonl \
        --custom-output-len 256 \
        ...

Or run grid search with real dataset:
    ./pd_exp/syn/run_grid_search_real.sh ./outputs/alpaca_prompts.jsonl 4
"""

import argparse
import json
import os

import numpy as np
from dataset_utils import (
    DEFAULT_SHAREGPT_PATH,
    apply_chat_template,
    load_alpaca_prompts,
    load_lmsys_prompts,
    load_longbench_prompts,
    load_numina_math_prompts,
    load_processbench_prompts,
    load_sharegpt_prompts,
)
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Export dataset to JSONL for vllm bench serve"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="alpaca",
        choices=[
            "alpaca",
            "sharegpt",
            "lmsys",
            "processbench",
            "numina_math",
            "longbench",
        ],
        help="Dataset to export",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model for tokenizer (used for chat template)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of samples to export"
    )
    parser.add_argument(
        "--sharegpt-path",
        type=str,
        default=DEFAULT_SHAREGPT_PATH,
        help="Path to ShareGPT JSON file",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output JSONL file path"
    )
    parser.add_argument(
        "--output-len-cap", type=int, default=None,
        help=(
            "Per-prompt output length cap (truncates dataset-native output_len). "
            "Paper §4.1 protocol: 500 for ShareGPT, 4000 for NuminaMath. When "
            "this flag is omitted, sharegpt and numina_math fall back to those "
            "paper defaults; pass an explicit value (e.g. 0/-1) to disable."
        ),
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template to prompts",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable thinking mode for Qwen3 models",
    )
    # ProcessBench specific options
    parser.add_argument(
        "--processbench-split",
        type=str,
        default="gsm8k",
        help="ProcessBench split to use (gsm8k, math, etc.)",
    )
    parser.add_argument(
        "--no-step-by-step",
        action="store_true",
        help='Do not append "Please think step by step" to prompts',
    )
    # NuminaMath specific options
    parser.add_argument(
        "--min-output-len",
        type=int,
        default=0,
        help="Minimum output length filter "
        "(for numina_math, use 800+ for decode-heavy)",
    )
    # LongBench specific options
    parser.add_argument(
        "--min-input-len",
        type=int,
        default=1000,
        help="Minimum input length filter (for longbench, prefill-heavy)",
    )
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=4000,
        help="Maximum input length filter (for longbench)",
    )

    args = parser.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "alpaca":
        prompts, input_lengths, output_lengths = load_alpaca_prompts(
            max_samples=args.num_samples, tokenizer=tokenizer
        )
    elif args.dataset == "sharegpt":
        # Paper §4.1 protocol caps ShareGPT outputs at 500 tokens. Apply that
        # default when the user didn't override --output-len-cap explicitly;
        # passing a non-positive value disables the cap.
        sharegpt_cap = args.output_len_cap
        if sharegpt_cap is None:
            sharegpt_cap = 500
        elif sharegpt_cap <= 0:
            sharegpt_cap = None
        prompts, input_lengths, output_lengths = load_sharegpt_prompts(
            json_path=args.sharegpt_path,
            max_samples=args.num_samples,
            output_len_cap=sharegpt_cap,
            tokenizer=tokenizer,
        )
    elif args.dataset == "lmsys":
        prompts, input_lengths, output_lengths = load_lmsys_prompts(
            max_samples=args.num_samples, tokenizer=tokenizer
        )
    elif args.dataset == "processbench":
        prompts, input_lengths, output_lengths = load_processbench_prompts(
            max_samples=args.num_samples,
            tokenizer=tokenizer,
            split=args.processbench_split,
            step_by_step=not args.no_step_by_step,
        )
    elif args.dataset == "numina_math":
        # Paper §4.1 protocol caps NuminaMath outputs at 4000 tokens. Apply
        # that default when --output-len-cap is omitted; non-positive disables.
        numina_cap = args.output_len_cap
        if numina_cap is None:
            numina_cap = 4000
        elif numina_cap <= 0:
            numina_cap = None
        prompts, input_lengths, output_lengths = load_numina_math_prompts(
            max_samples=args.num_samples,
            tokenizer=tokenizer,
            min_output_len=args.min_output_len,
            output_len_cap=numina_cap,
            step_by_step=not args.no_step_by_step,
        )
    elif args.dataset == "longbench":
        prompts, input_lengths, output_lengths = load_longbench_prompts(
            max_samples=args.num_samples,
            tokenizer=tokenizer,
            min_input_len=args.min_input_len,
            max_input_len=args.max_input_len,
        )

    # Optionally apply chat template
    if args.apply_chat_template:
        print("Applying chat template...")
        prompts = apply_chat_template(
            prompts, tokenizer, enable_thinking=not args.disable_thinking
        )

    # Export to JSONL (required by vllm CustomDataset)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Ensure output file ends with .jsonl
    output_path = args.output
    if not output_path.endswith(".jsonl"):
        output_path = output_path.rsplit(".", 1)[0] + ".jsonl"
        print("Note: Changed output extension to .jsonl (required by vllm)")

    print(f"Exporting {len(prompts)} prompts to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt, input_len, output_len in zip(
            prompts, input_lengths, output_lengths
        ):
            record = {
                "prompt": prompt,
                "input_len": input_len,
                "output_len": output_len,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Calculate statistics
    input_arr = np.array(input_lengths)
    output_arr = np.array(output_lengths)

    print(f"\n{'=' * 50}")
    print(f"Done! Exported {len(prompts)} prompts to {output_path}")
    print(f"{'=' * 50}")

    print("\n[Input Length] (actual):")
    print(f"  Mean:   {np.mean(input_arr):.1f} tokens")
    print(f"  Median: {np.median(input_arr):.1f} tokens")
    print(f"  Std:    {np.std(input_arr):.1f}")
    print(f"  Range:  [{np.min(input_arr)}, {np.max(input_arr)}]")

    print("\n[Output Length] (estimated from dataset):")
    print(f"  Mean:   {np.mean(output_arr):.1f} tokens")
    print(f"  Median: {np.median(output_arr):.1f} tokens")
    print(f"  Std:    {np.std(output_arr):.1f}")
    print(f"  Range:  [{np.min(output_arr)}, {np.max(output_arr)}]")

    ratio = np.mean(output_arr) / np.mean(input_arr) if np.mean(input_arr) > 0 else 0
    print("\n[Estimated Workload Type]:")
    print(f"  Output/Input Ratio: {ratio:.2f}x")
    print(f"  Type: {'DECODE-HEAVY' if ratio > 1 else 'PREFILL-HEAVY'} (estimated)")

    print("\nNote: Output length is estimated from dataset.")
    print("Actual output depends on model generation.")
    print("Run benchmark to get real statistics:")
    print("  python pd_exp/analyze_benchmark_stats.py <result_dir>")

    print("\n[Usage]:")
    print("  # Run grid search")
    print(f"  ./pd_exp/syn/run_grid_search_real.sh {output_path} 4")
    print("")
    print("  # Or manually with vllm bench serve")
    print(f"  vllm bench serve --dataset-name custom --dataset-path {output_path} ...")


if __name__ == "__main__":
    main()
