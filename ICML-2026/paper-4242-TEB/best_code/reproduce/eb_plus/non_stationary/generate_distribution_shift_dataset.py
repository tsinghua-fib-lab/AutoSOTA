#!/usr/bin/env python3
"""
Generate a multi-phase synthetic JSONL dataset for distribution shift experiments.

Each phase has controlled input/output token lengths, creating clear distribution
shifts for testing the IFR adaptive controller's convergence behavior.

Usage:
    python reproduce/eb_plus/non_stationary/ or long_context/ or disagg/generate_distribution_shift_dataset.py \
        --model Qwen/Qwen3-8B \
        --num-prompts-per-phase 2000 \
        --phases "1024:128,512:512,128:1024" \
        --output outputs/distribution_shift_3phase.jsonl

Phase format: "input_len:output_len,input_len:output_len,..."
    e.g., "1024:128,512:512,128:1024" means:
        Phase 1: prefill-heavy  (input~1024, output~128)
        Phase 2: balanced       (input~512,  output~512)
        Phase 3: decode-heavy   (input~128,  output~1024)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
from transformers import AutoTokenizer

# dataset_utils lives at reproduce/common/dataset_utils.py; this script is at
# reproduce/eb_plus/non_stationary/, so the relative path is ../../common.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common")
)
from dataset_utils import load_alpaca_prompts, load_sharegpt_prompts


def parse_phases(phases_str: str) -> list[tuple[int, int]]:
    """Parse phase string like '1024:128,512:512,128:1024'."""
    phases = []
    for part in phases_str.split(","):
        in_len, out_len = part.strip().split(":")
        phases.append((int(in_len), int(out_len)))
    return phases


def truncate_prompt_to_length(
    prompt: str,
    target_len: int,
    tokenizer,
) -> tuple[str, int]:
    """Truncate or select prompt to approximately target token length.

    Returns (prompt_text, actual_token_length).
    """
    tokens = tokenizer.encode(prompt)
    if len(tokens) >= target_len:
        # Truncate to target length
        truncated = tokens[:target_len]
        text = tokenizer.decode(truncated, skip_special_tokens=True)
        actual_len = len(tokenizer.encode(text))
        return text, actual_len
    else:
        # Prompt is shorter than target — repeat to fill
        repeated_tokens = tokens.copy()
        while len(repeated_tokens) < target_len:
            repeated_tokens.extend(tokens)
        repeated_tokens = repeated_tokens[:target_len]
        text = tokenizer.decode(repeated_tokens, skip_special_tokens=True)
        actual_len = len(tokenizer.encode(text))
        return text, actual_len


def generate_dataset(
    tokenizer,
    phases: list[tuple[int, int]],
    num_per_phase: int,
    variance: float,
    source_prompts: list[str],
    seed: int = 42,
) -> list[dict]:
    """Generate multi-phase JSONL records."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    records = []

    for phase_idx, (target_input, target_output) in enumerate(phases):
        phase_name = _phase_name(target_input, target_output)
        print(f"\nPhase {phase_idx + 1} ({phase_name}): "
              f"input~{target_input}, output~{target_output}, "
              f"n={num_per_phase}")

        # Sample output lengths with variance
        low = max(16, int(target_output * (1 - variance)))
        high = max(low + 1, int(target_output * (1 + variance)))
        output_lens = np_rng.randint(low, high + 1, size=num_per_phase)

        actual_input_lens = []
        actual_output_lens = []

        for i in range(num_per_phase):
            # Pick a random source prompt
            src = rng.choice(source_prompts)
            prompt, actual_input = truncate_prompt_to_length(
                src, target_input, tokenizer
            )
            out_len = int(output_lens[i])

            records.append({
                "prompt": prompt,
                "input_len": actual_input,
                "output_len": out_len,
            })
            actual_input_lens.append(actual_input)
            actual_output_lens.append(out_len)

        print(f"  Input:  mean={np.mean(actual_input_lens):.0f}, "
              f"std={np.std(actual_input_lens):.0f}")
        print(f"  Output: mean={np.mean(actual_output_lens):.0f}, "
              f"range=[{min(actual_output_lens)}, {max(actual_output_lens)}]")

    return records


def _phase_name(input_len: int, output_len: int) -> str:
    ratio = output_len / max(input_len, 1)
    if ratio > 1.5:
        return "decode-heavy"
    elif ratio < 0.5:
        return "prefill-heavy"
    else:
        return "balanced"


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-phase synthetic dataset for distribution shift experiments."
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-8B",
        help="Model name for tokenizer.",
    )
    parser.add_argument(
        "--num-prompts-per-phase", type=int, default=2000,
        help="Number of prompts per phase.",
    )
    parser.add_argument(
        "--phases", type=str, default="1024:128,512:512,128:1024",
        help="Phase definitions as 'input:output,...' (e.g., '1024:128,512:512,128:1024').",
    )
    parser.add_argument(
        "--variance", type=float, default=0.25,
        help="Relative variance for output_len (±fraction around mean).",
    )
    parser.add_argument(
        "--source-dataset", type=str, default="alpaca",
        choices=["alpaca", "sharegpt"],
        help="Source dataset for real prompts.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    phases = parse_phases(args.phases)
    total = args.num_prompts_per_phase * len(phases)

    print(f"Model: {args.model}")
    print(f"Phases: {len(phases)}")
    for i, (inp, out) in enumerate(phases):
        print(f"  Phase {i+1}: input={inp}, output={out} ({_phase_name(inp, out)})")
    print(f"Prompts per phase: {args.num_prompts_per_phase}")
    print(f"Total prompts: {total}")
    print(f"Output variance: ±{args.variance*100:.0f}%")

    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load source prompts (load plenty for variety)
    max_source = max(10000, total * 2)
    print(f"Loading {args.source_dataset} prompts (up to {max_source})...")
    if args.source_dataset == "alpaca":
        prompts, _, _ = load_alpaca_prompts(
            max_samples=max_source, tokenizer=tokenizer
        )
    else:
        prompts, _, _ = load_sharegpt_prompts(
            max_samples=max_source, tokenizer=tokenizer
        )

    if not prompts:
        print("Error: no source prompts loaded!")
        sys.exit(1)
    print(f"Loaded {len(prompts)} source prompts.")

    # Generate dataset
    records = generate_dataset(
        tokenizer=tokenizer,
        phases=phases,
        num_per_phase=args.num_prompts_per_phase,
        variance=args.variance,
        source_prompts=prompts,
        seed=args.seed,
    )

    # Write JSONL
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(records)} records to {args.output}")

    # Summary
    print(f"\n{'='*50}")
    print("Phase summary:")
    for i, (inp, out) in enumerate(phases):
        start = i * args.num_prompts_per_phase
        end = start + args.num_prompts_per_phase
        phase_records = records[start:end]
        out_lens = [r["output_len"] for r in phase_records]
        in_lens = [r["input_len"] for r in phase_records]
        print(f"  Phase {i+1} ({_phase_name(inp, out)}): "
              f"input mean={np.mean(in_lens):.0f}, "
              f"output mean={np.mean(out_lens):.0f} "
              f"[{min(out_lens)}, {max(out_lens)}]")

    print(f"\nUsage:")
    print(f"  vllm bench serve \\")
    print(f"      --dataset-name custom \\")
    print(f"      --dataset-path {args.output} \\")
    print(f"      --custom-output-len -1 \\")
    print(f"      --ignore-eos \\")
    print(f"      --num-prompts {total} ...")


if __name__ == "__main__":
    main()
