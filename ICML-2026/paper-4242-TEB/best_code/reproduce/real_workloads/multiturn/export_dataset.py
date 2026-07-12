#!/usr/bin/env python3
"""
Export multi-turn conversations for prefix cache testing.

This script exports conversations from WildChat-1M in a format compatible with
vLLM's multi-turn benchmark (benchmarks/multi_turn/benchmark_serving_multi_turn_threaded.py).

Usage:
    # Paper §4.1: WildChat 3,000 multi-turn conversations (~27,900 requests),
    # min 6 turns. Use these exact settings for paper-faithful reproduction —
    # smaller num-conversations under-reports queueing TTFT by 10-25x.
    python pd_exp/multiturn/export_dataset.py \
        --dataset wildchat \
        --model Qwen/Qwen3-8B \
        --num-conversations 3000 \
        --min-turns 6 \
        --output ./outputs/wildchat_multiturn.json

Then run the multi-turn benchmark:
    ./reproduce/real_workloads/multiturn/run_benchmark.sh ./outputs/wildchat_multiturn.json 4
"""

import argparse
import json
import os
import sys

# Add reproduce/common/ to path for dataset_utils.
# Script lives at reproduce/real_workloads/multiturn/; ../../common/ is the target.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "common"))

import numpy as np
from transformers import AutoTokenizer

from dataset_utils import load_wildchat_conversations


def main():
    parser = argparse.ArgumentParser(
        description='Export multi-turn conversations for prefix cache testing')

    parser.add_argument('--dataset', type=str, default='wildchat',
                        choices=['wildchat'],
                        help='Dataset to export')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-8B',
                        help='Model for tokenizer')
    parser.add_argument('--num-conversations', type=int, default=3000,
                        help='Number of conversations to export (paper §4.1: 3000)')
    parser.add_argument('--min-turns', type=int, default=6,
                        help='Minimum turns per conversation (paper §4.1: 6)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')

    args = parser.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'wildchat':
        conversations = load_wildchat_conversations(
            max_conversations=args.num_conversations,
            tokenizer=tokenizer,
            min_turns=args.min_turns,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if not conversations:
        print("Error: No conversations loaded!")
        sys.exit(1)

    # Export to JSON (format expected by multi-turn benchmark)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Ensure output file ends with .json
    output_path = args.output
    if not output_path.endswith('.json'):
        output_path = output_path.rsplit('.', 1)[0] + '.json'
        print(f"Note: Changed output extension to .json")

    print(f"Exporting {len(conversations)} conversations to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)

    # Calculate statistics
    turn_counts = [len(c['messages']) for c in conversations]
    user_turns = [len([m for m in c['messages'] if m['role'] == 'user']) for c in conversations]

    print(f"\n{'='*50}")
    print(f"Done! Exported {len(conversations)} conversations to {output_path}")
    print(f"{'='*50}")

    print(f"\n[Conversation Statistics]:")
    print(f"  Total conversations: {len(conversations)}")
    print(f"  Total messages: {sum(turn_counts)}")
    print(f"  Avg messages/conversation: {np.mean(turn_counts):.1f}")
    print(f"  Avg user turns/conversation: {np.mean(user_turns):.1f}")
    print(f"  Message range: [{np.min(turn_counts)}, {np.max(turn_counts)}]")

    # Estimate prefix cache benefit
    print(f"\n[Prefix Cache Benefit Estimate]:")
    print(f"  For each conversation with n user turns:")
    print(f"  - Turn 1: 0% cached (new conversation)")
    print(f"  - Turn 2: ~50% cached (turn 1 history)")
    print(f"  - Turn 3: ~67% cached (turn 1-2 history)")
    print(f"  - Turn n: ~(n-1)/n cached")
    avg_user = np.mean(user_turns)
    estimated_cache = (avg_user - 1) / avg_user * 100 if avg_user > 1 else 0
    print(f"  Estimated avg cache hit: {estimated_cache:.1f}%")

    print(f"\n[Usage]:")
    print(f"  # Run benchmark")
    print(f"  ./reproduce/real_workloads/multiturn/run_benchmark.sh {output_path} 4")


if __name__ == '__main__':
    main()
