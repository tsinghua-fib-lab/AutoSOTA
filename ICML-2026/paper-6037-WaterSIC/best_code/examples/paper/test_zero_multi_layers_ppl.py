"""Test PPL impact of zeroing out rows in multiple layers.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.test_zero_multi_layers_ppl \
        --model_name Qwen/Qwen3-8B \
        --zero_spec "6.w1:5723,8518;6.w3:5723,8518;16.w1:2271,1875;16.w3:2271,1875" \
        --init_dist
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_layerwise.data import get_wikitext2, split_dataset
from quant_layerwise.eval import eval_ppl
from quant_layerwise.pipeline import ensure_single_process_distributed, load_model_and_tokenizer


def parse_zero_spec(spec: str) -> dict:
    """Parse zero spec: '6.w1:5723,8518;6.w3:5723,8518' -> {(6, 'w1'): [5723, 8518], ...}"""
    result = {}
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid format: '{item}'. Expected 'layer.weight:row1,row2,...'")
        layer_weight = parts[0].strip()
        rows = [int(r.strip()) for r in parts[1].split(",") if r.strip()]

        lw_parts = layer_weight.split(".")
        if len(lw_parts) != 2:
            raise ValueError(f"Invalid layer.weight: '{layer_weight}'. Expected 'layer.weight'")
        layer_id = int(lw_parts[0])
        weight = lw_parts[1].strip()
        result[(layer_id, weight)] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description="Test PPL impact of zeroing rows in multiple layers")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--zero_spec", type=str, required=True,
                        help="Format: '6.w1:5723,8518;6.w3:5723,8518;16.w1:2271,1875;16.w3:2271,1875'")
    parser.add_argument("--nsamples", type=int, default=None,
                        help="Number of samples for PPL (None = all)")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--init_dist", action="store_true")

    args = parser.parse_args()

    zero_dict = parse_zero_spec(args.zero_spec)
    print(f"Will zero out: {zero_dict}")

    if args.init_dist:
        ensure_single_process_distributed(local_rank=0, master_port=29500)

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name, local_rank=0)
    model.eval()

    device = next(model.parameters()).device

    # Prepare eval data
    print("Preparing WikiText-2 test data...")
    eval_tokens = split_dataset(get_wikitext2(tokenizer, split="test"), args.seqlen)
    if args.nsamples:
        eval_tokens = eval_tokens[:args.nsamples]
    eval_tokens = eval_tokens.to(device)
    print(f"Eval samples: {eval_tokens.shape[0]}")

    # Resize KV caches
    batch_size = 1
    for layer in model.layers:
        attn = layer.attention
        if hasattr(attn, "cache_k"):
            old_shape = attn.cache_k.shape
            new_shape = (batch_size, old_shape[1], old_shape[2], old_shape[3])
            attn.cache_k = torch.zeros(new_shape, device=device, dtype=attn.cache_k.dtype)
            attn.cache_v = torch.zeros(new_shape, device=device, dtype=attn.cache_v.dtype)

    # Compute baseline PPL
    print("\nComputing baseline PPL...")
    ppl_baseline, nll_baseline = eval_ppl(model, eval_tokens)
    print(f"Baseline PPL: {ppl_baseline:.4f} (NLL: {nll_baseline:.4f})")

    # Store original weights and zero out specified rows
    originals = {}
    for (layer_id, weight), rows in zero_dict.items():
        layer = model.layers[layer_id]
        if weight == "w1":
            module = layer.feed_forward.w1
        elif weight == "w2":
            module = layer.feed_forward.w2
        elif weight == "w3":
            module = layer.feed_forward.w3
        else:
            raise ValueError(f"Unknown weight: {weight}")

        # Store original
        originals[(layer_id, weight)] = module.weight.data.clone()

        # Zero out rows
        with torch.no_grad():
            for row in rows:
                module.weight.data[row, :] = 0
        print(f"Zeroed L{layer_id}.{weight} rows {rows}")

    # Compute PPL with zeroed rows
    print("\nComputing PPL with zeroed rows...")
    ppl_zeroed, nll_zeroed = eval_ppl(model, eval_tokens)
    print(f"Zeroed PPL: {ppl_zeroed:.4f} (NLL: {nll_zeroed:.4f})")

    # Compute impact
    ppl_increase = ppl_zeroed - ppl_baseline
    ppl_increase_pct = (ppl_zeroed / ppl_baseline - 1) * 100

    print(f"\n{'='*60}")
    print("RESULTS")
    print('='*60)
    print("Zeroed specifications:")
    for (layer_id, weight), rows in zero_dict.items():
        print(f"  L{layer_id}.{weight}: rows {rows}")
    print(f"\nBaseline PPL:  {ppl_baseline:.4f}")
    print(f"Zeroed PPL:    {ppl_zeroed:.4f}")
    print(f"PPL increase:  {ppl_increase:.4f} ({ppl_increase_pct:+.2f}%)")

    if ppl_increase_pct < 1:
        print("\n✓ These rows have MINIMAL impact on PPL (<1% increase)")
        print("  → Safe to zero during quantization")
    elif ppl_increase_pct < 5:
        print("\n⚠ These rows have MODERATE impact on PPL (1-5% increase)")
    else:
        print("\n✗ These rows have SIGNIFICANT impact on PPL (>5% increase)")

    # Restore original weights
    for (layer_id, weight), original in originals.items():
        layer = model.layers[layer_id]
        if weight == "w1":
            module = layer.feed_forward.w1
        elif weight == "w2":
            module = layer.feed_forward.w2
        elif weight == "w3":
            module = layer.feed_forward.w3
        module.weight.data = original


if __name__ == "__main__":
    main()
