"""Scan w2 input norms across all layers to find outliers.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.scan_w2_input_norms \
        --model_name Qwen/Qwen3-8B \
        --nsamples 4 \
        --init_dist
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_layerwise.data import get_wikitext2, split_dataset, take_nseq
from quant_layerwise.pipeline import ensure_single_process_distributed, load_model_and_tokenizer


class W2InputCapture:
    """Capture w2 input activation."""

    def __init__(self):
        self.input_act = None
        self.handle = None

    def hook(self, module, input, output):
        self.input_act = input[0].detach().clone()

    def register(self, module):
        self.handle = module.register_forward_hook(self.hook)

    def remove(self):
        if self.handle:
            self.handle.remove()


def main():
    parser = argparse.ArgumentParser(description="Scan w2 input norms across all layers")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--init_dist", action="store_true")

    args = parser.parse_args()

    if args.init_dist:
        ensure_single_process_distributed(local_rank=0, master_port=29500)

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name, local_rank=0)
    model.eval()

    device = next(model.parameters()).device
    n_layers = len(model.layers)

    print(f"Model has {n_layers} layers")

    # Prepare data
    print(f"Preparing evaluation data (nsamples={args.nsamples})...")
    eval_tokens = split_dataset(get_wikitext2(tokenizer, split="test"), args.seqlen)
    eval_tokens = take_nseq(eval_tokens, args.nsamples)
    tokens = eval_tokens.to(device)

    # Resize KV caches
    batch_size = tokens.shape[0]
    for layer in model.layers:
        attn = layer.attention
        if hasattr(attn, "cache_k") and attn.cache_k.shape[0] < batch_size:
            old_shape = attn.cache_k.shape
            new_shape = (batch_size, old_shape[1], old_shape[2], old_shape[3])
            attn.cache_k = torch.zeros(new_shape, device=device, dtype=attn.cache_k.dtype)
            attn.cache_v = torch.zeros(new_shape, device=device, dtype=attn.cache_v.dtype)

    # Register hooks for all w2 modules
    captures = {}
    for layer_idx in range(n_layers):
        layer = model.layers[layer_idx]
        cap = W2InputCapture()
        cap.register(layer.feed_forward.w2)
        captures[layer_idx] = cap

    # Run forward pass
    print("Running forward pass...")
    with torch.no_grad():
        _ = model(tokens, start_pos=0)

    # Remove hooks
    for cap in captures.values():
        cap.remove()

    # Collect and print results
    print(f"\n{'='*60}")
    print("W2 INPUT ACTIVATION NORMS (REFERENCE MODEL)")
    print('='*60)
    print(f"{'Layer':<8} {'Norm':>12} {'Mean':>12} {'Std':>12} {'Max':>12}")
    print('-'*60)

    norms = []
    for layer_idx in range(n_layers):
        X = captures[layer_idx].input_act.float()
        norm = X.norm().item()
        mean = X.mean().item()
        std = X.std().item()
        abs_max = X.abs().max().item()
        norms.append((layer_idx, norm, mean, std, abs_max))
        print(f"L{layer_idx:<6} {norm:>12.2f} {mean:>12.4f} {std:>12.4f} {abs_max:>12.4f}")

    # Find outliers (> 2x median)
    norm_values = [n[1] for n in norms]
    median_norm = sorted(norm_values)[len(norm_values) // 2]
    mean_norm = sum(norm_values) / len(norm_values)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Median norm: {median_norm:.2f}")
    print(f"Mean norm: {mean_norm:.2f}")

    print(f"\nOUTLIERS (> 2x median = {2*median_norm:.2f}):")
    outliers = [(idx, norm) for idx, norm, _, _, _ in norms if norm > 2 * median_norm]
    if outliers:
        for idx, norm in sorted(outliers, key=lambda x: -x[1]):
            ratio = norm / median_norm
            print(f"  L{idx}: {norm:.2f} ({ratio:.1f}x median)")
    else:
        print("  None found")

    print(f"\nLOW OUTLIERS (< 0.5x median = {0.5*median_norm:.2f}):")
    low_outliers = [(idx, norm) for idx, norm, _, _, _ in norms if norm < 0.5 * median_norm]
    if low_outliers:
        for idx, norm in sorted(low_outliers, key=lambda x: x[1]):
            ratio = norm / median_norm
            print(f"  L{idx}: {norm:.2f} ({ratio:.1f}x median)")
    else:
        print("  None found")


if __name__ == "__main__":
    main()
