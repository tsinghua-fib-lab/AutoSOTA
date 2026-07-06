#!/usr/bin/env python3
"""
Compute Induction Score for Pythia-14M L3H3.

Protocol (from paper Appendix C and rubric):
- Generate N sequences, each of length 2*H where the second half repeats the first
- For each query position in the second half, identify the "induction target"
  key position: the token that follows the previous occurrence of the current token
- Induction score = mean attention from query positions to their induction-target keys
- n_repeats=2 (each sequence = prefix + repeated prefix)
- n_sequences=100
- induction_match="current", match_choice="last"

Paper baseline: 0.432 for Pythia-14M L3H3
Paper augmented: 0.485
"""

import sys
sys.path.insert(0, "/repo")

import os
import json
import time
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import (
    OFFICIAL_MODEL_NAMES, MODEL_ALIASES, make_model_alias_map
)
from model.hooks import PatternCache, setup_pattern_hooks


def compute_induction_score(
    model,
    layer: int,
    head: int,
    n_sequences: int = 100,
    seq_half_len: int = 32,
    seed: int = 12345,
    induction_match: str = "current",
    match_choice: str = "last",
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    Compute the induction (prefix-matching) score for a specific attention head.

    For each repeated sequence [prefix, prefix], at each query position q in the
    second copy, we find the previous occurrence of token[q] at position k (k < q,
    k in the first copy), and the induction target key position is k+1.
    The induction score is the attention weight from q to k+1, averaged over
    all valid (q, k+1) pairs.
    """
    device = next(model.parameters()).device
    inner = model.module if hasattr(model, "module") else model
    seq_len = 2 * seq_half_len

    # Register hooks to capture attention patterns
    pat_cache = PatternCache()
    hooks = setup_pattern_hooks(inner, layer, pat_cache)

    all_scores = []
    all_baselines = []
    valid_count = 0

    rng = np.random.default_rng(seed)

    try:
        for seq_idx in range(n_sequences):
            # Generate a random prefix and repeat it
            prefix = rng.integers(0, int(inner.cfg.d_vocab), size=seq_half_len)
            tokens_np = np.concatenate([prefix, prefix])  # [A, B, C, A, B, C]
            tokens = torch.from_numpy(tokens_np.astype(np.int64)).unsqueeze(0).to(device)

            # Forward pass to capture attention patterns
            pat_cache.clear()
            with torch.no_grad():
                _ = model(tokens)

            if pat_cache.pattern is None:
                raise RuntimeError(f"Attention pattern not captured for sequence {seq_idx}")

            # Attention pattern: [batch=1, n_heads, seq_len, seq_len]
            head_pattern = pat_cache.pattern[0, head]  # [seq_len, seq_len]

            # For each query position q in the second half:
            # token[q] matches token[q - half] (since sequence is repeated)
            # The induction target key position = (q - half) + 1 = q - half + 1
            # But also check for other matches using the general _find_match_p logic
            for q in range(1, seq_len):
                target_key = _find_match_p_general(
                    tokens_np, q, induction_match, match_choice
                )
                if target_key != -1 and target_key < q:
                    # Attention from query q to key position target_key
                    # In transformer, query position q-1 predicts token at position q
                    # So attention[q-1, target_key] is the relevant attention
                    attn_score = float(head_pattern[q - 1, target_key].item())

                    # Baseline: mean attention to all valid key positions (excluding self)
                    baseline_attn = float(
                        head_pattern[q - 1, :q].mean().item()
                    )

                    all_scores.append(attn_score)
                    all_baselines.append(baseline_attn)
                    valid_count += 1

    finally:
        try:
            inner.reset_hooks(hooks)
        except Exception:
            try:
                inner.remove_all_hook_fns()
            except Exception:
                pass

    if valid_count == 0:
        return {"induction_score": 0.0, "n_valid_positions": 0, "error": "No valid induction positions found"}

    scores = np.array(all_scores)
    baselines = np.array(all_baselines)

    mean_score = float(scores.mean())
    mean_baseline = float(baselines.mean())
    # Induction score: attention to induction target minus baseline
    induction_score = float((scores - baselines).mean())

    return {
        "induction_score": round(induction_score, 6),
        "mean_target_attention": round(mean_score, 6),
        "mean_baseline_attention": round(mean_baseline, 6),
        "n_valid_positions": valid_count,
        "n_sequences": n_sequences,
    }


def _find_match_p_general(tokens, t: int, induction_match: str, match_choice: str) -> int:
    """
    Find the key position of the induction target for query position t.
    Returns the position of the token that follows the previous occurrence of tokens[t].

    Args:
        tokens: 1D numpy array of token ids
        t: current query position
        induction_match: "current" (key = tokens[t]) or "previous" (key = tokens[t-1])
        match_choice: "last" or "first" (use last or first previous occurrence)
    Returns:
        key position (match_pos + 1), or -1 if no valid match
    """
    if induction_match == "previous":
        if t == 0:
            return -1
        key = int(tokens[t - 1])
        left = tokens[:t - 1]
    else:  # "current"
        key = int(tokens[t])
        left = tokens[:t]

    pos = np.where(left == key)[0]
    if len(pos) == 0:
        return -1

    match_pos = int(pos[-1] if match_choice == "last" else pos[0])
    next_pos = match_pos + 1
    if next_pos >= len(tokens):
        return -1
    return next_pos


def main():
    parser = argparse.ArgumentParser(description="Compute Induction Score for a target attention head")
    parser.add_argument("--model-path", type=str, default="/models/pythia-14m",
                        help="Path to model checkpoint")
    parser.add_argument("--layer", type=int, default=3, help="Target layer (0-indexed)")
    parser.add_argument("--head", type=int, default=3, help="Target head (0-indexed)")
    parser.add_argument("--n-sequences", type=int, default=100,
                        help="Number of repeated sequences for evaluation")
    parser.add_argument("--seq-half-len", type=int, default=32,
                        help="Half-length of each sequence (full seq = 2*half)")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--induction-match", type=str, default="current",
                        choices=["current", "previous"])
    parser.add_argument("--match-choice", type=str, default="last",
                        choices=["last", "first"])
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results")
    parser.add_argument("--dtype", type=str, default="float32",
                        help="Model dtype")
    args = parser.parse_args()

    # Environment setup
    os.environ.setdefault("HF_HOME", "/autosota_cache/hf")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    started = time.time()

    # Register local model path
    OFFICIAL_MODEL_NAMES.append(args.model_path)
    MODEL_ALIASES[args.model_path] = ["local-model"]
    make_model_alias_map()

    # Load tokenizer and model
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    print(f"Loading model from {args.model_path}...")
    model_dtype = getattr(torch, args.dtype)
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_path,
        dtype=model_dtype,
        tokenizer=tokenizer,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    model.cfg.use_attn_result = False
    model.eval()

    print(f"Model: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, "
          f"d_model={model.cfg.d_model}, d_head={model.cfg.d_head}")

    # Freeze parameters
    for p in model.parameters():
        p.requires_grad = False

    # Compute induction score
    print(f"\nComputing induction score for L{args.layer}H{args.head}...")
    print(f"  n_sequences={args.n_sequences}, seq_len={2*args.seq_half_len}")
    print(f"  induction_match={args.induction_match}, match_choice={args.match_choice}")

    result = compute_induction_score(
        model=model,
        layer=args.layer,
        head=args.head,
        n_sequences=args.n_sequences,
        seq_half_len=args.seq_half_len,
        seed=args.seed,
        induction_match=args.induction_match,
        match_choice=args.match_choice,
        dtype=model_dtype,
    )

    elapsed = time.time() - started

    # Also compute scores for all heads for reference
    print(f"\nComputing scores for all heads (reference)...")
    all_head_scores = {}
    for l in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            head_result = compute_induction_score(
                model=model, layer=l, head=h,
                n_sequences=args.n_sequences,
                seq_half_len=args.seq_half_len,
                seed=args.seed,
                induction_match=args.induction_match,
                match_choice=args.match_choice,
                dtype=model_dtype,
            )
            all_head_scores[f"L{l}H{h}"] = head_result["induction_score"]
            print(f"  L{l}H{h}: {head_result['induction_score']:.6f}")

    result["all_head_scores"] = all_head_scores
    result["elapsed_seconds"] = round(elapsed, 3)
    result["target_layer"] = args.layer
    result["target_head"] = args.head
    result["model_path"] = args.model_path

    print(f"\n{'='*60}")
    print(f"Target head L{args.layer}H{args.head}:")
    print(f"  Induction Score: {result['induction_score']:.6f}")
    print(f"  Mean Target Attention: {result['mean_target_attention']:.6f}")
    print(f"  Mean Baseline Attention: {result['mean_baseline_attention']:.6f}")
    print(f"  Valid positions: {result['n_valid_positions']}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")

    return result


if __name__ == "__main__":
    main()
