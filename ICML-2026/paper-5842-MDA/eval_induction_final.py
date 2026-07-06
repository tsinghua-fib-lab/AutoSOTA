#!/usr/bin/env python3
"""
Compute Induction Score for Pythia models.

Uses the transformer_lens induction head detection pattern matching
the Olsson et al. (2022) prefix-matching protocol.

Features:
- Multi-seed evaluation (--n-seeds) for robust measurement
- Optional float64 precision (--fp64) for numerical accuracy audit
- All-heads scanning (--all-heads)

Protocol:
- Generate sequences of length 2*seq_half_len with repeated prefix/suffix
- For each head, compute attention * induction_detection_pattern / total_attention
- Report the "mul" score

Paper: baseline=0.432, augmented=0.485
"""

import sys; sys.path.insert(0, "/repo")
import os, time, json, argparse
import numpy as np
import torch

os.environ["HF_HOME"] = "/autosota_cache/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import (
    OFFICIAL_MODEL_NAMES, MODEL_ALIASES, make_model_alias_map
)
from transformer_lens.head_detector import get_induction_head_detection_pattern
from model.hooks import PatternCache, setup_pattern_hooks


def compute_induction_score(
    model, layer: int, head: int,
    n_sequences: int = 100,
    seq_half_len: int = 32,
    seed: int = 12345,
    n_seeds: int = 1,
    use_fp64: bool = False,
) -> dict:
    """
    Compute induction score using transformer_lens detection pattern.

    When n_seeds > 1, runs the protocol with multiple independent seeds
    and reports mean +/- 2*sigma (95% CI) across seeds.
    """
    inner = model.module if hasattr(model, "module") else model

    seed_scores = []
    for seed_idx in range(n_seeds):
        current_seed = seed + seed_idx * 10000
        rng = np.random.default_rng(current_seed)
        all_scores = []
        pat_cache = PatternCache()
        hooks = setup_pattern_hooks(inner, layer, pat_cache)

        try:
            for si in range(n_sequences):
                prefix = rng.integers(0, int(inner.cfg.d_vocab), size=seq_half_len)
                tokens_np = np.concatenate([prefix, prefix])
                tokens = torch.from_numpy(tokens_np.astype(np.int64)).unsqueeze(0).cuda()

                det_pattern = get_induction_head_detection_pattern(tokens[0].cpu()).to(tokens.device)

                pat_cache.clear()
                with torch.no_grad():
                    _ = model(tokens)

                hp = pat_cache.pattern[0, head]  # [seq_len, seq_len]

                if use_fp64:
                    hp_d = hp.to(torch.float64)
                    dp_d = det_pattern.to(torch.float64)
                    score = (hp_d * dp_d).sum() / hp_d.sum()
                else:
                    score = (hp * det_pattern).sum() / hp.sum()

                all_scores.append(score.item())
        finally:
            try:
                inner.reset_hooks(hooks)
            except Exception:
                pass

        seed_mean = float(np.mean(all_scores))
        seed_std = float(np.std(all_scores))
        seed_scores.append({
            "seed": current_seed,
            "mean": seed_mean,
            "std": seed_std,
            "scores": all_scores,
        })

    # Aggregate across seeds
    if n_seeds > 1:
        means = np.array([s["mean"] for s in seed_scores])
        grand_mean = float(means.mean())
        grand_std = float(means.std())
        ci95 = 2.0 * grand_std  # 95% CI approximation
    else:
        grand_mean = seed_scores[0]["mean"]
        grand_std = seed_scores[0]["std"]
        ci95 = 2.0 * grand_std / np.sqrt(n_sequences)

    return {
        "induction_score": round(grand_mean, 8),
        "induction_score_std": round(grand_std, 8),
        "ci95_half_width": round(ci95, 8),
        "n_sequences": n_sequences,
        "n_seeds": n_seeds,
        "base_seed": seed,
        "seq_len": 2 * seq_half_len,
        "layer": layer,
        "head": head,
        "use_fp64": use_fp64,
        "per_seed": seed_scores if n_seeds > 1 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/models/pythia-14m")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--n-sequences", type=int, default=100)
    parser.add_argument("--seq-half-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of independent seeds for robust evaluation (default: 1)")
    parser.add_argument("--fp64", action="store_true",
                        help="Use float64 precision for accumulation (default: float32)")
    parser.add_argument("--output", default="/repo/outputs/induction_score.json")
    parser.add_argument("--all-heads", action="store_true")
    args = parser.parse_args()

    started = time.time()

    OFFICIAL_MODEL_NAMES.append(args.model_path)
    MODEL_ALIASES[args.model_path] = ["local-model"]
    make_model_alias_map()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_path,
        dtype=torch.float32,
        tokenizer=tokenizer,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    model.cfg.use_attn_result = False
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print("Model: {} layers, {} heads, d_model={}".format(
        model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_model))
    print("Eval config: n_seeds={}, n_sequences={}, fp64={}, base_seed={}".format(
        args.n_seeds, args.n_sequences, args.fp64, args.seed))

    # Compute target head score
    print("\nComputing induction score for L{}H{}...".format(args.layer, args.head))
    result = compute_induction_score(
        model, args.layer, args.head,
        n_sequences=args.n_sequences,
        seq_half_len=args.seq_half_len,
        seed=args.seed,
        n_seeds=args.n_seeds,
        use_fp64=args.fp64,
    )
    ci_str = " +/- {:.6f} (95% CI)".format(result["ci95_half_width"])
    print("  Induction Score: {:.8f} +/- {:.8f}{}".format(
        result["induction_score"], result["induction_score_std"], ci_str))
    if args.n_seeds > 1:
        seed_means = [s["mean"] for s in result["per_seed"]]
        seed_means_str = ", ".join("{:.6f}".format(m) for m in seed_means)
        print("  Per-seed means: [{}]".format(seed_means_str))

    # Optionally compute all heads
    if args.all_heads:
        print("\nAll head scores:")
        all_head_scores = {}
        for l in range(model.cfg.n_layers):
            for h in range(model.cfg.n_heads):
                hr = compute_induction_score(
                    model, l, h,
                    n_sequences=args.n_sequences,
                    seq_half_len=args.seq_half_len,
                    seed=args.seed,
                    n_seeds=1,
                    use_fp64=args.fp64,
                )
                key = "L{}H{}".format(l, h)
                all_head_scores[key] = hr["induction_score"]
                marker = " ***" if hr["induction_score"] > 0.1 else ""
                print("  {}: {:.6f}{}".format(key, hr["induction_score"], marker))
        result["all_head_scores"] = all_head_scores

    elapsed = time.time() - started
    result["elapsed_seconds"] = round(elapsed, 3)
    result["model_path"] = args.model_path

    sep = "=" * 50
    print("\n{}".format(sep))
    print("RESULT: Induction Score L{}H{} = {:.8f}".format(
        args.layer, args.head, result["induction_score"]))
    print("Paper baseline: 0.432")
    print("Paper augmented: 0.485")
    print("Elapsed: {:.1f}s".format(elapsed))
    print(sep)

    out_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to {}".format(args.output))


if __name__ == "__main__":
    main()
