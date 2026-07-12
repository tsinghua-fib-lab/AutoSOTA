#!/usr/bin/env python
"""Minimal end-to-end sanity check for the demo gallery.

Pulls the pre-computed embedding gallery from the Hub and greedily decodes
every row from its compression embedding (no other input). Prints the
reconstruction next to the original with a token-by-token ANSI diff so you
can see at a glance whether the round-trip holds. This is the same operation
the demo notebook does on each ▶ Reconstruct click -- collapsed into a script
so you can verify the Hub artifact before opening Colab.

The gallery can mix multiple frozen models (e.g. Llama-3.2-1B for the main
gallery + side-by-side, plus SmolLM2-360M for the second TC drift demo).
Without ``--model`` the script auto-switches the loaded frozen model by
``row["model_checkpoint"]`` and reconstructs every row in the gallery. Pass
``--model <ckpt>`` to restrict to one model's rows.

Usage:
    python scripts/reconstruct_demo_gallery.py                              # all rows, switching models
    python scripts/reconstruct_demo_gallery.py --dtype bfloat16             # A100 / H100
    python scripts/reconstruct_demo_gallery.py --model unsloth/Llama-3.2-1B # only Llama rows
"""

from __future__ import annotations

import argparse
import gc

import torch
from datasets import load_dataset, load_from_disk

from progressive_cramming.demo import load_frozen_model
from progressive_cramming.inference.generation import generate_from_compression

DEFAULT_REPO = "LarryLovestein/progressive_cramming_demo_gallery"

# ANSI colours -- match the notebook's green/red diff convention.
GREEN = "\033[32m"
RED = "\033[31m"
GRAY = "\033[90m"
RESET = "\033[0m"


def _strip_bos(ids: list[int], bos_id: int | None) -> list[int]:
    if bos_id is None or not ids or ids[0] != bos_id:
        return ids
    return ids[1:]


def _coloured_diff(gt_ids: list[int], gen_ids: list[int], tokenizer) -> str:
    """Render gen_ids token-by-token; colour each token by whether it matches gt at the same index."""
    parts: list[str] = []
    for i, tid in enumerate(gen_ids):
        piece = tokenizer.decode([tid]).replace("\n", "⏎")
        if i >= len(gt_ids):
            colour = GRAY  # past the original span -- free continuation
        elif tid == gt_ids[i]:
            colour = GREEN
        else:
            colour = RED
        parts.append(f"{colour}{piece}{RESET}")
    return "".join(parts)


def _release_model(model) -> None:
    """Drop a loaded model from GPU memory so the next checkpoint fits."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo_id", default=DEFAULT_REPO, help="HF dataset id to load (ignored if --local_path is set).")
    ap.add_argument(
        "--local_path", default=None,
        help="Inspect a local artifact dir produced by build_demo_gallery (--out_dir) "
             "instead of fetching from the Hub. Useful to verify a freshly trained run "
             "before pushing.",
    )
    ap.add_argument(
        "--model", default=None,
        help="Frozen model checkpoint. If omitted, the script auto-switches by "
             "row['model_checkpoint'] and reconstructs every row. If set, only "
             "rows matching this checkpoint are decoded.",
    )
    ap.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
        help="Inference dtype (float16 = T4/Colab; bfloat16 = A100/H100; float32 = CPU).",
    )
    ap.add_argument(
        "--extra_tokens", type=int, default=4,
        help="Decode `num_tokens + extra_tokens` per row so we also see the free continuation past the original.",
    )
    args = ap.parse_args()

    if args.local_path:
        print(f"Loading gallery from local artifact: {args.local_path}")
        ds = load_from_disk(args.local_path)
    else:
        print(f"Loading gallery from Hub: {args.repo_id}")
        ds = load_dataset(args.repo_id, split="train")
    print(f"  {len(ds)} rows")
    if args.model is not None:
        ds = ds.filter(lambda r: r["model_checkpoint"] == args.model)
        print(f"  --model filter: {len(ds)} rows match {args.model!r}")
    print()
    print("=" * 80)

    # Cache the most-recently loaded model; swap only when the row needs a
    # different checkpoint. With Llama-1B (2.5 GB) + SmolLM-360M (~700 MB),
    # holding one at a time keeps T4 VRAM headroom comfortable.
    cur_ckpt: str | None = None
    model = None
    tokenizer = None

    for r in ds:
        needed_ckpt = r["model_checkpoint"]
        if needed_ckpt != cur_ckpt:
            if model is not None:
                _release_model(model)
                model = None
            print(f"\nLoading frozen model: {needed_ckpt}  (dtype={args.dtype})")
            model, tokenizer = load_frozen_model(needed_ckpt, dtype=args.dtype)
            device = next(model.parameters()).device
            print(f"  device: {device}  hidden_size: {model.config.hidden_size}")
            cur_ckpt = needed_ckpt

        # Embedding is stored as a 2D nested list of float32; torch.tensor() gives [n_cram, hidden].
        emb = torch.tensor(r["embedding"], dtype=torch.float32)
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)  # [1, n_cram, hidden]

        # Greedy decode from the embedding alone, no text input. Pull the RAW
        # generated_ids straight from the generator -- the same path the defense
        # animation pipeline uses (compression_horizon/scripts/defense_demo/
        # generate_export.py). Decode -> re-tokenise round-trip used to lose 1-2
        # IDs to BPE whitespace cleanup, depressing the match metric on edge
        # cases (notably SmolLM at horizon).
        _texts, gen_ids_tensor = generate_from_compression(
            model=model,
            tokenizer=tokenizer,
            compression_token_embeddings=emb.to(next(model.parameters()).device),
            max_new_tokens=r["num_tokens"] + args.extra_tokens,
            num_return_sequences=1,
            return_generated_ids=True,
        )
        gen_ids = _strip_bos(gen_ids_tensor[0].cpu().tolist(), tokenizer.bos_token_id)
        # Re-render the generation for display (skip_special_tokens=True hides the
        # leading BOS in the printed string -- the metric above already ignores it).
        gen_text = tokenizer.decode(gen_ids_tensor[0], skip_special_tokens=True)

        # Ground-truth ids: clip the saved input_ids to the actual span (PC rows
        # may store padding past num_tokens; TC rows store only the real span).
        gt_ids = _strip_bos(list(r["input_ids"][: r["num_tokens"]]), tokenizer.bos_token_id)

        n_compare = min(len(gt_ids), len(gen_ids))
        matches = sum(1 for i in range(n_compare) if gt_ids[i] == gen_ids[i])
        pct = 100.0 * matches / n_compare if n_compare else 0.0

        kind = r["kind"]
        method = r["method"]
        domain = r["domain"]
        title = r["title"]
        print(f"[{kind:7s} {method:22s} {domain:10s}]  {title}")
        print(
            f"  model : {needed_ckpt}  hidden={r['hidden_size']}"
        )
        print(
            f"  stored: tok={r['num_tokens']:3d}  horizon={r['horizon']}  "
            f"conv={r['final_convergence']:.3f}  steps={r['steps_taken']}"
        )
        print(f"  decode match first {n_compare} tokens: {matches}/{n_compare} ({pct:.1f}%)")
        print(f"  original: {r['text']!r}")
        print(f"  decoded : {_coloured_diff(gt_ids, gen_ids, tokenizer)}")
        print("-" * 80)


if __name__ == "__main__":
    main()
