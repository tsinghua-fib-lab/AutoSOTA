#!/usr/bin/env python
"""Inspect a sweep of SmolLM TC/PC probes built by ``build_demo_gallery.py``.

Loads the SmolLM model **once**, then iterates over a set of probe artifact
directories (one per ``--smollm_seed``) and greedily decodes the TC and PC
embeddings stored there. Prints, per seed:

* TC: how many of the first 32 tokens match GT, where the first wrong token is,
  and the decoded continuation (so you can read whether the drift is the
  defense-style "first ~6 correct → coherent rephrase" or a pathological
  ``\n``-loop).
* PC: same fields for the progressive embedding.

Typical use:

    # 1. Build probes (one per seed)
    for seed in 7 13 100 314 2718; do
        rm -rf "runs/probe_seed_$seed"
        .venv/bin/python scripts/build_demo_gallery.py \\
            --skip_gallery --skip_llama_pair --smollm_seed "$seed" \\
            --out_dir "runs/probe_seed_$seed"
    done

    # 2. Inspect them all in one shot
    .venv/bin/python scripts/inspect_smollm_probes.py \\
        --runs runs/probe_seed_7 runs/probe_seed_13 runs/probe_seed_100 \\
              runs/probe_seed_314 runs/probe_seed_2718
"""

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_from_disk

from progressive_cramming.demo import load_frozen_model
from progressive_cramming.inference.generation import generate_from_compression


def _strip_bos(ids: list[int], bos_id: int | None) -> list[int]:
    if bos_id is None or not ids or ids[0] != bos_id:
        return ids
    return ids[1:]


def _decode_embedding(model, tokenizer, embedding_flat, *, max_new_tokens: int) -> list[int]:
    """Greedy decode raw IDs from one compression embedding row."""
    emb = torch.tensor(embedding_flat, dtype=torch.float32)
    if emb.dim() == 2:
        emb = emb.unsqueeze(0)  # [1, n_cram, hidden]
    emb = emb.to(next(model.parameters()).device)
    _texts, gen_ids = generate_from_compression(
        model=model,
        tokenizer=tokenizer,
        compression_token_embeddings=emb,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        return_generated_ids=True,
    )
    return _strip_bos(gen_ids[0].cpu().tolist(), tokenizer.bos_token_id)


def _summarise(label: str, gt: list[int], gen: list[int], *, row: dict,
               target_len: int, tokenizer) -> None:
    n = min(target_len, len(gt), len(gen))
    matches = sum(1 for i in range(n) if gt[i] == gen[i])
    first_wrong = next((i for i in range(n) if gt[i] != gen[i]), None)
    # `horizon` is None for TC rows; show num_tokens as the saved span there.
    stored_horizon = row.get("horizon")
    stored_horizon_str = f"horizon={stored_horizon}" if stored_horizon is not None else "horizon=N/A"
    print(
        f"  {label}: greedy_match={matches}/{n}  first_wrong=pos {first_wrong}  "
        f"|  stored: {stored_horizon_str} "
        f"num_tokens={row['num_tokens']} "
        f"conv={float(row['final_convergence']):.3f} "
        f"steps={int(row['steps_taken'])}"
    )
    print(f"  {label} decoded: {tokenizer.decode(gen, skip_special_tokens=True)!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="Probe directories (one per seed).")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--target_seq_len", type=int, default=32, help="GT prefix length to score against.")
    ap.add_argument("--extra_tokens", type=int, default=8, help="Decode target_seq_len + extra to see drift past horizon.")
    args = ap.parse_args()

    print(f"Loading frozen model: {args.model}  (dtype={args.dtype})")
    model, tokenizer = load_frozen_model(args.model, dtype=args.dtype)
    print(f"  device: {next(model.parameters()).device}  hidden_size: {model.config.hidden_size}")
    max_new = args.target_seq_len + args.extra_tokens

    for run_dir in args.runs:
        seed_label = os.path.basename(run_dir.rstrip("/"))
        print()
        print(f"=================== {seed_label} ===================")
        if not os.path.isdir(run_dir):
            print(f"  (skipped: not a directory)")
            continue
        ds = load_from_disk(run_dir)
        tc = next((r for r in ds if r["method"] == "full_cramming"), None)
        pc = next((r for r in ds if r["method"] == "progressive_cramming"), None)
        if tc is None or pc is None:
            print(f"  (skipped: missing TC or PC row in {run_dir})")
            continue

        gt = _strip_bos(list(tc["input_ids"]), tokenizer.bos_token_id)

        tc_gen = _decode_embedding(model, tokenizer, tc["embedding"], max_new_tokens=max_new)
        _summarise("TC", gt, tc_gen, row=tc, target_len=args.target_seq_len, tokenizer=tokenizer)

        pc_gen = _decode_embedding(model, tokenizer, pc["embedding"], max_new_tokens=max_new)
        _summarise("PC", gt, pc_gen, row=pc, target_len=args.target_seq_len, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
