#!/usr/bin/env python
"""Diagnose whether a saved PC embedding actually delivers its stored ``conv``.

For a given probe dir (one PC row + one TC row), runs **two** match-rate measurements
on the PC embedding:

1. **Teacher-forced**: build the same united_token_embeddings the trainer built --
   ``[compression_emb, embed(input_ids[:N])]`` -- forward once, compute the
   per-position argmax match rate exactly the way the trainer's compute_convergence
   does. This rate is what the trainer claims as ``final_convergence``; if our
   reproduced number disagrees, the saved embedding is NOT the one that passed
   the convergence check (== save-time drift bug).

2. **Greedy autoregressive**: rerun generate_from_compression and count matches
   against GT.

Both numbers are then compared against the row's stored ``final_convergence``.

Possible verdicts:

* stored=1.000, teacher=1.000, greedy=1.000 → everything consistent (the seed
  trained an embedding that passes both regimes).
* stored=1.000, teacher=1.000, greedy<1.000 → save IS consistent; greedy slips on
  numerical edge of the last token (typical when hidden dim is tight for the
  horizon, e.g. SmolLM-360M at 32 tokens).
* stored=1.000, teacher<1.000 → **save bug**: the embedding written to disk is
  not the one the trainer measured as conv=1.0 (extra optimizer step or wrong
  snapshot).
"""

from __future__ import annotations

import argparse

import torch
from datasets import load_from_disk

from progressive_cramming.demo import load_frozen_model
from progressive_cramming.inference.generation import generate_from_compression
from progressive_cramming.train.loss import token_argmax_match_rate_with_prefix


def _strip_bos(ids: list[int], bos_id: int | None) -> list[int]:
    if bos_id is None or not ids or ids[0] != bos_id:
        return ids
    return ids[1:]


@torch.no_grad()
def teacher_forced_match(model, tokenizer, embedding: torch.Tensor,
                         input_ids: list[int]) -> tuple[float, list[bool]]:
    """Return (match_rate, per_position_matches) for ``embedding`` against ``input_ids``
    using the same convergence formula the trainer uses."""
    device = next(model.parameters()).device
    weight_dtype = model.get_input_embeddings().weight.dtype

    ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)        # [1, N]
    attention_mask = torch.ones_like(ids_t)                                    # [1, N]
    seq_embs = model.get_input_embeddings()(ids_t)                             # [1, N, hidden]

    if embedding.dim() == 2:
        embedding = embedding.unsqueeze(0)                                     # [1, mem, hidden]
    embedding = embedding.to(device=device, dtype=weight_dtype)

    united = torch.cat([embedding, seq_embs], dim=1)                            # [1, mem+N, hidden]
    mem = embedding.shape[1]
    united_mask = torch.cat(
        [torch.ones((1, mem), dtype=torch.long, device=device), attention_mask], dim=1
    )

    outputs = model(inputs_embeds=united, attention_mask=united_mask)
    logits = outputs.logits                                                   # [1, mem+N, vocab]

    rate = token_argmax_match_rate_with_prefix(
        logits, ids_t, attention_mask,
        num_compression_tokens=mem, prefix_len=0,
    )[0].item()

    # Per-position breakdown so we can print where the mismatch is.
    prediction_ids = logits[:, mem - 1 : -1].argmax(dim=-1)[0].cpu().tolist()
    target_ids = input_ids
    n = min(len(prediction_ids), len(target_ids))
    per_pos = [prediction_ids[i] == target_ids[i] for i in range(n)]
    return rate, per_pos


@torch.no_grad()
def greedy_match(model, tokenizer, embedding: torch.Tensor,
                 gt_ids: list[int], max_new_tokens: int) -> tuple[float, list[bool], list[int]]:
    device = next(model.parameters()).device
    if embedding.dim() == 2:
        embedding = embedding.unsqueeze(0)
    embedding = embedding.to(device=device, dtype=torch.float32)
    _texts, gen_ids = generate_from_compression(
        model=model, tokenizer=tokenizer,
        compression_token_embeddings=embedding,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1, return_generated_ids=True,
    )
    raw_ids = gen_ids[0].cpu().tolist()
    stripped = _strip_bos(raw_ids, tokenizer.bos_token_id)
    n = min(len(stripped), len(gt_ids))
    per_pos = [stripped[i] == gt_ids[i] for i in range(n)]
    rate = sum(per_pos) / n if n else 0.0
    return rate, per_pos, stripped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="A probe dir produced by build_demo_gallery (1 TC + 1 PC).")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"],
                    help="Inference dtype. Match training dtype to factor out cast effects.")
    args = ap.parse_args()

    ds = load_from_disk(args.run_dir)
    pc = next((r for r in ds if r["method"] == "progressive_cramming"), None)
    if pc is None:
        raise RuntimeError(f"No PC row in {args.run_dir}")

    print(f"Loading frozen model: {args.model}  (dtype={args.dtype})")
    model, tokenizer = load_frozen_model(args.model, dtype=args.dtype)
    print(f"  device: {next(model.parameters()).device}  hidden_size: {model.config.hidden_size}")

    emb = torch.tensor(pc["embedding"], dtype=torch.float32)
    gt_ids = _strip_bos(list(pc["input_ids"][: pc["num_tokens"]]), tokenizer.bos_token_id)

    tf_rate, tf_per_pos = teacher_forced_match(model, tokenizer, emb, list(pc["input_ids"][: pc["num_tokens"]]))
    tf_matches = sum(tf_per_pos)
    tf_first_wrong = next((i for i, m in enumerate(tf_per_pos) if not m), None)

    gr_rate, gr_per_pos, gr_ids = greedy_match(
        model, tokenizer, emb, gt_ids, max_new_tokens=pc["num_tokens"] + 4
    )
    gr_matches = sum(gr_per_pos)
    gr_first_wrong = next((i for i, m in enumerate(gr_per_pos) if not m), None)

    print()
    print("=" * 80)
    print(f"  run_dir       : {args.run_dir}")
    print(f"  title         : {pc['title']}")
    print(f"  stored conv   : {float(pc['final_convergence']):.4f}  (horizon={pc['horizon']} num_tokens={pc['num_tokens']} steps={pc['steps_taken']})")
    print()
    print(f"  teacher-forced match: {tf_matches}/{len(tf_per_pos)} = {tf_rate:.4f}  first_wrong={tf_first_wrong}")
    print(f"  greedy match        : {gr_matches}/{len(gr_per_pos)} = {gr_rate:.4f}  first_wrong={gr_first_wrong}")
    print()
    trainer_dtype = str(pc.get("dtype", "")).strip().lower()
    same_dtype = (trainer_dtype == args.dtype)
    if not trainer_dtype:
        print(f"  note           : row has no 'dtype' field; assuming trainer used the default.")
    elif not same_dtype:
        print(f"  WARNING        : trainer trained in {trainer_dtype!r} but you're reconstructing in {args.dtype!r}.")
        print(f"                   Saved conv was computed in {trainer_dtype}; a {args.dtype} forward can disagree on")
        print(f"                   tokens that sat at the convergence boundary -- not a bug, an arithmetic-precision")
        print(f"                   effect (canonical for PC trained at threshold=1.0).")

    if same_dtype and abs(tf_rate - float(pc["final_convergence"])) < 1e-3:
        print("  verdict[save]  : OK -- teacher-forced reproduces stored conv exactly (same dtype as trainer).")
    elif same_dtype:
        print("  verdict[save]  : MISMATCH inside the same dtype -- this WOULD point to a save bug.")
        print("                   Drift-fix would be the place to look (extra optimizer step past the conv check).")
    else:
        print(f"  verdict[save]  : INCONCLUSIVE -- rerun in --dtype {trainer_dtype} to test save consistency strictly.")

    if tf_matches == len(tf_per_pos) and gr_matches < len(gr_per_pos):
        print("  verdict[greedy]: shape-noise edge -- teacher-forced 100% but greedy loses last token.")
        print("                   Same-dtype forward, but tensor-shape difference (compression+N vs compression+N+1)")
        print("                   slightly perturbs matmul kernel reductions, enough to flip the boundary token.")
    print("=" * 80)


if __name__ == "__main__":
    main()
