#!/usr/bin/env python
"""Toy quickstart: run all three cramming methods on a small model and print metrics.

Each method compresses a few short text samples into learnable *memory embeddings*
and reports its headline metric:

* **Full cramming**      -> mean token-reconstruction accuracy (``final_convergence``,
  1.0 = exact) and mean optimisation steps to converge.
* **Low-dim projection** -> mean reconstruction accuracy with the embedding optimised
  in a rank-``k`` subspace.
* **Progressive cramming** -> the converged *compression horizon*: the longest prefix
  (in tokens) reconstructed exactly, and the number of stages run.

Runs in a few minutes on a single GPU (CPU works but is slower). Results are written
to ``runs/quickstart/quickstart_results.json``.

Usage::

    python examples/quickstart.py                      # defaults below
    python examples/quickstart.py --model HuggingFaceTB/SmolLM2-360M --seq-len 96
    python examples/quickstart.py --methods full progressive
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from datasets import load_from_disk

from progressive_cramming.run import run_training
from progressive_cramming.train.arguments import MyTrainingArguments

OUT_ROOT = os.path.join("runs", "quickstart")


def _fresh_dir(path: str) -> str:
    """Remove any prior output at `path` so the trainer writes into a clean directory."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


def base_kwargs(args) -> dict:
    """Shared MyTrainingArguments fields for every method in the quickstart."""
    return dict(
        model_checkpoint=args.model,
        dataset_name=args.dataset,
        max_sequence_length=args.seq_len,
        limit_dataset_items=args.samples,
        number_of_mem_tokens=1,
        # Paper recipe: scaled-down random init + lr 0.1 converge fast on a small model.
        embedding_init_method="random0.02",
        learning_rate=0.1,
        warmup_steps=100,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-3},
        # float32 everywhere keeps the demo portable (CPU + any GPU) and deterministic.
        dtype="float32",
        attn_implementation="eager",
        random_seed=42,
        per_device_train_batch_size=1,
        report_to=[],
    )


def run_full(args) -> dict:
    out = _fresh_dir(os.path.join(OUT_ROOT, "full"))
    targs = MyTrainingArguments(
        output_dir=out,
        max_optimization_steps_per_sample=args.steps,
        full_cramming_convergence_threshold=1.0,
        **base_kwargs(args),
    )
    artifact = run_training(targs)
    ds = load_from_disk(artifact)
    conv = [r["final_convergence"] for r in ds]
    steps = [r["convergence_after_steps"] for r in ds]
    return {
        "method": "full_cramming",
        "samples": len(ds),
        "mean_reconstruction": sum(conv) / len(conv),
        "mean_steps_to_converge": sum(steps) / len(steps),
    }


def run_lowdim(args) -> dict:
    out = _fresh_dir(os.path.join(OUT_ROOT, "lowdim"))
    targs = MyTrainingArguments(
        output_dir=out,
        low_dim_train=True,
        low_dim_size=args.low_dim_size,
        max_optimization_steps_per_sample=args.steps,
        full_cramming_convergence_threshold=1.0,
        **base_kwargs(args),
    )
    artifact = run_training(targs)
    ds = load_from_disk(artifact)
    conv = [r["final_convergence"] for r in ds]
    return {
        "method": f"low_dim_projection(k={args.low_dim_size})",
        "samples": len(ds),
        "mean_reconstruction": sum(conv) / len(conv),
    }


def run_progressive(args) -> dict:
    out = _fresh_dir(os.path.join(OUT_ROOT, "progressive"))
    threshold = 1.0
    targs = MyTrainingArguments(
        output_dir=out,
        progressive_train=True,
        progressive_min_seq_len=1,
        progressive_step=args.progressive_step,
        progressive_convergence_threshold=threshold,
        max_optimization_steps_per_token=args.progressive_steps_per_token,
        max_optimization_steps_per_sample=args.steps,
        **base_kwargs(args),
    )
    artifact = run_training(targs)
    ds = load_from_disk(artifact)
    # Per sample: horizon = longest prefix reconstructed at/above threshold.
    horizons: dict[int, int] = {}
    stages: dict[int, int] = {}
    for r in ds:
        sid = r["sample_id"]
        stages[sid] = stages.get(sid, 0) + 1
        if r["final_convergence"] is not None and r["final_convergence"] >= threshold:
            horizons[sid] = max(horizons.get(sid, 0), r["stage_seq_len"])
    n = len(stages)
    mean_horizon = sum(horizons.get(s, 0) for s in stages) / n if n else 0.0
    mean_stages = sum(stages.values()) / n if n else 0.0
    return {
        "method": "progressive_cramming",
        "samples": n,
        "mean_converged_horizon_tokens": mean_horizon,
        "max_sequence_length": args.seq_len,
        "mean_stages": mean_stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--dataset", default="LarryLovestein/pg19_1k")
    parser.add_argument("--seq-len", type=int, default=64, dest="seq_len")
    parser.add_argument("--samples", type=int, default=2, help="number of text samples to cram")
    parser.add_argument("--steps", type=int, default=4000, help="max optimisation steps per sample")
    parser.add_argument("--low-dim-size", type=int, default=32, dest="low_dim_size")
    parser.add_argument("--progressive-step", type=int, default=8, dest="progressive_step")
    parser.add_argument("--progressive-steps-per-token", type=int, default=500, dest="progressive_steps_per_token")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["full", "lowdim", "progressive"],
        choices=["full", "lowdim", "progressive"],
    )
    args = parser.parse_args()

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'} | model: {args.model}")
    os.makedirs(OUT_ROOT, exist_ok=True)

    runners = {"full": run_full, "lowdim": run_lowdim, "progressive": run_progressive}
    results = []
    for m in args.methods:
        print(f"\n{'='*70}\nRunning: {m}\n{'='*70}")
        results.append(runners[m](args))

    print(f"\n{'='*70}\nQUICKSTART RESULTS\n{'='*70}")
    for r in results:
        print(json.dumps(r, indent=2))

    out_json = os.path.join(OUT_ROOT, "quickstart_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {out_json}")


if __name__ == "__main__":
    main()
