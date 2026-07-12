#!/usr/bin/env python
"""Build (and optionally upload) the demo notebook's pre-computed embedding gallery.

This script runs **only** the package's canonical training entry point --
:func:`progressive_cramming.run.run_training` -- on a handful of inline texts.
No reimplementation of the cramming loop here: trainer selection and the entire
optimisation are owned by the package, so any future fix (e.g. the drift fix in
``FullCrammingTrainer``) propagates automatically.

The dataset has two row kinds, both consumed by the Colab notebook:

* ``kind="gallery"``: 5 progressive-cramming runs on inline texts across distinct
  domains (literature / code / news / poetry / science). Each yields one row -- the
  embedding at the compression horizon -- click ``▶ Reconstruct`` in the notebook.
* ``kind="tc_pc"``: a total-cramming row + a progressive-cramming row on the same
  longer passage, for the side-by-side section.

Side-by-side passage: PG19 sample #7 from ``LarryLovestein/pg19_1k``, the same span
we use end-to-end in the defense / ICML pipeline.

Run on a GPU machine once; the demo notebook then reads the result::

    huggingface-cli login          # or set HF_TOKEN
    python scripts/build_demo_gallery.py --repo_id <user>/progressive_cramming_demo_gallery --push

Inspect locally without uploading::

    python scripts/build_demo_gallery.py --out_dir runs/demo_gallery
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer

from progressive_cramming.run import run_training
from progressive_cramming.train.arguments import MyTrainingArguments

# ── Default model ────────────────────────────────────────────────────────────
DEFAULT_MODEL = "unsloth/Llama-3.2-1B"

# ── Inline gallery texts (5 domains) ─────────────────────────────────────────
# Substantial paragraph, well below Llama-3.2-1B's compression horizon -- PC
# should reach the full span on every example with a comfortable step budget.
GALLERY: list[dict] = [
    {
        "domain": "literature",
        "title": "Austen — Pride and Prejudice (opening)",
        # Canonical, very familiar to any English LM -- a safer choice than a
        # less-frequent PG19 fragment we'd already happily showcased before.
        "text": (
            "It is a truth universally acknowledged, that a single man in possession "
            "of a good fortune, must be in want of a wife. However little known the "
            "feelings or views of such a man may be on his first entering a "
            "neighbourhood, this truth is so well fixed in the minds of the "
            "surrounding families."
        ),
    },
    {
        "domain": "code",
        "title": "Python: quicksort",
        "text": (
            "def quicksort(items):\n"
            "    if len(items) <= 1:\n"
            "        return items\n"
            "    pivot = items[len(items) // 2]\n"
            "    left = [x for x in items if x < pivot]\n"
            "    middle = [x for x in items if x == pivot]\n"
            "    right = [x for x in items if x > pivot]\n"
            "    return quicksort(left) + middle + quicksort(right)\n"
        ),
    },
    {
        "domain": "news",
        "title": "Newswire report",
        "text": (
            "Researchers reported on Tuesday that a single learned vector can store an "
            "entire paragraph of text, which a frozen language model then reconstructs "
            "token by token. The team said the method needs no retraining and could "
            "shed light on how transformers pack information into their representations."
        ),
    },
    {
        "domain": "poetry",
        "title": "Frost — The Road Not Taken (final stanza)",
        "text": (
            "I shall be telling this with a sigh\n"
            "Somewhere ages and ages hence:\n"
            "Two roads diverged in a wood, and I—\n"
            "I took the one less traveled by,\n"
            "And that has made all the difference."
        ),
    },
    {
        "domain": "science",
        "title": "The second law of thermodynamics",
        "text": (
            "The second law of thermodynamics states that the entropy of an isolated "
            "system never decreases over time. It remains constant only in an idealized "
            "reversible process and increases in every real, irreversible one, which "
            "sets a fundamental direction for the flow of time."
        ),
    },
]

# ── Side-by-side passage: PG19 sample #7 ─────────────────────────────────────
# Same span the defense demo / Llama gallery / generate_export pipeline use.
PAIR_DATASET = "LarryLovestein/pg19_1k"
PAIR_INDEX = 7
PAIR_DOMAIN = "literature"
PAIR_TITLE = "PG19 — sample #7 (travellers)"

# ── Side-by-side passage #2: SmolLM2-360M, 32 tokens ─────────────────────────
# Mirrors the defense-animation parameters
# (compression_horizon/artifacts/defense_demo/animation_data.json): TC reaches
# ~0.969 teacher-forced accuracy under threshold=0.95 (1 residual error out of
# 32), produces the first ~6 tokens correctly under greedy decoding, then
# cascades into a complete semantic drift. PC reaches 32/32. Same source text
# as the Llama pair (PG19 #7), just truncated -- so the side-by-side compares
# two failure modes (catastrophic early vs slow mid-sequence) on the same
# passage, not two different texts.
SMOLLM_DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"
SMOLLM_PAIR_MAX_SEQ_LEN = 32
SMOLLM_TC_THRESHOLD = 0.95
SMOLLM_LR = 0.01  # SmolLM family in run_jobs_progressive.py MODEL_CONFIGS


# ─────────────────────────────────────────────────────────────────────────────
# Single-text dataset construction
# ─────────────────────────────────────────────────────────────────────────────


def _tokenize_one_text(tokenizer, text: str, max_seq_len: int) -> Dataset:
    """Tokenize one text into a one-row HF Dataset that the trainers accept directly.

    Mirrors the layout produced by the package's
    :func:`progressive_cramming.data.tokenization.load_or_create_tokenized_dataset`
    so we can feed it straight to ``run_training(..., train_dataset=...)``.
    """
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
        add_special_tokens=True,
    )
    return Dataset.from_dict(
        {
            "input_ids": [enc["input_ids"]],
            "attention_mask": [enc["attention_mask"]],
        }
    ).with_format("torch")


def _load_pg19_sample(tokenizer, index: int, max_seq_len: int) -> Dataset:
    """Load one PG19 sample by index from the canonical HuggingFace dataset and
    tokenize it the same way :func:`_tokenize_one_text` does. Keeps the
    side-by-side passage byte-identical to the rest of the pipeline."""
    ds = load_dataset(PAIR_DATASET, split="train")
    return _tokenize_one_text(tokenizer, ds[index]["text"], max_seq_len)


# ─────────────────────────────────────────────────────────────────────────────
# Classical hyperparameter recipes
# ─────────────────────────────────────────────────────────────────────────────
# Match the paper's progressive-cramming protocol (Appendix A) and the
# repo's reproduction scripts (scripts/thesis_reproduction/experiments/progressive/*.sh):
#   - random0.02 init, lr=0.1 (Llama family), warmup=100, AdamW β1=β2=0.9, wd=0.01
#   - cross-entropy loss only (no alignment, no low-dim)
#   - progressive_min_seq_len=1, progressive_step=1 (token-by-token), threshold=1.0
#   - max_optimization_steps_per_token=1000  /  per_sample=10000


def _shared_kwargs(model_ckpt: str, max_seq_len: int, *,
                   learning_rate: float = 0.1, random_seed: int = 42) -> dict:
    """Fields shared by every cramming-arg recipe we build.

    ``learning_rate`` defaults to 0.1 (Llama-family value used by run_jobs_progressive.py);
    the SmolLM family in the same MODEL_CONFIGS table uses 0.01, so pass that
    explicitly when building SmolLM recipes. ``random_seed`` controls the trainer's
    init RNG -- override for SmolLM probes to explore different TC drift modes.
    """
    return dict(
        model_checkpoint=model_ckpt,
        # dataset_name is unused -- we always inject `train_dataset` ourselves.
        dataset_name="",
        max_sequence_length=max_seq_len,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        number_of_mem_tokens=1,
        embedding_init_method="random0.02",
        loss_type="cross_entropy",
        learning_rate=learning_rate,
        warmup_steps=100,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": 1e-3},
        # Train in fp32 (not bf16) so the convergence boundary is honest: an
        # embedding that passes threshold=1.0 in fp32 has a margin two orders of
        # magnitude wider than bf16 rounding noise, which means greedy is robust
        # under any downstream inference precision (bf16/fp16/fp32). Worth the
        # ~2x training-time cost for a once-off demo dataset; the Colab notebook
        # can then decode in fp16 on a T4 without precision surprises.
        dtype="float32",
        attn_implementation="eager",
        random_seed=random_seed,
        report_to=[],
    )


def progressive_args(model_ckpt: str, max_seq_len: int, output_dir: str,
                     *, learning_rate: float = 0.1,
                     random_seed: int = 42) -> MyTrainingArguments:
    """Classical progressive cramming: step=1, threshold=1.0 (paper Appendix A).

    ``max_optimization_steps_per_token`` is doubled vs the paper's 1000 because a
    single demo run cannot tolerate flaky stages: a stage that gets warm-started
    into a local minimum needs the extra budget to escape. The paper's 1000 is the
    statistical-protocol default, not a hard upper bound -- non-progressive
    variants of cramming routinely use >=2000 steps per sample.
    """
    return MyTrainingArguments(
        output_dir=output_dir,
        progressive_train=True,
        progressive_min_seq_len=1,
        progressive_step=1,
        progressive_convergence_threshold=1.0,
        max_optimization_steps_per_token=2000,
        max_optimization_steps_per_sample=20000,
        **_shared_kwargs(model_ckpt, max_seq_len,
                         learning_rate=learning_rate, random_seed=random_seed),
    )


def full_args(
    model_ckpt: str,
    max_seq_len: int,
    output_dir: str,
    *,
    convergence_threshold: float,
    learning_rate: float = 0.1,
    random_seed: int = 42,
) -> MyTrainingArguments:
    """Classical full (total) cramming with a configurable stop threshold.

    The TC side of the demo pair uses ``convergence_threshold=0.99`` -- the paper's
    nominal protocol. On a 128-token passage this permits at most 1 residual error
    (127/128=0.9922 satisfies; 126/128=0.984 does not), which is what makes the
    autoregressive cascade visible in the side-by-side section.
    """
    return MyTrainingArguments(
        output_dir=output_dir,
        max_optimization_steps_per_sample=10000,
        full_cramming_convergence_threshold=convergence_threshold,
        **_shared_kwargs(model_ckpt, max_seq_len,
                         learning_rate=learning_rate, random_seed=random_seed),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trainer-row -> gallery-row mapping (single source of truth for the schema)
# ─────────────────────────────────────────────────────────────────────────────


def _common_gallery_row(*, kind, domain, title, method, source, input_ids,
                        horizon, elapsed_s, model_ckpt, training_config) -> dict:
    return {
        "kind": kind,
        "domain": domain,
        "title": title,
        "method": method,
        "text": source["text"],
        "input_ids": input_ids,
        "embedding": source["embedding"],
        "n_cram": int(source["num_compression_tokens"]),
        "num_tokens": int(source["num_input_tokens"]),
        "hidden_size": int(source["hidden_size"]),
        "horizon": horizon,
        "final_convergence": float(source["final_convergence"]),
        "information_gain_bits": float(source["information_gain_bits"]),
        "steps_taken": int(source["steps_taken"]),
        "elapsed_s": float(elapsed_s),
        # Top-level so notebooks / external tools don't have to crack the
        # training_config JSON to reload the frozen model for reconstruction.
        "model_checkpoint": model_ckpt,
        "dtype": str(source.get("dtype", "")),
        "training_config": json.dumps(training_config),
    }


def _tc_steps_taken(source: dict) -> int:
    """FullCrammingTrainer saves ``convergence_after_steps`` (# steps spent below 1.0)
    rather than a single ``steps_taken`` field. Normalise to one name for the schema."""
    return int(source.get("convergence_after_steps", source.get("steps_taken", 0)))


# ─────────────────────────────────────────────────────────────────────────────
# High-level runs
# ─────────────────────────────────────────────────────────────────────────────


def _train_and_load_artifact(args: MyTrainingArguments, train_dataset: Dataset) -> tuple[Dataset, float]:
    """Run ``run_training`` and return the saved artifact loaded as a Dataset, plus elapsed seconds."""
    t0 = time.time()
    artifact = run_training(args, train_dataset=train_dataset)
    elapsed = time.time() - t0
    if artifact is None:
        raise RuntimeError("run_training returned None (no artifact saved). Check args.output_dir.")
    return load_from_disk(artifact), elapsed


def run_pc(tokenizer, model_ckpt: str, train_dataset: Dataset, max_seq_len: int,
           *, learning_rate: float = 0.1, random_seed: int = 42
           ) -> tuple[dict, int, float, MyTrainingArguments]:
    """Run classical PC on a pre-tokenised single-sample dataset; return
    (horizon_row, num_stages, elapsed_s, args). The horizon row is the saved
    stage with the largest ``stage_seq_len``."""
    with tempfile.TemporaryDirectory(prefix="pc_") as out:
        args = progressive_args(
            model_ckpt, max_seq_len, out,
            learning_rate=learning_rate, random_seed=random_seed,
        )
        ds, elapsed = _train_and_load_artifact(args, train_dataset)
        rows = sorted(list(ds), key=lambda r: int(r["stage_seq_len"]), reverse=True)
        return rows[0], len(rows), elapsed, args


def run_tc(tokenizer, model_ckpt: str, train_dataset: Dataset, max_seq_len: int,
           threshold: float, *, learning_rate: float = 0.1, random_seed: int = 42
           ) -> tuple[dict, float, MyTrainingArguments]:
    """Run classical TC on a pre-tokenised single-sample dataset."""
    with tempfile.TemporaryDirectory(prefix="tc_") as out:
        args = full_args(
            model_ckpt, max_seq_len, out,
            convergence_threshold=threshold,
            learning_rate=learning_rate, random_seed=random_seed,
        )
        ds, elapsed = _train_and_load_artifact(args, train_dataset)
        return list(ds)[0], elapsed, args


def build_pc_row(*, kind, item, source, num_stages, elapsed_s, input_ids,
                 model_ckpt, max_seq_len) -> dict:
    """Build a gallery-dataset row from a PC trainer artifact row."""
    horizon = int(source["stage_seq_len"])
    return _common_gallery_row(
        kind=kind, domain=item["domain"], title=item["title"],
        method="progressive_cramming",
        source=source,
        input_ids=input_ids,
        horizon=horizon,
        elapsed_s=elapsed_s,
        model_ckpt=model_ckpt,
        training_config={
            "method": "progressive_cramming",
            "model_checkpoint": model_ckpt,
            "num_mem_tokens": int(source["num_compression_tokens"]),
            "num_tokens": int(source["num_input_tokens"]),
            "hidden_size": int(source["hidden_size"]),
            "horizon": horizon,
            "num_stages": num_stages,
            "max_sequence_length": max_seq_len,
            "progressive_step": 1,
            "convergence_threshold": 1.0,
        },
    )


def build_tc_row(*, kind, item, source, threshold, elapsed_s,
                 model_ckpt, max_seq_len) -> dict:
    """Build a gallery-dataset row from a TC trainer artifact row."""
    # FullCrammingTrainer saves convergence_after_steps; normalise to "steps_taken".
    src = dict(source)
    src["steps_taken"] = _tc_steps_taken(source)
    return _common_gallery_row(
        kind=kind, domain=item["domain"], title=item["title"],
        method="full_cramming",
        source=src,
        input_ids=source["input_ids"],
        horizon=None,
        elapsed_s=elapsed_s,
        model_ckpt=model_ckpt,
        training_config={
            "method": "full_cramming",
            "model_checkpoint": model_ckpt,
            "num_mem_tokens": int(source["num_compression_tokens"]),
            "num_tokens": int(source["num_input_tokens"]),
            "hidden_size": int(source["hidden_size"]),
            "convergence_threshold": threshold,
            "max_sequence_length": max_seq_len,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


def build_gallery_rows(tokenizer, args) -> list[dict]:
    """Classical PC on each of the 5 inline gallery texts; return one row per text."""
    rows: list[dict] = []
    for i, item in enumerate(GALLERY):
        print(f"\n[gallery {i+1}/{len(GALLERY)}] {item['domain']}: {item['title']}")
        ds = _tokenize_one_text(tokenizer, item["text"], args.gallery_max_seq_len)
        # The tokenised input_ids carry the BOS prefix used at training time; pull
        # them out for the saved row (PC trainer doesn't persist them itself).
        input_ids = ds[0]["input_ids"].tolist()
        source, num_stages, elapsed, _targs = run_pc(
            tokenizer, args.model, ds, args.gallery_max_seq_len
        )
        print(
            f"  PC: horizon={source['stage_seq_len']}/{source['num_input_tokens']} tokens, "
            f"{num_stages} stages, {source['steps_taken']} steps, "
            f"{elapsed:.1f}s, reconstruction={float(source['final_convergence']):.3f}, "
            f"info_gain={float(source['information_gain_bits']):.1f} bits"
        )
        rows.append(
            build_pc_row(
                kind="gallery", item=item, source=source, num_stages=num_stages,
                elapsed_s=elapsed, input_ids=input_ids,
                model_ckpt=args.model, max_seq_len=args.gallery_max_seq_len,
            )
        )
    return rows


def build_tc_pc_rows(tokenizer, args) -> list[dict]:
    """Side-by-side: TC + PC on PG19 sample #7."""
    item = {"domain": PAIR_DOMAIN, "title": PAIR_TITLE}
    print(f"\n[tc_pc] using {PAIR_DATASET}[{PAIR_INDEX}] for the side-by-side pair")
    ds = _load_pg19_sample(tokenizer, PAIR_INDEX, args.pair_max_seq_len)
    input_ids = ds[0]["input_ids"].tolist()

    print("[tc_pc] full cramming...")
    tc_source, tc_elapsed, _ = run_tc(
        tokenizer, args.model, ds, args.pair_max_seq_len, args.tc_convergence_threshold
    )
    print(
        f"  TC: reconstruction={float(tc_source['final_convergence']):.3f} "
        f"(threshold={args.tc_convergence_threshold}), "
        f"steps={_tc_steps_taken(tc_source)}, {tc_elapsed:.1f}s"
    )

    print("[tc_pc] progressive cramming...")
    pc_source, pc_stages, pc_elapsed, _ = run_pc(
        tokenizer, args.model, ds, args.pair_max_seq_len
    )
    print(
        f"  PC: horizon={pc_source['stage_seq_len']}/{pc_source['num_input_tokens']} tokens, "
        f"{pc_stages} stages, {pc_elapsed:.1f}s"
    )

    return [
        build_tc_row(
            kind="tc_pc",
            item={**item, "title": f"{PAIR_TITLE} — full cramming"},
            source=tc_source, threshold=args.tc_convergence_threshold,
            elapsed_s=tc_elapsed,
            model_ckpt=args.model, max_seq_len=args.pair_max_seq_len,
        ),
        build_pc_row(
            kind="tc_pc",
            item={**item, "title": f"{PAIR_TITLE} — progressive cramming"},
            source=pc_source, num_stages=pc_stages,
            elapsed_s=pc_elapsed, input_ids=input_ids,
            model_ckpt=args.model, max_seq_len=args.pair_max_seq_len,
        ),
    ]


def build_smollm_tc_pc_rows(args) -> list[dict]:
    """Second side-by-side: SmolLM2-360M on PG19 #7 trimmed to 32 tokens.

    Shows the *gradual* TC failure mode (~6 correct tokens, then cascading drift)
    that the Llama-1B pair masks behind its catastrophic first-token miss. Same
    source text as the Llama pair, so the notebook can put them next to each
    other and contrast the two failure modes directly.

    Loads its own tokenizer (SmolLM uses a different vocabulary than Llama).
    """
    # SmolLM has its own BPE vocabulary -- can't reuse the Llama tokenizer.
    smollm_tokenizer = AutoTokenizer.from_pretrained(args.smollm_model)
    smollm_tokenizer.pad_token = smollm_tokenizer.eos_token
    smollm_tokenizer.padding_side = "right"

    item = {"domain": PAIR_DOMAIN, "title": PAIR_TITLE}
    print(f"\n[tc_pc smollm] using {PAIR_DATASET}[{PAIR_INDEX}] truncated to "
          f"{args.smollm_pair_max_seq_len} tokens with {args.smollm_model}")
    ds = _load_pg19_sample(smollm_tokenizer, PAIR_INDEX, args.smollm_pair_max_seq_len)
    input_ids = ds[0]["input_ids"].tolist()

    print(f"[tc_pc smollm] random_seed={args.smollm_seed}")
    print("[tc_pc smollm] full cramming...")
    tc_source, tc_elapsed, _ = run_tc(
        smollm_tokenizer, args.smollm_model, ds,
        args.smollm_pair_max_seq_len, args.smollm_tc_threshold,
        learning_rate=SMOLLM_LR, random_seed=args.smollm_seed,
    )
    print(
        f"  TC: reconstruction={float(tc_source['final_convergence']):.3f} "
        f"(threshold={args.smollm_tc_threshold}), "
        f"steps={_tc_steps_taken(tc_source)}, {tc_elapsed:.1f}s"
    )

    print("[tc_pc smollm] progressive cramming...")
    pc_source, pc_stages, pc_elapsed, _ = run_pc(
        smollm_tokenizer, args.smollm_model, ds, args.smollm_pair_max_seq_len,
        learning_rate=SMOLLM_LR, random_seed=args.smollm_seed,
    )
    print(
        f"  PC: horizon={pc_source['stage_seq_len']}/{pc_source['num_input_tokens']} tokens, "
        f"{pc_stages} stages, {pc_elapsed:.1f}s"
    )

    suffix = f" — SmolLM2-360M, {args.smollm_pair_max_seq_len} tok"
    return [
        build_tc_row(
            kind="tc_pc",
            item={**item, "title": f"{PAIR_TITLE}{suffix} — full cramming"},
            source=tc_source, threshold=args.smollm_tc_threshold,
            elapsed_s=tc_elapsed,
            model_ckpt=args.smollm_model, max_seq_len=args.smollm_pair_max_seq_len,
        ),
        build_pc_row(
            kind="tc_pc",
            item={**item, "title": f"{PAIR_TITLE}{suffix} — progressive cramming"},
            source=pc_source, num_stages=pc_stages,
            elapsed_s=pc_elapsed, input_ids=input_ids,
            model_ckpt=args.smollm_model, max_seq_len=args.smollm_pair_max_seq_len,
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--gallery_max_seq_len", type=int, default=96,
        help="Token cap per gallery span (kept generous; every text is well under it).",
    )
    ap.add_argument(
        "--pair_max_seq_len", type=int, default=128,
        help="Token cap for the side-by-side passage. 128 is the smallest length at "
             "which the paper's 0.99 threshold permits >0 residual errors "
             "(127/128 = 0.992 ≥ 0.99 > 126/128 = 0.984), guaranteeing the TC "
             "cascade is visible.",
    )
    ap.add_argument(
        "--tc_convergence_threshold", type=float, default=0.99,
        help="TC stop threshold for the side-by-side pair (paper's nominal protocol).",
    )
    # ── Second side-by-side: SmolLM2-360M @ 32 tokens, TC@0.95 (defense-demo recipe).
    ap.add_argument("--smollm_model", default=SMOLLM_DEFAULT_MODEL)
    ap.add_argument(
        "--smollm_pair_max_seq_len", type=int, default=SMOLLM_PAIR_MAX_SEQ_LEN,
        help="Length for the SmolLM pair. 32 matches the defense animation: under "
             "TC@0.95 final_conv ≈ 0.969 (31/32 = 1 residual error), and greedy "
             "decoding produces the first ~6 tokens correctly before cascading.",
    )
    ap.add_argument(
        "--smollm_tc_threshold", type=float, default=SMOLLM_TC_THRESHOLD,
        help="TC stop threshold for the SmolLM pair (paper's protocol for short spans).",
    )
    ap.add_argument(
        "--smollm_seed", type=int, default=42,
        help="random_seed for the SmolLM TC/PC pair only. Defense-style "
             "'first few correct then drift' is not reproducible deterministically "
             "across CUDA non-determinism, so use this to probe seeds (e.g. 7, 13, "
             "100) until the TC drift mode is visually pleasant. Llama gallery + "
             "Llama pair always use seed=42.",
    )
    # ── Skip-flags for fast iteration (e.g. probing SmolLM seeds without
    # rerunning the full Llama gallery, which takes ~2 min).
    ap.add_argument(
        "--skip_gallery", action="store_true",
        help="Skip the 5 Llama gallery PC runs (debug shortcut).",
    )
    ap.add_argument(
        "--skip_llama_pair", action="store_true",
        help="Skip the Llama-1B side-by-side pair (debug shortcut).",
    )
    ap.add_argument(
        "--skip_smollm_pair", action="store_true",
        help="Skip the second SmolLM2-360M side-by-side pair (debug shortcut).",
    )
    ap.add_argument("--out_dir", default="runs/demo_gallery")
    ap.add_argument("--repo_id", default=None, help="HF Hub dataset id to push to (with --push).")
    ap.add_argument("--push", action="store_true", help="Push the dataset to the Hub.")
    ap.add_argument("--private", action="store_true", help="Create the Hub dataset as private.")
    args = ap.parse_args()

    if args.push and not args.repo_id:
        ap.error("--push requires --repo_id")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | model: {args.model}")
    if device == "cpu":
        print("WARNING: no CUDA detected -- cramming Llama-3.2-1B on CPU is unusably slow.")

    # The tokenizer is needed both to tokenize inline texts and for ``run_training``
    # to load its own copy (we reuse the AutoTokenizer cache).
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    rows: list[dict] = []
    if not args.skip_gallery:
        rows.extend(build_gallery_rows(tokenizer, args))
    if not args.skip_llama_pair:
        rows.extend(build_tc_pc_rows(tokenizer, args))
    if not args.skip_smollm_pair:
        rows.extend(build_smollm_tc_pc_rows(args))

    dataset = Dataset.from_list(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    dataset.save_to_disk(args.out_dir)
    print(f"\nSaved {len(dataset)} rows to {args.out_dir}")

    if args.push:
        print(f"Pushing to https://huggingface.co/datasets/{args.repo_id} ...")
        dataset.push_to_hub(args.repo_id, private=args.private)
        print("Done.")
    else:
        print("Not pushing (pass --push --repo_id <id> to upload).")


if __name__ == "__main__":
    main()
