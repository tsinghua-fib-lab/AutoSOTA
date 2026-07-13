#!/usr/bin/env python3
"""
Run vLLM inference for R2-Router's RouterArena predictions.

Deploys one LLM at a time via vLLM, processes all its queries, then moves to next.
Order: largest → smallest (Qwen3-235B → Qwen3-80B → Qwen3-30B).

All entries use unlimited budget with a concise-answer prompt suffix.

Usage:
    # Run all 4 LLMs sequentially
    sbatch scripts/inference_routerarena.sbatch

    # Run specific model only
    .venv/bin/python scripts/inference_routerarena.py --model "qwen/qwen3-235b-a22b-2507"
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Config
# ============================================================================

ROUTERARENA_DIR = "/home/ah872032.ucf/jiaqi/RouterArena"
PREDICTION_FILE = os.path.join(ROUTERARENA_DIR, "router_inference/predictions/r2-router.json")

# Concise prompt: both system prompt AND user message suffix for maximum effect
CONCISE_SYSTEM_PROMPT = "Be concise, directly give the answer follow the format WITHOUT ANY explanation."
CONCISE_USER_SUFFIX = " Be concise, directly give the answer follow the format WITHOUT ANY explanation."

# LLMs to process, from largest to smallest
# (RouterArena name, HuggingFace model ID, tensor_parallel_size, max_model_len)
INFERENCE_ORDER = [
    ("qwen/qwen3-235b-a22b-2507", "Qwen/Qwen3-235B-A22B-Instruct-2507", 4, 32768),
    ("qwen/qwen3-coder-next", "Qwen/Qwen3-Coder-Next", 2, 32768),
    ("qwen/qwen3-next-80b-a3b-instruct", "Qwen/Qwen3-Next-80B-A3B-Instruct", 2, 32768),
    ("qwen/qwen3-30b-a3b-instruct-2507", "Qwen/Qwen3-30B-A3B-Instruct-2507", 1, 32768),
    ("qwen/qwen3-coder-30b-a3b-instruct", "Qwen/Qwen3-Coder-30B-A3B-Instruct", 1, 32768),
    ("mistralai/Ministral-3-14B-Instruct-2512", "mistralai/Ministral-3-14B-Instruct-2512", 1, 32768),
    ("mistralai/Ministral-3-8B-Instruct-2512", "mistralai/Ministral-3-8B-Instruct-2512", 1, 32768),
    ("mistralai/Ministral-3-3B-Instruct-2512", "mistralai/Ministral-3-3B-Instruct-2512", 1, 32768),
]

# Pool of LLMs (for filtering optimality entries)
POOL_MODELS = {ra_name for ra_name, _, _, _ in INFERENCE_ORDER}

# Per-model sampling params (Qwen3-Coder uses different best practices)
MODEL_SAMPLING_PARAMS = {
    "qwen/qwen3-coder-next": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_tokens": 2048,
        "repetition_penalty": 1.05,
    },
    "mistralai/Ministral-3-14B-Instruct-2512": {
        "temperature": 0.1,
        "max_tokens": 2048,
    },
    "mistralai/Ministral-3-8B-Instruct-2512": {
        "temperature": 0.1,
        "max_tokens": 2048,
    },
    "mistralai/Ministral-3-3B-Instruct-2512": {
        "temperature": 0.1,
        "max_tokens": 2048,
    },
}
# Default: Qwen3 recommended settings
DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": 2048,
    "repetition_penalty": 1.05,
}


# ============================================================================
# Chat message construction
# ============================================================================

def build_chat_messages(original_prompt: str, use_concise: bool = True):
    """Build chat messages for vLLM chat() API.

    Uses BOTH system prompt AND user message suffix for concise instruction.
    Qwen3 chat template will wrap these with <|im_start|>system/user/assistant tokens.
    """
    messages = []
    if use_concise:
        messages.append({"role": "system", "content": CONCISE_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": original_prompt + CONCISE_USER_SUFFIX})
    else:
        messages.append({"role": "user", "content": original_prompt})
    return messages


# ============================================================================
# vLLM inference for one model
# ============================================================================

def run_inference_for_model(
    ra_model_name: str,
    hf_model_id: str,
    tp_size: int,
    max_model_len: int,
    predictions: List[Dict],
    prediction_file: str,
    batch_size: int = 64,
    all_queries: bool = False,
    use_concise_prompt: bool = True,
):
    """Deploy vLLM for one model and run inference on its entries."""
    from vllm import LLM, SamplingParams

    # Collect entries for this model
    entries_to_run = []
    for i, entry in enumerate(predictions):
        # For baseline mode (all_queries), process ALL entries regardless of prediction
        if not all_queries and entry["prediction"] != ra_model_name:
            continue
        # Skip if already has successful result
        gr = entry.get("generated_result")
        if gr is not None and gr.get("success"):
            continue
        # Skip optimality entries for models not in our pool (unless all_queries mode)
        if not all_queries and entry.get("for_optimality") and ra_model_name not in POOL_MODELS:
            continue
        entries_to_run.append(i)

    if not entries_to_run:
        log(f"  No entries to process for {ra_model_name}. Skipping.")
        return 0

    # Count regular vs optimality
    n_regular = sum(1 for i in entries_to_run if not predictions[i].get("for_optimality"))
    n_optimality = sum(1 for i in entries_to_run if predictions[i].get("for_optimality"))
    log(f"  Entries: {len(entries_to_run)} ({n_regular} regular + {n_optimality} optimality)")

    # Build chat messages for each entry
    messages_list = []
    first_regular_logged = False
    for i in entries_to_run:
        entry = predictions[i]
        original_prompt = entry["prompt"]
        if not use_concise_prompt:
            # Baseline mode: no system prompt
            msgs = build_chat_messages(original_prompt, use_concise=False)
        elif entry.get("for_optimality"):
            # Optimality: no system prompt, original prompt as-is
            msgs = build_chat_messages(original_prompt, use_concise=False)
        else:
            # Regular: system prompt with concise instruction
            msgs = build_chat_messages(original_prompt, use_concise=True)
            # Log the first regular entry for verification
            if not first_regular_logged:
                log(f"  [VERIFY] First regular entry global_index: {entry.get('global index')}")
                log(f"  [VERIFY] Using chat API with system prompt: {repr(CONCISE_SYSTEM_PROMPT)}")
                log(f"  [VERIFY] Messages: {msgs}")
                first_regular_logged = True
        messages_list.append(msgs)

    # Load vLLM
    log(f"  Loading vLLM model: {hf_model_id} (TP={tp_size})...")
    llm = LLM(
        model=hf_model_id,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
    )

    # Sampling params: per-model or default
    sp = MODEL_SAMPLING_PARAMS.get(ra_model_name, DEFAULT_SAMPLING_PARAMS)
    sampling_params = SamplingParams(**sp)

    # Log sampling params for verification
    log(f"  [VERIFY] Sampling params: temperature={sampling_params.temperature}, top_p={sampling_params.top_p}, top_k={sampling_params.top_k}, max_tokens={sampling_params.max_tokens}, repetition_penalty={sampling_params.repetition_penalty}")

    # Run inference in batches, saving after each batch to avoid data loss on timeout
    log(f"  Running chat inference on {len(messages_list)} entries...")
    success_count = 0
    for start in range(0, len(messages_list), batch_size):
        batch = messages_list[start:start + batch_size]
        outputs = llm.chat(batch, sampling_params, use_tqdm=False)

        # Write results immediately after each batch
        for j, output in enumerate(outputs):
            pred_idx = entries_to_run[start + j]
            generated_text = output.outputs[0].text
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)

            predictions[pred_idx]["generated_result"] = {
                "generated_answer": generated_text,
                "success": True,
                "token_usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "provider": "local-vllm",
                "error": None,
            }
            success_count += 1

        done = min(start + batch_size, len(messages_list))
        log(f"    {done}/{len(messages_list)} done")

        # Save to disk after each batch
        with open(prediction_file, "w") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

    log(f"  Completed: {success_count}/{len(entries_to_run)}")

    # Free GPU memory
    del llm
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return success_count


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Only run this model (RouterArena name)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="vLLM batch size (default: 64)")
    parser.add_argument("--max-tp", type=int, default=None,
                        help="Skip models requiring TP > this value")
    parser.add_argument("--prediction-file", type=str, default=PREDICTION_FILE)
    parser.add_argument("--all-queries", action="store_true",
                        help="Run on ALL queries, ignoring prediction field (for baseline)")
    parser.add_argument("--no-concise-prompt", action="store_true",
                        help="Don't append concise prompt suffix (for baseline)")
    args = parser.parse_args()

    log("=== R2-Router RouterArena Inference ===")

    # Load predictions
    log(f"Loading predictions from {args.prediction_file}")
    with open(args.prediction_file, "r") as f:
        predictions = json.load(f)
    log(f"  Total entries: {len(predictions)}")

    # Determine which models to run
    if args.model:
        models_to_run = [(n, h, t, m) for n, h, t, m in INFERENCE_ORDER if n == args.model]
        if not models_to_run:
            log(f"ERROR: Model '{args.model}' not found in INFERENCE_ORDER")
            sys.exit(1)
    else:
        models_to_run = INFERENCE_ORDER

    # Filter by max-tp if specified
    if args.max_tp:
        skipped = [(n, t) for n, _, t, _ in models_to_run if t > args.max_tp]
        if skipped:
            log(f"  Skipping models with TP > {args.max_tp}: {[(n, f'TP={t}') for n, t in skipped]}")
        models_to_run = [(n, h, t, m) for n, h, t, m in models_to_run if t <= args.max_tp]

    # Process each model (largest first)
    total_success = 0
    for ra_name, hf_id, tp_size, max_len in models_to_run:
        log(f"\n{'='*60}")
        log(f"Model: {ra_name} ({hf_id})")
        log(f"{'='*60}")

        success = run_inference_for_model(
            ra_model_name=ra_name,
            hf_model_id=hf_id,
            tp_size=tp_size,
            max_model_len=max_len,
            predictions=predictions,
            prediction_file=args.prediction_file,
            batch_size=args.batch_size,
            all_queries=args.all_queries,
            use_concise_prompt=not args.no_concise_prompt,
        )
        total_success += success

        # Save after each model
        log(f"  Saving predictions...")
        with open(args.prediction_file, "w") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        log(f"  Saved.")

    log(f"\n=== Done! Total successful: {total_success} ===")


if __name__ == "__main__":
    main()
