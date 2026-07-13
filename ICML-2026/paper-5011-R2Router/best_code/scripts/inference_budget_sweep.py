#!/usr/bin/env python3
"""
Token budget sweep: collect LLM responses under different token limits.

Loads model ONCE, then iterates through all budget values {10,20,...,800,concise}.
For each budget N:
  - System prompt includes budget constraint + code format instruction
  - concise: concise prompt + code format instruction

Optimality entries are skipped (no response needed).
Code format instruction is universal — the LLM decides whether to apply it.

Usage:
    .venv/bin/python scripts/inference_budget_sweep.py \
        --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
        --model-short 30b \
        --output-dir /orange/qi855292.ucf/ah872032.ucf/budget_sweep/30b \
        --tp-size 1 \
        --batch-size 256
"""

import os
import json
import argparse
import gc
from datetime import datetime
from typing import Dict, List


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Config
# ============================================================================

TEMPLATE_FILE = "/home/ah872032.ucf/jiaqi/RouterArena/router_inference/predictions/r2-router-clean.json"

DEFAULT_BUDGETS = [10, 20, 40, 80, 150, 200, 400, 800, "concise"]

# Prompts
CONCISE_SYSTEM = "Be concise, directly give the answer follow the format WITHOUT ANY explanation."
CODE_FORMAT = " If this is a code generation task, you MUST wrap your code in a markdown code block using ```python and ```."

# Per-model sampling params (same as inference_routerarena.py)
MODEL_SAMPLING_PARAMS = {
    "Qwen/Qwen3-Coder-Next": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.05,
    },
    "mistralai/Ministral-3-14B-Instruct-2512": {"temperature": 0.1},
    "mistralai/Ministral-3-8B-Instruct-2512": {"temperature": 0.1},
    "mistralai/Ministral-3-3B-Instruct-2512": {"temperature": 0.1},
    "moonshotai/Kimi-K2.5": {"temperature": 0.6, "top_p": 0.95},
}
DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.05,
}


# ============================================================================
# Template & prompt helpers
# ============================================================================

def load_clean_template() -> List[Dict]:
    """Load r2-router-clean.json, keeping only regular entries (skip optimality)."""
    log(f"Loading template from {TEMPLATE_FILE}")
    with open(TEMPLATE_FILE) as f:
        raw = json.load(f)
    clean = []
    for entry in raw:
        if entry.get("for_optimality", False):
            continue
        clean.append({
            "global index": entry["global index"],
            "prompt": entry["prompt"],
            "generated_result": None,
        })
    log(f"  Template: {len(clean)} entries")
    return clean


def build_budget_messages(original_prompt: str, budget):
    """Build chat messages with budget constraint in system prompt only.

    Code format instruction is always included so the LLM self-determines
    whether to use markdown code blocks.
    """
    if budget == "concise":
        return [
            {"role": "system", "content": CONCISE_SYSTEM + CODE_FORMAT},
            {"role": "user", "content": original_prompt},
        ]

    # Numeric budget
    system = f"You MUST respond in {budget} tokens or fewer. Directly give the answer follow the format WITHOUT ANY explanation." + CODE_FORMAT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": original_prompt},
    ]


# ============================================================================
# Inference for one budget
# ============================================================================

def run_one_budget_vllm(llm, budget, entries: List[Dict], output_file: str,
                       sampling_base: dict, batch_size: int):
    """Run inference for a single budget value using vLLM."""
    from vllm import SamplingParams

    # Find entries that still need processing
    to_run = []
    for i, entry in enumerate(entries):
        gr = entry.get("generated_result")
        if gr is not None and gr.get("success"):
            continue
        to_run.append(i)

    if not to_run:
        log(f"  Budget {budget}: all entries already done, skipping")
        return

    log(f"  Budget {budget}: {len(to_run)} to process")

    # Build messages
    messages_list = []
    for i in to_run:
        msgs = build_budget_messages(entries[i]["prompt"], budget)
        messages_list.append(msgs)

    if messages_list:
        log(f"  [VERIFY] System: {messages_list[0][0]['content']}")

    # Sampling params: always use 2048 max_tokens, budget constraint is prompt-only (soft)
    sp = {**sampling_base, "max_tokens": 2048}
    sampling_params = SamplingParams(**sp)
    log(f"  [VERIFY] max_tokens={sampling_params.max_tokens}")

    # Run in batches
    success = 0
    for start in range(0, len(messages_list), batch_size):
        batch = messages_list[start:start + batch_size]
        outputs = llm.chat(batch, sampling_params, use_tqdm=False)

        for j, output in enumerate(outputs):
            idx = to_run[start + j]
            generated_text = output.outputs[0].text
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)

            entries[idx]["generated_result"] = {
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
            success += 1

        done = min(start + batch_size, len(messages_list))
        log(f"    {done}/{len(messages_list)} done")

        # Save checkpoint after each batch
        with open(output_file, "w") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)

    log(f"  Budget {budget}: {success}/{len(to_run)} completed")


def call_openrouter(client, model: str, messages: list, params: dict,
                    max_retries: int = 5) -> Dict:
    """Call OpenRouter API with retry and exponential backoff."""
    import time
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, **params)
            usage = response.usage
            return {
                "generated_answer": response.choices[0].message.content or "",
                "success": True,
                "token_usage": {
                    "input_tokens": getattr(usage, "prompt_tokens", 0),
                    "output_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                },
                "provider": "openrouter",
                "error": None,
            }
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower()
            wait = min(2 ** attempt * 5, 120) if is_rate_limit else 2 ** attempt * 2
            log(f"    {'Rate limited' if is_rate_limit else 'Error: ' + err_str[:200]}, "
                f"retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
            if attempt == max_retries - 1:
                return {
                    "generated_answer": "", "success": False,
                    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "provider": "openrouter", "error": err_str[:500],
                }
            time.sleep(wait)
    return {
        "generated_answer": "", "success": False,
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "provider": "openrouter", "error": "Max retries exceeded",
    }


def run_one_budget_api(client, api_model: str, budget, entries: List[Dict],
                       output_file: str, api_params: dict, save_every: int = 50):
    """Run inference for a single budget value using OpenRouter API."""
    import time

    to_run = []
    for i, entry in enumerate(entries):
        gr = entry.get("generated_result")
        if gr is not None and gr.get("success"):
            continue
        to_run.append(i)

    if not to_run:
        log(f"  Budget {budget}: all entries already done, skipping")
        return

    log(f"  Budget {budget}: {len(to_run)} to process via OpenRouter ({api_model})")

    success = 0
    errors = 0
    start_time = time.time()

    for idx_in_batch, pred_idx in enumerate(to_run):
        msgs = build_budget_messages(entries[pred_idx]["prompt"], budget)
        if idx_in_batch == 0:
            log(f"  [VERIFY] System: {msgs[0]['content']}")

        result = call_openrouter(client, api_model, msgs, api_params)
        entries[pred_idx]["generated_result"] = result

        if result["success"]:
            success += 1
        else:
            errors += 1

        done = idx_in_batch + 1
        if done % save_every == 0 or done == len(to_run):
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta_min = (len(to_run) - done) / rate / 60 if rate > 0 else 0
            log(f"    {done}/{len(to_run)} done ({success} ok, {errors} err) "
                f"[{rate:.1f} q/s, ETA {eta_min:.0f}min]")
            with open(output_file, "w") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)

    log(f"  Budget {budget}: {success}/{len(to_run)} completed ({errors} errors)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Token budget sweep inference")
    parser.add_argument("--model", required=True, help="HuggingFace or OpenRouter model ID")
    parser.add_argument("--model-short", required=True, help="Short name for folder/files")
    parser.add_argument("--output-dir", required=True, help="Output directory (on Orange)")
    parser.add_argument("--api", action="store_true",
                        help="Use OpenRouter API instead of vLLM (requires OPENROUTER_API_KEY)")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--budgets", type=str, default=None,
                        help="Comma-separated budgets (default: 10,20,...,800,concise)")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling top_p")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty")
    parser.add_argument("--block-size", type=int, default=None, help="KV cache block size (fix FlashInfer issues)")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N queries (API mode)")
    args = parser.parse_args()

    budgets = DEFAULT_BUDGETS
    if args.budgets:
        budgets = []
        for b in args.budgets.split(","):
            b = b.strip()
            if b == "concise":
                budgets.append(b)
            else:
                budgets.append(int(b))

    mode = "API (OpenRouter)" if args.api else "vLLM"
    log(f"=== Token Budget Sweep ({mode}) ===")
    log(f"Model: {args.model}")
    log(f"Short name: {args.model_short}")
    log(f"Output dir: {args.output_dir}")
    log(f"Budgets: {budgets}")

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load clean template
    template = load_clean_template()

    import copy

    if args.api:
        # ---- API mode (OpenRouter) ----
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=120.0,
        )
        api_params = {"temperature": args.temperature or 1.0, "max_tokens": 2048}
        if args.top_p is not None:
            api_params["top_p"] = args.top_p
        log(f"API params: {api_params}")

        for budget in budgets:
            log(f"\n{'='*60}")
            log(f"Budget: {budget}")
            log(f"{'='*60}")

            if budget == "concise":
                output_file = os.path.join(args.output_dir, "concise.json")
            else:
                output_file = os.path.join(args.output_dir, f"budget_{budget}.json")

            if os.path.exists(output_file):
                with open(output_file) as f:
                    entries = json.load(f)
                done_count = sum(1 for e in entries if (e.get("generated_result") or {}).get("success"))
                log(f"  Checkpoint found: {done_count}/{len(entries)} already done")
            else:
                entries = copy.deepcopy(template)

            run_one_budget_api(client, args.model, budget, entries,
                               output_file, api_params, args.save_every)

    else:
        # ---- vLLM mode ----
        log(f"TP size: {args.tp_size}, Batch size: {args.batch_size}")

        # Get sampling params (keyed by HF model ID)
        sampling_base = MODEL_SAMPLING_PARAMS.get(args.model, DEFAULT_SAMPLING_PARAMS).copy()
        if args.temperature is not None:
            sampling_base["temperature"] = args.temperature
        if args.top_p is not None:
            sampling_base["top_p"] = args.top_p
        if args.top_k is not None:
            sampling_base["top_k"] = args.top_k
        if args.repetition_penalty is not None:
            sampling_base["repetition_penalty"] = args.repetition_penalty
        sampling_base.pop("max_tokens", None)
        log(f"Sampling params: {sampling_base}")

        from vllm import LLM
        log(f"Loading vLLM model: {args.model} (TP={args.tp_size})...")
        llm_kwargs = {
            "model": args.model,
            "tensor_parallel_size": args.tp_size,
            "max_model_len": args.max_model_len,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.85,
        }
        if args.block_size is not None:
            llm_kwargs["block_size"] = args.block_size
            log(f"  Using block_size={args.block_size}")
        llm = LLM(**llm_kwargs)
        log("Model loaded.")

        for budget in budgets:
            log(f"\n{'='*60}")
            log(f"Budget: {budget}")
            log(f"{'='*60}")

            if budget == "concise":
                output_file = os.path.join(args.output_dir, "concise.json")
            else:
                output_file = os.path.join(args.output_dir, f"budget_{budget}.json")

            if os.path.exists(output_file):
                with open(output_file) as f:
                    entries = json.load(f)
                done_count = sum(1 for e in entries if (e.get("generated_result") or {}).get("success"))
                log(f"  Checkpoint found: {done_count}/{len(entries)} already done")
            else:
                entries = copy.deepcopy(template)

            run_one_budget_vllm(llm, budget, entries, output_file, sampling_base,
                                args.batch_size)

        del llm
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    log("\n=== Budget sweep complete! ===")


if __name__ == "__main__":
    main()
