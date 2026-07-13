#!/usr/bin/env python3
"""
Run closed-source LLM inference via OpenRouter for RouterArena.

Uses OpenRouter API (OpenAI-compatible) to call GPT-4o, Gemini Flash, etc.
Applies the same concise prompt suffix as the vLLM inference script.

Usage:
    .venv/bin/python scripts/inference_api.py --model gpt-4o --all-queries
    .venv/bin/python scripts/inference_api.py --model gemini-2.0-flash-001 --all-queries
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any


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

# Model configs: RouterArena name -> OpenRouter model ID + params
# Params use each model's defaults (best practice: don't over-tune)
#   GPT-4o: temperature=1.0 (default), no top_k support
#   Gemini 2.0 Flash: temperature=1.0 (default), top_p=0.95, top_k=40
MODEL_CONFIGS = {
    "gpt-4o": {
        "openrouter_model": "openai/gpt-4o",
        "params": {
            "temperature": 1.0,
            "max_tokens": 2048,
        },
    },
    "gemini-2.0-flash-001": {
        "openrouter_model": "google/gemini-2.0-flash-001",
        "params": {
            "temperature": 1.0,
            "max_tokens": 2048,
        },
    },
    "claude-haiku-4-5": {
        "openrouter_model": "anthropic/claude-haiku-4.5",
        "params": {
            "temperature": 1.0,
            "max_tokens": 2048,
        },
    },
}


def build_chat_messages(original_prompt: str, use_concise: bool = True):
    """Build chat messages with system prompt AND user suffix for concise instruction."""
    messages = []
    if use_concise:
        messages.append({"role": "system", "content": CONCISE_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": original_prompt + CONCISE_USER_SUFFIX})
    else:
        messages.append({"role": "user", "content": original_prompt})
    return messages


# ============================================================================
# OpenRouter API caller
# ============================================================================

def call_openrouter(client, model: str, messages: list, params: dict, max_retries: int = 5) -> Dict[str, Any]:
    """Call OpenRouter API with retry and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **params,
            )
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
            if is_rate_limit:
                wait = min(2 ** attempt * 5, 120)
                log(f"    Rate limited, waiting {wait}s... (attempt {attempt+1}/{max_retries})")
            else:
                wait = 2 ** attempt * 2
                log(f"    Error: {err_str[:200]}, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")

            if attempt == max_retries - 1:
                return {
                    "generated_answer": "",
                    "success": False,
                    "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "provider": "openrouter",
                    "error": err_str[:500],
                }
            time.sleep(wait)

    return {
        "generated_answer": "",
        "success": False,
        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "provider": "openrouter",
        "error": "Max retries exceeded",
    }


# ============================================================================
# Main inference loop
# ============================================================================

def run_inference(
    model_name: str,
    predictions: List[Dict],
    prediction_file: str,
    all_queries: bool = False,
    use_concise_prompt: bool = True,
    save_every: int = 50,
):
    from openai import OpenAI

    config = MODEL_CONFIGS[model_name]
    openrouter_model = config["openrouter_model"]
    params = config["params"]

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    # Collect entries to run
    entries_to_run = []
    for i, entry in enumerate(predictions):
        if not all_queries and entry["prediction"] != model_name:
            continue
        gr = entry.get("generated_result")
        if gr is not None and gr.get("success"):
            continue
        entries_to_run.append(i)

    if not entries_to_run:
        log(f"  No entries to process for {model_name}. Skipping.")
        return 0

    n_regular = sum(1 for i in entries_to_run if not predictions[i].get("for_optimality"))
    n_optimality = sum(1 for i in entries_to_run if predictions[i].get("for_optimality"))
    log(f"  Entries: {len(entries_to_run)} ({n_regular} regular + {n_optimality} optimality)")

    # Verification log
    log(f"  [VERIFY] Model: {model_name} (OpenRouter: {openrouter_model})")
    log(f"  [VERIFY] Params: {params}")
    log(f"  [VERIFY] Concise system prompt: {repr(CONCISE_SYSTEM_PROMPT)}")

    success_count = 0
    error_count = 0
    total_in_tokens = 0
    total_out_tokens = 0
    start_time = time.time()

    first_regular_logged = False
    for idx_in_batch, pred_idx in enumerate(entries_to_run):
        entry = predictions[pred_idx]
        original_prompt = entry["prompt"]

        # Build chat messages
        if not use_concise_prompt:
            messages = build_chat_messages(original_prompt, use_concise=False)
        elif entry.get("for_optimality"):
            messages = build_chat_messages(original_prompt, use_concise=False)
        else:
            messages = build_chat_messages(original_prompt, use_concise=True)
            if not first_regular_logged:
                log(f"  [VERIFY] First regular entry global_index: {entry.get('global index')}")
                log(f"  [VERIFY] Messages: {messages}")
                first_regular_logged = True

        # Call API
        result = call_openrouter(client, openrouter_model, messages, params)
        predictions[pred_idx]["generated_result"] = result

        if result["success"]:
            success_count += 1
            total_in_tokens += result["token_usage"]["input_tokens"]
            total_out_tokens += result["token_usage"]["output_tokens"]
        else:
            error_count += 1

        done = idx_in_batch + 1
        if done % save_every == 0 or done == len(entries_to_run):
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta_min = (len(entries_to_run) - done) / rate / 60 if rate > 0 else 0
            # Per-model pricing ($/M tokens)
            pricing = {"gpt-4o": (2.50, 10.00), "gemini-2.0-flash-001": (0.10, 0.40), "claude-haiku-4-5": (0.80, 4.00)}
            in_price, out_price = pricing.get(model_name, (1.0, 5.0))
            cost_in = total_in_tokens / 1e6 * in_price
            cost_out = total_out_tokens / 1e6 * out_price
            log(f"    {done}/{len(entries_to_run)} done "
                f"({success_count} ok, {error_count} err) "
                f"[{rate:.1f} q/s, ETA {eta_min:.0f}min] "
                f"cost: ${cost_in + cost_out:.4f}")

            with open(prediction_file, "w") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Final save
    with open(prediction_file, "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    log(f"  Completed: {success_count}/{len(entries_to_run)} "
        f"({error_count} errors) in {elapsed/60:.1f} min")
    log(f"  Total tokens: {total_in_tokens} input, {total_out_tokens} output")

    return success_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to run inference on")
    parser.add_argument("--prediction-file", type=str, default=PREDICTION_FILE)
    parser.add_argument("--all-queries", action="store_true")
    parser.add_argument("--no-concise-prompt", action="store_true")
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    log(f"=== R2-Router API Inference: {args.model} (via OpenRouter) ===")

    log(f"Loading predictions from {args.prediction_file}")
    with open(args.prediction_file) as f:
        predictions = json.load(f)
    log(f"  Total entries: {len(predictions)}")

    already_done = sum(1 for e in predictions
                       if e.get("generated_result") and e["generated_result"].get("success"))
    log(f"  Already completed: {already_done}")

    success = run_inference(
        model_name=args.model,
        predictions=predictions,
        prediction_file=args.prediction_file,
        all_queries=args.all_queries,
        use_concise_prompt=not args.no_concise_prompt,
        save_every=args.save_every,
    )

    log(f"\n=== Done! Total successful: {success} ===")


if __name__ == "__main__":
    main()
