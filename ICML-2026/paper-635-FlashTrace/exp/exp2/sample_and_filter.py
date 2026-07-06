#!/usr/bin/env python3
"""
Dataset sampler for Experiment 2.

Steps:
- Load a dataset item (MoreHopQA / HotpotQA / RULER niah / RULER vt).
- Call the generation model (qwen3-235b-a22b-2507) with a system prompt that
  asks for brief reasoning and a final answer wrapped in \\box{}.
- Enforce the output format: keep only generations that look like
  "<reasoning text> + final \\box{} answer" with nothing after the box.
- Call the judge model (deepseek-v3-1-terminus) to check whether the boxed
  answer matches the dataset reference answer; keep only judged True samples.
- Rebuild `target` as "<reasoning>\\n<answer text (no box)>" and store filtered
  samples to exp/exp2/data/<dataset>.jsonl (or a custom path) with inferred spans.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from transformers import AutoTokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp2.dataset_utils import (
    CachedExample,
    DatasetLoader,
    attach_spans_from_answer,
    split_boxed_generation,
)


class RateLimitError(RuntimeError):
    """Raised when API returns 429; carries a suggested wait time."""

    def __init__(self, wait_seconds: float, detail: str) -> None:
        super().__init__(detail)
        self.wait_seconds = wait_seconds

# GEN_SYSTEM_PROMPT = (
#     "You are a careful reasoning assistant. "
#     "Before answering, engage in an extremely detailed and exhaustive chain of thought. **No fewer than 2k tokens.** "
#     "Do not skip any logical steps, even if they seem obvious. "
#     "Process this freely and naturally without using specific headers or strict formatting. "
#     "When you reach the conclusion, wrap the entire final sentence containing the answer inside \\box{}. "
#     "Ensure the box wraps the **sentence** that naturally delivers the answer. DO NOT rewrite the answer word for the box separately."
# )

GEN_SYSTEM_PROMPT = (
    "You are a reasoning assistant. "
    "Before answering, engage in an chain of thought. "
    "Process this freely and naturally without using specific headers or strict formatting. "
    "When you reach the conclusion, wrap the entire final sentence containing the answer inside \\box{}. "
    "Ensure the box wraps the **sentence** that naturally delivers the answer. DO NOT rewrite the answer word for the box separately."
)

JUDGE_SYSTEM_PROMPT = (
    "You verify whether the model's boxed answer matches the reference answer. "
    "Reply strictly with True or False and nothing else."
)


def call_chat_api(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    timeout: int,
    max_tokens: int,
    temperature: float,
    cache_ttl: int,
    cache_namespace: Optional[str],
    rate_limit_delay: Optional[float] = None,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if cache_ttl > 0:
        cache_obj: Dict[str, Any] = {"ttl": cache_ttl}
        if cache_namespace:
            cache_obj["namespace"] = cache_namespace
        payload["cache"] = cache_obj

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            resp_bytes = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
        if e.code == 429:
            retry_after = None
            if hasattr(e, "headers") and e.headers:
                retry_after_header = e.headers.get("Retry-After")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        retry_after = None
            wait = retry_after or rate_limit_delay or 5.0
            raise RateLimitError(wait, f"API HTTP 429: {detail}") from e
        raise RuntimeError(f"API HTTP error {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"API request failed: {e}") from e

    try:
        response = json.loads(resp_bytes.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to decode API response: {resp_bytes!r}") from e

    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError(f"Empty choices from API: {response}")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Empty content from API: {response}")
    return content.strip()


def build_gen_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": GEN_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def build_judge_messages(reference_answer: str, candidate_answer: str) -> List[Dict[str, str]]:
    user = (
        "Decide if the model's boxed answer matches the reference answer.\n"
        f"Reference answer: {reference_answer}\n"
        f"Model boxed answer (only the content inside \\box{{}}): {candidate_answer}\n"
        "Output only True if they are semantically consistent; otherwise output False."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_bool(text: str) -> bool:
    first = text.strip().splitlines()[0].strip().lower()
    if first in {"true", "yes"}:
        return True
    if first in {"false", "no"}:
        return False
    # fallback: check substring
    if "true" in first and "false" not in first:
        return True
    if "false" in first:
        return False
    raise ValueError(f"Cannot parse boolean from: {text!r}")


def write_cache(out_path: Path, examples: Iterable[CachedExample]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            obj: Dict[str, Any] = {
                "prompt": ex.prompt,
                "target": ex.target,
                "indices_to_explain": ex.indices_to_explain,
                "attr_mask_indices": ex.attr_mask_indices,
                "sink_span": ex.sink_span,
                "thinking_span": ex.thinking_span,
                "metadata": ex.metadata,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser("Sample and filter dataset examples for exp2.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="morehopqa | hotpotqa_long | niah_* | vt_* | <morehopqa_json_path> | <ruler_jsonl_path>",
    )
    parser.add_argument("--max_examples", type=int, default=100, help="Number of raw examples to sample before filtering.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api_base", type=str, default="http://localhost:4000/v1", help="Chat API base URL.")
    parser.add_argument("--api_key", type=str, default=None, help="API key; defaults to FLASHTRACE_API_KEY/OPENAI_API_KEY.")
    parser.add_argument("--generator_model", type=str, default="qwen3-235b-a22b-2507")
    parser.add_argument("--judge_model", type=str, default="deepseek-v3-1-terminus")
    parser.add_argument("--api_timeout", type=int, default=300)
    parser.add_argument("--api_max_tokens", type=int, default=8192)
    parser.add_argument("--api_temperature", type=float, default=0.0)
    parser.add_argument("--api_cache_ttl", type=int, default=600)
    parser.add_argument("--api_cache_namespace", type=str, default="flashtrace-exp2")
    parser.add_argument("--retry_delay", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=2, help="Additional retries on API failure.")
    parser.add_argument("--request_interval", type=float, default=1.0, help="Sleep seconds between generation calls.")
    parser.add_argument("--judge_interval", type=float, default=1.0, help="Sleep seconds between judge calls.")
    parser.add_argument("--tokenizer_model", type=str, default=None, help="Tokenizer path for span extraction (default: generator model).")
    parser.add_argument("--data_root", type=str, default="exp/exp2/data", help="Output directory for filtered caches.")
    parser.add_argument("--out", type=str, default=None, help="Optional explicit output path (JSONL).")
    parser.add_argument("--rate_limit_delay", type=float, default=5.0, help="Seconds to wait on HTTP 429 before retrying.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("FLASHTRACE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set --api_key or FLASHTRACE_API_KEY/OPENAI_API_KEY for API access.")

    loader = DatasetLoader(seed=args.seed, data_root=args.data_root)
    # Load full dataset; we will stop early once enough kept examples are collected.
    raw_examples = loader.load_raw(args.dataset, sample=None)
    if not raw_examples:
        raise SystemExit("No examples loaded.")

    tok_name = args.tokenizer_model or args.generator_model
    tok_path = Path(tok_name)
    if tok_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tok_path.as_posix(), local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
    tokenizer.pad_token = tokenizer.eos_token

    kept: List[CachedExample] = []
    total = len(raw_examples)
    kept_bar = tqdm(total=args.max_examples, desc="Kept (judge=True)", position=1, leave=False)
    attempted = 0

    for idx, ex in enumerate(tqdm(raw_examples, total=total, desc="Sampling"), 1):
        if len(kept) >= args.max_examples:
            break
        reference_answer = ex.metadata.get("reference_answer") or ex.target or ""
        gen_messages = build_gen_messages(ex.prompt)
        attempted = idx

        # Step 1: generation
        for attempt in range(args.retries + 1):
            try:
                generation = call_chat_api(
                    args.api_base,
                    api_key,
                    args.generator_model,
                    gen_messages,
                    timeout=args.api_timeout,
                    max_tokens=args.api_max_tokens,
                    temperature=args.api_temperature,
                    cache_ttl=args.api_cache_ttl,
                    cache_namespace=args.api_cache_namespace,
                    rate_limit_delay=args.rate_limit_delay,
                )
                break
            except RateLimitError as e:
                if attempt >= args.retries:
                    raise
                time.sleep(e.wait_seconds)
            except Exception:  # noqa: BLE001
                if attempt >= args.retries:
                    raise
                time.sleep(args.retry_delay)
        if args.request_interval > 0:
            time.sleep(args.request_interval)

        parsed = split_boxed_generation(generation)
        if not parsed:
            print(f"[{idx}/{total}] skipped=format")
            continue

        thinking_text, boxed_segment, boxed_answer = parsed
        target_text = f"{thinking_text}\n{boxed_answer}" if thinking_text else boxed_answer
        judge_messages = build_judge_messages(reference_answer, boxed_answer)

        ok = False
        judge_resp = ""
        for attempt in range(args.retries + 1):
            try:
                judge_resp = call_chat_api(
                    args.api_base,
                    api_key,
                    args.judge_model,
                    judge_messages,
                    timeout=args.api_timeout,
                    max_tokens=64,
                    temperature=0.0,
                    cache_ttl=args.api_cache_ttl,
                    cache_namespace=args.api_cache_namespace,
                    rate_limit_delay=args.rate_limit_delay,
                )
                ok = parse_bool(judge_resp)
                break
            except RateLimitError as e:
                if attempt >= args.retries:
                    raise
                time.sleep(e.wait_seconds)
            except Exception:  # noqa: BLE001
                if attempt >= args.retries:
                    raise
                time.sleep(args.retry_delay)
        if args.judge_interval > 0:
            time.sleep(args.judge_interval)

        status = "kept" if ok else "filtered"
        print(f"[{idx}/{total}] judge={status}")
        if not ok:
            continue

        new_meta = dict(ex.metadata)
        new_meta["reference_answer"] = reference_answer
        new_meta["judge_response"] = judge_resp

        new_ex = CachedExample(
            prompt=ex.prompt,
            target=target_text,
            indices_to_explain=None,
            attr_mask_indices=ex.attr_mask_indices,
            sink_span=None,
            thinking_span=None,
            metadata=new_meta,
        )
        new_ex = attach_spans_from_answer(new_ex, tokenizer, boxed_answer)
        if not (isinstance(new_ex.sink_span, list) and len(new_ex.sink_span) == 2):
            print(f"[{idx}/{total}] skipped=span")
            continue

        # Token-level indices_to_explain: boxed-inner answer token span in target (closed interval).
        new_ex = CachedExample(
            prompt=new_ex.prompt,
            target=new_ex.target,
            indices_to_explain=new_ex.sink_span,
            attr_mask_indices=new_ex.attr_mask_indices,
            sink_span=new_ex.sink_span,
            thinking_span=new_ex.thinking_span,
            metadata=new_ex.metadata,
        )
        kept.append(new_ex)
        kept_bar.update(1)

    kept_bar.close()

    out_path = Path(args.out) if args.out else Path(args.data_root) / f"{args.dataset}.jsonl"
    written = write_cache(out_path, kept)
    attempted_total = attempted or 0
    print(f"Kept {written} / target {args.max_examples} (attempted {attempted_total} / {total}) -> {out_path}")


if __name__ == "__main__":
    main()
