#!/usr/bin/env python3
"""Prepare data/math_mine.json into an exp2 cached JSONL dataset.

This script supports two modes:

- map (offline): convert GSM8K-style math examples:

    {"question": "...", "answer": "... #### 18"}

  into exp2's cached JSONL format (one JSON object per line).

- resample (online): resample targets like exp/exp2/sample_and_filter.py:
  call a chat completion API to generate "<thinking> + final \\box{} answer",
  judge the boxed answer against the reference answer extracted from the raw
  GSM8K-style entry, and write only judge=True samples.

In both modes, exp2 expects token-level spans (NOT character spans):

  - indices_to_explain: [start_tok, end_tok] (generation-token indices, closed interval)
  - sink_span/thinking_span: token spans over tokenizer(target, add_special_tokens=False)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp2.dataset_utils import CachedExample, attach_spans_from_answer, split_boxed_generation  # noqa: E402


class RateLimitError(RuntimeError):
    """Raised when API returns 429; carries a suggested wait time."""

    def __init__(self, wait_seconds: float, detail: str) -> None:
        super().__init__(detail)
        self.wait_seconds = wait_seconds


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
    """Minimal OpenAI-compatible chat.completions client (no external deps)."""
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


def _load_tokenizer(tokenizer_model: str):
    tok_path = Path(tokenizer_model)
    if tok_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tok_path.as_posix(), local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _split_gsm8k_answer(answer: str) -> Optional[Tuple[str, str]]:
    """Return (thinking_text, final_answer) parsed from GSM8K `answer`."""
    text = (answer or "").strip()
    if not text:
        return None
    if "####" not in text:
        return None
    thinking, final = text.rsplit("####", 1)
    thinking = thinking.strip()
    final = final.strip()
    if not final:
        return None
    return thinking, final


def _is_token_span(span: Any) -> bool:
    return isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span)


def _build_cached_example(
    *,
    question: str,
    answer: str,
    tokenizer,
    example_idx: int,
    source_path: str,
) -> Optional[CachedExample]:
    parsed = _split_gsm8k_answer(answer)
    if parsed is None:
        return None
    thinking_text, final_answer = parsed

    prompt = question.strip()
    target = f"{thinking_text}\n{final_answer}" if thinking_text else final_answer

    example = CachedExample(
        prompt=prompt,
        target=target,
        indices_to_explain=None,
        attr_mask_indices=None,
        sink_span=None,
        thinking_span=None,
        metadata={
            "dataset": "math_mine",
            "source_path": source_path,
            "example_idx": int(example_idx),
            "raw_question": question,
            "raw_answer": answer,
            "reference_answer": final_answer,
            "boxed_answer": final_answer,
        },
    )
    example = attach_spans_from_answer(example, tokenizer, final_answer)
    if not _is_token_span(example.sink_span):
        return None

    # exp2 requires token-level indices_to_explain=[start_tok,end_tok] (closed interval).
    indices_to_explain = list(example.sink_span)
    thinking_span = example.thinking_span
    if thinking_span is not None and _is_token_span(thinking_span) and indices_to_explain[0] == 0:
        # No room for "thinking" tokens; avoid overlapping spans.
        thinking_span = None

    return CachedExample(
        prompt=example.prompt,
        target=example.target,
        indices_to_explain=indices_to_explain,
        attr_mask_indices=example.attr_mask_indices,
        sink_span=indices_to_explain,
        thinking_span=thinking_span,
        metadata=example.metadata,
    )


def _build_resampled_example(
    *,
    question: str,
    raw_answer: str,
    reference_answer: str,
    generation: str,
    tokenizer,
    example_idx: int,
    source_path: str,
    judge_response: str,
    generator_model: str,
    judge_model: str,
) -> Optional[CachedExample]:
    parsed = split_boxed_generation(generation)
    if not parsed:
        return None

    thinking_text, _boxed_segment, boxed_answer = parsed
    target_text = f"{thinking_text}\n{boxed_answer}" if thinking_text else boxed_answer

    example = CachedExample(
        prompt=question.strip(),
        target=target_text,
        indices_to_explain=None,
        attr_mask_indices=None,
        sink_span=None,
        thinking_span=None,
        metadata={
            "dataset": "math_mine",
            "source_path": source_path,
            "example_idx": int(example_idx),
            "raw_question": question,
            "raw_answer": raw_answer,
            "reference_answer": reference_answer,
            "judge_response": judge_response,
            "generator_model": generator_model,
            "judge_model": judge_model,
        },
    )
    example = attach_spans_from_answer(example, tokenizer, boxed_answer)
    if not _is_token_span(example.sink_span):
        return None

    indices_to_explain = list(example.sink_span)
    return CachedExample(
        prompt=example.prompt,
        target=example.target,
        indices_to_explain=indices_to_explain,
        attr_mask_indices=example.attr_mask_indices,
        sink_span=indices_to_explain,
        thinking_span=example.thinking_span,
        metadata=example.metadata,
    )


def _write_jsonl(path: Path, *, examples) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser("Prepare data/math_mine.json for exp2 cached JSONL.")
    ap.add_argument("--in_json", type=str, default="data/math_mine.json")
    ap.add_argument("--out_jsonl", type=str, default="exp/exp2/data/math.jsonl")
    ap.add_argument(
        "--tokenizer_model",
        type=str,
        required=True,
        help="Tokenizer name or local path; must match the tokenizer used in exp2 attribution.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        choices=["map", "resample"],
        default="map",
        help="map=offline mapping from GSM8K answers; resample=generate+judge like exp/exp2/sample_and_filter.py.",
    )

    # Resample (online) options (kept compatible with exp/exp2/sample_and_filter.py).
    ap.add_argument("--max_examples", type=int, default=100, help="Number of judge=True examples to keep (resample mode).")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed (only used with --shuffle).")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle examples before attempting (resample mode).")
    ap.add_argument("--api_base", type=str, default="http://localhost:4000/v1", help="Chat API base URL.")
    ap.add_argument("--api_key", type=str, default=None, help="API key; defaults to FLASHTRACE_API_KEY/OPENAI_API_KEY.")
    ap.add_argument("--generator_model", type=str, default="qwen3-235b-a22b-2507")
    ap.add_argument("--judge_model", type=str, default="deepseek-v3-1-terminus")
    ap.add_argument("--api_timeout", type=int, default=300)
    ap.add_argument("--api_max_tokens", type=int, default=8192)
    ap.add_argument("--api_temperature", type=float, default=0.0)
    ap.add_argument("--api_cache_ttl", type=int, default=600)
    ap.add_argument("--api_cache_namespace", type=str, default="flashtrace-exp2")
    ap.add_argument("--retry_delay", type=float, default=2.0)
    ap.add_argument("--retries", type=int, default=2, help="Additional retries on API failure.")
    ap.add_argument("--request_interval", type=float, default=1.0, help="Sleep seconds between generation calls.")
    ap.add_argument("--judge_interval", type=float, default=1.0, help="Sleep seconds between judge calls.")
    ap.add_argument("--rate_limit_delay", type=float, default=5.0, help="Seconds to wait on HTTP 429 before retrying.")
    args = ap.parse_args()

    in_path = Path(args.in_json)
    out_path = Path(args.out_jsonl)
    tokenizer = _load_tokenizer(args.tokenizer_model)

    raw = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"Expected a JSON array in {in_path}, got {type(raw).__name__}.")

    source_total = len(raw)
    total = 0
    kept = 0
    skipped_empty_q = 0
    skipped_empty_a = 0
    skipped_parse = 0
    skipped_span = 0

    examples = []
    if args.mode == "map":
        attempted = None
        skipped_format = None
        judged_false = None
        for idx, item in enumerate(raw):
            total += 1
            if not isinstance(item, dict):
                skipped_parse += 1
                continue

            question = str(item.get("question") or "")
            answer = str(item.get("answer") or "")
            if not question.strip():
                skipped_empty_q += 1
                continue
            if not answer.strip():
                skipped_empty_a += 1
                continue

            ex = _build_cached_example(
                question=question,
                answer=answer,
                tokenizer=tokenizer,
                example_idx=idx,
                source_path=str(in_path),
            )
            if ex is None:
                # distinguish parse-vs-span failure
                parsed = _split_gsm8k_answer(answer)
                if parsed is None:
                    skipped_parse += 1
                else:
                    skipped_span += 1
                continue

            examples.append(ex)
            kept += 1
    else:
        api_key = args.api_key or os.environ.get("FLASHTRACE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("resample mode requires --api_key or FLASHTRACE_API_KEY/OPENAI_API_KEY.")

        attempted = 0
        skipped_format = 0
        judged_false = 0

        indices = list(range(len(raw)))
        if bool(args.shuffle):
            import random

            rnd = random.Random(int(args.seed))
            rnd.shuffle(indices)

        kept_bar = tqdm(total=int(args.max_examples), desc="Kept (judge=True)", position=1, leave=False)
        for loop_idx in tqdm(indices, total=len(indices), desc="Resampling"):
            if kept >= int(args.max_examples):
                break

            total += 1
            item = raw[loop_idx]
            if not isinstance(item, dict):
                skipped_parse += 1
                continue

            question = str(item.get("question") or "")
            answer = str(item.get("answer") or "")
            if not question.strip():
                skipped_empty_q += 1
                continue
            if not answer.strip():
                skipped_empty_a += 1
                continue

            parsed = _split_gsm8k_answer(answer)
            if parsed is None:
                skipped_parse += 1
                continue
            _ref_thinking, reference_answer = parsed

            attempted += 1
            gen_messages = build_gen_messages(question.strip())

            # Step 1: generation
            for attempt in range(int(args.retries) + 1):
                try:
                    generation = call_chat_api(
                        str(args.api_base),
                        str(api_key),
                        str(args.generator_model),
                        gen_messages,
                        timeout=int(args.api_timeout),
                        max_tokens=int(args.api_max_tokens),
                        temperature=float(args.api_temperature),
                        cache_ttl=int(args.api_cache_ttl),
                        cache_namespace=str(args.api_cache_namespace) if args.api_cache_namespace else None,
                        rate_limit_delay=float(args.rate_limit_delay) if args.rate_limit_delay is not None else None,
                    )
                    break
                except RateLimitError as e:
                    if attempt >= int(args.retries):
                        raise
                    time.sleep(float(e.wait_seconds))
                except Exception:  # noqa: BLE001
                    if attempt >= int(args.retries):
                        raise
                    time.sleep(float(args.retry_delay))
            if float(args.request_interval) > 0:
                time.sleep(float(args.request_interval))

            parsed_gen = split_boxed_generation(generation)
            if not parsed_gen:
                skipped_format += 1
                print(f"[attempt={attempted}] skipped=format")
                continue

            thinking_text, _boxed_segment, boxed_answer = parsed_gen
            judge_messages = build_judge_messages(reference_answer, boxed_answer)

            ok = False
            judge_resp = ""
            for attempt in range(int(args.retries) + 1):
                try:
                    judge_resp = call_chat_api(
                        str(args.api_base),
                        str(api_key),
                        str(args.judge_model),
                        judge_messages,
                        timeout=int(args.api_timeout),
                        max_tokens=64,
                        temperature=0.0,
                        cache_ttl=int(args.api_cache_ttl),
                        cache_namespace=str(args.api_cache_namespace) if args.api_cache_namespace else None,
                        rate_limit_delay=float(args.rate_limit_delay) if args.rate_limit_delay is not None else None,
                    )
                    ok = parse_bool(judge_resp)
                    break
                except RateLimitError as e:
                    if attempt >= int(args.retries):
                        raise
                    time.sleep(float(e.wait_seconds))
                except Exception:  # noqa: BLE001
                    if attempt >= int(args.retries):
                        raise
                    time.sleep(float(args.retry_delay))
            if float(args.judge_interval) > 0:
                time.sleep(float(args.judge_interval))

            if not ok:
                judged_false += 1
                print(f"[attempt={attempted}] judge=filtered")
                continue

            ex = _build_resampled_example(
                question=question,
                raw_answer=answer,
                reference_answer=reference_answer,
                generation=generation,
                tokenizer=tokenizer,
                example_idx=int(loop_idx),
                source_path=str(in_path),
                judge_response=judge_resp,
                generator_model=str(args.generator_model),
                judge_model=str(args.judge_model),
            )
            if ex is None:
                skipped_span += 1
                print(f"[attempt={attempted}] skipped=span")
                continue

            examples.append(ex)
            kept += 1
            kept_bar.update(1)
            print(f"[attempt={attempted}] judge=kept")

        kept_bar.close()

    written = _write_jsonl(out_path, examples=examples)
    if written != kept:
        raise SystemExit(f"Internal error: written={written} != kept={kept}")

    print(
        json.dumps(
            {
                "in_json": str(in_path),
                "out_jsonl": str(out_path),
                "tokenizer_model": args.tokenizer_model,
                "mode": str(args.mode),
                "source_total": int(source_total),
                "visited": total,
                "kept": kept,
                "skipped_empty_question": skipped_empty_q,
                "skipped_empty_answer": skipped_empty_a,
                "skipped_parse": skipped_parse,
                "skipped_span": skipped_span,
                "attempted": attempted,
                "skipped_format": skipped_format,
                "judged_false": judged_false,
                "max_examples": int(args.max_examples) if str(args.mode) == "resample" else None,
                "api_base": str(args.api_base) if str(args.mode) == "resample" else None,
                "generator_model": str(args.generator_model) if str(args.mode) == "resample" else None,
                "judge_model": str(args.judge_model) if str(args.mode) == "resample" else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
