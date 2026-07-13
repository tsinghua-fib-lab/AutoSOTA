"""Utilities to record LLM parsing failures and aggregate attempt / usage stats."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional


_lock = threading.Lock()


def _default_stage_stats() -> Dict[str, int]:
    return {
        "attempts": 0,
        "failures": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }


def _default_stats() -> Dict[str, Dict[str, int]]:
    return {
        "generation": _default_stage_stats(),
        "evaluation": _default_stage_stats(),
    }


def _load_stats(path: str) -> Dict[str, Dict[str, int]]:
    if not os.path.exists(path):
        return _default_stats()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ensure all expected keys exist
        for stage, counts in _default_stats().items():
            data.setdefault(stage, counts)
            for key, value in counts.items():
                data[stage].setdefault(key, value)
        return data
    except Exception:
        return _default_stats()


def _write_stats(path: str, stats: Dict[str, Dict[str, int]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def record_attempt(output_dir: Optional[str], stage: str) -> None:
    """Increment attempt counter for a stage (generation/evaluation)."""
    if not output_dir:
        return
    stats_path = os.path.join(output_dir, "parse_stats.json")
    with _lock:
        stats = _load_stats(stats_path)
        stats.setdefault(stage, _default_stage_stats())
        stats[stage]["attempts"] = stats[stage].get("attempts", 0) + 1
        _write_stats(stats_path, stats)


def record_usage(output_dir: Optional[str], stage: str, usage: Optional[Any]) -> None:
    """Accumulate token usage for a stage from an OpenAI-style response.usage object or dict."""
    if not output_dir or usage is None:
        return

    def _get_value(obj: Any, key: str, default: int = 0) -> int:
        if obj is None:
            return default
        if isinstance(obj, dict):
            value = obj.get(key, default)
        else:
            value = getattr(obj, key, default)
        try:
            return int(value or 0)
        except Exception:
            return default

    prompt_tokens = _get_value(usage, "prompt_tokens")
    if prompt_tokens == 0:
        prompt_tokens = _get_value(usage, "input_tokens")

    completion_tokens = _get_value(usage, "completion_tokens")
    if completion_tokens == 0:
        completion_tokens = _get_value(usage, "output_tokens")

    total_tokens = _get_value(usage, "total_tokens")

    # Some providers expose reasoning tokens via nested details.
    reasoning_tokens = 0
    details = usage.get("completion_tokens_details") if isinstance(usage, dict) else getattr(usage, "completion_tokens_details", None)
    if details is not None:
        reasoning_tokens = _get_value(details, "reasoning_tokens")
    if reasoning_tokens == 0:
        details = usage.get("output_tokens_details") if isinstance(usage, dict) else getattr(usage, "output_tokens_details", None)
        if details is not None:
            reasoning_tokens = _get_value(details, "reasoning_tokens")

    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens

    stats_path = os.path.join(output_dir, "parse_stats.json")
    with _lock:
        stats = _load_stats(stats_path)
        stats.setdefault(stage, _default_stage_stats())
        stats[stage]["prompt_tokens"] = stats[stage].get("prompt_tokens", 0) + prompt_tokens
        stats[stage]["completion_tokens"] = stats[stage].get("completion_tokens", 0) + completion_tokens
        stats[stage]["total_tokens"] = stats[stage].get("total_tokens", 0) + total_tokens
        stats[stage]["reasoning_tokens"] = stats[stage].get("reasoning_tokens", 0) + reasoning_tokens
        _write_stats(stats_path, stats)


def record_failure(
    *,
    output_dir: Optional[str],
    stage: str,
    model: str,
    attempts: int,
    error: str,
    prompt: str,
    response: str,
    round_idx: Optional[int] = None,
    rule: Optional[str] = None,
    sample_idx: Optional[int] = None,
) -> None:
    """Record a parse failure with full prompt/response saved to disk."""
    if not output_dir:
        return

    failures_dir = os.path.join(output_dir, "parse_failures")
    os.makedirs(failures_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "parse_stats.json")
    jsonl_path = os.path.join(output_dir, "parse_failures.jsonl")

    with _lock:
        # Update stats
        stats = _load_stats(stats_path)
        stats.setdefault(stage, _default_stage_stats())
        stats[stage]["failures"] = stats[stage].get("failures", 0) + 1
        _write_stats(stats_path, stats)

        # Allocate id
        next_id_path = os.path.join(failures_dir, "next_id.txt")
        try:
            with open(next_id_path, "r", encoding="utf-8") as f:
                next_id = int(f.read().strip() or "1")
        except Exception:
            next_id = 1
        current_id = next_id
        with open(next_id_path, "w", encoding="utf-8") as f:
            f.write(str(next_id + 1))

    # Write prompt/response files (outside lock to minimize contention)
    prompt_file = os.path.join(failures_dir, f"fail_{current_id:06d}_prompt.txt")
    response_file = os.path.join(failures_dir, f"fail_{current_id:06d}_response.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(response_file, "w", encoding="utf-8") as f:
        f.write(response)

    record = {
        "id": current_id,
        "stage": stage,
        "round": round_idx,
        "rule": rule,
        "sample_idx": sample_idx,
        "attempts": attempts,
        "model": model,
        "error": error,
        "prompt_file": os.path.relpath(prompt_file, output_dir),
        "response_file": os.path.relpath(response_file, output_dir),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    with _lock:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
