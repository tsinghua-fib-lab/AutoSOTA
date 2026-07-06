#!/usr/bin/env python3
"""Migrate exp2 cached JSONL to token-span `indices_to_explain`.

This converts legacy caches that used sentence indices (e.g. `[-2]`) into the
token-span format:

    indices_to_explain = [start_tok, end_tok]

Where the span points to the boxed-inner (final answer) token span in `target`
under `tokenizer(target, add_special_tokens=False)`.

Rule:
1) If `sink_span` exists and looks valid -> copy it to `indices_to_explain`
2) Else try to recompute spans from `target` + `metadata.boxed_answer` using
   `exp/exp2/dataset_utils.attach_spans_from_answer`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import AutoTokenizer


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _is_token_span(span: Any) -> bool:
    return isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span)


def _load_tokenizer(tokenizer_model: str):
    tok_path = Path(tokenizer_model)
    if tok_path.exists():
        return AutoTokenizer.from_pretrained(tok_path.as_posix(), local_files_only=True)
    return AutoTokenizer.from_pretrained(tokenizer_model)


def _migrate_obj(obj: Dict[str, Any], tokenizer) -> tuple[Dict[str, Any], bool]:
    sink_span = obj.get("sink_span")
    if _is_token_span(sink_span):
        obj["indices_to_explain"] = sink_span
        return obj, True

    _ensure_repo_root_on_path()
    from exp.exp2.dataset_utils import CachedExample, attach_spans_from_answer  # noqa: E402

    example = CachedExample(
        prompt=obj.get("prompt") or "",
        target=obj.get("target"),
        indices_to_explain=obj.get("indices_to_explain"),
        attr_mask_indices=obj.get("attr_mask_indices"),
        sink_span=obj.get("sink_span"),
        thinking_span=obj.get("thinking_span"),
        metadata=obj.get("metadata") or {},
    )
    answer_text = (example.metadata.get("boxed_answer") or "").strip() or None
    migrated = attach_spans_from_answer(example, tokenizer, answer_text)
    if not _is_token_span(migrated.sink_span):
        return obj, False

    obj["sink_span"] = migrated.sink_span
    obj["thinking_span"] = migrated.thinking_span
    obj["indices_to_explain"] = migrated.sink_span
    obj["metadata"] = migrated.metadata
    return obj, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--tokenizer_model", type=str, required=True)
    ap.add_argument("--strict", action="store_true", help="Fail on any line that cannot be migrated.")
    args = ap.parse_args()

    tokenizer = _load_tokenizer(args.tokenizer_model)

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)

    try:
        same_path = in_path.resolve() == out_path.resolve()
    except FileNotFoundError:
        same_path = False

    tmp_out_path = out_path
    if same_path:
        tmp_out_path = out_path.with_name(out_path.name + ".tmp")
        if tmp_out_path.exists():
            tmp_out_path.unlink()

    tmp_out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    migrated_ok = 0
    bad = 0

    with in_path.open("r", encoding="utf-8") as fin, tmp_out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            total += 1
            obj: Dict[str, Any] = json.loads(line)
            new_obj, ok = _migrate_obj(obj, tokenizer)
            if ok:
                migrated_ok += 1
            else:
                bad += 1
                if args.strict:
                    raise RuntimeError(f"cannot migrate line {line_no}: cannot resolve sink_span token span")
            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

    if same_path:
        tmp_out_path.replace(out_path)
        print(f"[done] total={total} migrated_ok={migrated_ok} bad={bad} wrote={out_path} (in-place)")
    else:
        print(f"[done] total={total} migrated_ok={migrated_ok} bad={bad} wrote={out_path}")


if __name__ == "__main__":
    main()
