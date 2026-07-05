#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def make_key(entry: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    if "id" in entry and entry.get("id"):
        return ("id", entry.get("id"))
    run_id = entry.get("run_id")
    topic_slug = entry.get("topic_slug")
    question = entry.get("question")
    extras = []
    for field in (
        "question_author",
        "answer_author",
        "critic",
        "answer_model",
        "question_model",
        "alice_model",
        "bob_model",
    ):
        if field in entry:
            extras.append(entry.get(field))
    if run_id is None and topic_slug is None and not question:
        return None
    return (str(run_id) if run_id is not None else None, topic_slug, question, tuple(extras) or None)


def dedupe_entries(entries: List[Any]) -> Tuple[List[Any], int]:
    seen = set()
    kept: List[Any] = []
    removed = 0
    for entry in entries:
        if not entry or not isinstance(entry, dict):
            kept.append(entry)
            continue
        key = make_key(entry)
        if key and key in seen:
            removed += 1
            continue
        if key:
            seen.add(key)
        kept.append(entry)
    return kept, removed


def dedupe_payload(data: Any) -> Tuple[Optional[Any], int]:
    if isinstance(data, dict) and "decisions" in data and isinstance(data.get("decisions"), list):
        deduped, removed = dedupe_entries(data.get("decisions"))
        payload = dict(data)
        payload["decisions"] = deduped
        return payload, removed
    if isinstance(data, list):
        deduped, removed = dedupe_entries(data)
        return deduped, removed
    return None, 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate benchmark/answer/critique/debate/evaluation JSON files.")
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=[
            Path("benchmarks"),
            Path("answers"),
            Path("critiques"),
            Path("debates"),
            Path("automated_evaluations"),
            Path("evaluations"),
        ],
        help="Directories to scan for JSON files.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Apply changes to files. Without this flag, runs in dry-run mode.",
    )
    args = parser.parse_args()

    total_files = 0
    total_removed = 0
    for root in args.roots:
        if not root.exists():
            print(f"SKIP {root}: missing directory")
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception as exc:
                print(f"SKIP {path}: {exc}")
                continue
            deduped, removed = dedupe_payload(data)
            if deduped is None:
                print(f"SKIP {path}: unsupported JSON structure")
                continue
            if removed == 0:
                continue
            total_files += 1
            total_removed += removed
            print(f"DUP {path} remove={removed}")
            if args.real:
                path.write_text(json.dumps(deduped, indent=2))

    if not args.real:
        print("Dry run only. Re-run with --real to apply changes.")
    print(f"Files with dups: {total_files}, entries removed: {total_removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
