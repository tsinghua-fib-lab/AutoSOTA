#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _should_drop(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    if entry.get("status") != "failed":
        return False
    attempts = entry.get("attempts")
    return not attempts


def _filter_entries(entries: List[Any]) -> List[Any]:
    return [entry for entry in entries if not _should_drop(entry)]


def _process_file(path: Path, dry_run: bool) -> int:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return 0
    if not isinstance(data, list):
        return 0
    filtered = _filter_entries(data)
    removed = len(data) - len(filtered)
    if removed and not dry_run:
        path.write_text(json.dumps(filtered, indent=2))
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drop failed answer entries with no attempts."
    )
    parser.add_argument(
        "--answers-dir",
        type=Path,
        default=Path("answers"),
        help="Directory containing answer JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report removals without modifying files.",
    )
    args = parser.parse_args()

    if not args.answers_dir.exists():
        print(f"answers dir not found: {args.answers_dir}")
        return 1

    total_removed = 0
    files_touched = 0
    for path in args.answers_dir.rglob("*.json"):
        removed = _process_file(path, args.dry_run)
        if removed:
            files_touched += 1
            total_removed += removed
            print(f"{path}: removed {removed}")

    print(f"files_touched={files_touched} removed_total={total_removed}")
    if args.dry_run:
        print("dry_run=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
