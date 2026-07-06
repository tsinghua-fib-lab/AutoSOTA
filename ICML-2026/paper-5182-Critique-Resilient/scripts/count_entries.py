#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def count_json_entries(path: Path) -> Tuple[int, int]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return 0, 1
    if isinstance(data, list):
        return sum(1 for item in data if item), 0
    if isinstance(data, dict) and isinstance(data.get("decisions"), list):
        return sum(1 for item in data.get("decisions") if item), 0
    return 0, 1


def count_dir(root: Path) -> Tuple[int, int, int]:
    total = 0
    files = 0
    errors = 0
    if not root.exists():
        return 0, 0, 0
    for path in root.rglob("*.json"):
        files += 1
        count, err = count_json_entries(path)
        total += count
        errors += err
    return total, files, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Count entries in pipeline JSON files.")
    parser.add_argument("--benchmarks-dir", type=Path, default=Path("benchmarks"))
    parser.add_argument("--answers-dir", type=Path, default=Path("answers"))
    parser.add_argument("--critiques-dir", type=Path, default=Path("critiques"))
    parser.add_argument("--debates-dir", type=Path, default=Path("debates"))
    parser.add_argument("--evaluations-dir", type=Path, default=Path("automated_evaluations"))
    args = parser.parse_args()

    targets: Dict[str, Path] = {
        "benchmarks": args.benchmarks_dir,
        "answers": args.answers_dir,
        "critiques": args.critiques_dir,
        "debates": args.debates_dir,
        "automated_evaluations": args.evaluations_dir,
    }

    grand_total = 0
    for label, root in targets.items():
        total, files, errors = count_dir(root)
        grand_total += total
        print(f"{label}: entries={total} files={files} errors={errors}")
    print(f"total_entries={grand_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
