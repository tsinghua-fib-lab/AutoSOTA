#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)
        handle.write("\n")


def next_run_id(runs: dict) -> int:
    numeric_ids = []
    for key in runs.keys():
        try:
            numeric_ids.append(int(key))
        except (TypeError, ValueError):
            continue
    return max(numeric_ids, default=0) + 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Add missing core-mathematics topics to runs.json, "
            "preserving existing ID order."
        )
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("configs/runs.json"),
        help="Path to runs.json",
    )
    parser.add_argument(
        "--topics",
        type=Path,
        default=Path("configs/topic_info.json"),
        help="Path to topic_info.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path (defaults to --runs).",
    )
    args = parser.parse_args()

    runs_path = args.runs
    topics_path = args.topics
    out_path = args.out or runs_path

    runs = load_json(runs_path)
    topics = load_json(topics_path)

    existing_topics = set()
    for run in runs.values():
        if isinstance(run, dict) and "topic" in run:
            existing_topics.add(run["topic"])

    new_id = next_run_id(runs)
    added = []

    for topic in topics:
        if not isinstance(topic, dict):
            continue
        if not topic.get("core_mathematics"):
            continue
        slug = topic.get("slug")
        if not slug or slug in existing_topics:
            continue
        runs[str(new_id)] = {"topic": slug}
        added.append({"id": new_id, "slug": slug})
        new_id += 1

    write_json(out_path, runs)

    if added:
        print(f"Added {len(added)} runs:")
        for item in added:
            print(f"  {item['id']}: {item['slug']}")
    else:
        print("No missing core-mathematics topics found.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
