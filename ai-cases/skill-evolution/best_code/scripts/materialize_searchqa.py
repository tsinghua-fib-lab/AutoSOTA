"""Materialize runnable SearchQA splits from the released ID manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
REQUIRED_FIELDS = ("question", "context", "answers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "searchqa_id_split",
        help="Directory containing train/val/test ID manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "searchqa_split",
        help="Directory to write runnable train/val/test splits.",
    )
    parser.add_argument(
        "--dataset",
        default="lucadiliello/searchqa",
        help="Hugging Face dataset repository to load.",
    )
    return parser.parse_args()


def load_manifest_ids(manifest_dir: Path) -> dict[str, list[str]]:
    split_ids = {}
    for split in SPLITS:
        path = manifest_dir / split / "items.json"
        with path.open(encoding="utf-8") as file:
            items = json.load(file)
        split_ids[split] = [str(item["id"]) for item in items]
    return split_ids


def _reject_duplicate_manifest_ids(split_ids: Mapping[str, Iterable[str]]) -> None:
    seen: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for split, ids in split_ids.items():
        for item_id in ids:
            if item_id in seen:
                duplicates.setdefault(item_id, [seen[item_id]]).append(split)
            else:
                seen[item_id] = split

    if duplicates:
        preview = ", ".join(
            f"{item_id} ({'/'.join(splits)})"
            for item_id, splits in sorted(duplicates.items())[:5]
        )
        raise ValueError(
            "SearchQA split manifest contains duplicate IDs across splits. "
            f"First IDs: {preview}"
        )


def _iter_dataset_rows(dataset: Mapping[str, Iterable[dict]]) -> Iterable[dict]:
    for source_split in dataset.values():
        yield from source_split


def _normalize_row(row: dict) -> dict:
    try:
        key = str(row["key"])
    except KeyError as exc:
        raise ValueError("SearchQA source row is missing required field: key") from exc

    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"SearchQA source row {key!r} is missing required fields: {', '.join(missing)}")

    return {
        "id": key,
        "question": row["question"],
        "context": row["context"],
        "answers": row["answers"],
    }


def materialize_searchqa_splits(
    manifest_dir: Path,
    output_dir: Path,
    dataset: Mapping[str, Iterable[dict]],
    *,
    dataset_name: str,
) -> dict[str, int]:
    """Write runnable SearchQA train/val/test splits from a source dataset."""
    manifest_dir = manifest_dir.resolve()
    output_dir = output_dir.resolve()
    split_ids = load_manifest_ids(manifest_dir)
    _reject_duplicate_manifest_ids(split_ids)
    wanted_ids = {item_id for ids in split_ids.values() for item_id in ids}

    selected: dict[str, dict] = {}
    duplicate_ids: set[str] = set()
    for row in _iter_dataset_rows(dataset):
        key = str(row.get("key", ""))
        if key not in wanted_ids:
            continue
        if key in selected:
            duplicate_ids.add(key)
            continue
        selected[key] = _normalize_row(row)

    if duplicate_ids:
        preview = ", ".join(sorted(duplicate_ids)[:5])
        raise ValueError(f"SearchQA source dataset contains duplicate manifest IDs. First IDs: {preview}")

    missing = sorted(wanted_ids - selected.keys())
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"SearchQA source dataset is missing {len(missing)} manifest IDs. First IDs: {preview}")

    counts = {}
    for split, ids in split_ids.items():
        items = [selected[item_id] for item_id in ids]
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "items.json").open("w", encoding="utf-8") as file:
            json.dump(items, file, ensure_ascii=False, indent=2)
        counts[split] = len(items)

    manifest = {
        "source_manifest_dir": str(manifest_dir),
        "source_dataset": dataset_name,
        "counts": counts,
        "item_fields": ["id", *REQUIRED_FIELDS],
    }
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    return counts


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install it with:\n"
            "  python -m pip install 'skillopt[searchqa]'\n"
            "or:\n"
            "  python -m pip install datasets"
        ) from exc

    print(f"Loading {args.dataset}...")
    dataset = load_dataset(args.dataset)
    counts = materialize_searchqa_splits(
        args.manifest_dir,
        args.output_dir,
        dataset,
        dataset_name=args.dataset,
    )
    print(f"Wrote SearchQA splits to {args.output_dir.resolve()}: {counts}")


if __name__ == "__main__":
    main()
