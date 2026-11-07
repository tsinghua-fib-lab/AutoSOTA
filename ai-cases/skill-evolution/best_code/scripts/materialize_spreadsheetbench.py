"""Materialize runnable SpreadsheetBench splits from Verified 400 payload."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
REQUIRED_FIELDS = (
    "id",
    "instruction",
    "spreadsheet_path",
    "instruction_type",
    "answer_position",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "spreadsheetbench_id_split",
        help="Directory containing train/val/test SpreadsheetBench ID manifests.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "spreadsheetbench_verified_400",
        help="Directory containing dataset.json and spreadsheet/<id> directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "spreadsheetbench_split",
        help="Directory to write runnable train/val/test splits.",
    )
    return parser.parse_args()


def load_manifest_items(manifest_dir: Path) -> dict[str, list[dict]]:
    split_items = {}
    for split in SPLITS:
        path = manifest_dir / split / "items.json"
        with path.open(encoding="utf-8") as file:
            split_items[split] = json.load(file)
    return split_items


def _reject_duplicate_manifest_ids(split_items: Mapping[str, Iterable[dict]]) -> None:
    seen: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for split, items in split_items.items():
        for item in items:
            item_id = str(item["id"])
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
            "SpreadsheetBench split manifest contains duplicate IDs across splits. "
            f"First IDs: {preview}"
        )


def _normalize_row(row: dict, manifest_item: dict, data_root: Path) -> dict:
    item_id = str(row.get("id") or manifest_item["id"])
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"SpreadsheetBench source row {item_id!r} is missing fields: {', '.join(missing)}")

    spreadsheet_path = str(row.get("spreadsheet_path") or manifest_item.get("spreadsheet_path") or f"spreadsheet/{item_id}")
    task_dir = data_root / spreadsheet_path
    if not task_dir.exists():
        raise FileNotFoundError(f"SpreadsheetBench task directory not found for {item_id!r}: {task_dir}")

    normalized = dict(row)
    normalized.update(
        {
            "id": item_id,
            "spreadsheet_path": spreadsheet_path,
            "answer_sheet": str(row.get("answer_sheet", "")),
            "source_dataset": "KAKA22/SpreadsheetBench",
            "source_payload": "spreadsheetbench_verified_400.tar.gz",
        }
    )
    return normalized


def materialize_spreadsheetbench_splits(
    manifest_dir: Path,
    data_root: Path,
    output_dir: Path,
) -> dict[str, int]:
    manifest_dir = manifest_dir.resolve()
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()

    dataset_path = data_root / "dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing SpreadsheetBench dataset.json: {dataset_path}")

    with dataset_path.open(encoding="utf-8") as file:
        source_rows = json.load(file)

    split_items = load_manifest_items(manifest_dir)
    _reject_duplicate_manifest_ids(split_items)
    source_by_id = {str(row["id"]): row for row in source_rows}

    wanted_ids = {
        str(item["id"])
        for items in split_items.values()
        for item in items
    }
    missing = sorted(wanted_ids - source_by_id.keys())
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"SpreadsheetBench source dataset is missing {len(missing)} manifest IDs. First IDs: {preview}")

    counts = {}
    for split, manifest_items in split_items.items():
        items = [
            _normalize_row(source_by_id[str(item["id"])], item, data_root)
            for item in manifest_items
        ]
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "items.json").open("w", encoding="utf-8") as file:
            json.dump(items, file, ensure_ascii=False, indent=2)
        counts[split] = len(items)

    manifest = {
        "source_manifest_dir": str(manifest_dir),
        "data_root": str(data_root),
        "source_dataset": "KAKA22/SpreadsheetBench",
        "source_payload": "spreadsheetbench_verified_400.tar.gz",
        "counts": counts,
        "item_fields": list(REQUIRED_FIELDS),
    }
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    return counts


def main() -> None:
    args = parse_args()
    counts = materialize_spreadsheetbench_splits(args.manifest_dir, args.data_root, args.output_dir)
    print(f"Wrote SpreadsheetBench splits to {args.output_dir.resolve()}: {counts}")


if __name__ == "__main__":
    main()
