"""Materialize runnable DocVQA splits from the released ID manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "docvqa_id_split",
        help="Directory containing train/val/test DocVQA ID manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "docvqa" / "splits",
        help="Directory to write runnable train/val/test splits.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "docvqa_images",
        help="Directory to write selected DocVQA page images.",
    )
    parser.add_argument(
        "--dataset",
        default="lmms-lab/DocVQA",
        help="Hugging Face dataset repository to load.",
    )
    parser.add_argument(
        "--config",
        default="DocVQA",
        help="Hugging Face dataset config to load.",
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
            "DocVQA split manifest contains duplicate IDs across splits. "
            f"First IDs: {preview}"
        )


def _answers(row: dict) -> list[str]:
    raw = row.get("answers") or row.get("answer") or row.get("ground_truth") or []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw]


def _normalize_row(row: dict, manifest_item: dict, image_dir: Path) -> dict:
    question_id = str(row["questionId"])
    doc_id = str(row.get("docId", manifest_item.get("docId", "")))
    image_path = image_dir / f"q{question_id}_d{doc_id}.png"
    image = row.get("image")
    if image is None:
        raise ValueError(f"DocVQA row {question_id!r} is missing image payload")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        image.save(image_path)

    answers = _answers(row)
    return {
        "id": question_id,
        "questionId": question_id,
        "docId": str(doc_id),
        "question": str(row["question"]),
        "answer": answers[0] if answers else "",
        "answers": answers,
        "ground_truth": answers,
        "image_path": str(image_path),
        "image_paths": [str(image_path)],
        "question_types": row.get("question_types", []),
        "ucsf_document_id": str(row.get("ucsf_document_id", "")),
        "ucsf_document_page_no": str(row.get("ucsf_document_page_no", "")),
        "source_split": str(manifest_item.get("source_split") or row.get("data_split") or "validation"),
        "source_dataset": "lmms-lab/DocVQA",
        "source_config": "DocVQA",
    }


def materialize_docvqa_splits(
    manifest_dir: Path,
    output_dir: Path,
    image_dir: Path,
    dataset: Iterable[dict],
    *,
    dataset_name: str,
    dataset_config: str,
) -> dict[str, int]:
    manifest_dir = manifest_dir.resolve()
    output_dir = output_dir.resolve()
    image_dir = image_dir.resolve()
    split_items = load_manifest_items(manifest_dir)
    _reject_duplicate_manifest_ids(split_items)

    item_by_id = {
        str(item["id"]): item
        for items in split_items.values()
        for item in items
    }
    wanted_ids = set(item_by_id)

    selected: dict[str, dict] = {}
    duplicate_ids: set[str] = set()
    for row in dataset:
        question_id = str(row.get("questionId", ""))
        if question_id not in wanted_ids:
            continue
        if question_id in selected:
            duplicate_ids.add(question_id)
            continue
        selected[question_id] = _normalize_row(row, item_by_id[question_id], image_dir)

    if duplicate_ids:
        preview = ", ".join(sorted(duplicate_ids)[:5])
        raise ValueError(f"DocVQA source dataset contains duplicate manifest IDs. First IDs: {preview}")

    missing = sorted(wanted_ids - selected.keys())
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"DocVQA source dataset is missing {len(missing)} manifest IDs. First IDs: {preview}")

    counts = {}
    for split, manifest_items in split_items.items():
        items = [selected[str(item["id"])] for item in manifest_items]
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "items.json").open("w", encoding="utf-8") as file:
            json.dump(items, file, ensure_ascii=False, indent=2)
        counts[split] = len(items)

    manifest = {
        "source_manifest_dir": str(manifest_dir),
        "source_dataset": dataset_name,
        "source_config": dataset_config,
        "source_split": "validation",
        "counts": counts,
        "image_dir": str(image_dir),
        "item_fields": [
            "id",
            "question",
            "answer",
            "answers",
            "ground_truth",
            "image_path",
            "image_paths",
        ],
    }
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    return counts


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency 'datasets'. Install it with: python -m pip install datasets") from exc

    print(f"Loading {args.dataset}/{args.config} validation split...")
    dataset = load_dataset(args.dataset, args.config, split="validation")
    counts = materialize_docvqa_splits(
        args.manifest_dir,
        args.output_dir,
        args.image_dir,
        dataset,
        dataset_name=args.dataset,
        dataset_config=args.config,
    )
    print(f"Wrote DocVQA splits to {args.output_dir.resolve()}: {counts}")


if __name__ == "__main__":
    main()
