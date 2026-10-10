from __future__ import annotations

import json
from pathlib import Path

from meisenmeister.dataloading import MeisenmeisterROIDataset


def _get_splits_path(preprocessed_dataset_dir: Path) -> Path:
    return preprocessed_dataset_dir / "splits.json"


def _normalize_fold_entry(entry: object, fold_index: int) -> dict[str, list[str]]:
    if not isinstance(entry, dict):
        raise ValueError(f"Fold {fold_index} must be a JSON object")

    train_case_ids = entry.get("train")
    val_case_ids = entry.get("val")
    if not isinstance(train_case_ids, list) or not all(
        isinstance(case_id, str) for case_id in train_case_ids
    ):
        raise ValueError(f"Fold {fold_index} must define 'train' as a list of case ids")
    if not isinstance(val_case_ids, list) or not all(
        isinstance(case_id, str) for case_id in val_case_ids
    ):
        raise ValueError(f"Fold {fold_index} must define 'val' as a list of case ids")

    return {
        "train": sorted(set(train_case_ids)),
        "val": sorted(set(val_case_ids)),
    }


def _normalize_split_ids_to_sample_ids(
    dataset: MeisenmeisterROIDataset,
    split_ids: list[str],
    fold_index: int,
    split_name: str,
) -> list[str]:
    known_sample_ids = {sample["sample_id"] for sample in dataset.samples}
    case_id_to_sample_ids: dict[str, list[str]] = {}
    sample_id_to_case_id: dict[str, str] = {}
    for sample in dataset.samples:
        case_id_to_sample_ids.setdefault(sample["case_id"], []).append(
            sample["sample_id"]
        )
        sample_id_to_case_id[sample["sample_id"]] = sample["case_id"]

    normalized_sample_ids: list[str] = []
    unknown_ids: list[str] = []
    for split_id in split_ids:
        if split_id in known_sample_ids:
            normalized_sample_ids.append(split_id)
            continue
        if split_id in case_id_to_sample_ids:
            normalized_sample_ids.extend(case_id_to_sample_ids[split_id])
            continue
        unknown_ids.append(split_id)

    if unknown_ids:
        unknown_str = ", ".join(sorted(unknown_ids))
        raise ValueError(
            f"Fold {fold_index} contains unknown ids in '{split_name}': {unknown_str}"
        )

    return sorted(set(normalized_sample_ids))


def load_splits(preprocessed_dataset_dir: Path) -> list[dict[str, list[str]]]:
    splits_path = _get_splits_path(preprocessed_dataset_dir)
    if not splits_path.is_file():
        raise FileNotFoundError(
            f"Missing splits file: {splits_path}. Run mm_create_5fold first."
        )

    with splits_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("splits.json must contain a list of folds")

    return [
        _normalize_fold_entry(entry, fold_index)
        for fold_index, entry in enumerate(payload)
    ]


def get_fold_sample_ids(
    preprocessed_dataset_dir: Path,
    fold: int | str,
) -> dict[str, list[str]]:
    if fold == "all":
        dataset = MeisenmeisterROIDataset(preprocessed_dataset_dir)
        sample_ids = sorted(sample["sample_id"] for sample in dataset.samples)
        return {
            "train": sample_ids,
            "val": sample_ids,
        }

    if not isinstance(fold, int):
        raise ValueError(f"Fold must be an integer or 'all', got {fold!r}")

    splits = load_splits(preprocessed_dataset_dir)
    if fold < 0 or fold >= len(splits):
        raise ValueError(
            f"Fold {fold} does not exist in splits.json. Available folds: 0-{len(splits) - 1}"
        )

    dataset = MeisenmeisterROIDataset(preprocessed_dataset_dir)
    fold_entry = splits[fold]
    train_sample_ids = _normalize_split_ids_to_sample_ids(
        dataset,
        fold_entry["train"],
        fold,
        "train",
    )
    val_sample_ids = _normalize_split_ids_to_sample_ids(
        dataset,
        fold_entry["val"],
        fold,
        "val",
    )

    sample_id_to_case_id = {
        sample["sample_id"]: sample["case_id"] for sample in dataset.samples
    }
    train_case_ids = {sample_id_to_case_id[sample_id] for sample_id in train_sample_ids}
    val_case_ids = {sample_id_to_case_id[sample_id] for sample_id in val_sample_ids}
    overlap = train_case_ids & val_case_ids
    if overlap:
        overlap_str = ", ".join(sorted(overlap))
        raise ValueError(
            f"Fold {fold} leaks case ids between train and val: {overlap_str}"
        )

    return {
        "train": train_sample_ids,
        "val": val_sample_ids,
    }


def create_five_fold_splits(preprocessed_dataset_dir: Path) -> Path:
    dataset = MeisenmeisterROIDataset(preprocessed_dataset_dir)
    case_ids = sorted({sample["case_id"] for sample in dataset.samples})
    case_id_to_sample_ids: dict[str, list[str]] = {}
    for sample in dataset.samples:
        case_id_to_sample_ids.setdefault(sample["case_id"], []).append(
            sample["sample_id"]
        )
    num_folds = 5
    if len(case_ids) < num_folds:
        raise ValueError(
            f"Need at least {num_folds} unique case ids for clean 5-fold splits, got {len(case_ids)}"
        )

    val_case_ids_per_fold = [
        case_ids[fold_index::num_folds] for fold_index in range(num_folds)
    ]
    splits: list[dict[str, list[str]]] = []
    for fold_index, val_case_ids in enumerate(val_case_ids_per_fold):
        train_case_ids = [
            case_id
            for other_fold, other_case_ids in enumerate(val_case_ids_per_fold)
            if other_fold != fold_index
            for case_id in other_case_ids
        ]
        splits.append(
            {
                "train": sorted(
                    sample_id
                    for case_id in train_case_ids
                    for sample_id in case_id_to_sample_ids[case_id]
                ),
                "val": sorted(
                    sample_id
                    for case_id in val_case_ids
                    for sample_id in case_id_to_sample_ids[case_id]
                ),
            }
        )

    splits_path = _get_splits_path(preprocessed_dataset_dir)
    with splits_path.open("w", encoding="utf-8") as file:
        json.dump(splits, file, indent=2)
        file.write("\n")

    return splits_path
