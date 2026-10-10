from __future__ import annotations

import csv
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

SOURCE_ROOT = Path("")
OUTPUT_DATASET_DIR = Path("")

CHANNEL_FILES = {
    0: "Pre.nii.gz",
    1: "Post_1.nii.gz",
    2: "Post_2.nii.gz",
}
VALID_LABEL_VALUES = {0, 1, 2}


@dataclass(frozen=True)
class OdeliaCase:
    center: str
    case_id: str
    channel_paths: dict[int, Path]
    labels: dict[str, int]


@dataclass(frozen=True)
class CopyTask:
    source_path: Path
    target_path: Path


def _load_center_annotations(center_dir: Path) -> dict[str, dict[str, int]]:
    annotation_path = center_dir / "metadata" / "annotation.csv"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    annotations: dict[str, dict[str, int]] = {}
    with annotation_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            case_id = row.get("UID")
            if not case_id:
                raise ValueError(f"Missing UID value in {annotation_path}")
            if case_id in annotations:
                raise ValueError(
                    f"Duplicate annotation row for case '{case_id}' in {annotation_path}"
                )

            labels = {}
            for roi_name, column_name in (
                ("left", "Lesion_Left"),
                ("right", "Lesion_Right"),
            ):
                value = row.get(column_name)
                try:
                    label = int(value) if value is not None else None
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid label value '{value}' for {case_id} {roi_name} in {annotation_path}"
                    ) from exc
                if label not in VALID_LABEL_VALUES:
                    raise ValueError(
                        f"Unexpected label value '{value}' for {case_id} {roi_name} in {annotation_path}"
                    )
                labels[roi_name] = label

            annotations[case_id] = labels

    return annotations


def collect_odelia_cases(source_root: Path) -> list[OdeliaCase]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing source root: {source_root}")

    cases: list[OdeliaCase] = []
    seen_case_ids: set[str] = set()

    for center_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        data_dir = center_dir / "data"
        if not data_dir.is_dir():
            continue

        annotations = _load_center_annotations(center_dir)
        center_case_ids: list[str] = []

        for case_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
            case_id = case_dir.name
            if case_id in seen_case_ids:
                raise ValueError(f"Duplicate output case id detected: {case_id}")

            channel_paths: dict[int, Path] = {}
            missing_files: list[str] = []
            for channel_id, filename in CHANNEL_FILES.items():
                source_path = case_dir / filename
                if not source_path.is_file():
                    missing_files.append(filename)
                else:
                    channel_paths[channel_id] = source_path

            if missing_files:
                missing_str = ", ".join(missing_files)
                raise FileNotFoundError(
                    f"Case '{case_id}' is missing required files: {missing_str}"
                )

            if case_id not in annotations:
                raise ValueError(
                    f"Missing annotation row for converted case '{case_id}' in {center_dir / 'metadata' / 'annotation.csv'}"
                )

            cases.append(
                OdeliaCase(
                    center=center_dir.name,
                    case_id=case_id,
                    channel_paths=channel_paths,
                    labels=annotations[case_id],
                )
            )
            center_case_ids.append(case_id)
            seen_case_ids.add(case_id)

        unknown_annotations = sorted(set(annotations) - set(center_case_ids))
        if unknown_annotations:
            unknown_str = ", ".join(unknown_annotations)
            raise ValueError(
                f"Found annotations without matching case folders in {center_dir}: {unknown_str}"
            )

    if not cases:
        raise ValueError(f"No Odelia cases found in {source_root}")

    return cases


def build_dataset_json(num_cases: int) -> dict:
    return {
        "channel_names": {
            "0": "pre",
            "1": "post1",
            "2": "post2",
        },
        "file_ending": ".nii.gz",
        "problem_type": "classification",
        "labels": {
            "0": "healthy",
            "1": "benign",
            "2": "malignant",
        },
        "numTraining": num_cases,
    }


def build_labels_json(cases: list[OdeliaCase]) -> dict[str, int]:
    labels: dict[str, list[int]] = {}
    for case in cases:
        for roi_name, label in case.labels.items():
            one_hot = [0, 0, 0]
            one_hot[label] = 1
            labels[f"{case.case_id}_{roi_name}"] = one_hot
    return labels


def build_centerwise_splits(cases: list[OdeliaCase]) -> list[dict[str, list[str]]]:
    center_to_case_ids: dict[str, list[str]] = {}
    for case in cases:
        center_to_case_ids.setdefault(case.center, []).append(case.case_id)

    all_case_ids = sorted(case.case_id for case in cases)
    splits: list[dict[str, list[str]]] = []
    for center in sorted(center_to_case_ids):
        val_case_ids = sorted(center_to_case_ids[center])
        val_case_id_set = set(val_case_ids)
        train_case_ids = [
            case_id for case_id in all_case_ids if case_id not in val_case_id_set
        ]
        splits.append(
            {
                "train": train_case_ids,
                "val": val_case_ids,
            }
        )

    return splits


def build_copy_tasks(cases: list[OdeliaCase], images_tr_dir: Path) -> list[CopyTask]:
    copy_tasks: list[CopyTask] = []
    for case in cases:
        for channel_id, source_path in case.channel_paths.items():
            copy_tasks.append(
                CopyTask(
                    source_path=source_path,
                    target_path=images_tr_dir
                    / f"{case.case_id}_{channel_id:04d}.nii.gz",
                )
            )
    return copy_tasks


def _copy_file(task: CopyTask) -> bool:
    if task.target_path.exists():
        return False
    shutil.copy2(task.source_path, task.target_path)
    return True


def copy_case_files(copy_tasks: list[CopyTask], max_workers: int | None = None) -> None:
    if not copy_tasks:
        return

    resolved_max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
    resolved_max_workers = max(1, min(resolved_max_workers, len(copy_tasks)))

    if resolved_max_workers == 1:
        for task in tqdm(copy_tasks, desc="Copying Odelia images", unit="file"):
            _copy_file(task)
        return

    with ThreadPoolExecutor(max_workers=resolved_max_workers) as executor:
        futures = [executor.submit(_copy_file, task) for task in copy_tasks]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Copying Odelia images",
            unit="file",
        ):
            future.result()


def write_odelia_dataset(source_root: Path, output_dataset_dir: Path) -> Path:
    cases = collect_odelia_cases(source_root)
    images_tr_dir = output_dataset_dir / "imagesTr"
    images_tr_dir.mkdir(parents=True, exist_ok=True)

    copy_case_files(build_copy_tasks(cases, images_tr_dir))

    dataset_json = build_dataset_json(len(cases))
    labels_json = build_labels_json(cases)
    splits = build_centerwise_splits(cases)

    for file_name, payload in (
        ("dataset.json", dataset_json),
        ("labelsTr.json", labels_json),
        ("splits.json", splits),
    ):
        output_path = output_dataset_dir / file_name
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")

    return output_dataset_dir


def main() -> Path:
    return write_odelia_dataset(SOURCE_ROOT, OUTPUT_DATASET_DIR)


if __name__ == "__main__":
    main()
