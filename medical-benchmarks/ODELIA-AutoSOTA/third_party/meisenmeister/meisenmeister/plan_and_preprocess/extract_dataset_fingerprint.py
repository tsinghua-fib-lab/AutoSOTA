import json
import shutil

from meisenmeister.utils import (
    extract_roi_fingerprint_from_masks,
    find_dataset_dir,
    load_dataset_json,
    require_global_paths_set,
    verify_required_global_paths_set,
    verify_training_files_present,
)


@require_global_paths_set
def extract_dataset_fingerprint(d: int, num_workers: int = 4):
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")

    dataset_id = f"{d:03d}"
    paths = verify_required_global_paths_set()
    mm_raw = paths["mm_raw"]
    mm_preprocessed = paths["mm_preprocessed"]
    dataset_dir = find_dataset_dir(mm_raw, dataset_id)
    dataset_json = load_dataset_json(dataset_dir)
    case_files = verify_training_files_present(dataset_dir, dataset_json)
    dataset_fingerprint = extract_roi_fingerprint_from_masks(
        dataset_dir, dataset_json, case_files, num_workers=num_workers
    )

    output_dir = mm_preprocessed / dataset_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "dataset_fingerprint.json").open("w", encoding="utf-8") as file:
        json.dump(dataset_fingerprint, file, indent=2)

    shutil.copy2(dataset_dir / "dataset.json", output_dir / "dataset.json")

    print(output_dir / "dataset_fingerprint.json")
    return dataset_fingerprint
