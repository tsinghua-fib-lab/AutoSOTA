import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import median

from tqdm import tqdm

REQUIRED_DATASET_JSON_KEYS = (
    "channel_names",
    "file_ending",
    "problem_type",
    "labels",
    "numTraining",
)
OPTIONAL_MASKS_TR_METADATA_FILES = {
    "dataset.json",
    "plans.json",
    "predict_from_raw_data_args.json",
}


def load_dataset_json(
    dataset_dir: Path,
    *,
    resolve_training_cases: bool = True,
) -> dict:
    dataset_json_path = dataset_dir / "dataset.json"
    if not dataset_json_path.is_file():
        raise FileNotFoundError(f"Missing dataset.json in {dataset_dir}")

    with dataset_json_path.open("r", encoding="utf-8") as file:
        dataset_json = json.load(file)

    missing_keys = [
        key for key in REQUIRED_DATASET_JSON_KEYS if key not in dataset_json
    ]
    if missing_keys:
        missing_keys_str = ", ".join(missing_keys)
        raise ValueError(
            f"dataset.json in {dataset_dir} is missing required keys: {missing_keys_str}"
        )

    if resolve_training_cases and "training_cases" in dataset_json:
        _resolve_training_case_files(dataset_dir, dataset_json)

    return dataset_json


def _resolve_training_case_path(dataset_dir: Path, path_value: str) -> Path:
    source_path = Path(path_value)
    if source_path.is_absolute():
        return source_path.resolve(strict=False)
    return (dataset_dir / source_path).resolve(strict=False)


def _sort_channel_ids(channel_ids: set[str]) -> list[str]:
    return sorted(channel_ids, key=int)


def _resolve_training_case_files(
    dataset_dir: Path,
    dataset_json: dict,
) -> dict[str, list[Path]]:
    training_cases = dataset_json["training_cases"]
    if not isinstance(training_cases, dict):
        raise TypeError(
            "dataset.json key 'training_cases' must be a mapping of case ids to channel path mappings"
        )

    expected_num_cases = int(dataset_json["numTraining"])
    if len(training_cases) != expected_num_cases:
        raise ValueError(
            f"dataset.json declares numTraining={expected_num_cases}, but "
            f"training_cases contains {len(training_cases)} cases"
        )

    channel_names = dataset_json["channel_names"]
    expected_channel_ids = _sort_channel_ids(set(channel_names))
    expected_channel_id_set = set(expected_channel_ids)
    file_ending = dataset_json["file_ending"]

    resolved_case_files: dict[str, list[Path]] = {}
    for case_id, case_channels in sorted(training_cases.items()):
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(
                "dataset.json training_cases keys must be non-empty case id strings"
            )
        if not isinstance(case_channels, dict):
            raise TypeError(
                f"dataset.json training_cases['{case_id}'] must be a mapping of channel ids to file paths"
            )

        case_channel_ids = set(case_channels)
        missing_channel_ids = [
            channel_id
            for channel_id in expected_channel_ids
            if channel_id not in case_channel_ids
        ]
        extra_channel_ids = _sort_channel_ids(
            {
                channel_id
                for channel_id in case_channel_ids
                if channel_id not in expected_channel_id_set
            }
        )
        if missing_channel_ids or extra_channel_ids:
            raise ValueError(
                f"dataset.json training_cases['{case_id}'] must define exactly "
                f"channels {expected_channel_ids}; missing={missing_channel_ids}, "
                f"extra={extra_channel_ids}"
            )

        resolved_files: list[Path] = []
        for channel_id in expected_channel_ids:
            path_value = case_channels[channel_id]
            if not isinstance(path_value, str) or not path_value:
                raise TypeError(
                    f"dataset.json training_cases['{case_id}']['{channel_id}'] must be a non-empty filepath string"
                )

            resolved_path = _resolve_training_case_path(dataset_dir, path_value)
            if not resolved_path.name.endswith(file_ending):
                raise ValueError(
                    f"dataset.json training_cases['{case_id}']['{channel_id}'] must end "
                    f"with '{file_ending}', got '{resolved_path.name}'"
                )
            if not resolved_path.is_file():
                raise FileNotFoundError(
                    f"Missing training_cases file for case '{case_id}' channel "
                    f"'{channel_id}': {resolved_path}"
                )
            resolved_files.append(resolved_path)

        resolved_case_files[case_id] = resolved_files

    return resolved_case_files


def verify_training_files_present(
    dataset_dir: Path, dataset_json: dict
) -> dict[str, list[Path]]:
    if "training_cases" in dataset_json:
        return _resolve_training_case_files(dataset_dir, dataset_json)

    images_tr_dir = dataset_dir / "imagesTr"
    return discover_case_files(
        images_tr_dir,
        dataset_json,
        expected_num_cases=int(dataset_json["numTraining"]),
    )


def discover_case_files(
    images_dir: Path,
    dataset_json: dict,
    *,
    expected_num_cases: int | None = None,
) -> dict[str, list[Path]]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing input directory: {images_dir}")

    channel_names = dataset_json["channel_names"]
    file_ending = dataset_json["file_ending"]
    num_training = expected_num_cases

    print(
        f"Verifying input files in {images_dir}"
        + (f" for {num_training} cases" if num_training is not None else "")
        + " "
        f"with channels {channel_names} and file ending '{file_ending}'"
    )

    channel_ids = sorted(int(channel_id) for channel_id in channel_names)
    expected_suffixes = [
        f"_{channel_id:04d}{file_ending}" for channel_id in channel_ids
    ]
    expected_suffix_set = set(expected_suffixes)

    case_files: dict[str, list[Path]] = {}
    unexpected_files: list[str] = []
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue

        matched = False
        for suffix in expected_suffixes:
            if path.name.endswith(suffix):
                case_id = path.name[: -len(suffix)]
                if case_id:
                    case_files.setdefault(case_id, []).append(path)
                    matched = True
                break
        if not matched:
            unexpected_files.append(path.name)

    found_case_ids = sorted(case_files)
    print(f"Found case ids in {images_dir.name}: {found_case_ids}")

    if unexpected_files:
        unexpected_files_str = ", ".join(unexpected_files)
        raise ValueError(
            f"Found unexpected files in {images_dir}: {unexpected_files_str}"
        )

    if expected_num_cases is not None and len(case_files) != expected_num_cases:
        raise ValueError(
            f"Expected {expected_num_cases} cases in {images_dir}, found "
            f"{len(case_files)}: {found_case_ids}"
        )

    expected_channel_count = len(channel_ids)
    invalid_cases: list[str] = []
    for case_id, files in case_files.items():
        present_suffixes = sorted(path.name[len(case_id) :] for path in files)
        present_suffix_set = set(present_suffixes)
        missing_suffixes = sorted(
            suffix for suffix in expected_suffixes if suffix not in present_suffix_set
        )
        if len(files) != expected_channel_count:
            invalid_cases.append(
                f"{case_id}: found {len(files)} channel files, expected "
                f"{expected_channel_count}; present={present_suffixes}"
            )
            continue

        extra_suffixes = sorted(
            suffix for suffix in present_suffix_set if suffix not in expected_suffix_set
        )
        if missing_suffixes or extra_suffixes:
            parts: list[str] = [f"{case_id}: present={present_suffixes}"]
            if missing_suffixes:
                parts.append(f"missing={missing_suffixes}")
            if extra_suffixes:
                parts.append(f"unexpected={extra_suffixes}")
            invalid_cases.append("; ".join(parts))

    if invalid_cases:
        raise ValueError(
            "Input files are incomplete or inconsistent in "
            f"{images_dir}: {'; '.join(invalid_cases)}"
        )

    return case_files


def verify_roi_masks_present(
    dataset_dir: Path,
    dataset_json: dict,
    case_files: dict[str, list[Path]],
):
    masks_tr_dir = dataset_dir / "masksTr"
    if not masks_tr_dir.is_dir():
        raise FileNotFoundError(f"Missing masksTr directory in {dataset_dir}")

    file_ending = dataset_json["file_ending"]
    expected_mask_names = {f"{case_id}{file_ending}" for case_id in case_files}

    unexpected_masks = sorted(
        path.name
        for path in masks_tr_dir.iterdir()
        if path.is_file()
        and path.name not in expected_mask_names
        and path.name not in OPTIONAL_MASKS_TR_METADATA_FILES
    )
    if unexpected_masks:
        raise ValueError(
            f"Found unexpected mask files in {masks_tr_dir}: {', '.join(unexpected_masks)}"
        )

    invalid_cases: list[str] = []
    for case_id in sorted(case_files):
        mask_path = masks_tr_dir / f"{case_id}{file_ending}"
        if not mask_path.is_file():
            invalid_cases.append(f"{case_id}: missing mask file {mask_path.name}")
            continue

    if invalid_cases:
        raise ValueError(
            "ROI masks are incomplete or inconsistent in "
            f"{masks_tr_dir}: {'; '.join(invalid_cases)}"
        )


def _read_mask_geometry_for_label(mask_path: Path, roi_label: int):
    import numpy as np
    import SimpleITK as sitk

    mask = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayViewFromImage(mask)
    roi_voxels = np.argwhere(mask_array == roi_label)
    if roi_voxels.size == 0:
        raise ValueError(f"ROI label {roi_label} not present in {mask_path.name}")

    min_corner = roi_voxels.min(axis=0)
    max_corner = roi_voxels.max(axis=0)
    crop_shape = (max_corner - min_corner + 1).astype(int).tolist()
    full_shape = [int(i) for i in mask_array.shape]
    spacing = [float(i) for i in mask.GetSpacing()[::-1]]
    return spacing, full_shape, crop_shape


def extract_roi_fingerprint_from_masks(
    dataset_dir: Path,
    dataset_json: dict,
    case_files: dict[str, list[Path]],
    num_workers: int = 4,
) -> dict:
    verify_roi_masks_present(dataset_dir, dataset_json, case_files)

    file_ending = dataset_json["file_ending"]
    roi_labels = {"left": 1, "right": 2}
    spacings: list[list[float]] = []
    shapes_after_crop: list[list[int]] = []
    full_shapes: list[list[int]] = []

    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers}")

    def _process_case(case_id: str):
        mask_path = dataset_dir / "masksTr" / f"{case_id}{file_ending}"
        case_spacings: list[list[float]] = []
        case_full_shapes: list[list[int]] = []
        case_shapes_after_crop: list[list[int]] = []
        for roi_label in (1, 2):
            spacing, full_shape, crop_shape = _read_mask_geometry_for_label(
                mask_path, roi_label
            )
            case_spacings.append(spacing)
            case_full_shapes.append(full_shape)
            case_shapes_after_crop.append(crop_shape)
        return case_id, case_spacings, case_full_shapes, case_shapes_after_crop

    sorted_case_ids = sorted(case_files)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_case, case_id): case_id
            for case_id in sorted_case_ids
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Extracting ROI fingerprint",
        ):
            _, case_spacings, case_full_shapes, case_shapes_after_crop = future.result()
            spacings.extend(case_spacings)
            full_shapes.extend(case_full_shapes)
            shapes_after_crop.extend(case_shapes_after_crop)

    median_spacing = [
        float(median(axis_values)) for axis_values in zip(*spacings, strict=True)
    ]
    median_shape_after_crop = [
        int(round(median(axis_values)))
        for axis_values in zip(*shapes_after_crop, strict=True)
    ]

    return {
        "num_cases": len(case_files),
        "num_rois": len(shapes_after_crop),
        "foreground_labels": roi_labels,
        "spacings": spacings,
        "shapes_after_crop": shapes_after_crop,
        "full_shapes": full_shapes,
        "median_spacing": median_spacing,
        "median_shape_after_crop": median_shape_after_crop,
        "normalization": "per_case_zscore",
    }
