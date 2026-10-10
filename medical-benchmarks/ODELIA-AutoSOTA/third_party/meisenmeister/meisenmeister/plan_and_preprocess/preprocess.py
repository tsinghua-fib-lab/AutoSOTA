import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import blosc2
from tqdm import tqdm

from meisenmeister.plan_and_preprocess.preprocessing_utils import (
    load_case_image_data,
    load_mm_plans,
    preprocess_roi_array,
)
from meisenmeister.utils import (
    find_dataset_dir,
    load_dataset_json,
    require_global_paths_set,
    verify_required_global_paths_set,
    verify_training_files_present,
)


def _preprocess_case(
    case_id: str,
    case_file_paths: list[Path],
    dataset_json: dict,
    masks_tr_dir: Path,
    file_ending: str,
    plans: dict,
    output_dir: Path,
) -> None:
    mask_path = masks_tr_dir / f"{case_id}{file_ending}"
    import SimpleITK as sitk

    mask_image = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(mask_image)
    spacing = [float(i) for i in mask_image.GetSpacing()[::-1]]
    image_data = load_case_image_data(case_file_paths, dataset_json)

    target_spacing = [float(i) for i in plans["target_spacing"]]
    target_shape = [int(i) for i in plans["target_shape"]]
    margin_mm = [float(i) for i in plans["margin_mm"]]

    for roi_name, roi_label in plans["roi_labels"].items():
        output_path = output_dir / f"{case_id}_{roi_name}.b2nd"
        full_image = preprocess_roi_array(
            image_data,
            mask_array,
            spacing,
            roi_label=roi_label,
            target_spacing=target_spacing,
            target_shape=target_shape,
            margin_mm=margin_mm,
        )
        blosc2.asarray(
            full_image,
            urlpath=str(output_path),
            mode="w",
            chunks=full_image.shape,
        )


@require_global_paths_set
def preprocess(d: int, num_workers: int = 4) -> Path:
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers}")

    dataset_id = f"{d:03d}"
    paths = verify_required_global_paths_set()
    mm_raw = paths["mm_raw"]
    mm_preprocessed = paths["mm_preprocessed"]

    dataset_dir = find_dataset_dir(mm_raw, dataset_id)
    preprocessed_dataset_dir = mm_preprocessed / dataset_dir.name
    plans = load_mm_plans(preprocessed_dataset_dir / "mmPlans.json")
    dataset_json = load_dataset_json(dataset_dir)
    case_files = verify_training_files_present(dataset_dir, dataset_json)

    masks_tr_dir = dataset_dir / "masksTr"
    output_dir = preprocessed_dataset_dir / plans["output_folder_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_tr_path = dataset_dir / "labelsTr.json"
    if labels_tr_path.is_file():
        shutil.copy2(labels_tr_path, preprocessed_dataset_dir / "labelsTr.json")

    target_spacing = [float(i) for i in plans["target_spacing"]]
    target_shape = [int(i) for i in plans["target_shape"]]
    margin_mm = [float(i) for i in plans["margin_mm"]]
    file_ending = dataset_json["file_ending"]
    _ = target_spacing, target_shape, margin_mm

    sorted_case_ids = sorted(case_files)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _preprocess_case,
                case_id,
                case_files[case_id],
                dataset_json,
                masks_tr_dir,
                file_ending,
                plans,
                output_dir,
            ): case_id
            for case_id in sorted_case_ids
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Preprocessing ROI crops",
        ):
            future.result()

    print(output_dir)
    return output_dir
