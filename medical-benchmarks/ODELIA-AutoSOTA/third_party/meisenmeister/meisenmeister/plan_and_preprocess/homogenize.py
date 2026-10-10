import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm

from meisenmeister.utils import (
    find_dataset_dir,
    load_dataset_json,
    require_global_paths_set,
    verify_required_global_paths_set,
    verify_training_files_present,
)


def _resample_to_reference(
    moving_image: sitk.Image,
    reference_image: sitk.Image,
) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving_image)


def _matches_reference_geometry(
    moving_image: sitk.Image,
    reference_image: sitk.Image,
) -> bool:
    return (
        moving_image.GetSize() == reference_image.GetSize()
        and moving_image.GetSpacing() == reference_image.GetSpacing()
        and moving_image.GetOrigin() == reference_image.GetOrigin()
        and moving_image.GetDirection() == reference_image.GetDirection()
    )


def _homogenize_single_image(
    moving_path: Path,
    reference_path: Path,
) -> None:
    reference_image = sitk.ReadImage(str(reference_path))
    moving_image = sitk.ReadImage(str(moving_path))
    if _matches_reference_geometry(moving_image, reference_image):
        return
    resampled_image = _resample_to_reference(moving_image, reference_image)
    sitk.WriteImage(resampled_image, str(moving_path))


@require_global_paths_set
def homogenize(d: int) -> Path:
    if not 0 <= d <= 999:
        raise ValueError(f"Dataset id must be between 0 and 999, got {d}")

    dataset_id = f"{d:03d}"
    mm_raw = verify_required_global_paths_set()["mm_raw"]
    dataset_dir = find_dataset_dir(mm_raw, dataset_id)
    dataset_json = load_dataset_json(dataset_dir)
    case_files = verify_training_files_present(dataset_dir, dataset_json)
    file_ending = dataset_json["file_ending"]
    reference_suffix = f"_0000{file_ending}"
    work_items: list[tuple[Path, Path]] = []

    for case_id, files in sorted(case_files.items()):
        reference_path = next(
            (path for path in files if path.name.endswith(reference_suffix)),
            None,
        )
        if reference_path is None:
            raise FileNotFoundError(
                f"Missing reference image ending with '{reference_suffix}' for case {case_id}"
            )

        for path in files:
            if path == reference_path:
                continue
            work_items.append((path, reference_path))

    if work_items:
        max_workers = min(32, (os.cpu_count() or 1) + 4, len(work_items))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_homogenize_single_image, moving_path, reference_path)
                for moving_path, reference_path in work_items
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Homogenizing images",
            ):
                future.result()

    print(dataset_dir / "imagesTr")
    return dataset_dir / "imagesTr"
