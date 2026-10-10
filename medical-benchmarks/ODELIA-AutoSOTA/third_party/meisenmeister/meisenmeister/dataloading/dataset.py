import json
import os
from pathlib import Path

import blosc2
import numpy as np
import torch
from torch.utils.data import Dataset

from meisenmeister.data_augmentations import apply_augmentations


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_label_value(label_value):
    if isinstance(label_value, list):
        if len(label_value) == 0:
            raise ValueError("Encountered empty label list")
        return int(np.argmax(np.asarray(label_value)))
    if isinstance(label_value, int):
        return label_value
    raise TypeError(f"Unsupported label value type: {type(label_value).__name__}")


def _parse_sample_id(path: Path) -> tuple[str, str, str]:
    sample_id = path.stem
    if "_" not in sample_id:
        raise ValueError(
            f"Malformed sample filename '{path.name}'. Expected <case_id>_<roi>.b2nd"
        )

    case_id, roi_name = sample_id.rsplit("_", 1)
    if not case_id or not roi_name:
        raise ValueError(
            f"Malformed sample filename '{path.name}'. Expected <case_id>_<roi>.b2nd"
        )
    return sample_id, case_id, roi_name


class MeisenmeisterROIDataset(Dataset):
    def __init__(
        self,
        preprocessed_dataset_dir: Path,
        allowed_sample_ids: set[str] | None = None,
        allowed_case_ids: set[str] | None = None,
        augmentation_pipeline=None,
    ):
        self.preprocessed_dataset_dir = preprocessed_dataset_dir
        self.allowed_sample_ids = allowed_sample_ids
        self.allowed_case_ids = allowed_case_ids
        self.augmentation_pipeline = augmentation_pipeline
        self.plans = _load_json(preprocessed_dataset_dir / "mmPlans.json")
        target_shape = self.plans.get("target_shape")
        self.patch_size = (
            tuple(int(axis) for axis in target_shape)
            if target_shape is not None
            else None
        )
        self.labels = _load_json(preprocessed_dataset_dir / "labelsTr.json")
        self.data_dir = preprocessed_dataset_dir / self.plans["output_folder_name"]
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"Missing preprocessed data directory: {self.data_dir}"
            )

        self.samples = self._build_index()

    def _build_index(self) -> list[dict]:
        samples = []
        for path in sorted(self.data_dir.glob("*.b2nd")):
            sample_id, case_id, roi_name = _parse_sample_id(path)
            if (
                self.allowed_sample_ids is not None
                and sample_id not in self.allowed_sample_ids
            ):
                continue
            if (
                self.allowed_case_ids is not None
                and case_id not in self.allowed_case_ids
            ):
                continue
            if sample_id not in self.labels:
                raise KeyError(
                    f"Missing label for sample '{sample_id}' in labelsTr.json"
                )

            samples.append(
                {
                    "sample_id": sample_id,
                    "case_id": case_id,
                    "roi_name": roi_name,
                    "path": path,
                    "label": _normalize_label_value(self.labels[sample_id]),
                }
            )

        if not samples:
            raise ValueError(f"No .b2nd samples found in {self.data_dir}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        dparams = {"nthreads": 1}
        # mmap does not work on Windows for this loader path.
        mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}
        image = np.asarray(
            blosc2.open(
                urlpath=str(sample["path"]),
                mode="r",
                dparams=dparams,
                **mmap_kwargs,
            ),
            dtype=np.float32,
        )
        output = {
            "image": image,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "sample_id": sample["sample_id"],
            "case_id": sample["case_id"],
            "roi_name": sample["roi_name"],
        }
        if self.augmentation_pipeline is not None:
            if self.patch_size is None:
                raise KeyError(
                    "mmPlans.json must define target_shape when augmentation_pipeline is used"
                )
            output = apply_augmentations(
                output,
                pipeline=self.augmentation_pipeline,
                patch_size=self.patch_size,
            )
        output["image"] = torch.from_numpy(output["image"])
        return output
