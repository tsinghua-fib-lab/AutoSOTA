from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from meisenmeister.dataloading import MeisenmeisterROIDataset
from meisenmeister.training.splits import (
    create_five_fold_splits,
    get_fold_sample_ids,
    load_splits,
)


def _build_preprocessed_dataset(root: Path, num_cases: int = 5) -> Path:
    preprocessed_dataset_dir = root / "Dataset_001_Test"
    data_dir = preprocessed_dataset_dir / "mm_b2nd"
    data_dir.mkdir(parents=True)

    with (preprocessed_dataset_dir / "mmPlans.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump({"output_folder_name": "mm_b2nd"}, file)

    labels: dict[str, int] = {}
    for case_index in range(1, num_cases + 1):
        case_id = f"case_{case_index:03d}"
        for roi_name in ("left", "right"):
            sample_id = f"{case_id}_{roi_name}"
            labels[sample_id] = case_index % 2
            (data_dir / f"{sample_id}.b2nd").touch()

    with (preprocessed_dataset_dir / "labelsTr.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(labels, file)

    return preprocessed_dataset_dir


class SplitTests(unittest.TestCase):
    def test_create_five_fold_splits_groups_by_case_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            preprocessed_dataset_dir = _build_preprocessed_dataset(
                Path(temp_dir), num_cases=5
            )

            output_path = create_five_fold_splits(preprocessed_dataset_dir)
            splits = load_splits(preprocessed_dataset_dir)

        self.assertEqual(output_path.name, "splits.json")
        self.assertEqual(len(splits), 5)

        all_val_sample_ids = set()
        for fold in splits:
            self.assertEqual(len(fold["val"]), 2)
            self.assertEqual(len(fold["train"]), 8)
            self.assertTrue(set(fold["train"]).isdisjoint(set(fold["val"])))
            all_val_sample_ids.update(fold["val"])

        self.assertEqual(
            all_val_sample_ids,
            {
                f"case_{case_index:03d}_{roi_name}"
                for case_index in range(1, 6)
                for roi_name in ("left", "right")
            },
        )

    def test_get_fold_sample_ids_fails_for_missing_fold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            preprocessed_dataset_dir = _build_preprocessed_dataset(
                Path(temp_dir), num_cases=5
            )
            create_five_fold_splits(preprocessed_dataset_dir)

            with self.assertRaisesRegex(ValueError, "Fold 5 does not exist"):
                get_fold_sample_ids(preprocessed_dataset_dir, 5)

    def test_get_fold_sample_ids_all_uses_full_dataset_without_splits_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            preprocessed_dataset_dir = _build_preprocessed_dataset(
                Path(temp_dir), num_cases=5
            )

            split = get_fold_sample_ids(preprocessed_dataset_dir, "all")

        expected_sample_ids = [
            f"case_{case_index:03d}_{roi_name}"
            for case_index in range(1, 6)
            for roi_name in ("left", "right")
        ]
        self.assertEqual(split["train"], expected_sample_ids)
        self.assertEqual(split["val"], expected_sample_ids)

    def test_dataset_filter_keeps_only_selected_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            preprocessed_dataset_dir = _build_preprocessed_dataset(
                Path(temp_dir), num_cases=5
            )

            dataset = MeisenmeisterROIDataset(
                preprocessed_dataset_dir,
                allowed_sample_ids={"case_003_left"},
            )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(
            {sample["sample_id"] for sample in dataset.samples},
            {"case_003_left"},
        )

    def test_sample_based_split_entries_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            preprocessed_dataset_dir = _build_preprocessed_dataset(
                Path(temp_dir), num_cases=5
            )
            with (preprocessed_dataset_dir / "splits.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(
                    [
                        {
                            "train": [
                                "case_001_left",
                                "case_001_right",
                                "case_002_left",
                                "case_002_right",
                            ],
                            "val": [
                                "case_003_left",
                                "case_003_right",
                            ],
                        }
                    ],
                    file,
                )

            split = get_fold_sample_ids(preprocessed_dataset_dir, 0)

        self.assertEqual(
            split["train"],
            [
                "case_001_left",
                "case_001_right",
                "case_002_left",
                "case_002_right",
            ],
        )
        self.assertEqual(
            split["val"],
            ["case_003_left", "case_003_right"],
        )


if __name__ == "__main__":
    unittest.main()
