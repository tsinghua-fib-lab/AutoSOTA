from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from meisenmeister.dataset_conversion.odelia import (
    build_centerwise_splits,
    collect_odelia_cases,
    write_odelia_dataset,
)


class OdeliaConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def _write_center(
        self,
        center: str,
        cases: dict[str, tuple[int, int]],
        *,
        omit_files: dict[str, list[str]] | None = None,
    ) -> Path:
        center_dir = self.root / center
        data_dir = center_dir / "data"
        metadata_dir = center_dir / "metadata"
        data_dir.mkdir(parents=True)
        metadata_dir.mkdir()

        rows = ["UID,PatientID,Age,Lesion_Left,Lesion_Right,__index_level_0__"]
        for index, (case_id, (left_label, right_label)) in enumerate(
            sorted(cases.items())
        ):
            case_dir = data_dir / case_id
            case_dir.mkdir()
            omitted = set((omit_files or {}).get(case_id, []))
            for file_name in ("Pre.nii.gz", "Post_1.nii.gz", "Post_2.nii.gz"):
                if file_name in omitted:
                    continue
                (case_dir / file_name).write_text(file_name, encoding="utf-8")
            rows.append(f"{case_id},{case_id},10000,{left_label},{right_label},{index}")

        (metadata_dir / "annotation.csv").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        return center_dir

    def test_write_odelia_dataset_copies_images_and_writes_metadata(self) -> None:
        self._write_center(
            "CAM",
            {
                "CAM_001": (0, 1),
                "CAM_002": (2, 0),
            },
        )
        self._write_center(
            "RUMC",
            {
                "RUMC_001": (1, 2),
            },
        )
        output_dir = self.root / "mm_raw" / "Dataset_002_Odelia"

        write_odelia_dataset(self.root, output_dir)

        images_tr_dir = output_dir / "imagesTr"
        self.assertEqual(
            sorted(path.name for path in images_tr_dir.iterdir()),
            [
                "CAM_001_0000.nii.gz",
                "CAM_001_0001.nii.gz",
                "CAM_001_0002.nii.gz",
                "CAM_002_0000.nii.gz",
                "CAM_002_0001.nii.gz",
                "CAM_002_0002.nii.gz",
                "RUMC_001_0000.nii.gz",
                "RUMC_001_0001.nii.gz",
                "RUMC_001_0002.nii.gz",
            ],
        )

        dataset_json = json.loads((output_dir / "dataset.json").read_text())
        self.assertEqual(
            dataset_json,
            {
                "channel_names": {"0": "pre", "1": "post1", "2": "post2"},
                "file_ending": ".nii.gz",
                "problem_type": "classification",
                "labels": {"0": "healthy", "1": "benign", "2": "malignant"},
                "numTraining": 3,
            },
        )

        labels_json = json.loads((output_dir / "labelsTr.json").read_text())
        self.assertEqual(
            labels_json,
            {
                "CAM_001_left": [1, 0, 0],
                "CAM_001_right": [0, 1, 0],
                "CAM_002_left": [0, 0, 1],
                "CAM_002_right": [1, 0, 0],
                "RUMC_001_left": [0, 1, 0],
                "RUMC_001_right": [0, 0, 1],
            },
        )

        splits_json = json.loads((output_dir / "splits.json").read_text())
        self.assertEqual(
            splits_json,
            [
                {
                    "train": ["RUMC_001"],
                    "val": ["CAM_001", "CAM_002"],
                },
                {
                    "train": ["CAM_001", "CAM_002"],
                    "val": ["RUMC_001"],
                },
            ],
        )

    def test_build_centerwise_splits_uses_one_fold_per_center(self) -> None:
        self._write_center("CAM", {"CAM_001": (0, 0)})
        self._write_center("RUMC", {"RUMC_001": (1, 2)})
        self._write_center("UKA", {"UKA_001": (2, 1)})

        cases = collect_odelia_cases(self.root)
        splits = build_centerwise_splits(cases)

        self.assertEqual(len(splits), 3)
        self.assertEqual(
            splits[0], {"train": ["RUMC_001", "UKA_001"], "val": ["CAM_001"]}
        )
        self.assertEqual(
            splits[1], {"train": ["CAM_001", "UKA_001"], "val": ["RUMC_001"]}
        )
        self.assertEqual(
            splits[2], {"train": ["CAM_001", "RUMC_001"], "val": ["UKA_001"]}
        )

    def test_write_odelia_dataset_skips_existing_target_files(self) -> None:
        self._write_center(
            "CAM",
            {
                "CAM_001": (0, 1),
            },
        )
        output_dir = self.root / "mm_raw" / "Dataset_002_Odelia"
        images_tr_dir = output_dir / "imagesTr"
        images_tr_dir.mkdir(parents=True)
        target_path = images_tr_dir / "CAM_001_0000.nii.gz"
        target_path.write_text("already-there", encoding="utf-8")

        write_odelia_dataset(self.root, output_dir)

        self.assertEqual(target_path.read_text(encoding="utf-8"), "already-there")

    def test_collect_odelia_cases_fails_for_missing_required_phase(self) -> None:
        self._write_center(
            "CAM",
            {"CAM_001": (0, 1)},
            omit_files={"CAM_001": ["Post_2.nii.gz"]},
        )

        with self.assertRaisesRegex(
            FileNotFoundError,
            "Case 'CAM_001' is missing required files: Post_2.nii.gz",
        ):
            collect_odelia_cases(self.root)

    def test_collect_odelia_cases_fails_for_missing_annotation(self) -> None:
        center_dir = self._write_center("CAM", {"CAM_001": (0, 1)})
        (center_dir / "metadata" / "annotation.csv").write_text(
            "UID,PatientID,Age,Lesion_Left,Lesion_Right,__index_level_0__\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Missing annotation row for converted case 'CAM_001'",
        ):
            collect_odelia_cases(self.root)

    def test_collect_odelia_cases_fails_for_duplicate_case_ids(self) -> None:
        self._write_center("CAM", {"CASE_001": (0, 1)})
        self._write_center("RUMC", {"CASE_001": (1, 2)})

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate output case id detected: CASE_001",
        ):
            collect_odelia_cases(self.root)

    def test_collect_odelia_cases_fails_for_invalid_label_value(self) -> None:
        center_dir = self._write_center("CAM", {"CAM_001": (0, 1)})
        (center_dir / "metadata" / "annotation.csv").write_text(
            "\n".join(
                [
                    "UID,PatientID,Age,Lesion_Left,Lesion_Right,__index_level_0__",
                    "CAM_001,CAM_001,10000,3,1,0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unexpected label value '3' for CAM_001 left",
        ):
            collect_odelia_cases(self.root)


if __name__ == "__main__":
    unittest.main()
