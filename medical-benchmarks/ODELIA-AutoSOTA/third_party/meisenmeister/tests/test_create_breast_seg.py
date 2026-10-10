from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from meisenmeister.plan_and_preprocess.create_breast_seg import (
    _stage_primary_inputs,
    create_breast_segmentations,
)


class CreateBreastSegmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_stage_primary_inputs_uses_only_0000_files(self) -> None:
        staging_dir = self.root / "staging"
        staging_dir.mkdir()
        images_dir = self.root / "imagesTr"
        images_dir.mkdir()
        primary = images_dir / "case_001_0000.nii.gz"
        secondary = images_dir / "case_001_0001.nii.gz"
        primary.write_text("primary", encoding="utf-8")
        secondary.write_text("secondary", encoding="utf-8")

        staged_case_ids = _stage_primary_inputs(
            {"case_001": [primary, secondary]},
            staging_input_dir=staging_dir,
            file_ending=".nii.gz",
        )

        self.assertEqual(staged_case_ids, ["case_001"])
        self.assertTrue((staging_dir / "case_001_0000.nii.gz").exists())
        self.assertFalse((staging_dir / "case_001_0001.nii.gz").exists())

    def test_create_breast_segmentations_runs_directory_inference_from_temp_staging(
        self,
    ) -> None:
        dataset_dir = self.root / "Dataset_001_Test"
        images_tr_dir = dataset_dir / "imagesTr"
        images_tr_dir.mkdir(parents=True)
        (images_tr_dir / "case_001_0000.nii.gz").write_text("primary", encoding="utf-8")
        (images_tr_dir / "case_001_0001.nii.gz").write_text(
            "secondary", encoding="utf-8"
        )
        case_files = {
            "case_001": [
                images_tr_dir / "case_001_0000.nii.gz",
                images_tr_dir / "case_001_0001.nii.gz",
            ]
        }

        captured_calls: list[tuple[Path, Path]] = []

        class _Predictor:
            def predict(self, *, input_path, output_path):
                input_dir = Path(input_path)
                output_dir = Path(output_path)
                captured_calls.append((input_dir, output_dir))
                self_test = self  # keep local lint-free style simple
                del self_test
                staged_files = sorted(path.name for path in input_dir.iterdir())
                if staged_files != ["case_001_0000.nii.gz"]:
                    raise AssertionError(f"Unexpected staged files: {staged_files}")
                (output_dir / "case_001.nii.gz").write_text(
                    "mask",
                    encoding="utf-8",
                )

        with (
            patch(
                "meisenmeister.plan_and_preprocess.create_breast_seg.verify_required_global_paths_set",
                return_value={"mm_raw": self.root},
            ),
            patch(
                "meisenmeister.plan_and_preprocess.create_breast_seg.find_dataset_dir",
                return_value=dataset_dir,
            ),
            patch(
                "meisenmeister.plan_and_preprocess.create_breast_seg.load_dataset_json",
                return_value={"file_ending": ".nii.gz"},
            ),
            patch(
                "meisenmeister.plan_and_preprocess.create_breast_seg.verify_training_files_present",
                return_value=case_files,
            ),
            patch(
                "meisenmeister.plan_and_preprocess.create_breast_seg.get_breast_segmentation_predictor",
                return_value=_Predictor(),
            ),
        ):
            returned_images_tr_dir, returned_masks_tr_dir = create_breast_segmentations(
                1
            )

        self.assertEqual(returned_images_tr_dir, images_tr_dir)
        self.assertEqual(returned_masks_tr_dir, dataset_dir / "masksTr")
        self.assertEqual(
            (dataset_dir / "masksTr" / "case_001.nii.gz").read_text(encoding="utf-8"),
            "mask",
        )
        self.assertTrue((images_tr_dir / "case_001_0000.nii.gz").is_file())
        self.assertTrue((images_tr_dir / "case_001_0001.nii.gz").is_file())
        self.assertEqual(len(captured_calls), 1)
        input_dir, output_dir = captured_calls[0]
        self.assertEqual(output_dir, dataset_dir / "masksTr")
        self.assertFalse(input_dir.exists())

    def test_create_breast_segmentations_skips_cases_with_existing_masks(self) -> None:
        dataset_dir = self.root / "Dataset_001_Test"
        images_tr_dir = dataset_dir / "imagesTr"
        images_tr_dir.mkdir(parents=True)
        for filename in (
            "case_001_0000.nii.gz",
            "case_001_0001.nii.gz",
            "case_002_0000.nii.gz",
            "case_002_0001.nii.gz",
        ):
            (images_tr_dir / filename).write_text(filename, encoding="utf-8")

        masks_tr_dir = dataset_dir / "masksTr"
        masks_tr_dir.mkdir()
        (masks_tr_dir / "case_001.nii.gz").write_text("existing-mask", encoding="utf-8")

        case_files = {
            "case_001": [
                images_tr_dir / "case_001_0000.nii.gz",
                images_tr_dir / "case_001_0001.nii.gz",
            ],
            "case_002": [
                images_tr_dir / "case_002_0000.nii.gz",
                images_tr_dir / "case_002_0001.nii.gz",
            ],
        }

        captured_calls: list[tuple[Path, Path]] = []

        class _Predictor:
            def predict(self, *, input_path, output_path):
                input_dir = Path(input_path)
                output_dir = Path(output_path)
                captured_calls.append((input_dir, output_dir))
                staged_files = sorted(path.name for path in input_dir.iterdir())
                if staged_files != ["case_002_0000.nii.gz"]:
                    raise AssertionError(f"Unexpected staged files: {staged_files}")
                (output_dir / "case_002.nii.gz").write_text(
                    "new-mask", encoding="utf-8"
                )

        with (
            patch(
                "meisenmeister.plan_and_preprocess.create_breast_seg.verify_required_global_paths_set",
                return_value={"mm_raw": self.root},
            ),
            patch(
                "meisenmeister.plan_and_preprocess.create_breast_seg.find_dataset_dir",
                return_value=dataset_dir,
            ),
            patch(
                "meisenmeister.plan_and_preprocess.create_breast_seg.load_dataset_json",
                return_value={"file_ending": ".nii.gz"},
            ),
            patch(
                "meisenmeister.plan_and_preprocess.create_breast_seg.verify_training_files_present",
                return_value=case_files,
            ),
            patch(
                "meisenmeister.plan_and_preprocess.create_breast_seg.get_breast_segmentation_predictor",
                return_value=_Predictor(),
            ),
        ):
            returned_images_tr_dir, returned_masks_tr_dir = create_breast_segmentations(
                1
            )

        self.assertEqual(returned_images_tr_dir, images_tr_dir)
        self.assertEqual(returned_masks_tr_dir, masks_tr_dir)
        self.assertEqual(
            (masks_tr_dir / "case_001.nii.gz").read_text(encoding="utf-8"),
            "existing-mask",
        )
        self.assertEqual(
            (masks_tr_dir / "case_002.nii.gz").read_text(encoding="utf-8"),
            "new-mask",
        )
        self.assertEqual(len(captured_calls), 1)


if __name__ == "__main__":
    unittest.main()
