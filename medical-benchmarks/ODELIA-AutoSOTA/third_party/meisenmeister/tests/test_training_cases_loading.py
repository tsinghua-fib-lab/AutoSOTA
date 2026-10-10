from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from meisenmeister.utils.file_utils import (
    load_dataset_json,
    verify_training_files_present,
)


class TrainingCasesLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.dataset_dir = self.root / "Dataset_001_Test"

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _create_file(self, path: Path, content: str = "dummy") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _valid_dataset_json(self) -> dict:
        return {
            "channel_names": {"0": "pre", "1": "post1"},
            "file_ending": ".nii.gz",
            "problem_type": "classification",
            "labels": {"0": "healthy", "1": "malignant"},
            "numTraining": 1,
            "training_cases": {
                "case_001": {
                    "0": "../shared/case_001_0000.nii.gz",
                    "1": "../shared/case_001_0001.nii.gz",
                }
            },
        }

    def _write_dataset_json(self, payload: dict) -> None:
        self._write_json(self.dataset_dir / "dataset.json", payload)

    def test_load_dataset_json_and_verify_training_files_present_with_mixed_paths(
        self,
    ) -> None:
        relative_image = self._create_file(
            self.root / "shared" / "case_001_0000.nii.gz",
            "c0",
        )
        absolute_image = self._create_file(
            self.root / "absolute" / "case_001_0001.nii.gz",
            "c1",
        )
        payload = self._valid_dataset_json()
        payload["training_cases"]["case_001"]["1"] = str(absolute_image)
        self._write_dataset_json(payload)

        dataset_json = load_dataset_json(self.dataset_dir)
        resolved = verify_training_files_present(self.dataset_dir, dataset_json)

        self.assertEqual(sorted(resolved), ["case_001"])
        self.assertEqual(resolved["case_001"], [relative_image, absolute_image])

    def test_load_dataset_json_rejects_missing_declared_channel(self) -> None:
        self._create_file(self.root / "shared" / "case_001_0000.nii.gz")
        payload = self._valid_dataset_json()
        payload["training_cases"]["case_001"] = {"0": "../shared/case_001_0000.nii.gz"}
        self._write_dataset_json(payload)

        with self.assertRaisesRegex(ValueError, "must define exactly channels"):
            load_dataset_json(self.dataset_dir)

    def test_load_dataset_json_rejects_extra_undeclared_channel(self) -> None:
        self._create_file(self.root / "shared" / "case_001_0000.nii.gz")
        self._create_file(self.root / "shared" / "case_001_0001.nii.gz")
        self._create_file(self.root / "shared" / "case_001_0002.nii.gz")
        payload = self._valid_dataset_json()
        payload["training_cases"]["case_001"]["2"] = "../shared/case_001_0002.nii.gz"
        self._write_dataset_json(payload)

        with self.assertRaisesRegex(ValueError, "must define exactly channels"):
            load_dataset_json(self.dataset_dir)

    def test_load_dataset_json_rejects_num_training_mismatch(self) -> None:
        self._create_file(self.root / "shared" / "case_001_0000.nii.gz")
        self._create_file(self.root / "shared" / "case_001_0001.nii.gz")
        payload = self._valid_dataset_json()
        payload["numTraining"] = 2
        self._write_dataset_json(payload)

        with self.assertRaisesRegex(ValueError, "declares numTraining=2"):
            load_dataset_json(self.dataset_dir)

    def test_load_dataset_json_rejects_missing_referenced_file(self) -> None:
        self._create_file(self.root / "shared" / "case_001_0000.nii.gz")
        payload = self._valid_dataset_json()
        self._write_dataset_json(payload)

        with self.assertRaisesRegex(FileNotFoundError, "Missing training_cases file"):
            load_dataset_json(self.dataset_dir)

    def test_load_dataset_json_can_skip_training_case_resolution(self) -> None:
        payload = self._valid_dataset_json()
        self._write_dataset_json(payload)

        dataset_json = load_dataset_json(
            self.dataset_dir,
            resolve_training_cases=False,
        )

        self.assertEqual(dataset_json["file_ending"], ".nii.gz")
        self.assertIn("training_cases", dataset_json)

    def test_load_dataset_json_rejects_wrong_file_ending(self) -> None:
        self._create_file(self.root / "shared" / "case_001_0000.nii.gz")
        wrong_suffix = self._create_file(self.root / "shared" / "case_001_0001.nii")
        payload = self._valid_dataset_json()
        payload["training_cases"]["case_001"]["1"] = str(wrong_suffix)
        self._write_dataset_json(payload)

        with self.assertRaisesRegex(ValueError, "must end with '.nii.gz'"):
            load_dataset_json(self.dataset_dir)

    def test_load_dataset_json_rejects_non_mapping_training_cases(self) -> None:
        payload = self._valid_dataset_json()
        payload["training_cases"] = ["case_001"]
        self._write_dataset_json(payload)

        with self.assertRaisesRegex(TypeError, "training_cases' must be a mapping"):
            load_dataset_json(self.dataset_dir)

    def test_load_dataset_json_rejects_non_mapping_case_entry(self) -> None:
        payload = self._valid_dataset_json()
        payload["training_cases"]["case_001"] = "../shared/case_001_0000.nii.gz"
        self._write_dataset_json(payload)

        with self.assertRaisesRegex(
            TypeError,
            "training_cases\\['case_001'\\] must be a mapping",
        ):
            load_dataset_json(self.dataset_dir)

    def test_load_dataset_json_rejects_non_string_path_value(self) -> None:
        self._create_file(self.root / "shared" / "case_001_0001.nii.gz")
        payload = self._valid_dataset_json()
        payload["training_cases"]["case_001"]["0"] = 123
        self._write_dataset_json(payload)

        with self.assertRaisesRegex(
            TypeError,
            "must be a non-empty filepath string",
        ):
            load_dataset_json(self.dataset_dir)

    def test_load_dataset_json_rejects_empty_path_value(self) -> None:
        self._create_file(self.root / "shared" / "case_001_0001.nii.gz")
        payload = self._valid_dataset_json()
        payload["training_cases"]["case_001"]["0"] = ""
        self._write_dataset_json(payload)

        with self.assertRaisesRegex(
            TypeError,
            "must be a non-empty filepath string",
        ):
            load_dataset_json(self.dataset_dir)


if __name__ == "__main__":
    unittest.main()
