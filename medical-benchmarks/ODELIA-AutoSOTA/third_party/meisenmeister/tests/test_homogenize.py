from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from meisenmeister.plan_and_preprocess.homogenize import (
    _matches_reference_geometry,
    homogenize,
)


class HomogenizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def test_homogenize_resamples_only_non_reference_channels(self) -> None:
        dataset_dir = self.root / "Dataset_001_Test"
        images_tr_dir = dataset_dir / "imagesTr"
        images_tr_dir.mkdir(parents=True)
        reference_path = images_tr_dir / "case_001_0000.nii.gz"
        moving_path_1 = images_tr_dir / "case_001_0001.nii.gz"
        moving_path_2 = images_tr_dir / "case_001_0002.nii.gz"
        for path in (reference_path, moving_path_1, moving_path_2):
            path.write_text("placeholder", encoding="utf-8")

        case_files = {
            "case_001": [reference_path, moving_path_1, moving_path_2],
        }
        read_calls: list[str] = []
        write_calls: list[str] = []

        def _fake_read(path: str):
            read_calls.append(Path(path).name)
            return path

        def _fake_write(image, path: str):
            del image
            write_calls.append(Path(path).name)

        with (
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.verify_required_global_paths_set",
                return_value={"mm_raw": self.root},
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.find_dataset_dir",
                return_value=dataset_dir,
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.load_dataset_json",
                return_value={"file_ending": ".nii.gz"},
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.verify_training_files_present",
                return_value=case_files,
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.sitk.ReadImage",
                side_effect=_fake_read,
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.sitk.WriteImage",
                side_effect=_fake_write,
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize._matches_reference_geometry",
                return_value=False,
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize._resample_to_reference",
                side_effect=lambda moving_image, reference_image: (
                    moving_image,
                    reference_image,
                ),
            ),
        ):
            output_path = homogenize(1)

        self.assertEqual(output_path, images_tr_dir)
        self.assertEqual(
            sorted(write_calls),
            ["case_001_0001.nii.gz", "case_001_0002.nii.gz"],
        )
        self.assertEqual(read_calls.count("case_001_0000.nii.gz"), 2)
        self.assertEqual(read_calls.count("case_001_0001.nii.gz"), 1)
        self.assertEqual(read_calls.count("case_001_0002.nii.gz"), 1)

    def test_homogenize_skips_already_aligned_channels(self) -> None:
        dataset_dir = self.root / "Dataset_001_Test"
        images_tr_dir = dataset_dir / "imagesTr"
        images_tr_dir.mkdir(parents=True)
        reference_path = images_tr_dir / "case_001_0000.nii.gz"
        moving_path_1 = images_tr_dir / "case_001_0001.nii.gz"
        moving_path_2 = images_tr_dir / "case_001_0002.nii.gz"
        for path in (reference_path, moving_path_1, moving_path_2):
            path.write_text("placeholder", encoding="utf-8")

        case_files = {
            "case_001": [reference_path, moving_path_1, moving_path_2],
        }

        with (
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.verify_required_global_paths_set",
                return_value={"mm_raw": self.root},
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.find_dataset_dir",
                return_value=dataset_dir,
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.load_dataset_json",
                return_value={"file_ending": ".nii.gz"},
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.verify_training_files_present",
                return_value=case_files,
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.sitk.ReadImage",
                side_effect=lambda path: path,
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize._matches_reference_geometry",
                return_value=True,
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize._resample_to_reference",
            ) as resample_mock,
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.sitk.WriteImage",
            ) as write_mock,
        ):
            output_path = homogenize(1)

        self.assertEqual(output_path, images_tr_dir)
        resample_mock.assert_not_called()
        write_mock.assert_not_called()

    def test_homogenize_fails_for_missing_reference_channel(self) -> None:
        dataset_dir = self.root / "Dataset_001_Test"
        images_tr_dir = dataset_dir / "imagesTr"
        images_tr_dir.mkdir(parents=True)
        moving_path = images_tr_dir / "case_001_0001.nii.gz"
        moving_path.write_text("placeholder", encoding="utf-8")

        case_files = {
            "case_001": [moving_path],
        }

        with (
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.verify_required_global_paths_set",
                return_value={"mm_raw": self.root},
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.find_dataset_dir",
                return_value=dataset_dir,
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.load_dataset_json",
                return_value={"file_ending": ".nii.gz"},
            ),
            patch(
                "meisenmeister.plan_and_preprocess.homogenize.verify_training_files_present",
                return_value=case_files,
            ),
        ):
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Missing reference image ending with '_0000.nii.gz' for case case_001",
            ):
                homogenize(1)

    def test_matches_reference_geometry_requires_full_geometry_match(self) -> None:
        class _Image:
            def __init__(self, *, size, spacing, origin, direction) -> None:
                self._size = size
                self._spacing = spacing
                self._origin = origin
                self._direction = direction

            def GetSize(self):
                return self._size

            def GetSpacing(self):
                return self._spacing

            def GetOrigin(self):
                return self._origin

            def GetDirection(self):
                return self._direction

        reference = _Image(
            size=(10, 11, 12),
            spacing=(1.0, 1.0, 1.5),
            origin=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        matching = _Image(
            size=(10, 11, 12),
            spacing=(1.0, 1.0, 1.5),
            origin=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        different_spacing = _Image(
            size=(10, 11, 12),
            spacing=(1.0, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
            direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )

        self.assertTrue(_matches_reference_geometry(matching, reference))
        self.assertFalse(_matches_reference_geometry(different_spacing, reference))


if __name__ == "__main__":
    unittest.main()
