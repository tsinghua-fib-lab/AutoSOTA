from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
import torch

from meisenmeister.training import prediction_pipeline
from meisenmeister.utils import prediction_inference


def _write_image(path: Path, array: np.ndarray) -> None:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.6, 0.7, 0.8))
    sitk.WriteImage(image, str(path))


class PredictionPipelineTests(unittest.TestCase):
    def test_validate_folds_normalizes_unique_sorted_values(self) -> None:
        self.assertEqual(prediction_pipeline.validate_folds([3, 1, 2]), [1, 2, 3])

    def test_validate_folds_rejects_empty_negative_and_duplicate_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one fold"):
            prediction_pipeline.validate_folds([])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            prediction_pipeline.validate_folds([0, -1])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            prediction_pipeline.validate_folds([1, 1])

    def test_normalize_prediction_folds_handles_all_and_rejects_invalid_forms(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            (experiment_dir / "fold_all").mkdir()

            self.assertEqual(
                prediction_pipeline.normalize_prediction_folds(
                    ["all"],
                    experiment_dir=experiment_dir,
                ),
                ["all"],
            )

            with self.assertRaisesRegex(
                ValueError,
                "'all' cannot be combined with explicit fold indices",
            ):
                prediction_pipeline.normalize_prediction_folds(
                    ["all", 0],
                    experiment_dir=experiment_dir,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "fold_all does not exist"):
                prediction_pipeline.normalize_prediction_folds(
                    ["all"],
                    experiment_dir=experiment_dir,
                )

    def test_stage_breastdivider_primary_inputs_converts_mha_primary_channel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staging_input_dir = root / "staging"
            staging_input_dir.mkdir()
            primary_path = root / "case_001_0000.mha"
            secondary_path = root / "case_001_0001.mha"
            _write_image(primary_path, np.ones((2, 3, 4), dtype=np.float32))
            _write_image(secondary_path, np.full((2, 3, 4), 2.0, dtype=np.float32))

            staged_case_ids = prediction_pipeline.stage_breastdivider_primary_inputs(
                {"case_001": [secondary_path, primary_path]},
                staging_input_dir=staging_input_dir,
                file_ending=".mha",
            )

            self.assertEqual(staged_case_ids, ["case_001"])
            staged_path = staging_input_dir / "case_001_0000.nii.gz"
            self.assertTrue(staged_path.is_file())
            staged_image = sitk.ReadImage(str(staged_path))
            self.assertEqual(staged_image.GetSize(), (4, 3, 2))

    def test_stage_breastdivider_primary_inputs_rejects_missing_primary_and_bad_suffix(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staging_input_dir = root / "staging"
            staging_input_dir.mkdir()
            secondary_path = root / "case_001_0001.mha"
            _write_image(secondary_path, np.ones((2, 2, 2), dtype=np.float32))

            with self.assertRaisesRegex(FileNotFoundError, "Missing input file"):
                prediction_pipeline.stage_breastdivider_primary_inputs(
                    {"case_001": [secondary_path]},
                    staging_input_dir=staging_input_dir,
                    file_ending=".mha",
                )

            with self.assertRaisesRegex(ValueError, "Unsupported prediction file ending"):
                prediction_pipeline.stage_breastdivider_primary_inputs(
                    {"case_001": [secondary_path]},
                    staging_input_dir=staging_input_dir,
                    file_ending=".nii",
                )

    def test_generate_breast_masks_for_cases_raises_for_missing_predictor_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            output_dir.mkdir()
            primary_path = root / "case_001_0000.nii.gz"
            primary_path.write_text("image", encoding="utf-8")

            with patch(
                "meisenmeister.training.prediction_pipeline.predict_breast_segmentation"
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "Breast segmentation output missing for case case_001",
                ):
                    prediction_pipeline.generate_breast_masks_for_cases(
                        case_files_by_case_id={"case_001": [primary_path]},
                        dataset_json={"file_ending": ".nii.gz"},
                        predictor=object(),
                        output_dir=output_dir,
                    )

    def test_prepare_case_prediction_inputs_saves_roi_masks_and_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            breast_mask_path = root / "case_001_breast_mask.nii.gz"
            output_dir = root / "out"
            output_dir.mkdir()

            mask_array = np.zeros((2, 2, 2), dtype=np.uint8)
            mask_array[:, 0, :] = 1
            mask_array[:, 1, :] = 2
            _write_image(breast_mask_path, mask_array)

            preprocess_calls: list[dict] = []

            def _fake_preprocess_roi_array(*args, **kwargs):
                preprocess_calls.append(kwargs)
                roi_label = kwargs["roi_label"]
                return np.full((1, 2, 2, 2), roi_label, dtype=np.float32)

            with (
                patch(
                    "meisenmeister.training.prediction_pipeline.get_case_channel_files"
                ) as mock_get_case_channel_files,
                patch(
                    "meisenmeister.training.prediction_pipeline.load_case_image_data",
                    return_value={"image": "loaded"},
                ) as mock_load_case_image_data,
                patch(
                    "meisenmeister.training.prediction_pipeline.preprocess_roi_array",
                    side_effect=_fake_preprocess_roi_array,
                ),
            ):
                roi_tensors, artifact_paths = (
                    prediction_pipeline.prepare_case_prediction_inputs(
                        case_id="case_001",
                        case_files=[root / "case_001_0000.nii.gz"],
                        breast_mask_path=breast_mask_path,
                        dataset_json={"file_ending": ".nii.gz"},
                        plans={
                            "roi_labels": {"left": 1, "right": 2},
                            "target_spacing": [1.0, 1.0, 1.0],
                            "target_shape": [2, 2, 2],
                        },
                        output_dir=output_dir,
                    )
                )

            mock_get_case_channel_files.assert_called_once()
            mock_load_case_image_data.assert_called_once()
            self.assertEqual(sorted(roi_tensors), ["left", "right"])
            self.assertTrue(torch.equal(roi_tensors["left"], torch.ones((1, 2, 2, 2))))
            self.assertTrue(
                torch.equal(roi_tensors["right"], torch.full((1, 2, 2, 2), 2.0))
            )
            self.assertEqual(len(preprocess_calls), 2)
            self.assertTrue((output_dir / "case_001_left_mask.nii.gz").is_file())
            self.assertTrue((output_dir / "case_001_right_mask.nii.gz").is_file())
            self.assertEqual(
                artifact_paths["breast_mask"],
                str(breast_mask_path),
            )
            self.assertEqual(preprocess_calls[0]["margin_mm"], [0.0, 0.0, 0.0])

    def test_prepare_case_prediction_inputs_rejects_missing_roi_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            breast_mask_path = root / "case_001_breast_mask.nii.gz"
            output_dir = root / "out"
            output_dir.mkdir()

            mask_array = np.ones((2, 2, 2), dtype=np.uint8)
            _write_image(breast_mask_path, mask_array)

            with (
                patch(
                    "meisenmeister.training.prediction_pipeline.get_case_channel_files"
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.load_case_image_data",
                    return_value={"image": "loaded"},
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.preprocess_roi_array",
                    return_value=np.ones((1, 2, 2, 2), dtype=np.float32),
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "ROI label 2 \\(right\\) not present for case_001",
                ):
                    prediction_pipeline.prepare_case_prediction_inputs(
                        case_id="case_001",
                        case_files=[root / "case_001_0000.nii.gz"],
                        breast_mask_path=breast_mask_path,
                        dataset_json={"file_ending": ".nii.gz"},
                        plans={
                            "roi_labels": {"left": 1, "right": 2},
                            "target_spacing": [1.0, 1.0, 1.0],
                            "target_shape": [2, 2, 2],
                        },
                        output_dir=output_dir,
                    )

    def test_get_experiment_metadata_falls_back_to_directory_name_and_requires_config(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = (
                Path(temp_dir) / "Dataset_001_Test" / "mmTrainer_ResNet3D18"
            )
            fold_dir = experiment_dir / "fold_0"
            fold_dir.mkdir(parents=True)

            with patch(
                "meisenmeister.training.prediction_pipeline.load_checkpoint_payload",
                return_value={
                    "trainer_config": {
                        "dataset_id": "001",
                        "trainer_name": "mmTrainer",
                        "architecture_name": "ResNet3D18",
                    }
                },
            ):
                metadata = prediction_pipeline.get_experiment_metadata(
                    experiment_dir,
                    [0],
                    "best",
                )

            self.assertEqual(metadata["dataset_name"], "Dataset_001_Test")

            with patch(
                "meisenmeister.training.prediction_pipeline.load_checkpoint_payload",
                return_value={"trainer_config": {"dataset_id": "001"}},
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "missing trainer_name or architecture_name",
                ):
                    prediction_pipeline.get_experiment_metadata(
                        experiment_dir,
                        [0],
                        "best",
                    )

    def test_load_model_from_checkpoint_payload_requires_architecture_metadata(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "missing architecture metadata required for portable inference",
        ):
            prediction_inference.load_model_from_checkpoint_payload(
                checkpoint_payload={
                    "trainer_config": {"in_channels": 1},
                    "model_state_dict": {},
                },
                architecture_name="ResNet3D18",
                device=torch.device("cpu"),
                compile_model=False,
            )

    def test_load_fold_predictors_from_experiment_dir_loads_each_requested_fold(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = Path(temp_dir)

            with (
                patch(
                    "meisenmeister.utils.prediction_inference.load_checkpoint_payload",
                    side_effect=[{"model_state_dict": {}}, {"model_state_dict": {}}],
                ) as mock_load_payload,
                patch(
                    "meisenmeister.utils.prediction_inference.load_model_from_checkpoint_payload",
                    side_effect=[
                        (object(), False, "compile disabled"),
                        (object(), True, "compiled"),
                    ],
                ) as mock_load_model,
            ):
                predictors = prediction_inference.load_fold_predictors_from_experiment_dir(
                    experiment_dir=experiment_dir,
                    architecture_name="ResNet3D18",
                    folds=[0, 2],
                    checkpoint="last",
                    compile_model=False,
                )

            self.assertEqual([item["fold"] for item in predictors], [0, 2])
            self.assertEqual(
                [Path(item["checkpoint_path"]).name for item in predictors],
                ["model_last.pt", "model_last.pt"],
            )
            self.assertEqual(mock_load_payload.call_count, 2)
            self.assertEqual(mock_load_model.call_count, 2)


if __name__ == "__main__":
    unittest.main()
