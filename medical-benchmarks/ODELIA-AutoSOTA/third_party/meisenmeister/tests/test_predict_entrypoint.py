from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
import torch

from meisenmeister.utils.file_utils import discover_case_files

predict_module = importlib.import_module("meisenmeister.training.predict")
prediction_inference_module = importlib.import_module(
    "meisenmeister.utils.prediction_inference"
)
prediction_pipeline_module = importlib.import_module(
    "meisenmeister.training.prediction_pipeline"
)
prediction_utils_module = importlib.import_module(
    "meisenmeister.utils.prediction_utils"
)


class _ConstantModel(torch.nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("_logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        del x
        return self._logits.unsqueeze(0)


class _PortableModel(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int, **kwargs) -> None:
        super().__init__()
        self.received = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            **kwargs,
        }
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        del x
        return torch.zeros((1, 2), dtype=torch.float32)


class PredictEntrypointTests(unittest.TestCase):
    def test_resolve_prediction_file_ending_detects_mha(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / "case_001_0000.mha").touch()
            (input_dir / "case_001_0001.mha").touch()

            file_ending = prediction_utils_module.resolve_prediction_file_ending(
                input_dir,
                {
                    "channel_names": {"0": "a", "1": "b"},
                    "file_ending": ".nii.gz",
                },
            )

        self.assertEqual(file_ending, ".mha")

    def test_resolve_prediction_file_ending_rejects_mixed_supported_suffixes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / "case_001_0000.nii.gz").touch()
            (input_dir / "case_001_0001.nii.gz").touch()
            (input_dir / "case_002_0000.mha").touch()
            (input_dir / "case_002_0001.mha").touch()

            with self.assertRaisesRegex(
                ValueError,
                "Mixed prediction input suffixes detected",
            ):
                prediction_utils_module.resolve_prediction_file_ending(
                    input_dir,
                    {
                        "channel_names": {"0": "a", "1": "b"},
                        "file_ending": ".nii.gz",
                    },
                )

    def test_resolve_prediction_file_ending_rejects_unsupported_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / "case_001_0000.nii").touch()
            (input_dir / "case_001_0001.nii").touch()

            with self.assertRaisesRegex(
                ValueError,
                "Could not detect a supported prediction input suffix",
            ):
                prediction_utils_module.resolve_prediction_file_ending(
                    input_dir,
                    {
                        "channel_names": {"0": "a", "1": "b"},
                        "file_ending": ".nii.gz",
                    },
                )

    def test_iter_prepared_case_prediction_inputs_yields_sorted_cases(self) -> None:
        case_files = {
            "case_002": [Path("/tmp/case_002_0000.nii.gz")],
            "case_001": [Path("/tmp/case_001_0000.nii.gz")],
        }
        breast_mask_paths = {
            "case_001": Path("/tmp/case_001_breast_mask.nii.gz"),
            "case_002": Path("/tmp/case_002_breast_mask.nii.gz"),
        }

        def _fake_prepare_case_prediction_inputs(**kwargs):
            case_id = kwargs["case_id"]
            return (
                {"left": f"tensor-{case_id}"},
                {"breast_mask": str(kwargs["breast_mask_path"])},
            )

        with patch(
            "meisenmeister.training.prediction_pipeline.prepare_case_prediction_inputs",
            side_effect=_fake_prepare_case_prediction_inputs,
        ):
            prepared = list(
                prediction_pipeline_module.iter_prepared_case_prediction_inputs(
                    case_files_by_case_id=case_files,
                    breast_mask_paths=breast_mask_paths,
                    dataset_json={"file_ending": ".nii.gz"},
                    plans={"roi_labels": {"left": 1}},
                    output_dir=Path("/tmp/output"),
                    num_workers=1,
                )
            )

        self.assertEqual([item[0] for item in prepared], ["case_001", "case_002"])
        self.assertEqual(prepared[0][1]["left"], "tensor-case_001")
        self.assertEqual(
            prepared[1][2]["breast_mask"],
            "/tmp/case_002_breast_mask.nii.gz",
        )

    def test_generate_breast_masks_for_cases_runs_single_directory_inference(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            for filename in (
                "case_001_0000.nii.gz",
                "case_001_0001.nii.gz",
                "case_002_0000.nii.gz",
                "case_002_0001.nii.gz",
            ):
                (input_dir / filename).write_text(filename, encoding="utf-8")

            case_files = {
                "case_001": [
                    input_dir / "case_001_0000.nii.gz",
                    input_dir / "case_001_0001.nii.gz",
                ],
                "case_002": [
                    input_dir / "case_002_0000.nii.gz",
                    input_dir / "case_002_0001.nii.gz",
                ],
            }
            captured_calls: list[tuple[Path, Path]] = []

            class _Predictor:
                def predict(self, *, input_path, output_path):
                    input_path = Path(input_path)
                    output_path = Path(output_path)
                    captured_calls.append((input_path, output_path))
                    staged_files = sorted(path.name for path in input_path.iterdir())
                    if staged_files != ["case_001_0000.nii.gz", "case_002_0000.nii.gz"]:
                        raise AssertionError(f"Unexpected staged files: {staged_files}")
                    (output_path / "case_001.nii.gz").write_text(
                        "mask-1", encoding="utf-8"
                    )
                    (output_path / "case_002.nii.gz").write_text(
                        "mask-2", encoding="utf-8"
                    )

            mask_paths = prediction_pipeline_module.generate_breast_masks_for_cases(
                case_files_by_case_id=case_files,
                dataset_json={"file_ending": ".nii.gz"},
                predictor=_Predictor(),
                output_dir=output_dir,
            )
            self.assertEqual(len(captured_calls), 1)
            self.assertEqual(
                sorted(path.name for path in mask_paths.values()),
                ["case_001_breast_mask.nii.gz", "case_002_breast_mask.nii.gz"],
            )
            self.assertEqual(
                (output_dir / "case_001_breast_mask.nii.gz").read_text(
                    encoding="utf-8"
                ),
                "mask-1",
            )
            self.assertEqual(
                (output_dir / "case_002_breast_mask.nii.gz").read_text(
                    encoding="utf-8"
                ),
                "mask-2",
            )

    def test_generate_breast_masks_for_cases_skips_existing_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            for filename in (
                "case_001_0000.nii.gz",
                "case_001_0001.nii.gz",
                "case_002_0000.nii.gz",
                "case_002_0001.nii.gz",
            ):
                (input_dir / filename).write_text(filename, encoding="utf-8")
            (output_dir / "case_001_breast_mask.nii.gz").write_text(
                "existing-mask", encoding="utf-8"
            )

            case_files = {
                "case_001": [
                    input_dir / "case_001_0000.nii.gz",
                    input_dir / "case_001_0001.nii.gz",
                ],
                "case_002": [
                    input_dir / "case_002_0000.nii.gz",
                    input_dir / "case_002_0001.nii.gz",
                ],
            }
            captured_calls: list[tuple[Path, Path]] = []

            class _Predictor:
                def predict(self, *, input_path, output_path):
                    input_path = Path(input_path)
                    output_path = Path(output_path)
                    captured_calls.append((input_path, output_path))
                    staged_files = sorted(path.name for path in input_path.iterdir())
                    if staged_files != ["case_002_0000.nii.gz"]:
                        raise AssertionError(f"Unexpected staged files: {staged_files}")
                    (output_path / "case_002.nii.gz").write_text(
                        "new-mask", encoding="utf-8"
                    )

            mask_paths = prediction_pipeline_module.generate_breast_masks_for_cases(
                case_files_by_case_id=case_files,
                dataset_json={"file_ending": ".nii.gz"},
                predictor=_Predictor(),
                output_dir=output_dir,
            )

            self.assertEqual(len(captured_calls), 1)
            self.assertEqual(
                (output_dir / "case_001_breast_mask.nii.gz").read_text(
                    encoding="utf-8"
                ),
                "existing-mask",
            )
            self.assertEqual(
                (output_dir / "case_002_breast_mask.nii.gz").read_text(
                    encoding="utf-8"
                ),
                "new-mask",
            )
            self.assertEqual(
                mask_paths["case_001"],
                output_dir / "case_001_breast_mask.nii.gz",
            )
            self.assertEqual(
                mask_paths["case_002"],
                output_dir / "case_002_breast_mask.nii.gz",
            )

    def test_generate_breast_masks_for_cases_converts_mha_inputs_for_breastdivider(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            for filename in (
                "case_001_0000.mha",
                "case_001_0001.mha",
            ):
                image = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.float32))
                sitk.WriteImage(image, str(input_dir / filename))

            case_files = {
                "case_001": [
                    input_dir / "case_001_0000.mha",
                    input_dir / "case_001_0001.mha",
                ],
            }
            captured_calls: list[tuple[Path, Path]] = []

            class _Predictor:
                def predict(self, *, input_path, output_path):
                    input_path = Path(input_path)
                    output_path = Path(output_path)
                    captured_calls.append((input_path, output_path))
                    staged_files = sorted(path.name for path in input_path.iterdir())
                    if staged_files != ["case_001_0000.nii.gz"]:
                        raise AssertionError(f"Unexpected staged files: {staged_files}")
                    image = sitk.GetImageFromArray(np.ones((2, 2, 2), dtype=np.uint8))
                    sitk.WriteImage(image, str(output_path / "case_001.nii.gz"))

            mask_paths = prediction_pipeline_module.generate_breast_masks_for_cases(
                case_files_by_case_id=case_files,
                dataset_json={"file_ending": ".mha"},
                predictor=_Predictor(),
                output_dir=output_dir,
            )

            self.assertEqual(len(captured_calls), 1)
            self.assertTrue((output_dir / "case_001_breast_mask.mha").is_file())
            self.assertFalse((output_dir / "case_001.nii.gz").exists())
            self.assertEqual(
                mask_paths["case_001"],
                output_dir / "case_001_breast_mask.mha",
            )

    def test_write_concise_prediction_output_uses_written_predictions_json(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions_path = root / "predictions.json"
            concise_output_path = root / "nested" / "concise.json"
            predictions_path.write_text(
                json.dumps(
                    {
                        "cases": {
                            "case_001": {
                                "rois": {
                                    "left": {
                                        "probabilities": [0.987, 0.02, 0.001],
                                    },
                                    "right": {
                                        "probabilities": [0.001, 0.01, 0.988],
                                    },
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            written_path = prediction_utils_module.write_concise_prediction_output(
                predictions_path,
                dataset_json={
                    "labels": {"0": "normal", "1": "benign", "2": "malignant"}
                },
                concise_output_path=str(concise_output_path),
            )

            payload = json.loads(concise_output_path.read_text(encoding="utf-8"))
            self.assertEqual(written_path, concise_output_path)
            self.assertEqual(
                payload,
                {
                    "left": {
                        "normal": 0.987,
                        "benign": 0.02,
                        "malignant": 0.001,
                    },
                    "right": {
                        "normal": 0.001,
                        "benign": 0.01,
                        "malignant": 0.988,
                    },
                },
            )

    def test_write_concise_prediction_output_rejects_multiple_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions_path = root / "predictions.json"
            predictions_path.write_text(
                json.dumps(
                    {
                        "cases": {
                            "case_001": {"rois": {}},
                            "case_002": {"rois": {}},
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "Concise output requires predictions.json to contain exactly one case",
            ):
                prediction_utils_module.write_concise_prediction_output(
                    predictions_path,
                    dataset_json={
                        "labels": {"0": "normal", "1": "benign", "2": "malignant"}
                    },
                    concise_output_path=str(root / "concise.json"),
                )

    def test_predict_case_from_files_uses_input_suffix_instead_of_dataset_suffix(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            experiment_dir = root / "experiment"
            experiment_dir.mkdir()
            pre_path = root / "pre.mha"
            post1_path = root / "post1.mha"
            post2_path = root / "post2.mha"
            for path in (pre_path, post1_path, post2_path):
                path.write_text(path.name, encoding="utf-8")

            def _fake_predict_from_modelfolder(model_folder, **kwargs):
                del model_folder
                staged_input_dir = Path(kwargs["input_dir"])
                staged_names = sorted(path.name for path in staged_input_dir.iterdir())
                self.assertEqual(
                    staged_names,
                    [
                        "case_000_0000.mha",
                        "case_000_0001.mha",
                        "case_000_0002.mha",
                    ],
                )
                output_dir = Path(kwargs["output_dir"])
                mask_path = output_dir / "case_000_breast_mask.mha"
                mask_path.write_text("mask", encoding="utf-8")
                concise_path = Path(kwargs["concise_output_path"])
                concise_path.write_text(
                    json.dumps({"left": {"normal": 0.9}}),
                    encoding="utf-8",
                )
                predictions_path = output_dir / "predictions.json"
                predictions_path.write_text(json.dumps({"cases": {}}), encoding="utf-8")
                return predictions_path

            with (
                patch(
                    "meisenmeister.training.predict.load_dataset_json",
                    return_value={
                        "channel_names": {"0": "pre", "1": "post1", "2": "post2"},
                        "file_ending": ".nii.gz",
                    },
                ),
                patch(
                    "meisenmeister.training.predict.predict_from_modelfolder",
                    side_effect=_fake_predict_from_modelfolder,
                ),
            ):
                result = predict_module.predict_case_from_files(
                    str(experiment_dir),
                    str(pre_path),
                    str(post1_path),
                    str(post2_path),
                    folds=[0],
                )

        self.assertEqual(result["case_id"], "case_000")
        self.assertEqual(Path(result["mask_path"]).suffix, ".mha")
        self.assertEqual(result["concise_predictions"], {"left": {"normal": 0.9}})

    def test_predict_case_from_files_rejects_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            experiment_dir = root / "experiment"
            experiment_dir.mkdir()
            pre_path = root / "pre.nii.gz"
            post1_path = root / "post1.nii.gz"
            pre_path.write_text("pre", encoding="utf-8")
            post1_path.write_text("post1", encoding="utf-8")

            with patch(
                "meisenmeister.training.predict.load_dataset_json",
                return_value={
                    "channel_names": {"0": "pre", "1": "post1", "2": "post2"},
                    "file_ending": ".nii.gz",
                },
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "Single-case prediction input files not found",
                ):
                    predict_module.predict_case_from_files(
                        str(experiment_dir),
                        str(pre_path),
                        str(post1_path),
                        str(root / "missing.nii.gz"),
                        folds=[0],
                    )

    def test_predict_case_from_files_rejects_mixed_supported_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            experiment_dir = root / "experiment"
            experiment_dir.mkdir()
            pre_path = root / "pre.nii.gz"
            post1_path = root / "post1.mha"
            post2_path = root / "post2.nii.gz"
            for path in (pre_path, post1_path, post2_path):
                path.write_text(path.name, encoding="utf-8")

            with patch(
                "meisenmeister.training.predict.load_dataset_json",
                return_value={
                    "channel_names": {"0": "pre", "1": "post1", "2": "post2"},
                    "file_ending": ".nii.gz",
                },
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Mixed prediction input suffixes detected for single-case prediction",
                ):
                    predict_module.predict_case_from_files(
                        str(experiment_dir),
                        str(pre_path),
                        str(post1_path),
                        str(post2_path),
                        folds=[0],
                    )

    def test_predict_roi_with_tta_averages_all_flip_variants(self) -> None:
        model = _ConstantModel([1.0, 3.0])
        roi_tensor = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4)

        probabilities = prediction_utils_module.predict_roi_with_tta(
            model,
            roi_tensor,
            device=torch.device("cpu"),
            use_tta=True,
        )

        expected = torch.softmax(torch.tensor([1.0, 3.0]), dim=0).numpy()
        np.testing.assert_allclose(probabilities, expected)
        self.assertEqual(len(prediction_utils_module.get_flip_axes(True)), 8)

    def test_discover_case_files_groups_multiple_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            for filename in (
                "case_001_0000.nii.gz",
                "case_001_0001.nii.gz",
                "case_002_0000.nii.gz",
                "case_002_0001.nii.gz",
            ):
                (input_dir / filename).touch()

            dataset_json = {
                "channel_names": {"0": "a", "1": "b"},
                "file_ending": ".nii.gz",
                "problem_type": "classification",
                "labels": {"0": "neg", "1": "pos"},
                "numTraining": 2,
            }

            discovered = discover_case_files(input_dir, dataset_json)

        self.assertEqual(sorted(discovered), ["case_001", "case_002"])
        self.assertEqual(len(discovered["case_001"]), 2)
        self.assertEqual(len(discovered["case_002"]), 2)

    def test_discover_case_files_groups_multiple_mha_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            for filename in (
                "case_001_0000.mha",
                "case_001_0001.mha",
                "case_002_0000.mha",
                "case_002_0001.mha",
            ):
                (input_dir / filename).touch()

            dataset_json = {
                "channel_names": {"0": "a", "1": "b"},
                "file_ending": ".mha",
                "problem_type": "classification",
                "labels": {"0": "neg", "1": "pos"},
                "numTraining": 2,
            }

            discovered = discover_case_files(input_dir, dataset_json)

        self.assertEqual(sorted(discovered), ["case_001", "case_002"])
        self.assertEqual(len(discovered["case_001"]), 2)
        self.assertEqual(len(discovered["case_002"]), 2)

    def test_discover_case_files_rejects_missing_channel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir)
            (input_dir / "case_001_0000.nii.gz").touch()
            dataset_json = {
                "channel_names": {"0": "a", "1": "b"},
                "file_ending": ".nii.gz",
                "problem_type": "classification",
                "labels": {"0": "neg", "1": "pos"},
                "numTraining": 1,
            }

            with self.assertRaisesRegex(ValueError, "incomplete or inconsistent"):
                discover_case_files(input_dir, dataset_json)

    def test_predict_writes_combined_json_and_flat_mask_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "predictions"
            input_dir = root / "input"
            input_dir.mkdir()
            for filename in (
                "case_001_0000.nii.gz",
                "case_002_0000.nii.gz",
            ):
                (input_dir / filename).write_text(filename, encoding="utf-8")
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            preprocessed_dir = preprocessed_root / dataset_dir.name
            dataset_dir.mkdir()
            preprocessed_dir.mkdir(parents=True)

            case_files = {
                "case_001": [input_dir / "case_001_0000.nii.gz"],
                "case_002": [input_dir / "case_002_0000.nii.gz"],
            }

            def _fake_prepare_case_prediction_inputs(**kwargs):
                case_id = kwargs["case_id"]
                case_output_dir = kwargs["output_dir"]
                breast_mask = case_output_dir / f"{case_id}_breast_mask.nii.gz"
                left_mask = case_output_dir / f"{case_id}_left_mask.nii.gz"
                right_mask = case_output_dir / f"{case_id}_right_mask.nii.gz"
                for path in (breast_mask, left_mask, right_mask):
                    path.write_text("mask", encoding="utf-8")
                return {
                    "left": torch.ones((1, 2, 2, 2), dtype=torch.float32),
                    "right": torch.full((1, 2, 2, 2), 2.0, dtype=torch.float32),
                }, {
                    "breast_mask": str(breast_mask),
                    "left_mask": str(left_mask),
                    "right_mask": str(right_mask),
                }

            with (
                patch(
                    "meisenmeister.training.predict.resolve_trainer_architecture_name",
                    return_value="ResNet3D18",
                ),
                patch(
                    "meisenmeister.training.predict.verify_required_global_paths_set",
                    return_value={
                        "mm_raw": root,
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.predict.find_dataset_dir",
                    return_value=dataset_dir,
                ),
                patch(
                    "meisenmeister.training.predict.load_dataset_json",
                    return_value={
                        "channel_names": {"0": "image"},
                        "file_ending": ".nii.gz",
                        "problem_type": "classification",
                        "labels": {"0": "neg", "1": "pos"},
                        "numTraining": 2,
                    },
                ),
                patch(
                    "meisenmeister.training.predict.load_mm_plans",
                    return_value={
                        "roi_labels": {"left": 1, "right": 2},
                        "target_spacing": [1.0, 1.0, 1.0],
                        "target_shape": [2, 2, 2],
                    },
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.discover_case_files",
                    return_value=case_files,
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.get_breast_segmentation_predictor",
                    return_value=object(),
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.generate_breast_masks_for_cases",
                    return_value={
                        "case_001": output_dir / "case_001_breast_mask.nii.gz",
                        "case_002": output_dir / "case_002_breast_mask.nii.gz",
                    },
                ),
                patch(
                    "meisenmeister.training.predict.load_fold_predictors",
                    return_value=[
                        {
                            "fold": 0,
                            "device": torch.device("cpu"),
                            "checkpoint_path": "/tmp/fold_0.pt",
                            "model": object(),
                        },
                        {
                            "fold": 1,
                            "device": torch.device("cpu"),
                            "checkpoint_path": "/tmp/fold_1.pt",
                            "model": object(),
                        },
                    ],
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.prepare_case_prediction_inputs",
                    side_effect=_fake_prepare_case_prediction_inputs,
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.predict_roi_with_tta",
                    side_effect=[
                        np.array([0.2, 0.8], dtype=np.float32),
                        np.array([0.4, 0.6], dtype=np.float32),
                        np.array([0.7, 0.3], dtype=np.float32),
                        np.array([0.6, 0.4], dtype=np.float32),
                        np.array([0.1, 0.9], dtype=np.float32),
                        np.array([0.3, 0.7], dtype=np.float32),
                        np.array([0.8, 0.2], dtype=np.float32),
                        np.array([0.5, 0.5], dtype=np.float32),
                    ],
                ),
            ):
                predictions_path = predict_module.predict.__wrapped__(
                    1,
                    input_dir=str(input_dir),
                    output_dir=str(output_dir),
                    folds=[0, 1],
                    trainer_name="mmTrainer",
                    checkpoint="best",
                    use_tta=True,
                    num_workers=8,
                    concise_output_path=None,
                )

            payload = json.loads(predictions_path.read_text(encoding="utf-8"))
            self.assertEqual(predictions_path, output_dir / "predictions.json")
            self.assertTrue((output_dir / "case_001_breast_mask.nii.gz").is_file())
            self.assertTrue((output_dir / "case_001_left_mask.nii.gz").is_file())
            self.assertTrue((output_dir / "case_001_right_mask.nii.gz").is_file())
            self.assertIn("case_001", payload["cases"])
            self.assertIn("case_002", payload["cases"])
            self.assertTrue(payload["config"]["tta_enabled"])
            self.assertEqual(payload["config"]["checkpoint"], "best")
            left_case_001 = payload["cases"]["case_001"]["rois"]["left"]
            self.assertEqual(left_case_001["prediction"], 1)
            self.assertEqual(sorted(left_case_001["per_fold"]), ["0", "1"])

    def test_predict_detects_mha_inputs_and_writes_mha_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "predictions"
            input_dir = root / "input"
            input_dir.mkdir()
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            preprocessed_dir = preprocessed_root / dataset_dir.name
            dataset_dir.mkdir()
            preprocessed_dir.mkdir(parents=True)
            for filename in (
                "case_001_0000.mha",
                "case_001_0001.mha",
                "case_002_0000.mha",
                "case_002_0001.mha",
            ):
                (input_dir / filename).write_text(filename, encoding="utf-8")

            case_files = {
                "case_001": [
                    input_dir / "case_001_0000.mha",
                    input_dir / "case_001_0001.mha",
                ],
                "case_002": [
                    input_dir / "case_002_0000.mha",
                    input_dir / "case_002_0001.mha",
                ],
            }

            def _fake_prepare_case_prediction_inputs(**kwargs):
                case_id = kwargs["case_id"]
                case_output_dir = kwargs["output_dir"]
                breast_mask = case_output_dir / f"{case_id}_breast_mask.mha"
                left_mask = case_output_dir / f"{case_id}_left_mask.mha"
                right_mask = case_output_dir / f"{case_id}_right_mask.mha"
                for path in (breast_mask, left_mask, right_mask):
                    path.write_text("mask", encoding="utf-8")
                return {
                    "left": torch.ones((1, 2, 2, 2), dtype=torch.float32),
                    "right": torch.full((1, 2, 2, 2), 2.0, dtype=torch.float32),
                }, {
                    "breast_mask": str(breast_mask),
                    "left_mask": str(left_mask),
                    "right_mask": str(right_mask),
                }

            with (
                patch(
                    "meisenmeister.training.predict.resolve_trainer_architecture_name",
                    return_value="ResNet3D18",
                ),
                patch(
                    "meisenmeister.training.predict.verify_required_global_paths_set",
                    return_value={
                        "mm_raw": root,
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.predict.find_dataset_dir",
                    return_value=dataset_dir,
                ),
                patch(
                    "meisenmeister.training.predict.load_dataset_json",
                    return_value={
                        "channel_names": {"0": "image", "1": "image2"},
                        "file_ending": ".nii.gz",
                        "problem_type": "classification",
                        "labels": {"0": "neg", "1": "pos"},
                        "numTraining": 2,
                    },
                ),
                patch(
                    "meisenmeister.training.predict.load_mm_plans",
                    return_value={
                        "roi_labels": {"left": 1, "right": 2},
                        "target_spacing": [1.0, 1.0, 1.0],
                        "target_shape": [2, 2, 2],
                    },
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.discover_case_files",
                    return_value=case_files,
                ) as mock_discover_case_files,
                patch(
                    "meisenmeister.training.prediction_pipeline.get_breast_segmentation_predictor",
                    return_value=object(),
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.generate_breast_masks_for_cases",
                    return_value={
                        "case_001": output_dir / "case_001_breast_mask.mha",
                        "case_002": output_dir / "case_002_breast_mask.mha",
                    },
                ),
                patch(
                    "meisenmeister.training.predict.load_fold_predictors",
                    return_value=[
                        {
                            "fold": 0,
                            "device": torch.device("cpu"),
                            "checkpoint_path": "/tmp/fold_0.pt",
                            "model": object(),
                        }
                    ],
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.prepare_case_prediction_inputs",
                    side_effect=_fake_prepare_case_prediction_inputs,
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.predict_roi_with_tta",
                    side_effect=[
                        np.array([0.2, 0.8], dtype=np.float32),
                        np.array([0.7, 0.3], dtype=np.float32),
                        np.array([0.1, 0.9], dtype=np.float32),
                        np.array([0.8, 0.2], dtype=np.float32),
                    ],
                ),
            ):
                predictions_path = predict_module.predict.__wrapped__(
                    1,
                    input_dir=str(input_dir),
                    output_dir=str(output_dir),
                    folds=[0],
                    trainer_name="mmTrainer",
                    checkpoint="best",
                    use_tta=True,
                    num_workers=8,
                )

            payload = json.loads(predictions_path.read_text(encoding="utf-8"))
            self.assertEqual(
                mock_discover_case_files.call_args.args[1]["file_ending"],
                ".mha",
            )
            self.assertEqual(predictions_path, output_dir / "predictions.json")
            self.assertTrue((output_dir / "case_001_breast_mask.mha").is_file())
            self.assertTrue((output_dir / "case_001_left_mask.mha").is_file())
            self.assertTrue((output_dir / "case_001_right_mask.mha").is_file())
            self.assertEqual(
                payload["cases"]["case_001"]["artifacts"]["breast_mask"],
                str(output_dir / "case_001_breast_mask.mha"),
            )
            self.assertEqual(
                payload["cases"]["case_001"]["artifacts"]["left_mask"],
                str(output_dir / "case_001_left_mask.mha"),
            )

    def test_predict_uses_literal_fold_all_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            preprocessed_dir = preprocessed_root / dataset_dir.name
            results_dir = root / "results"
            experiment_dir = results_dir / dataset_dir.name / "mmTrainer_ResNet3D18"
            input_dir = root / "input"
            dataset_dir.mkdir()
            input_dir.mkdir()
            preprocessed_dir.mkdir(parents=True)
            (experiment_dir / "fold_all").mkdir(parents=True)
            (input_dir / "case_001_0000.nii.gz").write_text(
                "case_001_0000.nii.gz",
                encoding="utf-8",
            )

            with (
                patch(
                    "meisenmeister.training.predict.resolve_trainer_architecture_name",
                    return_value="ResNet3D18",
                ),
                patch(
                    "meisenmeister.training.predict.verify_required_global_paths_set",
                    return_value={
                        "mm_raw": root,
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": results_dir,
                    },
                ),
                patch(
                    "meisenmeister.training.predict.find_dataset_dir",
                    return_value=dataset_dir,
                ),
                patch(
                    "meisenmeister.training.predict.load_dataset_json",
                    return_value={
                        "file_ending": ".nii.gz",
                        "channel_names": {"0": "image"},
                    },
                ),
                patch(
                    "meisenmeister.training.predict.load_mm_plans",
                    return_value={"roi_labels": {"left": 1}},
                ),
                patch(
                    "meisenmeister.training.predict.load_fold_predictors",
                    return_value=[],
                ) as mock_load_fold_predictors,
                patch(
                    "meisenmeister.training.predict.run_prediction",
                    return_value=Path("/tmp/predictions.json"),
                ) as mock_run_prediction,
            ):
                predict_module.predict.__wrapped__(
                    1,
                    input_dir=str(input_dir),
                    output_dir=str(root / "output"),
                    folds=["all"],
                    num_workers=8,
                    concise_output_path=None,
                )

        self.assertEqual(mock_load_fold_predictors.call_args.kwargs["folds"], ["all"])
        self.assertEqual(mock_run_prediction.call_args.kwargs["folds"], ["all"])

    def test_predict_from_modelfolder_writes_outputs_without_global_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            experiment_dir = (
                root / "results" / "Dataset_001_Test" / "mmTrainer_ResNet3D18"
            )
            output_dir = root / "predictions"
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "case_001_0000.nii.gz").write_text(
                "case_001_0000.nii.gz",
                encoding="utf-8",
            )
            experiment_dir.mkdir(parents=True)
            (experiment_dir / "dataset.json").write_text(
                json.dumps(
                    {
                        "channel_names": {"0": "image"},
                        "file_ending": ".nii.gz",
                        "problem_type": "classification",
                        "labels": {"0": "neg", "1": "pos"},
                        "numTraining": 2,
                    }
                ),
                encoding="utf-8",
            )
            (experiment_dir / "mmPlans.json").write_text(
                json.dumps(
                    {
                        "roi_labels": {"left": 1, "right": 2},
                        "target_spacing": [1.0, 1.0, 1.0],
                        "target_shape": [2, 2, 2],
                    }
                ),
                encoding="utf-8",
            )

            case_files = {
                "case_001": [input_dir / "case_001_0000.nii.gz"],
            }

            def _fake_prepare_case_prediction_inputs(**kwargs):
                case_id = kwargs["case_id"]
                case_output_dir = kwargs["output_dir"]
                breast_mask = case_output_dir / f"{case_id}_breast_mask.nii.gz"
                left_mask = case_output_dir / f"{case_id}_left_mask.nii.gz"
                right_mask = case_output_dir / f"{case_id}_right_mask.nii.gz"
                for path in (breast_mask, left_mask, right_mask):
                    path.write_text("mask", encoding="utf-8")
                return {
                    "left": torch.ones((1, 2, 2, 2), dtype=torch.float32),
                    "right": torch.full((1, 2, 2, 2), 2.0, dtype=torch.float32),
                }, {
                    "breast_mask": str(breast_mask),
                    "left_mask": str(left_mask),
                    "right_mask": str(right_mask),
                }

            with (
                patch(
                    "meisenmeister.training.prediction_pipeline.discover_case_files",
                    return_value=case_files,
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.get_breast_segmentation_predictor",
                    return_value=object(),
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.generate_breast_masks_for_cases",
                    return_value={
                        "case_001": output_dir / "case_001_breast_mask.nii.gz",
                    },
                ),
                patch(
                    "meisenmeister.training.predict.get_experiment_metadata",
                    return_value={
                        "dataset_id": "001",
                        "dataset_name": "Dataset_001_Test",
                        "trainer_name": "mmTrainer",
                        "architecture_name": "ResNet3D18",
                        "experiment_postfix": None,
                    },
                ),
                patch(
                    "meisenmeister.training.predict.load_fold_predictors_from_experiment_dir",
                    return_value=[
                        {
                            "fold": 0,
                            "device": torch.device("cpu"),
                            "checkpoint_path": str(
                                experiment_dir / "fold_0" / "model_best.pt"
                            ),
                            "model": object(),
                        }
                    ],
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.prepare_case_prediction_inputs",
                    side_effect=_fake_prepare_case_prediction_inputs,
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.predict_roi_with_tta",
                    side_effect=[
                        np.array([0.2, 0.8], dtype=np.float32),
                        np.array([0.7, 0.3], dtype=np.float32),
                    ],
                ),
            ):
                predictions_path = predict_module.predict_from_modelfolder(
                    str(experiment_dir),
                    input_dir=str(input_dir),
                    output_dir=str(output_dir),
                    folds=[0],
                    checkpoint="best",
                    use_tta=True,
                    num_workers=8,
                    concise_output_path=None,
                )

            payload = json.loads(predictions_path.read_text(encoding="utf-8"))
            self.assertEqual(predictions_path, output_dir / "predictions.json")
            self.assertEqual(payload["config"]["dataset_id"], "001")
            self.assertEqual(payload["config"]["trainer_name"], "mmTrainer")
            self.assertEqual(sorted(payload["cases"]), ["case_001"])

    def test_predict_from_modelfolder_writes_concise_output_for_single_case(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            experiment_dir = (
                root / "results" / "Dataset_001_Test" / "mmTrainer_ResNet3D18"
            )
            output_dir = root / "predictions"
            concise_output_path = root / "concise" / "single_case.json"
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "case_001_0000.nii.gz").write_text(
                "case_001_0000.nii.gz",
                encoding="utf-8",
            )
            experiment_dir.mkdir(parents=True)
            (experiment_dir / "dataset.json").write_text(
                json.dumps(
                    {
                        "channel_names": {"0": "image"},
                        "file_ending": ".nii.gz",
                        "problem_type": "classification",
                        "labels": {
                            "0": "normal",
                            "1": "benign",
                            "2": "malignant",
                        },
                        "numTraining": 2,
                    }
                ),
                encoding="utf-8",
            )
            (experiment_dir / "mmPlans.json").write_text(
                json.dumps(
                    {
                        "roi_labels": {"left": 1, "right": 2},
                        "target_spacing": [1.0, 1.0, 1.0],
                        "target_shape": [2, 2, 2],
                    }
                ),
                encoding="utf-8",
            )

            case_files = {
                "case_001": [input_dir / "case_001_0000.nii.gz"],
            }

            def _fake_prepare_case_prediction_inputs(**kwargs):
                case_id = kwargs["case_id"]
                case_output_dir = kwargs["output_dir"]
                breast_mask = case_output_dir / f"{case_id}_breast_mask.nii.gz"
                left_mask = case_output_dir / f"{case_id}_left_mask.nii.gz"
                right_mask = case_output_dir / f"{case_id}_right_mask.nii.gz"
                for path in (breast_mask, left_mask, right_mask):
                    path.write_text("mask", encoding="utf-8")
                return {
                    "left": torch.ones((1, 2, 2, 2), dtype=torch.float32),
                    "right": torch.full((1, 2, 2, 2), 2.0, dtype=torch.float32),
                }, {
                    "breast_mask": str(breast_mask),
                    "left_mask": str(left_mask),
                    "right_mask": str(right_mask),
                }

            with (
                patch(
                    "meisenmeister.training.prediction_pipeline.discover_case_files",
                    return_value=case_files,
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.get_breast_segmentation_predictor",
                    return_value=object(),
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.generate_breast_masks_for_cases",
                    return_value={
                        "case_001": output_dir / "case_001_breast_mask.nii.gz",
                    },
                ),
                patch(
                    "meisenmeister.training.predict.get_experiment_metadata",
                    return_value={
                        "dataset_id": "001",
                        "dataset_name": "Dataset_001_Test",
                        "trainer_name": "mmTrainer",
                        "architecture_name": "ResNet3D18",
                        "experiment_postfix": None,
                    },
                ),
                patch(
                    "meisenmeister.training.predict.load_fold_predictors_from_experiment_dir",
                    return_value=[
                        {
                            "fold": 0,
                            "device": torch.device("cpu"),
                            "checkpoint_path": str(
                                experiment_dir / "fold_0" / "model_best.pt"
                            ),
                            "model": object(),
                        }
                    ],
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.prepare_case_prediction_inputs",
                    side_effect=_fake_prepare_case_prediction_inputs,
                ),
                patch(
                    "meisenmeister.training.prediction_pipeline.predict_roi_with_tta",
                    side_effect=[
                        np.array([0.987, 0.02, 0.001], dtype=np.float32),
                        np.array([0.001, 0.01, 0.988], dtype=np.float32),
                    ],
                ),
            ):
                predict_module.predict_from_modelfolder(
                    str(experiment_dir),
                    input_dir=str(input_dir),
                    output_dir=str(output_dir),
                    folds=[0],
                    checkpoint="best",
                    use_tta=True,
                    num_workers=8,
                    concise_output_path=str(concise_output_path),
                )

            payload = json.loads(concise_output_path.read_text(encoding="utf-8"))
            self.assertEqual(set(payload), {"left", "right"})
            self.assertAlmostEqual(payload["left"]["normal"], 0.987, places=6)
            self.assertAlmostEqual(payload["left"]["benign"], 0.02, places=6)
            self.assertAlmostEqual(payload["left"]["malignant"], 0.001, places=6)
            self.assertAlmostEqual(payload["right"]["normal"], 0.001, places=6)
            self.assertAlmostEqual(payload["right"]["benign"], 0.01, places=6)
            self.assertAlmostEqual(payload["right"]["malignant"], 0.988, places=6)

    def test_get_experiment_metadata_reads_checkpoint_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_dir = (
                Path(temp_dir) / "Dataset_001_Test" / "mmTrainer_ResNet3D18"
            )
            fold_dir = experiment_dir / "fold_0"
            fold_dir.mkdir(parents=True)
            torch.save(
                {
                    "trainer_config": {
                        "dataset_id": "001",
                        "dataset_name": "Dataset_001_Test",
                        "trainer_name": "mmTrainer",
                        "architecture_name": "ResNet3D18",
                        "experiment_postfix": "portable",
                        "in_channels": 1,
                        "num_classes": 2,
                    },
                    "model_state_dict": {},
                },
                fold_dir / "model_best.pt",
            )

            metadata = prediction_pipeline_module.get_experiment_metadata(
                experiment_dir,
                [0],
                "best",
            )

        self.assertEqual(metadata["dataset_id"], "001")
        self.assertEqual(metadata["dataset_name"], "Dataset_001_Test")
        self.assertEqual(metadata["trainer_name"], "mmTrainer")
        self.assertEqual(metadata["architecture_name"], "ResNet3D18")
        self.assertEqual(metadata["experiment_postfix"], "portable")

    def test_load_model_from_checkpoint_payload_passes_architecture_kwargs(
        self,
    ) -> None:
        checkpoint_payload = {
            "trainer_config": {
                "in_channels": 1,
                "num_classes": 2,
                "architecture_kwargs": {
                    "input_shape": (16, 16, 16),
                    "patch_embed_size": (8, 8, 8),
                },
            },
            "model_state_dict": _PortableModel(1, 2).state_dict(),
        }

        with (
            patch(
                "meisenmeister.utils.prediction_inference.get_architecture_class",
                return_value=_PortableModel,
            ),
            patch(
                "meisenmeister.utils.prediction_inference.maybe_compile_model",
                side_effect=lambda model, device, enabled: (
                    model,
                    False,
                    "compile disabled",
                ),
            ),
        ):
            model, compile_applied, compile_status = (
                prediction_inference_module.load_model_from_checkpoint_payload(
                    checkpoint_payload=checkpoint_payload,
                    architecture_name="PrimusMClsNetwork",
                    device=torch.device("cpu"),
                    compile_model=False,
                )
            )

        self.assertFalse(compile_applied)
        self.assertEqual(compile_status, "compile disabled")
        self.assertEqual(
            model.received,
            {
                "in_channels": 1,
                "num_classes": 2,
                "input_shape": (16, 16, 16),
                "patch_embed_size": (8, 8, 8),
            },
        )

    def test_predict_uses_single_identity_variant_when_tta_disabled(self) -> None:
        with patch(
            "meisenmeister.utils.prediction_utils.get_flip_axes",
            return_value=[()],
        ):
            model = _ConstantModel([2.0, 1.0])
            roi_tensor = torch.ones((1, 2, 2, 2), dtype=torch.float32)

            probabilities = prediction_utils_module.predict_roi_with_tta(
                model,
                roi_tensor,
                device=torch.device("cpu"),
                use_tta=False,
            )

        expected = torch.softmax(torch.tensor([2.0, 1.0]), dim=0).numpy()
        np.testing.assert_allclose(probabilities, expected)
