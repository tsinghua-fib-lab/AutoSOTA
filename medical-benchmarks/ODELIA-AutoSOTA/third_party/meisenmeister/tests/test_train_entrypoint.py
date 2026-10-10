from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

train_module = importlib.import_module("meisenmeister.training.train")


class _MockTrainer:
    last_instance = None
    ARCHITECTURE_NAME = "MockArchitecture"

    def __init__(
        self,
        dataset_id: str,
        fold: int | str,
        dataset_dir: Path,
        preprocessed_dataset_dir: Path,
        results_dir: Path,
        architecture_name: str,
        num_workers: int | None = None,
        continue_training: bool = False,
        weights_path: Path | None = None,
        experiment_postfix: str | None = None,
        compile_enabled: bool = True,
        grad_cam_enabled: bool = False,
    ):
        self.dataset_id = dataset_id
        self.fold = fold
        self.dataset_dir = dataset_dir
        self.preprocessed_dataset_dir = preprocessed_dataset_dir
        self.results_dir = results_dir
        self.architecture_name = architecture_name
        self.num_workers = num_workers
        self.continue_training = continue_training
        self.weights_path = weights_path
        self.experiment_postfix = experiment_postfix
        self.compile_enabled = compile_enabled
        self.grad_cam_enabled = grad_cam_enabled
        self.device = torch.device("cpu")
        self.fit_called = False
        self.grad_cam_checked = False
        type(self).last_instance = self

    def fit(self) -> None:
        self.fit_called = True

    def get_architecture(self):
        class _Arch:
            def __init__(self, outer):
                self.outer = outer

            def load_state_dict(self, state_dict):
                self.outer.loaded_state_dict = state_dict

        return _Arch(self)

    def ensure_grad_cam_available(self) -> None:
        self.grad_cam_checked = True


class TrainEntrypointTests(unittest.TestCase):
    def test_train_accepts_all_fold_and_runs_selected_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            dataset_dir.mkdir()
            preprocessed_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.train.verify_required_global_paths_set",
                    return_value={
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.train.find_dataset_dir",
                    return_value=preprocessed_root / dataset_dir.name,
                ),
                patch(
                    "meisenmeister.training.train.get_fold_sample_ids",
                    return_value={
                        "train": ["case_001_left", "case_001_right"],
                        "val": ["case_001_left", "case_001_right"],
                    },
                ) as mock_get_fold_sample_ids,
                patch(
                    "meisenmeister.training.train.get_trainer_class",
                    return_value=_MockTrainer,
                ),
            ):
                train_module.train(
                    1,
                    fold="all",
                    trainer_name="mmTrainer_Debug",
                )

        mock_get_fold_sample_ids.assert_called_once_with(
            preprocessed_root / dataset_dir.name,
            "all",
        )
        self.assertEqual(_MockTrainer.last_instance.fold, "all")
        self.assertTrue(_MockTrainer.last_instance.fit_called)

    def test_train_resolves_and_runs_selected_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            dataset_dir.mkdir()
            preprocessed_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.train.verify_required_global_paths_set",
                    return_value={
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.train.find_dataset_dir",
                    return_value=preprocessed_root / dataset_dir.name,
                ),
                patch(
                    "meisenmeister.training.train.get_fold_sample_ids",
                    return_value={
                        "train": ["case_001_left", "case_001_right"],
                        "val": ["case_002_left", "case_002_right"],
                    },
                ) as mock_get_fold_sample_ids,
                patch(
                    "meisenmeister.training.train.get_trainer_class",
                    return_value=_MockTrainer,
                ) as mock_get_trainer_class,
            ):
                train_module.train(
                    1,
                    fold=0,
                    trainer_name="mmTrainer_Debug",
                )

        mock_get_fold_sample_ids.assert_called_once_with(
            preprocessed_root / dataset_dir.name,
            0,
        )
        mock_get_trainer_class.assert_called_once_with("mmTrainer_Debug")
        self.assertIsNotNone(_MockTrainer.last_instance)
        self.assertEqual(_MockTrainer.last_instance.dataset_id, "001")
        self.assertEqual(_MockTrainer.last_instance.fold, 0)
        self.assertEqual(
            _MockTrainer.last_instance.architecture_name, "MockArchitecture"
        )
        self.assertEqual(
            _MockTrainer.last_instance.dataset_dir,
            preprocessed_root / dataset_dir.name,
        )
        self.assertEqual(
            _MockTrainer.last_instance.preprocessed_dataset_dir,
            preprocessed_root / dataset_dir.name,
        )
        self.assertEqual(_MockTrainer.last_instance.results_dir, root / "results")
        self.assertFalse(_MockTrainer.last_instance.continue_training)
        self.assertIsNone(_MockTrainer.last_instance.weights_path)
        self.assertIsNone(_MockTrainer.last_instance.experiment_postfix)
        self.assertFalse(_MockTrainer.last_instance.grad_cam_enabled)
        self.assertFalse(_MockTrainer.last_instance.grad_cam_checked)
        self.assertTrue(_MockTrainer.last_instance.fit_called)

    def test_train_val_last_loads_last_checkpoint_and_runs_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            dataset_dir.mkdir()
            preprocessed_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.train.verify_required_global_paths_set",
                    return_value={
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.train.find_dataset_dir",
                    return_value=preprocessed_root / dataset_dir.name,
                ),
                patch(
                    "meisenmeister.training.train.get_fold_sample_ids",
                    return_value={
                        "train": ["case_001_left", "case_001_right"],
                        "val": ["case_002_left", "case_002_right"],
                    },
                ),
                patch(
                    "meisenmeister.training.train.get_trainer_class",
                    return_value=_MockTrainer,
                ),
                patch(
                    "meisenmeister.training.train.torch.load",
                    return_value={"model_state_dict": {"weight": 1}},
                ) as mock_torch_load,
                patch(
                    "meisenmeister.training.train.run_final_validation_evaluation"
                ) as mock_run_eval,
            ):
                train_module.train(
                    1,
                    fold=0,
                    trainer_name="mmTrainer_Debug",
                    val="last",
                )

        self.assertFalse(_MockTrainer.last_instance.fit_called)
        mock_torch_load.assert_called_once()
        self.assertIn(
            "model_last.pt",
            str(
                mock_torch_load.call_args.kwargs["map_location"]
                if False
                else mock_torch_load.call_args.args[0]
            ),
        )
        self.assertEqual(_MockTrainer.last_instance.loaded_state_dict, {"weight": 1})
        self.assertIn(
            "eval_last.json", str(mock_run_eval.call_args.kwargs["output_path"])
        )

    def test_train_val_last_uses_fold_all_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            dataset_dir.mkdir()
            preprocessed_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.train.verify_required_global_paths_set",
                    return_value={
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.train.find_dataset_dir",
                    return_value=preprocessed_root / dataset_dir.name,
                ),
                patch(
                    "meisenmeister.training.train.get_fold_sample_ids",
                    return_value={
                        "train": ["case_001_left", "case_001_right"],
                        "val": ["case_001_left", "case_001_right"],
                    },
                ),
                patch(
                    "meisenmeister.training.train.get_trainer_class",
                    return_value=_MockTrainer,
                ),
                patch(
                    "meisenmeister.training.train.torch.load",
                    return_value={"model_state_dict": {"weight": 1}},
                ) as mock_torch_load,
                patch(
                    "meisenmeister.training.train.run_final_validation_evaluation"
                ) as mock_run_eval,
            ):
                train_module.train(
                    1,
                    fold="all",
                    trainer_name="mmTrainer_Debug",
                    val="last",
                )

        self.assertIn("fold_all", str(mock_torch_load.call_args.args[0]))
        self.assertIn("fold_all", str(mock_run_eval.call_args.kwargs["output_path"]))

    def test_train_val_best_loads_best_checkpoint_and_runs_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            dataset_dir.mkdir()
            preprocessed_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.train.verify_required_global_paths_set",
                    return_value={
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.train.find_dataset_dir",
                    return_value=preprocessed_root / dataset_dir.name,
                ),
                patch(
                    "meisenmeister.training.train.get_fold_sample_ids",
                    return_value={
                        "train": ["case_001_left", "case_001_right"],
                        "val": ["case_002_left", "case_002_right"],
                    },
                ),
                patch(
                    "meisenmeister.training.train.get_trainer_class",
                    return_value=_MockTrainer,
                ),
                patch(
                    "meisenmeister.training.train.torch.load",
                    return_value={"model_state_dict": {"weight": 2}},
                ) as mock_torch_load,
                patch(
                    "meisenmeister.training.train.run_final_validation_evaluation"
                ) as mock_run_eval,
            ):
                train_module.train(
                    1,
                    fold=0,
                    trainer_name="mmTrainer_Debug",
                    val="best",
                )

        self.assertFalse(_MockTrainer.last_instance.fit_called)
        self.assertIn("model_best.pt", str(mock_torch_load.call_args.args[0]))
        self.assertEqual(_MockTrainer.last_instance.loaded_state_dict, {"weight": 2})
        self.assertIn(
            "eval_best.json", str(mock_run_eval.call_args.kwargs["output_path"])
        )
        self.assertIsNone(mock_run_eval.call_args.kwargs["grad_cam_output_dir"])

    def test_train_passes_continue_training_to_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            dataset_dir.mkdir()
            preprocessed_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.train.verify_required_global_paths_set",
                    return_value={
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.train.find_dataset_dir",
                    return_value=preprocessed_root / dataset_dir.name,
                ),
                patch(
                    "meisenmeister.training.train.get_fold_sample_ids",
                    return_value={
                        "train": ["case_001_left", "case_001_right"],
                        "val": ["case_002_left", "case_002_right"],
                    },
                ),
                patch(
                    "meisenmeister.training.train.get_trainer_class",
                    return_value=_MockTrainer,
                ),
            ):
                train_module.train(
                    1,
                    fold=0,
                    trainer_name="mmTrainer_Debug",
                    continue_training=True,
                )

        self.assertTrue(_MockTrainer.last_instance.continue_training)

    def test_train_only_checks_grad_cam_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            dataset_dir.mkdir()
            preprocessed_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.train.verify_required_global_paths_set",
                    return_value={
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.train.find_dataset_dir",
                    return_value=preprocessed_root / dataset_dir.name,
                ),
                patch(
                    "meisenmeister.training.train.get_fold_sample_ids",
                    return_value={"train": ["a"], "val": ["b"]},
                ),
                patch(
                    "meisenmeister.training.train.get_trainer_class",
                    return_value=_MockTrainer,
                ),
            ):
                train_module.train(1, fold=0, trainer_name="mmTrainer_Debug")

        self.assertFalse(_MockTrainer.last_instance.grad_cam_checked)

    def test_train_checks_grad_cam_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            dataset_dir.mkdir()
            preprocessed_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.train.verify_required_global_paths_set",
                    return_value={
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.train.find_dataset_dir",
                    return_value=preprocessed_root / dataset_dir.name,
                ),
                patch(
                    "meisenmeister.training.train.get_fold_sample_ids",
                    return_value={"train": ["a"], "val": ["b"]},
                ),
                patch(
                    "meisenmeister.training.train.get_trainer_class",
                    return_value=_MockTrainer,
                ),
            ):
                train_module.train(
                    1,
                    fold=0,
                    trainer_name="mmTrainer_Debug",
                    grad_cam_enabled=True,
                )

        self.assertTrue(_MockTrainer.last_instance.grad_cam_enabled)
        self.assertTrue(_MockTrainer.last_instance.grad_cam_checked)

    def test_train_val_with_grad_cam_passes_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            dataset_dir.mkdir()
            preprocessed_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.train.verify_required_global_paths_set",
                    return_value={
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.train.find_dataset_dir",
                    return_value=preprocessed_root / dataset_dir.name,
                ),
                patch(
                    "meisenmeister.training.train.get_fold_sample_ids",
                    return_value={
                        "train": ["case_001_left"],
                        "val": ["case_002_left"],
                    },
                ),
                patch(
                    "meisenmeister.training.train.get_trainer_class",
                    return_value=_MockTrainer,
                ),
                patch(
                    "meisenmeister.training.train.torch.load",
                    return_value={"model_state_dict": {"weight": 1}},
                ),
                patch(
                    "meisenmeister.training.train.run_final_validation_evaluation"
                ) as mock_run_eval,
            ):
                train_module.train(
                    1,
                    fold=0,
                    trainer_name="mmTrainer_Debug",
                    val="last",
                    grad_cam_enabled=True,
                )

        self.assertTrue(_MockTrainer.last_instance.grad_cam_checked)
        self.assertIn(
            "grad_cam_last",
            str(mock_run_eval.call_args.kwargs["grad_cam_output_dir"]),
        )
        self.assertEqual(
            mock_run_eval.call_args.kwargs["grad_cam_checkpoint_kind"], "last"
        )

    def test_train_passes_weights_and_postfix_to_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "Dataset_001_Test"
            preprocessed_root = root / "preprocessed"
            dataset_dir.mkdir()
            preprocessed_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.train.verify_required_global_paths_set",
                    return_value={
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": root / "results",
                    },
                ),
                patch(
                    "meisenmeister.training.train.find_dataset_dir",
                    return_value=preprocessed_root / dataset_dir.name,
                ),
                patch(
                    "meisenmeister.training.train.get_fold_sample_ids",
                    return_value={
                        "train": ["case_001_left", "case_001_right"],
                        "val": ["case_002_left", "case_002_right"],
                    },
                ),
                patch(
                    "meisenmeister.training.train.get_trainer_class",
                    return_value=_MockTrainer,
                ),
            ):
                train_module.train(
                    1,
                    fold=0,
                    trainer_name="mmTrainer_Debug",
                    weights_path="/tmp/pretrained.pt",
                    experiment_postfix="finetuningNNSSL",
                )

        self.assertEqual(
            _MockTrainer.last_instance.weights_path, Path("/tmp/pretrained.pt")
        )
        self.assertEqual(
            _MockTrainer.last_instance.experiment_postfix, "finetuningNNSSL"
        )

    def test_train_rejects_continue_training_with_weights(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Cannot use --continue-training and --weights together",
        ):
            train_module.train(
                1,
                fold=0,
                continue_training=True,
                weights_path="/tmp/pretrained.pt",
            )

    def test_train_rejects_val_with_continue_training(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Cannot use --val together with --continue-training",
        ):
            train_module.train(
                1,
                fold=0,
                continue_training=True,
                val="last",
            )

    def test_train_requires_only_preprocessed_and_results_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preprocessed_root = root / "preprocessed"
            preprocessed_dataset_dir = preprocessed_root / "Dataset_001_Test"
            results_root = root / "results"
            preprocessed_dataset_dir.mkdir(parents=True)
            results_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.train.verify_required_global_paths_set",
                    return_value={
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": results_root,
                    },
                ),
                patch(
                    "meisenmeister.training.train.get_fold_sample_ids",
                    return_value={
                        "train": ["case_001_left", "case_001_right"],
                        "val": ["case_002_left", "case_002_right"],
                    },
                ),
                patch(
                    "meisenmeister.training.train.get_trainer_class",
                    return_value=_MockTrainer,
                ),
            ):
                train_module.train(
                    1,
                    fold=0,
                    trainer_name="mmTrainer_Debug",
                )

        self.assertEqual(
            _MockTrainer.last_instance.dataset_dir,
            preprocessed_dataset_dir,
        )


if __name__ == "__main__":
    unittest.main()
