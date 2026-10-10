from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import blosc2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from meisenmeister.architectures import BaseArchitecture, PrimusMClsNetwork
from meisenmeister.data_augmentations import Compose3D, RandomShiftWithinMargin3D
from meisenmeister.training.trainers.mm_trainer import mmTrainer
from meisenmeister.training.trainers.networks.nnunet_encoder import (
    mmTrainer_NNUNetEncoder,
    mmTrainer_NNUNetEncoder_Finetune_ClassBalanced,
    mmTrainer_NNUNetEncoder_Finetune_TorchIO,
)
from meisenmeister.training.trainers.networks.primus import mmTrainer_PrimusM
from meisenmeister.utils.training import (
    build_final_validation_evaluation,
    compute_stratified_bootstrap_interval,
    run_final_validation_evaluation,
    save_da_preview,
)


class _TinyROIDataset(Dataset):
    def __init__(self, labels: list[int]) -> None:
        self.labels = labels
        self.samples = [
            {"sample_id": f"case_{index:03d}_left", "label": label}
            for index, label in enumerate(labels)
        ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        label = self.labels[index]
        return {
            "image": torch.full((1, 2, 2, 2), float(index + 1), dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": f"case_{index:03d}_left",
            "case_id": f"case_{index:03d}",
            "roi_name": "left",
        }


class _TinyNet(BaseArchitecture):
    def __init__(self) -> None:
        super().__init__(in_channels=1, num_classes=2)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc(x.flatten(1))


class _TinyGradCamNet(BaseArchitecture):
    def __init__(self) -> None:
        super().__init__(in_channels=1, num_classes=2)
        self.conv = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def get_grad_cam_target_layer(self) -> nn.Module:
        return self.conv


class _TinyPrimusLikeNet(_TinyNet):
    def get_init_kwargs(self) -> dict:
        return {
            "input_shape": (16, 16, 16),
            "patch_embed_size": (8, 8, 8),
        }


class MMTrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def _make_trainer(
        self,
        *,
        continue_training: bool = False,
        num_epochs: int = 2,
        weights_path: Path | None = None,
        experiment_postfix: str | None = None,
    ) -> mmTrainer:
        dataset_dir = self.root / "Dataset_001_Test"
        preprocessed_dataset_dir = self.root / "Dataset_001_Test"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset.json").write_text("{}", encoding="utf-8")
        (preprocessed_dataset_dir / "mmPlans.json").write_text("{}", encoding="utf-8")
        with patch(
            "meisenmeister.training.trainers.mm_trainer.get_fold_sample_ids",
            return_value={
                "train": ["case_000_left", "case_001_left"],
                "val": ["case_002_left", "case_003_left"],
            },
        ):
            trainer = mmTrainer(
                dataset_id="001",
                fold=0,
                dataset_dir=dataset_dir,
                preprocessed_dataset_dir=preprocessed_dataset_dir,
                results_dir=self.root / "results",
                architecture_name="ResNet3D18",
                num_epochs=num_epochs,
                batch_size=2,
                num_workers=0,
                shuffle=False,
                continue_training=continue_training,
                weights_path=weights_path,
                experiment_postfix=experiment_postfix,
            )

        trainer.device = torch.device("cpu")
        trainer._train_dataset = _TinyROIDataset([0, 1])
        trainer._val_dataset = _TinyROIDataset([1, 0])
        trainer._architecture = _TinyNet().to(trainer.device)
        trainer._loss = nn.CrossEntropyLoss()
        trainer._optimizer = None
        trainer._scheduler = None
        return trainer

    def _make_nnunet_trainer(
        self, *, target_shape: list[int]
    ) -> mmTrainer_NNUNetEncoder:
        dataset_dir = self.root / "Dataset_001_Test"
        preprocessed_dataset_dir = self.root / "Dataset_001_Test"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset.json").write_text("{}", encoding="utf-8")
        (preprocessed_dataset_dir / "mmPlans.json").write_text(
            json.dumps(
                {
                    "target_shape": target_shape,
                    "target_spacing": [1.0, 1.0, 1.0],
                    "margin_mm": [0.0, 0.0, 0.0],
                }
            ),
            encoding="utf-8",
        )
        with patch(
            "meisenmeister.training.trainers.mm_trainer.get_fold_sample_ids",
            return_value={
                "train": ["case_000_left", "case_001_left"],
                "val": ["case_002_left", "case_003_left"],
            },
        ):
            trainer = mmTrainer_NNUNetEncoder(
                dataset_id="001",
                fold=0,
                dataset_dir=dataset_dir,
                preprocessed_dataset_dir=preprocessed_dataset_dir,
                results_dir=self.root / "results",
            )
        trainer.device = torch.device("cpu")
        trainer._train_dataset = _TinyROIDataset([0, 1])
        trainer._val_dataset = _TinyROIDataset([1, 0])
        return trainer

    def _make_class_balanced_finetune_trainer(
        self, *, train_labels: list[int]
    ) -> mmTrainer_NNUNetEncoder_Finetune_ClassBalanced:
        dataset_dir = self.root / "Dataset_001_Test"
        preprocessed_dataset_dir = self.root / "Dataset_001_Test"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset.json").write_text("{}", encoding="utf-8")
        (preprocessed_dataset_dir / "mmPlans.json").write_text(
            json.dumps(
                {
                    "target_shape": [128, 160, 192],
                    "target_spacing": [1.0, 1.0, 1.0],
                    "margin_mm": [0.0, 0.0, 0.0],
                }
            ),
            encoding="utf-8",
        )
        with patch(
            "meisenmeister.training.trainers.mm_trainer.get_fold_sample_ids",
            return_value={
                "train": [
                    f"case_{index:03d}_left" for index in range(len(train_labels))
                ],
                "val": ["case_100_left", "case_101_left"],
            },
        ):
            trainer = mmTrainer_NNUNetEncoder_Finetune_ClassBalanced(
                dataset_id="001",
                fold=0,
                dataset_dir=dataset_dir,
                preprocessed_dataset_dir=preprocessed_dataset_dir,
                results_dir=self.root / "results",
                num_workers=0,
            )
        trainer.device = torch.device("cpu")
        trainer._train_dataset = _TinyROIDataset(train_labels)
        trainer._val_dataset = _TinyROIDataset([1, 0])
        return trainer

    def _make_torchio_finetune_trainer(
        self,
    ) -> mmTrainer_NNUNetEncoder_Finetune_TorchIO:
        dataset_dir = self.root / "Dataset_001_Test"
        preprocessed_dataset_dir = self.root / "Dataset_001_Test"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset.json").write_text("{}", encoding="utf-8")
        (preprocessed_dataset_dir / "mmPlans.json").write_text(
            json.dumps(
                {
                    "target_shape": [128, 160, 192],
                    "target_spacing": [1.0, 1.0, 1.0],
                    "margin_mm": [0.0, 0.0, 0.0],
                }
            ),
            encoding="utf-8",
        )
        with patch(
            "meisenmeister.training.trainers.mm_trainer.get_fold_sample_ids",
            return_value={
                "train": ["case_000_left", "case_001_left"],
                "val": ["case_002_left", "case_003_left"],
            },
        ):
            trainer = mmTrainer_NNUNetEncoder_Finetune_TorchIO(
                dataset_id="001",
                fold=0,
                dataset_dir=dataset_dir,
                preprocessed_dataset_dir=preprocessed_dataset_dir,
                results_dir=self.root / "results",
                num_workers=0,
            )
        trainer.device = torch.device("cpu")
        trainer._train_dataset = _TinyROIDataset([0, 1])
        trainer._val_dataset = _TinyROIDataset([1, 0])
        return trainer

    def _make_primus_trainer(
        self,
        *,
        target_shape: list[int],
        num_epochs: int = 100,
    ) -> mmTrainer_PrimusM:
        dataset_dir = self.root / "Dataset_001_Test"
        preprocessed_dataset_dir = self.root / "Dataset_001_Test"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "dataset.json").write_text("{}", encoding="utf-8")
        (preprocessed_dataset_dir / "mmPlans.json").write_text(
            json.dumps(
                {
                    "target_shape": target_shape,
                    "target_spacing": [1.0, 1.0, 1.0],
                    "margin_mm": [0.0, 0.0, 0.0],
                }
            ),
            encoding="utf-8",
        )
        with patch(
            "meisenmeister.training.trainers.mm_trainer.get_fold_sample_ids",
            return_value={
                "train": ["case_000_left", "case_001_left"],
                "val": ["case_002_left", "case_003_left"],
            },
        ):
            trainer = mmTrainer_PrimusM(
                dataset_id="001",
                fold=0,
                dataset_dir=dataset_dir,
                preprocessed_dataset_dir=preprocessed_dataset_dir,
                results_dir=self.root / "results",
                num_epochs=num_epochs,
                num_workers=0,
                compile_enabled=False,
            )
        trainer.device = torch.device("cpu")
        trainer._train_dataset = _TinyROIDataset([0, 1])
        trainer._val_dataset = _TinyROIDataset([1, 0])
        return trainer

    def test_fit_copies_portable_inference_metadata_to_experiment_dir(self) -> None:
        trainer = self._make_trainer(num_epochs=0)

        with (
            patch("sys.stdout", new_callable=io.StringIO),
            patch(
                "meisenmeister.training.trainers.mm_trainer.run_final_validation_evaluation"
            ) as mock_final_eval,
        ):
            trainer.fit()

        self.assertTrue((trainer.experiment_dir / "dataset.json").is_file())
        self.assertTrue((trainer.experiment_dir / "mmPlans.json").is_file())
        mock_final_eval.assert_called_once()

    def test_save_da_preview_writes_and_overwrites_png(self) -> None:
        plot_path = self.root / "da_preview.png"
        samples = [
            {
                "image": torch.full(
                    (3, 4, 5, 6), float(index + 1), dtype=torch.float32
                ),
                "sample_id": f"case_{index:03d}_left",
            }
            for index in range(3)
        ]

        save_da_preview(samples, plot_path)
        first_size = plot_path.stat().st_size

        updated_samples = [
            {
                "image": torch.full(
                    (3, 4, 5, 6),
                    float(index + 11),
                    dtype=torch.float32,
                ),
                "sample_id": f"case_{index:03d}_left",
            }
            for index in range(3)
        ]
        save_da_preview(updated_samples, plot_path)

        self.assertTrue(plot_path.is_file())
        self.assertGreater(first_size, 0)
        self.assertGreater(plot_path.stat().st_size, 0)

    def test_nnunet_trainer_rejects_incompatible_target_shape_early(self) -> None:
        trainer = self._make_nnunet_trainer(target_shape=[129, 165, 184])

        with self.assertRaisesRegex(
            ValueError,
            r"ResidualEncoderClsNetwork requires target_shape divisible by \[16, 32, 32\], got \[129, 165, 184\]",
        ):
            trainer.fit()

    def test_primus_trainer_rejects_incompatible_target_shape_early(self) -> None:
        trainer = self._make_primus_trainer(target_shape=[15, 16, 16])

        with self.assertRaisesRegex(
            ValueError,
            r"PrimusMClsNetwork requires target_shape divisible by \[8, 8, 8\], got \[15, 16, 16\]",
        ):
            trainer.fit()

    def test_primus_trainer_builds_architecture_kwargs_from_plans(self) -> None:
        trainer = self._make_primus_trainer(target_shape=[16, 24, 32])

        self.assertEqual(
            trainer.get_architecture_kwargs(),
            {
                "input_shape": (16, 24, 32),
                "patch_embed_size": (8, 8, 8),
            },
        )

    def test_primus_trainer_uses_adamw_transformer_defaults(self) -> None:
        trainer = self._make_primus_trainer(target_shape=[16, 16, 16], num_epochs=1)
        trainer._architecture = _TinyPrimusLikeNet().to(trainer.device)

        optimizer = trainer.get_optimizer()

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(float(optimizer.param_groups[0]["lr"]), 3e-4)
        self.assertEqual(float(optimizer.param_groups[0]["weight_decay"]), 5e-2)
        self.assertEqual(tuple(optimizer.param_groups[0]["betas"]), (0.9, 0.98))

    def test_class_balanced_finetune_trainer_weights_minority_class_higher(
        self,
    ) -> None:
        trainer = self._make_class_balanced_finetune_trainer(train_labels=[0, 0, 0, 1])

        loss = trainer.get_loss()

        self.assertIsInstance(loss, nn.CrossEntropyLoss)
        self.assertIsNotNone(loss.weight)
        self.assertGreater(float(loss.weight[1]), float(loss.weight[0]))

    def test_torchio_finetune_trainer_builds_torchio_pipeline(self) -> None:
        trainer = self._make_torchio_finetune_trainer()
        sentinel_pipeline = object()

        with patch(
            "meisenmeister.training.trainers.networks.nnunet_encoder.build_default_mri_torchio_pipeline",
            return_value=sentinel_pipeline,
        ) as mock_build_pipeline:
            pipeline = trainer.get_train_augmentation_pipeline()

        self.assertIsInstance(pipeline, Compose3D)
        self.assertEqual(len(pipeline.augmentations), 2)
        self.assertIsInstance(pipeline.augmentations[0], RandomShiftWithinMargin3D)
        self.assertIs(pipeline.augmentations[1], sentinel_pipeline)
        mock_build_pipeline.assert_called_once_with()

    def _make_validation_metrics(
        self,
        *,
        loss: float,
        labels: list[int],
        predictions: list[int],
        probabilities: list[list[float]],
        sample_ids: list[str] | None = None,
    ) -> dict:
        resolved_sample_ids = sample_ids or [
            f"case_{index + 2:03d}_left" for index in range(len(labels))
        ]
        return {
            "loss": float(loss),
            "num_samples": len(labels),
            "num_correct": sum(
                int(prediction == label)
                for prediction, label in zip(predictions, labels, strict=True)
            ),
            "labels": torch.tensor(labels),
            "predictions": torch.tensor(predictions),
            "probabilities": torch.tensor(probabilities, dtype=torch.float32),
            "sample_ids": resolved_sample_ids,
            "case_ids": [
                sample_id.rsplit("_", 1)[0] for sample_id in resolved_sample_ids
            ],
            "roi_names": [
                sample_id.rsplit("_", 1)[1] for sample_id in resolved_sample_ids
            ],
        }

    def test_get_optimizer_uses_model_parameters(self) -> None:
        trainer = self._make_trainer()

        optimizer = trainer.get_optimizer()

        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(
            len(optimizer_parameters),
            len(list(trainer.get_architecture().parameters())),
        )
        self.assertSetEqual(
            {id(parameter) for parameter in optimizer_parameters},
            {id(parameter) for parameter in trainer.get_architecture().parameters()},
        )

    def test_train_step_returns_loss_and_accuracy_metrics(self) -> None:
        trainer = self._make_trainer()
        batch = next(iter(DataLoader(trainer.get_train_dataset(), batch_size=2)))
        before = trainer.get_architecture().fc.weight.detach().clone()

        metrics = trainer.train_step(batch, batch_idx=1)

        self.assertTrue(torch.isfinite(metrics["loss_sum"]))
        self.assertEqual(metrics["num_samples"], 2)
        self.assertGreaterEqual(int(metrics["num_correct"]), 0)
        self.assertLessEqual(int(metrics["num_correct"]), 2)
        self.assertFalse(
            torch.equal(before, trainer.get_architecture().fc.weight.detach())
        )

    def test_validate_step_runs_without_recording_gradients(self) -> None:
        trainer = self._make_trainer()
        batch = next(iter(DataLoader(trainer.get_val_dataset(), batch_size=2)))

        metrics = trainer.validate_step(batch, batch_idx=1)

        self.assertTrue(torch.isfinite(metrics["loss_sum"]))
        self.assertEqual(metrics["num_samples"], 2)
        self.assertGreaterEqual(int(metrics["num_correct"]), 0)
        self.assertLessEqual(int(metrics["num_correct"]), 2)
        self.assertEqual(tuple(metrics["labels"].shape), (2,))
        self.assertEqual(tuple(metrics["probabilities"].shape), (2, 2))
        self.assertEqual(metrics["sample_ids"], ["case_000_left", "case_001_left"])
        self.assertEqual(metrics["case_ids"], ["case_000", "case_001"])
        self.assertEqual(metrics["roi_names"], ["left", "left"])
        for parameter in trainer.get_architecture().parameters():
            self.assertIsNone(parameter.grad)

    def test_base_architecture_grad_cam_target_layer_raises_by_default(self) -> None:
        model = _TinyNet()

        with self.assertRaisesRegex(NotImplementedError, "Grad-CAM is not available"):
            model.get_grad_cam_target_layer()

    def test_primus_trainer_grad_cam_enablement_fails_clearly(self) -> None:
        trainer = self._make_primus_trainer(target_shape=[16, 16, 16])
        trainer._architecture = PrimusMClsNetwork(
            in_channels=1,
            num_classes=2,
            input_shape=(16, 16, 16),
        )

        with self.assertRaisesRegex(ValueError, "Grad-CAM is not available"):
            trainer.ensure_grad_cam_available()

    def test_run_final_validation_evaluation_writes_grad_cam_outputs(self) -> None:
        trainer = self._make_trainer(num_epochs=0)
        trainer._architecture = _TinyGradCamNet().to(trainer.device)
        output_path = trainer.fold_dir / "eval_last.json"
        grad_cam_dir = trainer.fold_dir / "grad_cam_last"

        payload = run_final_validation_evaluation(
            trainer,
            output_path=output_path,
            log_path=trainer.log_path,
            grad_cam_output_dir=grad_cam_dir,
            grad_cam_checkpoint_kind="last",
        )

        self.assertTrue(output_path.is_file())
        self.assertEqual(payload["summary"]["num_samples"], 2)
        self.assertTrue((grad_cam_dir / "metadata.json").is_file())
        self.assertTrue((grad_cam_dir / "case_000_left.b2nd").is_file())
        metadata = json.loads(
            (grad_cam_dir / "metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(metadata["checkpoint_kind"], "last")
        self.assertEqual(metadata["target_policy"], "predicted_class")
        heatmap = blosc2.open(
            urlpath=str(grad_cam_dir / "case_000_left.b2nd"), mode="r"
        )
        self.assertEqual(tuple(heatmap.shape), (2, 2, 2))
        self.assertGreaterEqual(float(heatmap[:].min()), 0.0)
        self.assertLessEqual(float(heatmap[:].max()), 1.0)

    def test_fit_creates_artifacts_and_logs_overwrite_warning(self) -> None:
        trainer = self._make_trainer()
        trainer.fold_dir.mkdir(parents=True, exist_ok=True)
        (trainer.fold_dir / "stale.txt").write_text("stale", encoding="utf-8")

        train_loader = [object()]
        val_loader = [object()]
        train_step_metrics = [
            {"loss": 0.8, "num_samples": 2, "num_correct": 1},
            {"loss": 0.7, "num_samples": 2, "num_correct": 2},
        ]
        validate_step_metrics = [
            self._make_validation_metrics(
                loss=0.6,
                labels=[0, 1],
                predictions=[0, 1],
                probabilities=[[0.9, 0.1], [0.1, 0.9]],
            ),
            self._make_validation_metrics(
                loss=0.5,
                labels=[0, 1],
                predictions=[0, 1],
                probabilities=[[0.95, 0.05], [0.05, 0.95]],
            ),
            self._make_validation_metrics(
                loss=0.5,
                labels=[0, 1],
                predictions=[0, 1],
                probabilities=[[0.95, 0.05], [0.05, 0.95]],
            ),
        ]

        with (
            patch.object(trainer, "get_train_dataloader", return_value=train_loader),
            patch.object(trainer, "get_val_dataloader", return_value=val_loader),
            patch.object(trainer, "train_step", side_effect=train_step_metrics),
            patch.object(trainer, "validate_step", side_effect=validate_step_metrics),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            trainer.fit()

        output = stdout.getvalue()
        self.assertIn("WARNING: YOU ARE OVERWRITING", output)
        self.assertTrue(trainer.log_path.is_file())
        self.assertTrue(trainer.last_checkpoint_path.is_file())
        self.assertTrue(trainer.best_checkpoint_path.is_file())
        self.assertTrue(trainer.eval_last_path.is_file())
        self.assertTrue(trainer.plot_path.is_file())
        eval_payload = json.loads(trainer.eval_last_path.read_text(encoding="utf-8"))

        checkpoint = torch.load(
            trainer.last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(checkpoint["last_completed_epoch"], 2)
        self.assertEqual(
            checkpoint["trainer_config"]["fold_dir"], str(trainer.fold_dir)
        )
        self.assertEqual(checkpoint["history"]["epoch"], [1, 2])
        self.assertEqual(checkpoint["best_state"]["epoch"], 2)
        self.assertIn("Saved new best model at epoch 1", trainer.log_path.read_text())
        self.assertIn("Saved new best model at epoch 2", trainer.log_path.read_text())
        self.assertIsNone(checkpoint["trainer_config"]["source_weights_path"])
        self.assertIsNone(checkpoint["trainer_config"]["experiment_postfix"])
        self.assertEqual(checkpoint["trainer_config"]["architecture_kwargs"], {})
        self.assertEqual(
            set(eval_payload["predictions"]), {"case_002_left", "case_003_left"}
        )
        self.assertEqual(eval_payload["summary"]["num_samples"], 2)
        self.assertEqual(eval_payload["summary"]["balanced_accuracy"], 1.0)
        self.assertEqual(eval_payload["summary"]["macro_auc"], 1.0)

    def test_experiment_postfix_changes_experiment_directory_name(self) -> None:
        trainer = self._make_trainer(experiment_postfix="finetuningNNSSL")

        self.assertEqual(
            trainer.experiment_dir.name,
            "mmTrainer_ResNet3D18_finetuningNNSSL",
        )

    def test_fit_loads_external_plain_state_dict_weights(self) -> None:
        source_model = _TinyNet()
        with torch.no_grad():
            source_model.fc.weight.fill_(1.25)
            source_model.fc.bias.fill_(0.5)
        weights_path = self.root / "pretrained_state_dict.pt"
        torch.save(source_model.state_dict(), weights_path)

        trainer = self._make_trainer(
            num_epochs=1,
            weights_path=weights_path,
            experiment_postfix="finetuningNNSSL",
        )
        train_loader = [object()]
        val_loader = [object()]

        with (
            patch.object(trainer, "get_train_dataloader", return_value=train_loader),
            patch.object(trainer, "get_val_dataloader", return_value=val_loader),
            patch.object(
                trainer,
                "train_step",
                return_value={"loss": 0.9, "num_samples": 2, "num_correct": 1},
            ),
            patch.object(
                trainer,
                "validate_step",
                return_value=self._make_validation_metrics(
                    loss=0.7,
                    labels=[0, 0],
                    predictions=[0, 0],
                    probabilities=[[0.9, 0.1], [0.8, 0.2]],
                ),
            ),
        ):
            trainer.fit()

        self.assertTrue(
            torch.allclose(
                trainer.get_architecture().fc.weight.detach(),
                source_model.fc.weight.detach(),
            )
        )
        checkpoint = torch.load(
            trainer.last_checkpoint_path, map_location="cpu", weights_only=False
        )
        self.assertEqual(
            checkpoint["trainer_config"]["source_weights_path"], str(weights_path)
        )
        self.assertEqual(
            checkpoint["trainer_config"]["experiment_postfix"], "finetuningNNSSL"
        )

    def test_fit_saves_architecture_kwargs_for_portable_inference(self) -> None:
        trainer = self._make_primus_trainer(
            target_shape=[16, 16, 16],
            num_epochs=1,
        )
        trainer._architecture = _TinyPrimusLikeNet().to(trainer.device)
        train_loader = [object()]
        val_loader = [object()]

        with (
            patch.object(trainer, "get_train_dataloader", return_value=train_loader),
            patch.object(trainer, "get_val_dataloader", return_value=val_loader),
            patch.object(
                trainer,
                "train_step",
                return_value={"loss": 0.9, "num_samples": 2, "num_correct": 1},
            ),
            patch.object(
                trainer,
                "validate_step",
                return_value=self._make_validation_metrics(
                    loss=0.7,
                    labels=[0, 0],
                    predictions=[0, 0],
                    probabilities=[[0.9, 0.1], [0.8, 0.2]],
                ),
            ),
        ):
            trainer.fit()

        checkpoint = torch.load(
            trainer.last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(
            checkpoint["trainer_config"]["architecture_kwargs"],
            {
                "input_shape": (16, 16, 16),
                "patch_embed_size": (8, 8, 8),
            },
        )

    def test_fit_loads_external_checkpoint_model_state_dict_weights(self) -> None:
        source_model = _TinyNet()
        with torch.no_grad():
            source_model.fc.weight.fill_(0.75)
            source_model.fc.bias.fill_(0.25)
        weights_path = self.root / "pretrained_checkpoint.pt"
        torch.save({"model_state_dict": source_model.state_dict()}, weights_path)

        trainer = self._make_trainer(num_epochs=0, weights_path=weights_path)
        trainer.fold_dir.mkdir(parents=True, exist_ok=True)
        trainer.log_path.write_text("", encoding="utf-8")

        trainer.get_architecture().load_initial_weights(
            path=weights_path,
            device=trainer.device,
        )

        self.assertTrue(
            torch.allclose(
                trainer.get_architecture().fc.weight.detach(),
                source_model.fc.weight.detach(),
            )
        )

    def test_fit_resume_continues_same_experiment_directory(self) -> None:
        initial_trainer = self._make_trainer(num_epochs=1)
        train_loader = [object()]
        val_loader = [object()]

        initial_train_step_metrics = [{"loss": 0.8, "num_samples": 2, "num_correct": 1}]
        initial_validate_step_metrics = [
            self._make_validation_metrics(
                loss=0.6,
                labels=[0, 1],
                predictions=[0, 1],
                probabilities=[[0.9, 0.1], [0.1, 0.9]],
            ),
            self._make_validation_metrics(
                loss=0.6,
                labels=[0, 1],
                predictions=[0, 1],
                probabilities=[[0.9, 0.1], [0.1, 0.9]],
            ),
        ]

        with (
            patch.object(
                initial_trainer, "get_train_dataloader", return_value=train_loader
            ),
            patch.object(
                initial_trainer, "get_val_dataloader", return_value=val_loader
            ),
            patch.object(
                initial_trainer, "train_step", side_effect=initial_train_step_metrics
            ),
            patch.object(
                initial_trainer,
                "validate_step",
                side_effect=initial_validate_step_metrics,
            ),
        ):
            initial_trainer.fit()

        resumed_trainer = self._make_trainer(continue_training=True, num_epochs=2)
        resumed_train_step_metrics = [{"loss": 0.4, "num_samples": 2, "num_correct": 2}]
        resumed_validate_step_metrics = [
            self._make_validation_metrics(
                loss=0.4,
                labels=[0, 1],
                predictions=[0, 1],
                probabilities=[[0.98, 0.02], [0.02, 0.98]],
            ),
            self._make_validation_metrics(
                loss=0.4,
                labels=[0, 1],
                predictions=[0, 1],
                probabilities=[[0.98, 0.02], [0.02, 0.98]],
            ),
        ]

        with (
            patch.object(
                resumed_trainer, "get_train_dataloader", return_value=train_loader
            ),
            patch.object(
                resumed_trainer, "get_val_dataloader", return_value=val_loader
            ),
            patch.object(
                resumed_trainer, "train_step", side_effect=resumed_train_step_metrics
            ),
            patch.object(
                resumed_trainer,
                "validate_step",
                side_effect=resumed_validate_step_metrics,
            ),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            resumed_trainer.fit()

        output = stdout.getvalue()
        self.assertIn("Resuming training from epoch 2", output)
        checkpoint = torch.load(
            resumed_trainer.last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(checkpoint["history"]["epoch"], [1, 2])
        self.assertEqual(checkpoint["last_completed_epoch"], 2)
        self.assertEqual(checkpoint["best_state"]["epoch"], 2)

    def test_fit_resume_uses_matching_postfix_experiment_directory(self) -> None:
        initial_trainer = self._make_trainer(
            num_epochs=1,
            experiment_postfix="finetuningNNSSL",
        )
        train_loader = [object()]
        val_loader = [object()]

        with (
            patch.object(
                initial_trainer, "get_train_dataloader", return_value=train_loader
            ),
            patch.object(
                initial_trainer, "get_val_dataloader", return_value=val_loader
            ),
            patch.object(
                initial_trainer,
                "train_step",
                return_value={"loss": 0.8, "num_samples": 2, "num_correct": 1},
            ),
            patch.object(
                initial_trainer,
                "validate_step",
                return_value=self._make_validation_metrics(
                    loss=0.6,
                    labels=[0, 1],
                    predictions=[0, 1],
                    probabilities=[[0.9, 0.1], [0.1, 0.9]],
                ),
            ),
        ):
            initial_trainer.fit()

        resumed_trainer = self._make_trainer(
            continue_training=True,
            num_epochs=2,
            experiment_postfix="finetuningNNSSL",
        )
        with (
            patch.object(
                resumed_trainer, "get_train_dataloader", return_value=train_loader
            ),
            patch.object(
                resumed_trainer, "get_val_dataloader", return_value=val_loader
            ),
            patch.object(
                resumed_trainer,
                "train_step",
                return_value={"loss": 0.4, "num_samples": 2, "num_correct": 2},
            ),
            patch.object(
                resumed_trainer,
                "validate_step",
                return_value=self._make_validation_metrics(
                    loss=0.4,
                    labels=[0, 1],
                    predictions=[0, 1],
                    probabilities=[[0.98, 0.02], [0.02, 0.98]],
                ),
            ),
        ):
            resumed_trainer.fit()

        checkpoint = torch.load(
            resumed_trainer.last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(
            checkpoint["trainer_config"]["experiment_postfix"], "finetuningNNSSL"
        )

    def test_resume_falls_back_to_best_checkpoint_if_last_is_corrupted(self) -> None:
        initial_trainer = self._make_trainer(num_epochs=1)
        train_loader = [object()]
        val_loader = [object()]

        with (
            patch.object(
                initial_trainer, "get_train_dataloader", return_value=train_loader
            ),
            patch.object(
                initial_trainer, "get_val_dataloader", return_value=val_loader
            ),
            patch.object(
                initial_trainer,
                "train_step",
                return_value={"loss": 0.8, "num_samples": 2, "num_correct": 1},
            ),
            patch.object(
                initial_trainer,
                "validate_step",
                return_value=self._make_validation_metrics(
                    loss=0.6,
                    labels=[0, 1],
                    predictions=[0, 1],
                    probabilities=[[0.9, 0.1], [0.1, 0.9]],
                ),
            ),
        ):
            initial_trainer.fit()

        initial_trainer.last_checkpoint_path.write_bytes(b"corrupted")

        resumed_trainer = self._make_trainer(continue_training=True, num_epochs=2)
        with (
            patch.object(
                resumed_trainer, "get_train_dataloader", return_value=train_loader
            ),
            patch.object(
                resumed_trainer, "get_val_dataloader", return_value=val_loader
            ),
            patch.object(
                resumed_trainer,
                "train_step",
                return_value={"loss": 0.4, "num_samples": 2, "num_correct": 2},
            ),
            patch.object(
                resumed_trainer,
                "validate_step",
                return_value=self._make_validation_metrics(
                    loss=0.4,
                    labels=[0, 1],
                    predictions=[0, 1],
                    probabilities=[[0.98, 0.02], [0.02, 0.98]],
                ),
            ),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            resumed_trainer.fit()

        output = stdout.getvalue()
        self.assertIn("resuming from fallback checkpoint", output.lower())
        checkpoint = torch.load(
            resumed_trainer.last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(checkpoint["history"]["epoch"], [1, 2])

    def test_fit_handles_undefined_macro_auc_without_crashing(self) -> None:
        trainer = self._make_trainer(num_epochs=1)
        train_loader = [object()]
        val_loader = [object()]

        with (
            patch.object(
                trainer,
                "get_train_dataloader",
                return_value=train_loader,
            ),
            patch.object(
                trainer,
                "get_val_dataloader",
                return_value=val_loader,
            ),
            patch.object(
                trainer,
                "train_step",
                return_value={"loss": 0.9, "num_samples": 2, "num_correct": 1},
            ),
            patch.object(
                trainer,
                "validate_step",
                return_value=self._make_validation_metrics(
                    loss=0.7,
                    labels=[0, 0],
                    predictions=[0, 0],
                    probabilities=[[0.9, 0.1], [0.8, 0.2]],
                ),
            ),
        ):
            trainer.fit()

        checkpoint = torch.load(
            trainer.last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        eval_payload = json.loads(trainer.eval_last_path.read_text(encoding="utf-8"))
        self.assertTrue(math.isnan(checkpoint["history"]["val_macro_auc"][0]))
        self.assertIn("val_macro_auc undefined", trainer.log_path.read_text())
        self.assertIsNone(eval_payload["summary"]["macro_auc"])
        self.assertFalse(eval_payload["summary"]["macro_auc_defined"])
        self.assertFalse(eval_payload["summary"]["macro_auc_ci"]["defined"])

    def test_eval_last_uses_last_model_outputs_not_best_epoch_outputs(self) -> None:
        trainer = self._make_trainer(num_epochs=2)
        train_loader = [object()]
        val_loader = [object()]
        validate_step_metrics = [
            self._make_validation_metrics(
                loss=0.3,
                labels=[0, 1],
                predictions=[0, 1],
                probabilities=[[0.99, 0.01], [0.01, 0.99]],
            ),
            self._make_validation_metrics(
                loss=0.9,
                labels=[0, 1],
                predictions=[0, 0],
                probabilities=[[0.9, 0.1], [0.9, 0.1]],
            ),
            self._make_validation_metrics(
                loss=0.9,
                labels=[0, 1],
                predictions=[0, 0],
                probabilities=[[0.9, 0.1], [0.9, 0.1]],
            ),
        ]

        with (
            patch.object(trainer, "get_train_dataloader", return_value=train_loader),
            patch.object(trainer, "get_val_dataloader", return_value=val_loader),
            patch.object(
                trainer,
                "train_step",
                side_effect=[
                    {"loss": 0.3, "num_samples": 2, "num_correct": 2},
                    {"loss": 0.9, "num_samples": 2, "num_correct": 1},
                ],
            ),
            patch.object(trainer, "validate_step", side_effect=validate_step_metrics),
        ):
            trainer.fit()

        checkpoint = torch.load(
            trainer.last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        eval_payload = json.loads(trainer.eval_last_path.read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["best_state"]["epoch"], 1)
        self.assertEqual(eval_payload["summary"]["balanced_accuracy"], 0.5)
        self.assertEqual(eval_payload["predictions"]["case_003_left"]["prediction"], 0)

    def test_completed_resume_recreates_missing_eval_last_json(self) -> None:
        initial_trainer = self._make_trainer(num_epochs=1)
        train_loader = [object()]
        val_loader = [object()]

        with (
            patch.object(
                initial_trainer, "get_train_dataloader", return_value=train_loader
            ),
            patch.object(
                initial_trainer, "get_val_dataloader", return_value=val_loader
            ),
            patch.object(
                initial_trainer,
                "train_step",
                return_value={"loss": 0.8, "num_samples": 2, "num_correct": 1},
            ),
            patch.object(
                initial_trainer,
                "validate_step",
                return_value=self._make_validation_metrics(
                    loss=0.6,
                    labels=[0, 1],
                    predictions=[0, 1],
                    probabilities=[[0.9, 0.1], [0.1, 0.9]],
                ),
            ),
        ):
            initial_trainer.fit()

        initial_trainer.eval_last_path.unlink()

        resumed_trainer = self._make_trainer(continue_training=True, num_epochs=1)
        with (
            patch.object(
                resumed_trainer, "get_val_dataloader", return_value=val_loader
            ),
            patch.object(
                resumed_trainer,
                "validate_step",
                return_value=self._make_validation_metrics(
                    loss=0.6,
                    labels=[0, 1],
                    predictions=[0, 1],
                    probabilities=[[0.9, 0.1], [0.1, 0.9]],
                ),
            ),
        ):
            resumed_trainer.fit()

        self.assertTrue(resumed_trainer.eval_last_path.is_file())
        eval_payload = json.loads(
            resumed_trainer.eval_last_path.read_text(encoding="utf-8")
        )
        self.assertEqual(eval_payload["summary"]["balanced_accuracy"], 1.0)

    def test_build_final_validation_evaluation_formats_summary_and_predictions(
        self,
    ) -> None:
        payload = build_final_validation_evaluation(
            [
                self._make_validation_metrics(
                    loss=0.4,
                    labels=[0, 1],
                    predictions=[0, 1],
                    probabilities=[[0.95, 0.05], [0.05, 0.95]],
                    sample_ids=["case_002_left", "case_003_right"],
                )
            ],
            n_bootstrap=32,
            seed=7,
        )

        self.assertEqual(payload["summary"]["num_samples"], 2)
        self.assertEqual(payload["summary"]["balanced_accuracy"], 1.0)
        self.assertEqual(payload["predictions"]["case_002_left"]["case_id"], "case_002")
        self.assertEqual(payload["predictions"]["case_003_right"]["roi_name"], "right")
        self.assertTrue(payload["summary"]["balanced_accuracy_ci"]["defined"])
        self.assertIsNotNone(payload["summary"]["balanced_accuracy_paper"])

    def test_compute_stratified_bootstrap_interval_is_deterministic(self) -> None:
        labels = torch.tensor([0, 0, 1, 1]).numpy()
        predictions = torch.tensor([0, 0, 1, 1]).numpy()
        probabilities = torch.tensor(
            [[0.95, 0.05], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
            dtype=torch.float32,
        ).numpy()

        first = compute_stratified_bootstrap_interval(
            labels,
            predictions,
            probabilities,
            metric_fn=lambda y_true, y_pred, _probs: float(
                sum(int(a == b) for a, b in zip(y_true, y_pred, strict=True))
            )
            / len(y_true),
            n_bootstrap=64,
            confidence_level=0.95,
            seed=11,
        )
        second = compute_stratified_bootstrap_interval(
            labels,
            predictions,
            probabilities,
            metric_fn=lambda y_true, y_pred, _probs: float(
                sum(int(a == b) for a, b in zip(y_true, y_pred, strict=True))
            )
            / len(y_true),
            n_bootstrap=64,
            confidence_level=0.95,
            seed=11,
        )

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
