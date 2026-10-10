from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from meisenmeister.training import (
    BaseTrainer,
    get_available_architecture_names,
    get_trainer_class,
    get_trainer_registry,
)
from meisenmeister.training.trainers.mm_trainer import mmTrainer
from meisenmeister.training.trainers.networks.nnunet_encoder import (
    mmTrainer_NNUNetEncoder,
    mmTrainer_NNUNetEncoder_Finetune_ClassBalanced,
    mmTrainer_NNUNetEncoder_Finetune_TorchIO,
)
from meisenmeister.training.trainers.networks.primus import (
    mmTrainer_PrimusM,
    mmTrainer_PrimusM_bs4,
)


class TrainingRegistryTests(unittest.TestCase):
    def test_registry_contains_only_sorted_trainer_classes(self) -> None:
        registry = get_trainer_registry()

        self.assertEqual(list(registry), sorted(registry))
        self.assertTrue(registry)
        for name, trainer_class in registry.items():
            with self.subTest(name=name):
                self.assertTrue(name.startswith("mmTrainer"))
                self.assertTrue(issubclass(trainer_class, BaseTrainer))

    def test_unknown_trainer_error_lists_available_names(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Unknown trainer 'missing'.*mmTrainer.*mmTrainer_Debug.*mmTrainer_NNUNetEncoder",
        ):
            get_trainer_class("missing")

    def test_base_trainer_cannot_be_instantiated_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(TypeError):
                BaseTrainer("001", 0, root, root)

    def test_representative_trainers_satisfy_base_contract(self) -> None:
        trainer_classes = [
            mmTrainer,
            mmTrainer_NNUNetEncoder,
            mmTrainer_NNUNetEncoder_Finetune_ClassBalanced,
            mmTrainer_NNUNetEncoder_Finetune_TorchIO,
            mmTrainer_PrimusM,
            mmTrainer_PrimusM_bs4,
        ]
        sample_ids = {
            "train": ["case_001_left", "case_001_right"],
            "val": ["case_002_left", "case_002_right"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mmPlans.json").write_text(
                '{"target_shape": [16, 16, 16]}',
                encoding="utf-8",
            )
            with patch(
                "meisenmeister.training.trainers.mm_trainer.get_fold_sample_ids",
                return_value=sample_ids,
            ):
                for trainer_class in trainer_classes:
                    with self.subTest(trainer=trainer_class.__name__):
                        trainer = trainer_class("001", 0, root, root, root)
                        self.assertIsInstance(trainer, BaseTrainer)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "mmPlans.json").write_text(
                '{"target_shape": [16, 16, 16]}',
                encoding="utf-8",
            )
            with patch(
                "meisenmeister.training.trainers.mm_trainer.get_fold_sample_ids",
                return_value=sample_ids,
            ):
                trainer = mmTrainer_PrimusM_bs4("001", 0, root, root, root)

        self.assertEqual(trainer.batch_size, 4)

    def test_architecture_exports_include_expected_names(self) -> None:
        available_names = get_available_architecture_names()

        self.assertIn("ResNet3D18", available_names)
        self.assertIn("ResidualEncoderClsNetwork", available_names)
        self.assertIn("PrimusMClsNetwork", available_names)


if __name__ == "__main__":
    unittest.main()
