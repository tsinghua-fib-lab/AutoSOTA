from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

preview_da_module = importlib.import_module("meisenmeister.training.preview_da")


class _PreviewDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict:
        return {
            "image": torch.full((2, 4, 5, 6), float(index + 1), dtype=torch.float32),
            "sample_id": f"case_{index:03d}_left",
            "case_id": f"case_{index:03d}",
            "roi_name": "left",
            "label": torch.tensor(index % 2, dtype=torch.long),
        }


class _MockTrainer:
    ARCHITECTURE_NAME = "MockArchitecture"

    def __init__(
        self,
        dataset_id: str,
        fold: int,
        dataset_dir: Path,
        preprocessed_dataset_dir: Path,
        results_dir: Path,
        architecture_name: str,
        num_workers: int | None = None,
        continue_training: bool = False,
        weights_path: Path | None = None,
        experiment_postfix: str | None = None,
        compile_enabled: bool = True,
    ) -> None:
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
        self._train_dataset = _PreviewDataset()

    def get_train_dataset(self):
        return self._train_dataset


class PreviewDAEntrypointTests(unittest.TestCase):
    def test_preview_da_uses_selected_trainer_and_default_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preprocessed_root = root / "preprocessed"
            dataset_dir = preprocessed_root / "Dataset_001_Test"
            results_root = root / "results"
            dataset_dir.mkdir(parents=True)
            results_root.mkdir()

            with (
                patch(
                    "meisenmeister.training.preview_da.verify_required_global_paths_set",
                    return_value={
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": results_root,
                    },
                ),
                patch(
                    "meisenmeister.training.preview_da.find_dataset_dir",
                    return_value=dataset_dir,
                ),
                patch(
                    "meisenmeister.training.preview_da.get_fold_sample_ids",
                    return_value={
                        "train": ["case_000_left", "case_001_left"],
                        "val": ["case_002_left", "case_003_left"],
                    },
                ),
                patch(
                    "meisenmeister.training.preview_da.get_trainer_class",
                    return_value=_MockTrainer,
                ),
                patch(
                    "meisenmeister.training.preview_da.random.sample",
                    return_value=[0, 2, 3],
                ),
                patch(
                    "meisenmeister.training.preview_da.save_da_preview"
                ) as mock_save_da_preview,
            ):
                output_path = preview_da_module.preview_da(
                    1,
                    fold=0,
                    trainer_name="mmTrainer_Debug",
                    num_samples=3,
                )

        self.assertEqual(
            output_path,
            results_root
            / "Dataset_001_Test"
            / "_MockTrainer_MockArchitecture"
            / "fold_0"
            / "da_preview.png",
        )
        self.assertEqual(mock_save_da_preview.call_args.args[1], output_path)
        self.assertEqual(len(mock_save_da_preview.call_args.args[0]), 3)

    def test_preview_da_honors_explicit_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preprocessed_root = root / "preprocessed"
            dataset_dir = preprocessed_root / "Dataset_001_Test"
            results_root = root / "results"
            dataset_dir.mkdir(parents=True)
            results_root.mkdir()
            explicit_output = root / "custom" / "preview.png"

            with (
                patch(
                    "meisenmeister.training.preview_da.verify_required_global_paths_set",
                    return_value={
                        "mm_preprocessed": preprocessed_root,
                        "mm_results": results_root,
                    },
                ),
                patch(
                    "meisenmeister.training.preview_da.find_dataset_dir",
                    return_value=dataset_dir,
                ),
                patch(
                    "meisenmeister.training.preview_da.get_fold_sample_ids",
                    return_value={
                        "train": ["case_000_left", "case_001_left"],
                        "val": ["case_002_left", "case_003_left"],
                    },
                ),
                patch(
                    "meisenmeister.training.preview_da.get_trainer_class",
                    return_value=_MockTrainer,
                ),
                patch(
                    "meisenmeister.training.preview_da.random.sample",
                    return_value=[1],
                ),
                patch(
                    "meisenmeister.training.preview_da.save_da_preview"
                ) as mock_save_da_preview,
            ):
                output_path = preview_da_module.preview_da(
                    1,
                    fold=0,
                    output_path=str(explicit_output),
                    num_samples=1,
                )

        self.assertEqual(output_path, explicit_output)
        self.assertEqual(mock_save_da_preview.call_args.args[1], explicit_output)


if __name__ == "__main__":
    unittest.main()
