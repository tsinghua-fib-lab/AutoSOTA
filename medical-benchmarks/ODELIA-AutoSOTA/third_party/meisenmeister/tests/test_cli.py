from __future__ import annotations

import argparse
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from meisenmeister import cli


class CliTests(unittest.TestCase):
    def test_parse_fold_accepts_integer_and_all(self) -> None:
        self.assertEqual(cli._parse_fold("3"), 3)
        self.assertEqual(cli._parse_fold("all"), "all")

    def test_parse_fold_rejects_invalid_value(self) -> None:
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError,
            "Use a non-negative integer or 'all'",
        ):
            cli._parse_fold("fold-zero")

    def test_mm_homogenize_requires_explicit_confirmation(self) -> None:
        with (
            patch("sys.argv", ["mm_homogenize", "-d", "1"]),
            patch("builtins.input", return_value="no"),
            patch("meisenmeister.cli.homogenize") as mock_homogenize,
        ):
            with self.assertRaisesRegex(SystemExit, "Aborted."):
                cli.mm_homogenize()

        mock_homogenize.assert_not_called()

    def test_mm_train_forwards_non_default_options(self) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "mm_train",
                    "-d",
                    "1",
                    "-f",
                    "all",
                    "--trainer",
                    "mmTrainer_Debug",
                    "--num-workers",
                    "4",
                    "-c",
                    "-w",
                    "/tmp/pretrained.pt",
                    "--postfix",
                    "portable",
                    "--val",
                    "best",
                    "--disable-compile",
                    "--grad-cam",
                ],
            ),
            patch("meisenmeister.cli.train") as mock_train,
        ):
            cli.mm_train()

        mock_train.assert_called_once_with(
            1,
            fold="all",
            trainer_name="mmTrainer_Debug",
            num_workers=4,
            continue_training=True,
            weights_path="/tmp/pretrained.pt",
            experiment_postfix="portable",
            val="best",
            compile_enabled=False,
            grad_cam_enabled=True,
        )

    def test_mm_preview_da_prints_output_path(self) -> None:
        with (
            patch("sys.argv", ["mm_preview_da", "-d", "1", "-f", "0"]),
            patch(
                "meisenmeister.cli.preview_da",
                return_value=Path("/tmp/da_preview.png"),
            ),
            patch("sys.stdout", new_callable=io.StringIO) as mock_stdout,
        ):
            cli.mm_preview_da()

        self.assertEqual(mock_stdout.getvalue().strip(), "/tmp/da_preview.png")

    def test_mm_predict_forwards_all_special_options(self) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "mm_predict",
                    "-d",
                    "1",
                    "-i",
                    "/tmp/in",
                    "-o",
                    "/tmp/out",
                    "-f",
                    "all",
                    "--trainer",
                    "mmTrainer_Debug",
                    "--postfix",
                    "portable",
                    "--checkpoint",
                    "last",
                    "--no-tta",
                    "--no-compile",
                    "--num-workers",
                    "2",
                    "--concise-output",
                    "/tmp/concise.json",
                ],
            ),
            patch("meisenmeister.cli.predict") as mock_predict,
        ):
            cli.mm_predict()

        mock_predict.assert_called_once_with(
            1,
            input_dir="/tmp/in",
            output_dir="/tmp/out",
            folds=["all"],
            trainer_name="mmTrainer_Debug",
            experiment_postfix="portable",
            checkpoint="last",
            use_tta=False,
            compile_model=False,
            num_workers=2,
            concise_output_path="/tmp/concise.json",
        )

    def test_mm_predict_from_modelfolder_forwards_options(self) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "mm_predict_from_modelfolder",
                    "/tmp/model",
                    "-i",
                    "/tmp/in",
                    "-o",
                    "/tmp/out",
                    "-f",
                    "0",
                    "2",
                    "--checkpoint",
                    "last",
                    "--no-tta",
                    "--no-compile",
                    "--num-workers",
                    "3",
                    "--concise-output",
                    "/tmp/concise.json",
                ],
            ),
            patch("meisenmeister.cli.predict_from_modelfolder") as mock_predict,
        ):
            cli.mm_predict_from_modelfolder()

        mock_predict.assert_called_once_with(
            "/tmp/model",
            input_dir="/tmp/in",
            output_dir="/tmp/out",
            folds=[0, 2],
            checkpoint="last",
            use_tta=False,
            compile_model=False,
            num_workers=3,
            concise_output_path="/tmp/concise.json",
        )

    def test_mm_evaluate_predictions_forwards_paths(self) -> None:
        with (
            patch(
                "sys.argv",
                [
                    "mm_evaluate_predictions",
                    "-t",
                    "/tmp/labels.json",
                    "-p",
                    "/tmp/predictions.json",
                    "-o",
                    "/tmp/reports",
                ],
            ),
            patch("meisenmeister.cli.evaluate_predictions") as mock_evaluate,
        ):
            cli.mm_evaluate_predictions()

        mock_evaluate.assert_called_once_with(
            targets_path="/tmp/labels.json",
            predictions_path="/tmp/predictions.json",
            output_path="/tmp/reports",
        )

    def test_mm_create_5fold_resolves_dataset_and_prints_output_path(self) -> None:
        with (
            patch("sys.argv", ["mm_create_5fold", "-d", "1"]),
            patch(
                "meisenmeister.cli.verify_required_global_paths_set",
                return_value={
                    "mm_raw": Path("/tmp/raw"),
                    "mm_preprocessed": Path("/tmp/pre"),
                    "mm_results": Path("/tmp/results"),
                },
            ),
            patch(
                "meisenmeister.cli.find_dataset_dir",
                return_value=Path("/tmp/raw/Dataset_001_Test"),
            ) as mock_find_dataset_dir,
            patch(
                "meisenmeister.cli.create_five_fold_splits",
                return_value=Path("/tmp/pre/Dataset_001_Test/splits.json"),
            ) as mock_create_splits,
            patch("sys.stdout", new_callable=io.StringIO) as mock_stdout,
        ):
            cli.mm_create_5fold.__wrapped__()

        mock_find_dataset_dir.assert_called_once_with(Path("/tmp/raw"), "001")
        mock_create_splits.assert_called_once_with(Path("/tmp/pre/Dataset_001_Test"))
        self.assertEqual(
            mock_stdout.getvalue().strip(),
            "/tmp/pre/Dataset_001_Test/splits.json",
        )


if __name__ == "__main__":
    unittest.main()
