import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from odelia_autosota import heads
from odelia_autosota.cli import main
from odelia_autosota.data import _softmax, load_training_archive
from odelia_autosota.metrics import compute_metrics
from tests.fixtures import training_archive_fixture

torch.set_num_threads(1)


class WorkflowTests(unittest.TestCase):
    def cli(self, *args):
        with (
            patch("sys.argv", ["odelia-autosota", *map(str, args)]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(), 0)

    def test_cache_pack_preserves_features_logits_and_case_ids(self):
        from odelia_autosota.data import FEATURE_KEYS

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original = training_archive_fixture()
            for fold in range(7):
                folder = root / "cache" / f"fold_{fold}"
                (folder / "samples").mkdir(parents=True)
                records = []
                for prefix in ("train", "val"):
                    for i, sample_id in enumerate(original[f"{prefix}_sample_ids"]):
                        vector = original[f"{prefix}_features"][i, fold]
                        blocks = {}
                        offset = 0
                        for key, width in zip(FEATURE_KEYS, [256] * 3 + [320] * 6):
                            blocks[key] = vector[offset : offset + width]
                            offset += width
                        np.savez(
                            folder / "samples" / f"{sample_id}.npz",
                            released_logits=original[f"{prefix}_released_logits"][
                                i, fold
                            ],
                            **blocks,
                        )
                        records.append(
                            {
                                "sample_id": str(sample_id),
                                "case_id": str(original[f"{prefix}_case_ids"][i]),
                                "label": int(original[f"{prefix}_labels"][i]),
                                "partition": prefix,
                            }
                        )
                (folder / "manifest.json").write_text(
                    json.dumps({"fold": fold, "records": records})
                )
            self.cli(
                "pack", "--cache-root", root / "cache", "--output", root / "packed.npz"
            )
            packed = load_training_archive(root / "packed.npz")
            for key, value in original.items():
                np.testing.assert_array_equal(value, packed[key])

    def test_train_save_load_baseline_and_aligned_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = training_archive_fixture()
            np.savez(root / "train.npz", **a)
            self.cli(
                "train",
                "--packed-cache",
                root / "train.npz",
                "--output-dir",
                root / "run",
                "--device",
                "cpu",
                "--epochs",
                1,
                "--batch-size",
                12,
            )
            ids = a["val_sample_ids"]
            z = a["val_released_logits"]
            x = a["val_features"]
            np.savez(root / "input.npz", sample_ids=ids, features=x, released_logits=z)
            self.cli(
                "baseline",
                "--inputs",
                root / "input.npz",
                "--output-dir",
                root / "base",
            )
            with np.load(root / "base/predictions.npz") as data:
                np.testing.assert_allclose(
                    data["ensemble_probabilities"],
                    _softmax(z).mean(axis=1),
                    rtol=1e-12,
                    atol=1e-12,
                )
            for name in ("out1", "out2"):
                self.cli(
                    "infer",
                    "--inputs",
                    root / "input.npz",
                    "--run-dir",
                    root / "run",
                    "--output-dir",
                    root / name,
                    "--device",
                    "cpu",
                )
            with (
                np.load(root / "out1/predictions.npz") as a1,
                np.load(root / "out2/predictions.npz") as a2,
            ):
                np.testing.assert_array_equal(
                    a1["ensemble_probabilities"], a2["ensemble_probabilities"]
                )
                expected = compute_metrics(
                    a["val_labels"], a1["ensemble_probabilities"]
                )
            np.savez(
                root / "labels.npz", sample_ids=ids[::-1], labels=a["val_labels"][::-1]
            )
            self.cli(
                "evaluate",
                "--prediction-archive",
                root / "out1/predictions.npz",
                "--labels",
                root / "labels.npz",
                "--output",
                root / "eval.json",
            )
            got = json.loads((root / "eval.json").read_text())["metrics"]
            self.assertEqual(
                got, {k.removeprefix("val_"): v for k, v in expected.items()}
            )
            for fold in range(7):
                checkpoint = torch.load(
                    root / f"run/fold_{fold}/head_best.pth", weights_only=True
                )
                self.assertEqual(checkpoint["method"], heads.METHOD_NAME)

    def test_historical_mode_rejects_wrong_counts(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            np.savez(p / "t.npz", **training_archive_fixture())
            with self.assertRaises(ValueError):
                with patch(
                    "sys.argv",
                    [
                        "odelia-autosota",
                        "train",
                        "--packed-cache",
                        str(p / "t.npz"),
                        "--output-dir",
                        str(p / "out"),
                        "--strict-historical",
                    ],
                ):
                    main()

    def test_legacy_probability_alignment_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            probs = np.asarray([0.2, 0.3, 0.5])
            rows = {}
            for name in ("c1", "c2"):
                rows[name] = {
                    "rois": {
                        "left": {
                            "per_fold": {
                                str(f): {"probabilities": probs.tolist()}
                                for f in range(7)
                            }
                        }
                    }
                }
            (root / "pred.json").write_text(json.dumps({"cases": rows}))
            np.savez(
                root / "features.npz",
                sample_ids=np.asarray(["c2_left", "c1_left"]),
                block_features=np.zeros((2, 7, 2688), dtype=np.float32),
            )
            self.cli(
                "baseline",
                "--predictions",
                root / "pred.json",
                "--block-features",
                root / "features.npz",
                "--output-dir",
                root / "base",
            )
            with np.load(root / "base/predictions.npz") as data:
                self.assertEqual(data["sample_ids"].tolist(), ["c1_left", "c2_left"])
                np.testing.assert_allclose(
                    data["ensemble_probabilities"], np.tile(probs, (2, 1))
                )
