import tempfile
import unittest
from pathlib import Path
import numpy as np
from odelia_autosota.data import load_training_archive, load_inference_archive
from tests.fixtures import training_archive_fixture


class DataTests(unittest.TestCase):
    def check_bad(self, transform):
        with tempfile.TemporaryDirectory() as d:
            arrays = training_archive_fixture()
            transform(arrays)
            p = Path(d) / "bad.npz"
            np.savez(p, **arrays)
            with self.assertRaises(ValueError):
                load_training_archive(p)

    def test_bilateral_case_leakage_rejected(self):
        self.check_bad(
            lambda a: a["val_case_ids"].__setitem__(0, a["train_case_ids"][0])
        )

    def test_duplicate_roi_rejected(self):
        self.check_bad(
            lambda a: a["val_sample_ids"].__setitem__(0, a["val_sample_ids"][1])
        )

    def test_nonfinite_features_rejected(self):
        self.check_bad(lambda a: a["train_features"].__setitem__((0, 0, 0), np.nan))

    def test_inference_labels_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a = training_archive_fixture()
            p = Path(d) / "in.npz"
            np.savez(
                p,
                sample_ids=a["val_sample_ids"],
                features=a["val_features"],
                released_logits=a["val_released_logits"],
                labels=a["val_labels"],
            )
            with self.assertRaises(ValueError):
                load_inference_archive(p)
