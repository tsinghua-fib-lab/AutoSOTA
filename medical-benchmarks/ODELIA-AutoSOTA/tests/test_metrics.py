import unittest
import numpy as np
from odelia_autosota.metrics import compute_metrics


class MetricsTests(unittest.TestCase):
    def test_perfect_known_predictions(self):
        labels = np.asarray([0, 1, 2] * 3)
        probabilities = np.eye(3)[labels]
        metrics = compute_metrics(labels, probabilities)
        for key, value in metrics.items():
            self.assertAlmostEqual(value, 0 if key == "val_nll" else 1)

    def test_invalid_distribution_rejected(self):
        with self.assertRaises(ValueError):
            compute_metrics(np.asarray([0, 1, 2]), np.ones((3, 3)))
