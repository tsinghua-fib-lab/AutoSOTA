import unittest

import numpy as np
import torch

from verl.trainer.ppo.strategy_archive import (
    StrategyArchiveConfig,
    StrategyArchiveController,
    expression_signature,
)


class StrategyArchiveTest(unittest.TestCase):
    def test_commutative_equivalents_share_signature(self):
        self.assertEqual(
            expression_signature("<answer>1 + 2</answer>"),
            expression_signature("<answer>2 + 1</answer>"),
        )

    def test_different_trees_have_different_signatures(self):
        self.assertNotEqual(
            expression_signature("<answer>(1 + 2) * 3</answer>"),
            expression_signature("<answer>1 + (2 * 3)</answer>"),
        )

    def test_residual_is_zero_mean_and_only_on_correct(self):
        controller = StrategyArchiveController(StrategyArchiveConfig())
        residual, metrics = controller.observe(
            np.array([7, 7, 7, 7]),
            torch.tensor([True, True, True, False]),
            [
                "<answer>1 + 2</answer>",
                "<answer>2 + 1</answer>",
                "<answer>3 * 1</answer>",
                "garbage",
            ],
        )
        self.assertAlmostEqual(float(residual[:3].sum()), 0.0, places=5)
        self.assertEqual(float(residual[3]), 0.0)
        self.assertEqual(metrics["strategy_archive/active_group_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
