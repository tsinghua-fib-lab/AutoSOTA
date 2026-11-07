import unittest

import numpy as np
import torch

from verl.trainer.ppo.frontier_scaling import (
    FrontierScalingConfig,
    FrontierScalingController,
)


class FrontierScalingControllerTest(unittest.TestCase):
    def test_first_observation_is_identity(self):
        controller = FrontierScalingController(FrontierScalingConfig())
        weights, metrics = controller.observe(
            np.array([1, 1, 1, 1]), torch.tensor([True, False, True, False])
        )
        self.assertTrue(torch.equal(weights, torch.ones(4)))
        self.assertEqual(metrics["frontier_scaling/active_group_fraction"], 0.0)

    def test_uncertain_problem_is_upweighted(self):
        controller = FrontierScalingController(FrontierScalingConfig())
        ids = np.array([2, 2, 2, 2])
        controller.observe(ids, torch.tensor([True, False, True, False]))
        weights, _ = controller.observe(ids, torch.tensor([True, False, True, False]))
        self.assertTrue(torch.all(weights > 1.0))

    def test_mastered_problem_is_downweighted(self):
        controller = FrontierScalingController(FrontierScalingConfig())
        ids = np.array([3, 3, 3, 3])
        for _ in range(8):
            controller.observe(ids, torch.ones(4, dtype=torch.bool))
        weights, metrics = controller.observe(ids, torch.ones(4, dtype=torch.bool))
        self.assertTrue(torch.all(weights < 1.0))
        self.assertGreater(metrics["frontier_scaling/downweighted_group_fraction"], 0.0)

    def test_scaling_preserves_zero_and_sign(self):
        base = torch.tensor([[1.0], [-1.0], [0.0], [0.0]])
        weights = torch.tensor([1.5, 1.5, 0.25, 0.25]).unsqueeze(-1)
        scaled = base * weights
        self.assertGreater(scaled[0].item(), 0.0)
        self.assertLess(scaled[1].item(), 0.0)
        self.assertEqual(scaled[2].item(), 0.0)
        self.assertEqual(scaled[3].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
