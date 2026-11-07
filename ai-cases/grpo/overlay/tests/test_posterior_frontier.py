import unittest

import numpy as np
import torch

from verl.trainer.ppo.posterior_frontier import (
    PosteriorFrontierConfig,
    PosteriorFrontierController,
)


class PosteriorFrontierControllerTest(unittest.TestCase):
    def test_first_observation_is_cross_fitted_out(self):
        controller = PosteriorFrontierController(PosteriorFrontierConfig())
        residual, metrics = controller.observe(
            np.array([7, 7, 7, 7]), torch.ones(4, dtype=torch.bool)
        )
        self.assertTrue(torch.equal(residual, torch.zeros(4)))
        self.assertEqual(metrics["posterior_frontier/active_group_fraction"], 0.0)

    def test_repeated_all_correct_group_gets_positive_signal(self):
        controller = PosteriorFrontierController(PosteriorFrontierConfig())
        ids = np.array([7, 7, 7, 7])
        controller.observe(ids, torch.tensor([True, False, False, False]))
        residual, metrics = controller.observe(ids, torch.ones(4, dtype=torch.bool))
        self.assertTrue(torch.all(residual > 0))
        self.assertEqual(metrics["posterior_frontier/all_correct_group_fraction"], 1.0)

    def test_repeated_all_wrong_group_gets_negative_signal(self):
        controller = PosteriorFrontierController(PosteriorFrontierConfig())
        ids = np.array([9, 9, 9, 9])
        controller.observe(ids, torch.tensor([True, True, True, False]))
        residual, metrics = controller.observe(ids, torch.zeros(4, dtype=torch.bool))
        self.assertTrue(torch.all(residual < 0))
        self.assertEqual(metrics["posterior_frontier/all_wrong_group_fraction"], 1.0)

    def test_mixed_groups_keep_ordinary_advantage_untouched(self):
        controller = PosteriorFrontierController(PosteriorFrontierConfig())
        ids = np.array([3, 3, 3, 3])
        controller.observe(ids, torch.ones(4, dtype=torch.bool))
        residual, metrics = controller.observe(
            ids, torch.tensor([True, False, True, False])
        )
        self.assertTrue(torch.equal(residual, torch.zeros(4)))
        self.assertEqual(metrics["posterior_frontier/active_group_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
