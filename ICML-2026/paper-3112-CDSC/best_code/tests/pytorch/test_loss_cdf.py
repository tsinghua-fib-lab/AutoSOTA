import unittest

import numpy as np
import torch

from cenreg.pytorch.loss_cdf import NegativeLogLikelihoodInterval


class TestNegativeLogLikelihoodInterval(unittest.TestCase):
    def test_proportional(self):
        bins = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        loss_fn = NegativeLogLikelihoodInterval(y_bins=bins, proportional=True)

        lb = torch.tensor([0.0, 0.5, 3.5])
        ub = torch.tensor([1.0, 1.5, 4.0])
        pred = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        loss = loss_fn.loss(pred, lb, ub)
        self.assertEqual(loss.shape, (3,))
        self.assertAlmostEqual(loss[0].item(), -np.log(0.1), places=2)
        self.assertAlmostEqual(loss[1].item(), -np.log(0.15), places=2)
        self.assertAlmostEqual(loss[2].item(), -np.log(0.4), places=2)

    def test_nonproportional(self):
        bins = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        loss_fn = NegativeLogLikelihoodInterval(y_bins=bins, proportional=False)

        lb = torch.tensor([0.0, 0.5, 3.5])
        ub = torch.tensor([1.0, 1.5, 4.0])
        pred = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        loss = loss_fn.loss(pred, lb, ub)
        self.assertEqual(loss.shape, (3,))
        self.assertAlmostEqual(loss[0].item(), -np.log(0.1), places=2)
        self.assertAlmostEqual(loss[1].item(), -np.log(0.3), places=2)
        self.assertAlmostEqual(loss[2].item(), -np.log(0.4), places=2)


if __name__ == "__main__":
    unittest.main()
