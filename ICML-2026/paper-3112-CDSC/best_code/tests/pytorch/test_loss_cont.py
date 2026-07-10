import unittest

import numpy as np
import torch

from cenreg.pytorch.loss_cont import NegativeLogLikelihoodInterval


class TestNegativeLogLikelihoodInterval(unittest.TestCase):
    def test1(self):
        loss_fn = NegativeLogLikelihoodInterval()

        F_lb = torch.tensor([0.0, 0.5, 0.8])
        F_ub = torch.tensor([0.3, 1.0, 1.0])
        loss = loss_fn.loss(F_lb, F_ub)
        self.assertEqual(loss.shape, (3,))
        self.assertAlmostEqual(loss[0].item(), -np.log(0.3), places=3)
        self.assertAlmostEqual(loss[1].item(), -np.log(0.5), places=3)
        self.assertAlmostEqual(loss[2].item(), -np.log(0.2), places=3)


if __name__ == "__main__":
    unittest.main()
