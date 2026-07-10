import unittest

import numpy as np
import torch

from cenreg.pytorch.distribution import CumulativeDist


class TestCumulativeDist(unittest.TestCase):
    def test_left1(self):
        dist = CumulativeDist(
            b=torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            cum_p=torch.tensor([0.1, 0.3, 0.7, 0.9]),
            interpolate="left",
        )

        ret = dist.cdf(torch.tensor([-np.inf, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, np.inf]))
        self.assertEqual(ret.shape, (13,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 0.0)
        self.assertAlmostEqual(ret[2].item(), 0.0)
        self.assertAlmostEqual(ret[3].item(), 0.1)
        self.assertAlmostEqual(ret[4].item(), 0.1)
        self.assertAlmostEqual(ret[5].item(), 0.3)
        self.assertAlmostEqual(ret[6].item(), 0.3)
        self.assertAlmostEqual(ret[7].item(), 0.7)
        self.assertAlmostEqual(ret[8].item(), 0.7)
        self.assertAlmostEqual(ret[9].item(), 0.9)
        self.assertAlmostEqual(ret[10].item(), 0.9)
        self.assertAlmostEqual(ret[11].item(), 1.0)
        self.assertAlmostEqual(ret[12].item(), 1.0)

        ret = dist.icdf(torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]))
        self.assertEqual(ret.shape, (7,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 0.0)
        self.assertAlmostEqual(ret[2].item(), 1.0)
        self.assertAlmostEqual(ret[3].item(), 2.0)
        self.assertAlmostEqual(ret[4].item(), 2.0)
        self.assertAlmostEqual(ret[5].item(), 3.0)
        self.assertAlmostEqual(ret[6].item(), 4.0)

    def test_left2(self):
        dist = CumulativeDist(
            b=torch.tensor([0.0, 1.0, 2.0, 3.0]),
            cum_p=torch.tensor([0.0, 0.3, 1.0]),
            interpolate="left",
        )

        ret = dist.cdf(torch.tensor([-np.inf, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, np.inf]))
        self.assertEqual(ret.shape, (11,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 0.0)
        self.assertAlmostEqual(ret[2].item(), 0.0)
        self.assertAlmostEqual(ret[3].item(), 0.0)
        self.assertAlmostEqual(ret[4].item(), 0.0)
        self.assertAlmostEqual(ret[5].item(), 0.3)
        self.assertAlmostEqual(ret[6].item(), 0.3)
        self.assertAlmostEqual(ret[7].item(), 1.0)
        self.assertAlmostEqual(ret[8].item(), 1.0)
        self.assertAlmostEqual(ret[9].item(), 1.0)
        self.assertAlmostEqual(ret[10].item(), 1.0)

        ret = dist.icdf(torch.tensor([0.0, 0.1, 0.3, 0.5, 1.0]))
        self.assertEqual(ret.shape, (5,))
        self.assertAlmostEqual(ret[0].item(), 1.0)
        self.assertAlmostEqual(ret[1].item(), 1.0)
        self.assertAlmostEqual(ret[2].item(), 1.0)
        self.assertAlmostEqual(ret[3].item(), 2.0)
        self.assertAlmostEqual(ret[4].item(), 2.0)

    def test_left3(self):
        omega = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
        cum_p = torch.tensor([0.0, 0.15, 0.55, 0.85, 1.0])
        dist = CumulativeDist(b=omega, cum_p=cum_p, interpolate="left")

        ret = dist.cdf(torch.tensor([-np.inf, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, np.inf]))
        self.assertEqual(ret.shape, (8,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 0.0)
        self.assertAlmostEqual(ret[2].item(), 0.0)
        self.assertAlmostEqual(ret[3].item(), 0.15)
        self.assertAlmostEqual(ret[4].item(), 0.55)
        self.assertAlmostEqual(ret[5].item(), 0.85)
        self.assertAlmostEqual(ret[6].item(), 1.0)
        self.assertAlmostEqual(ret[7].item(), 1.0)

    def test_right1(self):
        dist = CumulativeDist(
            b=torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            cum_p=torch.tensor([0.1, 0.3, 0.7, 0.9]),
            interpolate="right",
        )

        ret = dist.cdf(torch.tensor([-np.inf, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, np.inf]))
        self.assertEqual(ret.shape, (13,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 0.0)
        self.assertAlmostEqual(ret[2].item(), 0.1)
        self.assertAlmostEqual(ret[3].item(), 0.1)
        self.assertAlmostEqual(ret[4].item(), 0.3)
        self.assertAlmostEqual(ret[5].item(), 0.3)
        self.assertAlmostEqual(ret[6].item(), 0.7)
        self.assertAlmostEqual(ret[7].item(), 0.7)
        self.assertAlmostEqual(ret[8].item(), 0.9)
        self.assertAlmostEqual(ret[9].item(), 0.9)
        self.assertAlmostEqual(ret[10].item(), 1.0)
        self.assertAlmostEqual(ret[11].item(), 1.0)
        self.assertAlmostEqual(ret[12].item(), 1.0)

        ret = dist.icdf(torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]))
        self.assertEqual(ret.shape, (7,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 0.0)
        self.assertAlmostEqual(ret[2].item(), 1.0)
        self.assertAlmostEqual(ret[3].item(), 2.0)
        self.assertAlmostEqual(ret[4].item(), 2.0)
        self.assertAlmostEqual(ret[5].item(), 3.0)
        self.assertAlmostEqual(ret[6].item(), 4.0)

    def test_right2(self):
        dist = CumulativeDist(
            b=torch.tensor([0.0, 1.0, 2.0, 3.0]),
            cum_p=torch.tensor([0.0, 0.3, 1.0]),
            interpolate="right",
        )

        ret = dist.cdf(torch.tensor([-np.inf, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, np.inf]))
        self.assertEqual(ret.shape, (11,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 0.0)
        self.assertAlmostEqual(ret[2].item(), 0.0)
        self.assertAlmostEqual(ret[3].item(), 0.0)
        self.assertAlmostEqual(ret[4].item(), 0.3)
        self.assertAlmostEqual(ret[5].item(), 0.3)
        self.assertAlmostEqual(ret[6].item(), 1.0)
        self.assertAlmostEqual(ret[7].item(), 1.0)
        self.assertAlmostEqual(ret[8].item(), 1.0)
        self.assertAlmostEqual(ret[9].item(), 1.0)
        self.assertAlmostEqual(ret[10].item(), 1.0)

        ret = dist.icdf(torch.tensor([0.0, 0.1, 0.3, 0.5, 1.0]))
        self.assertEqual(ret.shape, (5,))
        self.assertAlmostEqual(ret[0].item(), 1.0)
        self.assertAlmostEqual(ret[1].item(), 1.0)
        self.assertAlmostEqual(ret[2].item(), 1.0)
        self.assertAlmostEqual(ret[3].item(), 2.0)
        self.assertAlmostEqual(ret[4].item(), 2.0)

    def test_right3(self):
        dist = CumulativeDist(
            b=torch.tensor([0.0, 1.0, 2.0, 3.0]),
            cum_p=torch.tensor([0.0, 0.3, 1.0]),
            interpolate="right",
        )

        ret = dist.cdf(torch.tensor([[-0.5, 0.0, 0.5, 1.0, 1.5], [1.5, 2.0, 2.5, 3.0, 3.5]]))
        self.assertEqual(ret.shape, (2, 5))
        self.assertAlmostEqual(ret[0, 0].item(), 0.0)
        self.assertAlmostEqual(ret[0, 1].item(), 0.0)
        self.assertAlmostEqual(ret[0, 2].item(), 0.0)
        self.assertAlmostEqual(ret[0, 3].item(), 0.3)
        self.assertAlmostEqual(ret[0, 4].item(), 0.3)
        self.assertAlmostEqual(ret[1, 0].item(), 0.3)
        self.assertAlmostEqual(ret[1, 1].item(), 1.0)
        self.assertAlmostEqual(ret[1, 2].item(), 1.0)
        self.assertAlmostEqual(ret[1, 3].item(), 1.0)
        self.assertAlmostEqual(ret[1, 4].item(), 1.0)

    def test_linear1(self):
        dist = CumulativeDist(
            b=torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]),
            cum_p=torch.tensor([0.1, 0.3, 0.7, 0.9]),
            interpolate="linear",
        )

        ret = dist.cdf(torch.tensor([-np.inf, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, np.inf]))
        self.assertEqual(ret.shape, (13,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 0.0)
        self.assertAlmostEqual(ret[2].item(), 0.0)
        self.assertAlmostEqual(ret[3].item(), 0.05)
        self.assertAlmostEqual(ret[4].item(), 0.1)
        self.assertAlmostEqual(ret[5].item(), 0.2)
        self.assertAlmostEqual(ret[6].item(), 0.3)
        self.assertAlmostEqual(ret[7].item(), 0.5)
        self.assertAlmostEqual(ret[8].item(), 0.7)
        self.assertAlmostEqual(ret[9].item(), 0.8)
        self.assertAlmostEqual(ret[10].item(), 0.9)
        self.assertAlmostEqual(ret[11].item(), 1.0)
        self.assertAlmostEqual(ret[12].item(), 1.0)

        ret = dist.icdf(torch.tensor([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]))
        self.assertEqual(ret.shape, (7,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 1.0)
        self.assertAlmostEqual(ret[2].item(), 2.0)
        self.assertAlmostEqual(ret[3].item(), 2.5)
        self.assertAlmostEqual(ret[4].item(), 3.0)
        self.assertAlmostEqual(ret[5].item(), 4.0)
        self.assertAlmostEqual(ret[6].item(), 4.0)

    def test_linear2(self):
        dist = CumulativeDist(
            b=torch.tensor([0.0, 1.0, 2.0, 3.0]),
            cum_p=torch.tensor([0.0, 0.3, 1.0]),
            interpolate="linear",
        )

        ret = dist.cdf(torch.tensor([-np.inf, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, np.inf]))
        self.assertEqual(ret.shape, (11,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 0.0)
        self.assertAlmostEqual(ret[2].item(), 0.0)
        self.assertAlmostEqual(ret[3].item(), 0.0)
        self.assertAlmostEqual(ret[4].item(), 0.0)
        self.assertAlmostEqual(ret[5].item(), 0.15)
        self.assertAlmostEqual(ret[6].item(), 0.3)
        self.assertAlmostEqual(ret[7].item(), 0.65)
        self.assertAlmostEqual(ret[8].item(), 1.0)
        self.assertAlmostEqual(ret[9].item(), 1.0)
        self.assertAlmostEqual(ret[10].item(), 1.0)

        ret = dist.icdf(torch.tensor([0.0, 0.15, 0.3, 0.65, 1.0]))
        self.assertEqual(ret.shape, (5,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 1.5)
        self.assertAlmostEqual(ret[2].item(), 2.0)
        self.assertAlmostEqual(ret[3].item(), 2.5)
        self.assertAlmostEqual(ret[4].item(), 3.0)

    def test_linear3(self):
        dist = CumulativeDist(
            b=torch.tensor([0.0, 1.0, 2.0, 3.0]),
            cum_p=torch.tensor([[0.0, 0.3, 1.0], [0.0, 0.4, 1.0]]),
            interpolate="linear",
        )

        ret = dist.cdf(torch.tensor([-np.inf, 2.0, 2.5, 3.0, np.inf]))
        self.assertEqual(ret.shape, (2, 5))
        self.assertAlmostEqual(ret[0, 0].item(), 0.0)
        self.assertAlmostEqual(ret[0, 1].item(), 0.3)
        self.assertAlmostEqual(ret[0, 2].item(), 0.65)
        self.assertAlmostEqual(ret[0, 3].item(), 1.0)
        self.assertAlmostEqual(ret[0, 4].item(), 1.0)
        self.assertAlmostEqual(ret[1, 0].item(), 0.0)
        self.assertAlmostEqual(ret[1, 1].item(), 0.4)
        self.assertAlmostEqual(ret[1, 2].item(), 0.7)
        self.assertAlmostEqual(ret[1, 3].item(), 1.0)
        self.assertAlmostEqual(ret[1, 4].item(), 1.0)

        ret = dist.icdf(torch.tensor([0.3, 0.4]))
        self.assertEqual(ret.shape, (2, 2))
        self.assertAlmostEqual(ret[0, 0].item(), 2.0)
        self.assertAlmostEqual(ret[0, 1].item(), 2.0 + 1.0 / 7.0, places=5)
        self.assertAlmostEqual(ret[1, 0].item(), 1.75)
        self.assertAlmostEqual(ret[1, 1].item(), 2.0)

    def test_linear4(self):
        dist = CumulativeDist(
            b=torch.tensor([0.0, 1.0, 2.0, 3.0]),
            cum_p=torch.tensor([[0.0, 0.3, 1.0], [0.0, 0.3, 1.0], [0.0, 0.4, 1.0], [0.0, 0.4, 1.0]]),
            interpolate="linear",
        )

        ret = dist.cdf(torch.tensor([[-np.inf], [2.0], [2.5], [np.inf]]))
        self.assertEqual(ret.shape, (4, 1))
        self.assertAlmostEqual(ret[0, 0].item(), 0.0)
        self.assertAlmostEqual(ret[1, 0].item(), 0.3)
        self.assertAlmostEqual(ret[2, 0].item(), 0.7)
        self.assertAlmostEqual(ret[3, 0].item(), 1.0)

        ret = dist.icdf(torch.tensor([[0.3], [0.3], [0.4], [0.4]]))
        self.assertEqual(ret.shape, (4, 1))
        self.assertAlmostEqual(ret[0, 0].item(), 2.0)
        self.assertAlmostEqual(ret[1, 0].item(), 2.0)
        self.assertAlmostEqual(ret[2, 0].item(), 2.0)
        self.assertAlmostEqual(ret[3, 0].item(), 2.0)

    def test_linear5(self):
        dist = CumulativeDist(
            b=torch.tensor([0.0, 1.0, 2.0, 3.0]),
            cum_p=torch.tensor([[0.2, 0.5, 1.0], [0.2, 0.5, 1.0]]),
            interpolate="linear",
        )

        ret = dist.cdf(torch.tensor([[2.0], [2.5]]))
        self.assertEqual(ret.shape, (2, 1))
        self.assertAlmostEqual(ret[0, 0].item(), 0.5)
        self.assertAlmostEqual(ret[1, 0].item(), 0.75)

        ret = dist.icdf(torch.tensor([[0.2], [0.75]]))
        self.assertEqual(ret.shape, (2, 1))
        self.assertAlmostEqual(ret[0, 0].item(), 1.0)
        self.assertAlmostEqual(ret[1, 0].item(), 2.5)


if __name__ == "__main__":
    unittest.main()
