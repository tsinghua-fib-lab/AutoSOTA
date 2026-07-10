import unittest

import numpy as np

from cenreg.distribution.quantile import QuantileDist


class TestQuantileDist(unittest.TestCase):
    def test_linear1(self):
        dist = QuantileDist(
            q=np.array([0.0, 0.1, 0.3, 0.7, 1.0]),
            v=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            interpolate="linear",
        )

        ret = dist.cdf(np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]))
        self.assertEqual(ret.shape, (11,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 0.0)
        self.assertAlmostEqual(ret[2].item(), 0.05)
        self.assertAlmostEqual(ret[3].item(), 0.1)
        self.assertAlmostEqual(ret[4].item(), 0.2)
        self.assertAlmostEqual(ret[5].item(), 0.3)
        self.assertAlmostEqual(ret[6].item(), 0.5)
        self.assertAlmostEqual(ret[7].item(), 0.7)
        self.assertAlmostEqual(ret[8].item(), 0.85)
        self.assertAlmostEqual(ret[9].item(), 1.0)
        self.assertAlmostEqual(ret[10].item(), 1.0)

        ret = dist.icdf(np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0]))
        self.assertEqual(ret.shape, (7,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 1.0)
        self.assertAlmostEqual(ret[2].item(), 2.0)
        self.assertAlmostEqual(ret[3].item(), 2.5)
        self.assertAlmostEqual(ret[4].item(), 3.0)
        self.assertAlmostEqual(ret[5].item(), 3.5)
        self.assertAlmostEqual(ret[6].item(), 4.0)

    def test_linear2(self):
        dist = QuantileDist(
            q=np.array([0.0, 0.3, 1.0]),
            v=np.array([1.0, 2.0, 3.0]),
            interpolate="linear",
        )

        ret = dist.cdf(np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]))
        self.assertEqual(ret.shape, (9,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 0.0)
        self.assertAlmostEqual(ret[2].item(), 0.0)
        self.assertAlmostEqual(ret[3].item(), 0.0)
        self.assertAlmostEqual(ret[4].item(), 0.15)
        self.assertAlmostEqual(ret[5].item(), 0.3)
        self.assertAlmostEqual(ret[6].item(), 0.65)
        self.assertAlmostEqual(ret[7].item(), 1.0)
        self.assertAlmostEqual(ret[8].item(), 1.0)

        ret = dist.icdf(np.array([0.0, 0.15, 0.3, 0.65, 1.0]))
        self.assertEqual(ret.shape, (5,))
        self.assertAlmostEqual(ret[0].item(), 1.0)
        self.assertAlmostEqual(ret[1].item(), 1.5)
        self.assertAlmostEqual(ret[2].item(), 2.0)
        self.assertAlmostEqual(ret[3].item(), 2.5)
        self.assertAlmostEqual(ret[4].item(), 3.0)


if __name__ == "__main__":
    unittest.main()
