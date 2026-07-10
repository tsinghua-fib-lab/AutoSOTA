import unittest

import numpy as np

from cenreg.distribution.cdf import CumulativeDist
from cenreg.metric.cdf import km_calibration


class TestKMCalibration(unittest.TestCase):
    def test1(self):
        pred = np.array([[0.1, 0.2, 0.6, 0.8, 1.0]])
        dist = CumulativeDist(np.array([0, 1, 2, 3, 4, 5], dtype=float), cum_p=pred)

        observed_times = np.array([1, 2, 2, 3, 4])
        uncensored = np.array([True, True, False, True, True])

        kmcal = km_calibration(dist, observed_times, uncensored)
        kmcal_expected = (
            0.2 * np.log(0.2 / 0.1) + 0.2 * np.log(0.2 / 0.1) + 0.3 * np.log(0.3 / 0.4) + 0.3 * np.log(0.3 / 0.2)
        )
        self.assertAlmostEqual(kmcal, kmcal_expected, places=3)


if __name__ == "__main__":
    unittest.main()
