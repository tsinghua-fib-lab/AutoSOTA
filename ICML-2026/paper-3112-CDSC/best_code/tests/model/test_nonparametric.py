import unittest

import numpy as np

from cenreg.model.copula_np import IndependenceCopula
from cenreg.model.nonparametric import (
    empirical_cdf_estimator,
    kaplan_meier_estimator,
    li_watkins_yu_estimator,
    turnbull_estimator,
    zheng_klein_estimator,
)
from cenreg.utils import adjust_exact_observations


class TestComputeEmpiricalCDF(unittest.TestCase):
    def test1(self):
        a = np.array([1, 2, 2, 4, 3], dtype=float)
        dist = empirical_cdf_estimator(a, y_min=0, y_max=5)

        self.assertEqual(dist.b.shape, (6,))
        self.assertEqual(dist.cum_p.shape, (5,))
        self.assertTrue(np.allclose(dist.b, np.array([0, 1, 2, 3, 4, 5], dtype=float)))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 0.2, 0.6, 0.8, 1.0])))

        ret = dist.cdf(np.array([0, 1, 2, 2.5, 3, 4, 5]))
        self.assertEqual(ret.shape, (7,))
        self.assertAlmostEqual(ret[0].item(), 0.0)
        self.assertAlmostEqual(ret[1].item(), 0.2)
        self.assertAlmostEqual(ret[2].item(), 0.6)
        self.assertAlmostEqual(ret[3].item(), 0.6)
        self.assertAlmostEqual(ret[4].item(), 0.8)
        self.assertAlmostEqual(ret[5].item(), 1.0)
        self.assertAlmostEqual(ret[6].item(), 1.0)

        ret = dist.icdf(np.array([0.0, 0.1, 0.5, 0.7, 0.9, 1.0]))
        self.assertEqual(ret.shape, (6,))
        self.assertAlmostEqual(ret[0].item(), 1.0)
        self.assertAlmostEqual(ret[1].item(), 1.0)
        self.assertAlmostEqual(ret[2].item(), 2.0)
        self.assertAlmostEqual(ret[3].item(), 3.0)
        self.assertAlmostEqual(ret[4].item(), 4.0)
        self.assertAlmostEqual(ret[5].item(), 4.0)

    def test2(self):
        a = np.array([5, 5, 5, 5], dtype=float)
        dist = empirical_cdf_estimator(a)

        self.assertEqual(dist.b.shape, (3,))
        self.assertEqual(dist.cum_p.shape, (2,))
        self.assertTrue(np.allclose(dist.b, np.array([4.5, 5, 5.5], dtype=float)))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 1.0], dtype=float)))
        self.assertTrue(
            np.allclose(
                dist.cdf(np.array([4, 5, 6], dtype=float)),
                [0.0, 1.0, 1.0],
            )
        )
        self.assertTrue(
            np.allclose(
                dist.icdf(np.array([0.0, 0.5, 1.0], dtype=float)),
                [5, 5, 5],
            )
        )


class TestKaplanMeierEstimator(unittest.TestCase):
    def test1(self):
        observed_times = np.array([1, 2, 2, 4, 3], dtype=float)
        uncensored = np.array([True, True, False, True, True])
        dist = kaplan_meier_estimator(observed_times, uncensored)

        self.assertEqual(dist.confidence_interval, None)
        self.assertEqual(dist.b.shape, (5,))
        self.assertEqual(dist.cum_p.shape, (4,))
        self.assertTrue(np.allclose(dist.b, np.array([0, 1, 2, 3, 4], dtype=float)))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 0.2, 0.4, 0.7], dtype=float)))
        self.assertTrue(
            np.allclose(
                dist.cdf(np.array([0, 1, 2, 2.5, 3, 4, 5], dtype=float)),
                [0.0, 0.2, 0.4, 0.4, 0.7, 1.0, 1.0],
            )
        )
        ret = dist.icdf(np.array([0.0, 0.1, 0.5, 0.7, 0.9, 1.0], dtype=float))
        self.assertEqual(ret.shape, (6,))
        self.assertAlmostEqual(ret[0].item(), 1.0)
        self.assertAlmostEqual(ret[1].item(), 1.0)
        self.assertAlmostEqual(ret[2].item(), 3.0)
        self.assertAlmostEqual(ret[3].item(), 3.0)
        self.assertAlmostEqual(ret[4].item(), 4.0)
        self.assertAlmostEqual(ret[5].item(), 4.0)

    def test2(self):
        observed_times = np.array([5, 5, 5, 5], dtype=float)
        uncensored = np.array([False, False, True, False])
        dist = kaplan_meier_estimator(observed_times, uncensored, y_max=10)

        self.assertEqual(dist.confidence_interval, None)
        self.assertEqual(dist.b.shape, (3,))
        self.assertEqual(dist.cum_p.shape, (2,))
        self.assertTrue(np.allclose(dist.b, np.array([0, 5, 10])))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 0.25])))
        self.assertTrue(
            np.allclose(
                dist.cdf(np.array([4, 5, 6, 10, 11], dtype=float)),
                [0.0, 0.25, 0.25, 1.0, 1.0],
            )
        )
        self.assertTrue(
            np.allclose(
                dist.icdf(np.array([0.0, 0.5, 1.0], dtype=float)),
                [5, 10.0, 10.0],
            )
        )

    def test3(self):
        observed_times = np.array([5, 5, 5, 6], dtype=float)
        uncensored = np.array([True, True, True, False])
        dist = kaplan_meier_estimator(observed_times, uncensored)

        self.assertEqual(dist.confidence_interval, None)
        self.assertEqual(dist.b.shape, (3,))
        self.assertEqual(dist.cum_p.shape, (2,))
        self.assertTrue(np.allclose(dist.b, np.array([0, 5, 6])))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 0.75])))
        self.assertTrue(
            np.allclose(
                dist.cdf(np.array([4, 5, 6], dtype=float)),
                [0.0, 0.75, 1.0],
            )
        )
        self.assertTrue(
            np.allclose(
                dist.icdf(np.array([0.0, 0.5, 1.0], dtype=float)),
                [5, 5, 6],
            )
        )


class ZhengKleinEstimator(unittest.TestCase):
    def test1(self):
        observed_times = np.array([1, 2, 2, 4, 3], dtype=float)
        uncensored = np.array([True, True, False, True, True], dtype=bool)
        copula = IndependenceCopula()
        dist = zheng_klein_estimator(observed_times, uncensored, copula, y_max=5)

        self.assertEqual(dist.confidence_interval, None)
        self.assertEqual(dist.b.shape, (6,))
        self.assertEqual(dist.cum_p.shape, (5,))
        self.assertTrue(np.allclose(dist.b, np.array([0, 1, 2, 3, 4, 5], dtype=float)))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 0.2, 0.4, 0.7, 1.0], dtype=float), rtol=0.01))
        self.assertTrue(
            np.allclose(
                dist.cdf(np.array([0, 1, 2, 2.5, 3, 4, 5], dtype=float)),
                [0.0, 0.2, 0.4, 0.4, 0.7, 1.0, 1.0],
                rtol=0.01,
            )
        )
        ret = dist.icdf(np.array([0.0, 0.1, 0.5, 0.7, 0.9, 1.0], dtype=float))
        self.assertEqual(ret.shape, (6,))
        self.assertAlmostEqual(ret[0].item(), 1.0)
        self.assertAlmostEqual(ret[1].item(), 1.0)
        self.assertAlmostEqual(ret[2].item(), 3.0)
        self.assertAlmostEqual(ret[3].item(), 3.0)
        self.assertAlmostEqual(ret[4].item(), 4.0)
        self.assertAlmostEqual(ret[5].item(), 5.0)

    def test2(self):
        observed_times = np.array([5, 5, 5, 5], dtype=float)
        uncensored = np.array([False, False, True, False])
        copula = IndependenceCopula()
        dist = zheng_klein_estimator(observed_times, uncensored, copula, y_max=10)

        self.assertEqual(dist.confidence_interval, None)
        self.assertEqual(dist.b.shape, (3,))
        self.assertEqual(dist.cum_p.shape, (2,))
        self.assertTrue(np.allclose(dist.b, np.array([0, 5, 10], dtype=float)))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 0.25], dtype=float), rtol=0.01))
        self.assertTrue(
            np.allclose(
                dist.cdf(np.array([4, 5, 6, 10, 11], dtype=float)),
                [0.0, 0.25, 0.25, 1.0, 1.0],
                rtol=0.01,
            )
        )
        self.assertTrue(
            np.allclose(
                dist.icdf(np.array([0.0, 0.5, 1.0], dtype=float)),
                [5, 10.0, 10.0],
            )
        )

    def test3(self):
        observed_times = np.array([5, 5, 5, 6], dtype=float)
        uncensored = np.array([True, True, True, False])
        copula = IndependenceCopula()
        dist = zheng_klein_estimator(observed_times, uncensored, copula)

        self.assertEqual(dist.confidence_interval, None)
        self.assertEqual(dist.b.shape, (3,))
        self.assertEqual(dist.cum_p.shape, (2,))
        self.assertTrue(np.allclose(dist.b, np.array([0, 5, 6], dtype=float)))
        self.assertTrue(np.allclose(dist.cum_p, np.array([0.0, 0.75], dtype=float), rtol=0.01))
        self.assertTrue(
            np.allclose(
                dist.cdf(np.array([4, 5, 6], dtype=float)),
                [0.0, 0.75, 1.0],
                rtol=0.01,
            )
        )
        self.assertTrue(
            np.allclose(
                dist.icdf(np.array([0.0, 0.5, 1.0], dtype=float)),
                [5, 5, 6],
            )
        )


class TestTurnbullEstimator(unittest.TestCase):
    def test1(self):
        lb = np.array([1, 2, 2, 4, 3])
        ub = np.array([1, 2, np.inf, 4, 3])
        lb, ub = adjust_exact_observations(lb, ub)
        cdf = turnbull_estimator(lb, ub, y_min=0.0, y_max=5.0)

        self.assertEqual(cdf.b.shape, (10,))
        self.assertEqual(cdf.cum_p.shape, (9,))
        self.assertTrue(
            np.allclose(
                cdf.b,
                np.array([0.0, 0.99999, 1, 1.99999, 2, 2.99999, 3, 3.99999, 4, 5.0]),
            )
        )
        self.assertTrue(
            np.allclose(
                cdf.cum_p,
                np.array([0.0, 0.2, 0.2, 0.4, 0.4, 0.7, 0.7, 1.0, 1.0]),
                rtol=0.01,
            )
        )
        self.assertTrue(
            np.allclose(
                cdf.cdf(np.array([0, 1, 2, 2.5, 3, 4, 5], dtype=float)),
                [0.0, 0.2, 0.4, 0.4, 0.7, 1.0, 1.0],
                rtol=0.01,
            )
        )

        ret = cdf.icdf(np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=float))
        self.assertEqual(ret.shape, (5,))
        self.assertAlmostEqual(ret[0].item(), 0.99999)
        self.assertAlmostEqual(ret[1].item(), 0.99999)
        self.assertAlmostEqual(ret[2].item(), 2.99999)
        self.assertAlmostEqual(ret[3].item(), 3.99999)
        self.assertAlmostEqual(ret[4].item(), 4.0)


class TestLiWatkinsYuEstimator(unittest.TestCase):
    def test1(self):
        lb = np.array([1, 2, 2, 4, 3], dtype=float)
        ub = np.array([1, 2, np.inf, 4, 3], dtype=float)
        cdf = li_watkins_yu_estimator(lb, ub, y_min=0.0, y_max=5.0)

        self.assertEqual(cdf.b.shape, (6,))
        self.assertEqual(cdf.cum_p.shape, (5,))
        self.assertAlmostEqual(cdf.b[0], 0.0, places=3)
        self.assertAlmostEqual(cdf.b[1], 1.0, places=3)
        self.assertAlmostEqual(cdf.b[2], 2.0, places=3)
        self.assertAlmostEqual(cdf.b[3], 3.0, places=3)
        self.assertAlmostEqual(cdf.b[4], 4.0, places=3)
        self.assertAlmostEqual(cdf.b[5], 5.0, places=3)
        self.assertAlmostEqual(cdf.cum_p[0], 0.2, places=3)
        self.assertAlmostEqual(cdf.cum_p[1], 0.4, places=3)
        self.assertAlmostEqual(cdf.cum_p[2], 0.7, places=3)
        self.assertAlmostEqual(cdf.cum_p[3], 1.0, places=3)
        self.assertAlmostEqual(cdf.cum_p[4], 1.0, places=3)

        ret = cdf.cdf(np.array([0, 1, 2, 2.5, 3, 4, 5], dtype=float))
        self.assertAlmostEqual(ret[0], 0.0, places=3)
        self.assertAlmostEqual(ret[1], 0.2, places=3)
        self.assertAlmostEqual(ret[2], 0.4, places=3)
        self.assertAlmostEqual(ret[3], 0.7, places=3)
        self.assertAlmostEqual(ret[4], 0.7, places=3)
        self.assertAlmostEqual(ret[5], 1.0, places=3)
        self.assertAlmostEqual(ret[6], 1.0, places=3)

        ret = cdf.icdf(np.array([0.0, 0.1, 0.5, 0.9, 1.0]))
        self.assertEqual(ret.shape, (5,))
        self.assertAlmostEqual(ret[0], 0.0, places=3)
        self.assertAlmostEqual(ret[1], 0.0, places=3)
        self.assertAlmostEqual(ret[2], 2.0, places=3)
        self.assertAlmostEqual(ret[3], 3.0, places=3)
        self.assertAlmostEqual(ret[4], 4.0, places=3)

    def test2(self):
        lb = np.array([0, 2, 0, 1, 3, 1, 1, 1, 2, 2], dtype=float)
        ub = np.array([2, np.inf, 1, 3, np.inf, 2, 2, 2, 3, 3], dtype=float)
        cdf = li_watkins_yu_estimator(lb, ub)

        self.assertEqual(cdf.b.shape, (6,))
        self.assertEqual(cdf.cum_p.shape, (5,))
        self.assertAlmostEqual(cdf.cum_p[0].item(), 0.0, places=6)
        self.assertAlmostEqual(cdf.cum_p[1].item(), 0.121855, places=6)
        self.assertAlmostEqual(cdf.cum_p[2].item(), 0.558281, places=6)
        self.assertAlmostEqual(cdf.cum_p[3].item(), 0.870696, places=6)
        self.assertAlmostEqual(cdf.cum_p[4].item(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
