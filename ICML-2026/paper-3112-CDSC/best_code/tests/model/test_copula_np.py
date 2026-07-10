import unittest

import numpy as np

from cenreg.model.copula_np import IndependenceCopula, SurvivalCopula


class TestCopulaNp(unittest.TestCase):
    def test_independence_copula(self):
        a = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
        copula = IndependenceCopula()
        ret = copula.cdf(a)

        self.assertEqual(ret.shape, (3,))
        self.assertAlmostEqual(ret[0].item(), 0.02)
        self.assertAlmostEqual(ret[1].item(), 0.06)
        self.assertAlmostEqual(ret[2].item(), 0.12)

    def test_survival_copula(self):
        a = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
        copula = IndependenceCopula()
        survival_copula = SurvivalCopula(copula)
        ret = survival_copula.cdf(a)

        self.assertEqual(ret.shape, (3,))
        self.assertAlmostEqual(ret[0].item(), 0.02)
        self.assertAlmostEqual(ret[1].item(), 0.06)
        self.assertAlmostEqual(ret[2].item(), 0.12)


if __name__ == "__main__":
    unittest.main()
