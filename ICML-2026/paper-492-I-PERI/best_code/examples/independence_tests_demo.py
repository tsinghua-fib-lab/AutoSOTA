"""Demo: usage examples for independence tests in pyciphod

Three scenarios:
 - continuous: FisherZTest, LinearRegressionCoefficientTTest, CIMhTest (permutation p-value)
 - discrete: GsqTest (statistic + p-value), CIMhTest used in discrete mode
 - mixed: CIMhTest treating X as discrete

Run from repository root with: PYTHONPATH=src python3 examples/independence_tests_demo.py
"""

import numpy as np
import pandas as pd
import time

from pyciphod.utils.stat_tests.independence_tests import (
    FisherZTest,
    LinearRegressionCoefficientTTest,
    GsqTest,
    CIMhTest,
)

np.random.seed(0)


def continuous_scenario():
    print('\n=== Continuous scenario ===')
    n = 500
    x = np.random.normal(size=n)
    z = 0.7 * x + 0.3 * np.random.normal(size=n)
    y = 1.8 * z + 0.1 * np.random.normal(size=n)
    df = pd.DataFrame({"x": x, "z": z, "y": y})

    # Fisher Z test (partial correlation test)
    fz = FisherZTest("x", "y", cond_list=["z"])
    dep = fz.get_dependence(df)
    pval = fz.get_pvalue(df)
    print(f"FisherZTest dependence (partial r): {dep}")
    print(f"FisherZTest p-value: {pval}")

    # Linear regression t-test for coefficient
    lrtest = LinearRegressionCoefficientTTest("x", "y", cond_list=["z"])
    beta = lrtest.get_dependence(df)
    p_beta = lrtest.get_pvalue(df)
    print(f"LinearRegressionCoefficientTTest beta (x): {beta}")
    print(f"LinearRegressionCoefficientTTest p-value: {p_beta}")

    # CMI-based test with permutation p-value (CIMhTest uses get_pvalue_by_permutation)
    cim = CIMhTest("x", "y", cond_list=["z"])
    # configure CMI estimator attributes expected by CMIh
    cim.discrete_vars = None
    cim.k = None
    cmi_stat = cim.get_dependence(df)
    print(f"CIMhTest dependence (CMI): {cmi_stat}")
    t0 = time.time()
    p_perm = cim.get_pvalue_by_permutation(df, n_permutations=200, seed=1)
    t1 = time.time()
    print(f"CIMhTest permutation p-value (200 iters): {p_perm} (took {t1-t0:.2f}s)")


def discrete_scenario():
    print('\n=== Discrete scenario ===')
    n = 600
    x = np.random.randint(0, 3, size=n)
    z = x.copy()  # mediator
    y_ind = np.random.randint(0, 3, size=n)
    y_dep = x.copy()

    df_ind = pd.DataFrame({"x": x, "y": y_ind, "z": z})
    df_dep = pd.DataFrame({"x": x, "y": y_dep, "z": z})

    # G-squared test
    gtest_ind = GsqTest("x", "y")
    g_ind_stat = gtest_ind.get_dependence(df_ind)
    g_ind_p = gtest_ind.get_pvalue(df_ind)
    print(f"GsqTest independent statistic: {g_ind_stat}, p-value: {g_ind_p}")

    gtest_dep = GsqTest("x", "y")
    g_dep_stat = gtest_dep.get_dependence(df_dep)
    g_dep_p = gtest_dep.get_pvalue(df_dep)
    print(f"GsqTest dependent statistic: {g_dep_stat}, p-value: {g_dep_p}")

    # Conditioned G-square (conditioning removes dependence when z==x)
    g_cond = GsqTest("x", "y", cond_list=["z"])
    g_cond_stat = g_cond.get_dependence(df_dep)
    g_cond_p = g_cond.get_pvalue(df_dep)
    print(f"GsqTest conditioned on z statistic: {g_cond_stat}, p-value: {g_cond_p}")

    # CMI on discrete variables using CIMhTest (configure discrete_vars)
    cim = CIMhTest("x", "y")
    cim.discrete_vars = ["x", "y", "z"]
    cim.k = None
    cmi_dep = cim.get_dependence(df_dep.assign(z=z))
    p_cmi = cim.get_pvalue_by_permutation(df_dep.assign(z=z), n_permutations=200, seed=2)
    print(f"CIMhTest (discrete mode) dependence: {cmi_dep}, permutation p-value: {p_cmi}")


def mixed_scenario():
    print('\n=== Mixed scenario ===')
    n = 500
    x = np.random.randint(0, 4, size=n)  # discrete
    y = 1.2 * x + 0.4 * np.random.normal(size=n)  # continuous dependent
    z = np.random.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y, "z": z})

    cim = CIMhTest("x", "y", cond_list=["z"])
    cim.discrete_vars = ["x"]
    cim.k = None
    cmi_stat = cim.get_dependence(df)
    p_cmi = cim.get_pvalue_by_permutation(df, n_permutations=200, seed=3)
    print(f"CIMhTest (mixed) dependence: {cmi_stat}, permutation p-value: {p_cmi}")


if __name__ == '__main__':
    continuous_scenario()
    discrete_scenario()
    mixed_scenario()
    print('\nIndependence demo finished.')
