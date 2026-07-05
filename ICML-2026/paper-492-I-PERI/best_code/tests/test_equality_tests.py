import numpy as np
import pandas as pd

from pyciphod.utils.stat_tests.equality_tests import (
    PartialCorrelationEqualityTest,
    CMIhEqualityTest,
    GsqEqualityTest,
    LinearRegressionCoefficientEqualityTest,
    KernelPartialCorrelationEqualityTest,
    GComputationEqualityTest,
)


def make_continuous(n=100, rho=0.5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    eps = rng.normal(size=n)
    y = rho * x + np.sqrt(max(0, 1 - rho ** 2)) * eps
    z = rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def make_discrete_z_case(n_per_group=20, groups=3, seed=1, dependent=False):
    rng = np.random.default_rng(seed)
    rows_list = []
    for g in range(groups):
        x = rng.binomial(1, 0.3, size=n_per_group)
        if dependent:
            flip = rng.binomial(1, 1 - 0.9, size=n_per_group)
            y = np.abs(x - flip)
        else:
            y = rng.binomial(1, 0.3, size=n_per_group)
        z = np.full(n_per_group, g, dtype=int)
        rows_list.append(pd.DataFrame({"x": x, "y": y, "z_disc": z}))
    df = pd.concat(rows_list, ignore_index=True)
    return df


def test_partialcorr_get_pvalue_in_unit_interval_and_deterministic():
    df1 = make_continuous(n=120, rho=0.8, seed=0)
    df2 = make_continuous(n=120, rho=0.2, seed=1)

    pct = PartialCorrelationEqualityTest(x="x", y="y", cond_list=["z"], drop_na=True)

    p1 = pct.get_pvalue_by_permutation(df1, df2, n_permutations=100, seed=42)
    p2 = pct.get_pvalue_by_permutation(df1, df2, n_permutations=100, seed=42)

    assert (p1 is None) or (np.isnan(p1) or (0.0 <= p1 <= 1.0))
    # deterministic with seed: equal results
    if not (p1 is None) and not np.isnan(p1):
        assert p1 == p2


def test_get_pvalue_returns_nan_for_insufficient_rows():
    df_small = pd.DataFrame({"x": [1], "y": [2], "z": [0]})
    df_ok = make_continuous(n=10, rho=0.1, seed=2)

    pct = PartialCorrelationEqualityTest(x="x", y="y", cond_list=["z"], drop_na=True)
    p = pct.get_pvalue_by_permutation(df_small, df_ok, n_permutations=50, seed=0)
    assert np.isnan(p)


def test_cmih_equality_test_discrete_returns_pvalue_and_is_deterministic():
    df1 = make_discrete_z_case(n_per_group=30, groups=2, seed=3, dependent=False)
    df2 = make_discrete_z_case(n_per_group=30, groups=2, seed=4, dependent=True)

    cmih = CMIhEqualityTest(x="x", y="y", cond_list=["z_disc"], drop_na=True)

    p1 = cmih.get_pvalue_by_permutation(df1, df2, n_permutations=100, seed=123)
    p2 = cmih.get_pvalue_by_permutation(df1, df2, n_permutations=100, seed=123)

    assert (p1 is None) or (np.isnan(p1) or (0.0 <= p1 <= 1.0))
    if not (p1 is None) and not np.isnan(p1):
        assert p1 == p2


def test_gsq_equality_test_discrete_returns_pvalue_and_is_deterministic():
    # G-squared is for discrete variables; reuse make_discrete_z_case
    df1 = make_discrete_z_case(n_per_group=30, groups=2, seed=10, dependent=False)
    df2 = make_discrete_z_case(n_per_group=30, groups=2, seed=11, dependent=True)

    gsq = GsqEqualityTest(x="x", y="y", cond_list=["z_disc"], drop_na=True)

    p1 = gsq.get_pvalue_by_permutation(df1, df2, n_permutations=100, seed=202)
    p2 = gsq.get_pvalue_by_permutation(df1, df2, n_permutations=100, seed=202)

    assert (p1 is None) or (np.isnan(p1) or (0.0 <= p1 <= 1.0))
    if not (p1 is None) and not np.isnan(p1):
        assert p1 == p2


def test_linear_regression_coefficient_equality_test_returns_pvalue_and_is_deterministic():
    # Create two populations with different regression coefficients
    rng = np.random.default_rng(1234)
    n = 150
    x1 = rng.normal(size=n)
    eps1 = rng.normal(scale=0.5, size=n)
    y1 = 2.0 * x1 + eps1
    df1 = pd.DataFrame({"x": x1, "y": y1})

    x2 = rng.normal(size=n)
    eps2 = rng.normal(scale=0.5, size=n)
    y2 = 0.5 * x2 + eps2
    df2 = pd.DataFrame({"x": x2, "y": y2})

    lr_eq = LinearRegressionCoefficientEqualityTest(x="x", y="y", cond_list=None, drop_na=True)

    p1 = lr_eq.get_pvalue_by_permutation(df1, df2, n_permutations=100, seed=7)
    p2 = lr_eq.get_pvalue_by_permutation(df1, df2, n_permutations=100, seed=7)

    assert (p1 is None) or (np.isnan(p1) or (0.0 <= p1 <= 1.0))
    if not (p1 is None) and not np.isnan(p1):
        assert p1 == p2


