import numpy as np
import pandas as pd
import pytest

from pyciphod.utils.stat_tests.dependency_measures import (
    PartialCorrelation,
    LinearRegressionCoefficient,
    Gsq,
    CMIh,
)

np.random.seed(0)


def test_partial_correlation_no_cond():
    n = 500
    x = np.random.normal(size=n)
    z = x + 0.1 * np.random.normal(size=n)
    y = 2.0 * z + 0.1 * np.random.normal(size=n)
    df = pd.DataFrame({"x": x, "z": z, "y": y})

    pc = PartialCorrelation("x", "y")
    r = pc.get_dependence(df)
    assert isinstance(r, float)
    assert np.isfinite(r)
    assert r > 0.1


def test_partial_correlation_with_cond_zero():
    n = 1000
    x = np.random.normal(size=n)
    z = x + 0.05 * np.random.normal(size=n)
    y = 3.0 * z + 0.05 * np.random.normal(size=n)
    df = pd.DataFrame({"x": x, "z": z, "y": y})

    pc = PartialCorrelation("x", "y", cond_list=["z"])
    r = pc.get_dependence(df)
    # After conditioning on z, x and y should be nearly independent
    assert isinstance(r, float)
    assert np.isfinite(r)
    assert abs(r) < 0.15


def test_partial_correlation_small_sample_nan():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    pc = PartialCorrelation("x", "y")
    val = pc.get_dependence(df)
    assert np.isnan(val)


def test_linear_regression_coefficient_recovery():
    n = 800
    x = np.random.normal(size=n)
    z = np.random.normal(size=n)
    y = 3.0 * x + 2.0 * z + 0.2 * np.random.normal(size=n)
    df = pd.DataFrame({"x": x, "z": z, "y": y})

    lrcoef = LinearRegressionCoefficient("x", "y", cond_list=["z"])
    beta = lrcoef.get_dependence(df)
    assert isinstance(beta, float)
    assert np.isfinite(beta)
    assert pytest.approx(3.0, rel=0.1) == beta


def test_linear_regression_insufficient_samples_nan():
    # two samples and two predictors (x and z) -> should return nan
    df = pd.DataFrame({"x": [1.0, 2.0], "z": [1.0, 2.0], "y": [1.0, 2.0]})
    lrcoef = LinearRegressionCoefficient("x", "y", cond_list=["z"])
    val = lrcoef.get_dependence(df)
    assert np.isnan(val)


def test_gsq_independence_and_dependence():
    n = 500
    x = np.random.randint(0, 2, size=n)
    y_ind = np.random.randint(0, 2, size=n)
    y_dep = x.copy()
    df_ind = pd.DataFrame({"x": x, "y": y_ind})
    df_dep = pd.DataFrame({"x": x, "y": y_dep})

    g_ind = Gsq("x", "y").get_dependence(df_ind)
    g_dep = Gsq("x", "y").get_dependence(df_dep)

    assert isinstance(g_ind, float)
    assert isinstance(g_dep, float)
    assert g_ind >= 0
    assert g_dep >= 0
    # dependent case should have noticeably larger G² than independent
    assert g_dep > max(5.0, g_ind * 3 + 1.0)

    # With conditioning on z that equals x, dependence should vanish
    df_cond = pd.DataFrame({"x": x, "y": y_dep, "z": x})
    g_cond = Gsq("x", "y", cond_list=["z"]).get_dependence(df_cond)
    assert isinstance(g_cond, float)
    assert g_cond >= 0
    assert g_cond < 1.0


def test_cmih_continuous_degenerate_and_indep():
    # dependent continuous
    n = 300
    x = np.random.normal(size=n)
    y = x + 0.1 * np.random.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y})
    cmi = CMIh("x", "y").get_dependence(df)
    assert isinstance(cmi, float)
    assert not np.isnan(cmi)
    assert cmi > 0.0

    # independent continuous
    x2 = np.random.normal(size=n)
    y2 = np.random.normal(size=n)
    df2 = pd.DataFrame({"x": x2, "y": y2})
    cmi2 = CMIh("x", "y").get_dependence(df2)
    assert isinstance(cmi2, float)
    # independent case should be small (close to 0)
    assert cmi2 >= 0
    assert cmi2 < cmi

    # degenerate repeated points -> should return nan because kNN estimator degenerates
    df_deg = pd.DataFrame({"x": np.ones(10), "y": np.ones(10)})
    cmi_deg = CMIh("x", "y").get_dependence(df_deg)
    assert np.isnan(cmi_deg)
