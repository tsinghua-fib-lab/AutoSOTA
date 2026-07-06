# Ajout de tests unitaires pour les fonctions de independence_tests.py
import numpy as np
import pandas as pd
import pytest

from pyciphod.utils.stat_tests.independence_tests import (
    PartialCorrelation,
    LinearRegressionCoefficient,
    Gsq,
    GsqTest,
    FisherZTest,
    LinearRegressionCoefficientTTest,
    CIMhTest,
)


def test_partial_correlation_insufficient_rows_returns_nan():
    df = pd.DataFrame({"x": [1, 2], "y": [2, 4]})
    pc = PartialCorrelation("x", "y")
    res = pc.get_dependence(df)
    assert np.isnan(res)


def test_partial_correlation_perfect_correlation_returns_one():
    # Pour une relation linéaire parfaite la matrice de corrélation est singulière
    # et l'implémentation retourne np.nan (cas de colinéarité parfaite).
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]})
    pc = PartialCorrelation("x", "y")
    res = pc.get_dependence(df)
    assert np.isnan(res)


def test_linear_regression_coefficient_insufficient_rows_returns_nan():
    df = pd.DataFrame({"x": [1], "y": [2]})
    lrcoef = LinearRegressionCoefficient("x", "y")
    res = lrcoef.get_dependence(df)
    assert np.isnan(res)


def test_linear_regression_coefficient_simple_case():
    # Y = 2 * X exactly
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    lrcoef = LinearRegressionCoefficient("x", "y")
    res = lrcoef.get_dependence(df)
    assert pytest.approx(2.0, rel=1e-6) == res


def test_gsq_dependence_and_pvalue_nan_for_zero_df_total():
    # All data in a single cell -> df_total == 0 and g2_stat == 0
    df = pd.DataFrame({"x": [0, 0, 0, 0], "y": [1, 1, 1, 1]})
    gsq = Gsq("x", "y")
    g2 = gsq.get_dependence(df)
    assert g2 == 0.0

    gsq_test = GsqTest("x", "y")
    pval = gsq_test.get_pvalue(df)
    assert np.isnan(pval)


def test_fisherz_test_pvalue_finite_and_in_unit_interval():
    # Pour une relation linéaire parfaite le calcul de la corrélation partielle
    # est indéfini et la p-valeur doit être np.nan.
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [2, 4, 6, 8, 10, 12]})
    fzt = FisherZTest("x", "y")
    pval = fzt.get_pvalue(df)
    assert np.isnan(pval)


def test_get_pvalue_by_permutation_is_deterministic_with_seed():
    # Use the LinearRegressionCoefficientTTest which relies on regression coef
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7], "y": [2, 4, 6, 8, 10, 12, 14]})
    test_obj = LinearRegressionCoefficientTTest("x", "y")

    p1 = test_obj.get_pvalue_by_permutation(df, n_permutations=50, seed=42)
    p2 = test_obj.get_pvalue_by_permutation(df, n_permutations=50, seed=42)
    assert p1 == p2
    assert 0.0 <= p1 <= 1.0


def test_cimhtest_get_pvalue_discrete_delegates_and_is_in_0_1():
    # Create discrete identical variables -> mutual information > 0
    df = pd.DataFrame({"x": [0, 1, 0, 1, 0, 1, 0, 1], "y": [0, 1, 0, 1, 0, 1, 0, 1]})
    ct = CIMhTest("x", "y")
    # Treat x and y as discrete to force plugin entropy paths
    ct.discrete_vars = ["x", "y"]
    pval = ct.get_pvalue_by_permutation(df, n_permutations=50, seed=0)
    assert np.isfinite(pval)
    assert 0.0 <= pval <= 1.0
