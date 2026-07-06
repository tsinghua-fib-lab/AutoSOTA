#!/usr/bin/env python
# encoding: utf-8
import pandas as pd
from scipy import stats
import math
from warnings import warn

import numpy as np
import pandas as pd

from scipy import stats


def chi_square_test(df, var1, var2, condition_vars=None, alaph=0.01, **kwargs):
    """
    Test for the independence condition (var1 _|_ var2 | condition_vars) in df.

    Parameters
    ----------
    df: pandas Dataframe
        The dataset on which to test the independence condition.

    var1: str
        First variable in the independence condition.

    var2: str
        Second variable in the independence condition

    condition_vars: list
        List of variable names in given variables.

    Returns
    -------
    chi_stat: float
        The chi-square statistic for the test.

    p_value: float
        The p-value of the test

    dof: int
        Degrees of Freedom
    """
    var1 = str(var1)
    var2 = str(var2)
    condition_vars = [str(i) for i in condition_vars]
    if "state_names" in kwargs.keys():
        state_names = kwargs["state_names"]
    else:
        state_names = {
            var_name: df.loc[:, var_name].unique() for var_name in df.columns
        }
    num_params = (
        (len(state_names[var1]) - 1)
        * (len(state_names[var2]) - 1)
        * np.prod([len(state_names[z]) for z in condition_vars])
    )
    sufficient_data = len(df) >= num_params * 5
    if not condition_vars:
        observed = pd.crosstab(df[str(var1)], df[str(var2)])
        chi_stat, p_value, dof, expected = stats.chi2_contingency(observed)

    else:
        observed_combinations = df.groupby(condition_vars).size().reset_index()

        chi_stat = 0
        dof = 0
        for combination in range(len(observed_combinations)):
            df_conditioned = df.copy()
            for condition_var in condition_vars:
                df_conditioned = df_conditioned.loc[df_conditioned.loc[:,
                                                                       condition_var] == observed_combinations.loc[combination,
                                                                                                                   condition_var]]
            observed = pd.crosstab(df_conditioned[var1], df_conditioned[var2])
            chi, _, freedom, _ = stats.chi2_contingency(observed)
            chi_stat += chi
            dof += freedom
        p_value = 1.0 - stats.chi2.cdf(x=chi_stat, df=dof)

    if dof <= 0:
        dof = 1
    # print("chi_stat is ", chi_stat, "dof is ", dof)
    if math.isnan(p_value):
        chi_stat = 1.0
        dep = 2.0 + chi_stat / num_params
    elif p_value > alaph:
        dep = -2.0 - chi_stat / num_params
    else:
        dep = 2.0 + chi_stat / num_params

    # n = df.shape[0]
    # rmsea = sqrt((chi_stat / dof - 1) / (n - 1))

    p_value = round(p_value, 6)
    # if p_value > alaph:
    #     p_valueX, p_valueY = str(p_value).split('.')
    #     p_value = float(p_valueX + '.' + p_valueY[0:2])

    return chi_stat, p_value, dof, 1-p_value



def chi_square(X, Y, Z, data, alpha, **kwargs):
    """
    Chi-square conditional independence test.
    Tests the null hypothesis that X is independent from Y given Zs.

    This is done by comparing the observed frequencies with the expected
    frequencies if X,Y were conditionally independent, using a chisquare
    deviance statistic. The expected frequencies given independence are
    `P(X,Y,Zs) = P(X|Zs)*P(Y|Zs)*P(Zs)`. The latter term can be computed
    as `P(X,Zs)*P(Y,Zs)/P(Zs).

    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set
    Y: int, string, hashable object
        A variable name contained in the data set, different from X
    Zs: list of variable names
        A list of variable names contained in the data set, different from X and Y.
        This is the separating set that (potentially) makes X and Y independent.
        Default: []

    Returns
    -------
    chi2: float
        The chi2 test statistic.
    p_value: float
        The p_value, i.e. the probability of observing the computed chi2
        statistic (or an even higher value), given the null hypothesis
        that X _|_ Y | Zs.
    sufficient_data: bool
        A flag that indicates if the sample size is considered sufficient.
        As in [4], require at least 5 samples per parameter (on average).
        That is, the size of the data set must be greater than
        `5 * (c(X) - 1) * (c(Y) - 1) * prod([c(Z) for Z in Zs])`
        (c() denotes the variable cardinality).


    References
    ----------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.2.2.3 (page 789)
    [2] Neapolitan, Learning Bayesian Networks, Section 10.3 (page 600ff)
        http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
    [3] Chi-square test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Test_of_independence
    [4] Tsamardinos et al., The max-min hill-climbing BN structure learning algorithm, 2005, Section 4

    Examples
    --------
    # >>> import pandas as pd
    # >>> import numpy as np
    # >>> from pgmpy.estimators import ConstraintBasedEstimator
    # >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
    # >>> data['E'] = data['A'] + data['B'] + data['C']
    # >>> c = ConstraintBasedEstimator(data)
    # >>> print(c.test_conditional_independence('A', 'C'))  # independent
    # True
    # >>> print(c.test_conditional_independence('A', 'B', 'D'))  # independent
    # True
    # >>> print(c.test_conditional_independence('A', 'B', ['D', 'E']))  # dependent
    False
    """
    X = str(X)
    Y = str(Y)
    Z = [str(i) for i in Z]
    if isinstance(Z, (frozenset, list, set, tuple)):
        Z = list(Z)
    else:
        Z = [Z]

    if "state_names" in kwargs.keys():
        state_names = kwargs["state_names"]
    else:
        state_names = {
            var_name: data.loc[:, var_name].unique() for var_name in data.columns
        }

    num_params = (
        (len(state_names[X]) - 1)
        * (len(state_names[Y]) - 1)
        * np.prod([len(state_names[z]) for z in Z])
    )
    sufficient_data = len(data) >= num_params * 5

    if not sufficient_data:
        # warn(
        #     "Insufficient data for testing {0} _|_ {1} | {2}. ".format(X, Y, Z)
        #     + "At least {0} samples recommended, {1} present.".format(
        #         5 * num_params, len(data)
        #     )
        # )
        chi2 = 1
        p_value = 1
    else:
        # compute actual frequency/state_count table:
        # = P(X,Y,Zs)
        XYZ_state_counts = pd.crosstab(
            index=data[X], columns=[data[Y]] + [data[z] for z in Z]
        )
        # reindex to add missing rows & columns (if some values don't appear in data)
        row_index = state_names[X]
        column_index = pd.MultiIndex.from_product(
            [state_names[Y]] + [state_names[z] for z in Z], names=[Y] + Z
        )
        if not isinstance(XYZ_state_counts.columns, pd.MultiIndex):
            XYZ_state_counts.columns = pd.MultiIndex.from_arrays([XYZ_state_counts.columns])
        XYZ_state_counts = XYZ_state_counts.reindex(
            index=row_index, columns=column_index
        ).fillna(0)

        # compute the expected frequency/state_count table if X _|_ Y | Zs:
        # = P(X|Zs)*P(Y|Zs)*P(Zs) = P(X,Zs)*P(Y,Zs)/P(Zs)
        if Z:
            XZ_state_counts = XYZ_state_counts.sum(axis=1, level=Z)  # marginalize out Y
            YZ_state_counts = XYZ_state_counts.sum().unstack(Z)  # marginalize out X
        else:
            XZ_state_counts = XYZ_state_counts.sum(axis=1)

            YZ_state_counts = XYZ_state_counts.sum()
        Z_state_counts = YZ_state_counts.sum()  # marginalize out both

        XYZ_expected = pd.DataFrame(
            index=XYZ_state_counts.index, columns=XYZ_state_counts.columns
        )
        for X_val in XYZ_expected.index:
            if Z:
                for Y_val in XYZ_expected.columns.levels[0]:
                    XYZ_expected.loc[X_val, Y_val] = (
                            XZ_state_counts.loc[X_val]
                            * YZ_state_counts.loc[Y_val]
                            / Z_state_counts
                    ).values
            else:
                for Y_val in XYZ_expected.columns:
                    XYZ_expected.loc[X_val, Y_val] = (
                            XZ_state_counts.loc[X_val]
                            * YZ_state_counts.loc[Y_val]
                            / float(Z_state_counts)
                    )

        observed = XYZ_state_counts.values.flatten()
        expected = XYZ_expected.fillna(0).values.flatten()
        # remove elements where the expected value is 0;
        # this also corrects the degrees of freedom for chisquare
        observed, expected = zip(
            *((o, e) for o, e in zip(observed, expected) if not e == 0)
        )

        chi2, p_value = stats.chisquare(observed, expected)
        # p_value = 1.0 - stats.chi2.cdf(x=chi2, df=num_params)

    if p_value >= alpha:
        dep = (-2.0) - chi2 / num_params
    else:
        dep = 2.0 + chi2 / num_params
    # print(significance_level," chi2 is: ", chi2," num_params is: ",num_params)

    return chi2, num_params, dep, p_value

