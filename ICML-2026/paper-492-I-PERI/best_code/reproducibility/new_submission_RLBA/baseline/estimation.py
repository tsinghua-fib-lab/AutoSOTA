"""
Coded by Charles Assaad
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

import scipy as scp
from sklearn.linear_model import LinearRegression as lr
from sklearn.feature_selection import f_regression as fr


# from scipy import special
# from scipy.spatial import cKDTree
# import itertools
# from joblib import Parallel, delayed, cpu_count
# import statsmodels.api as sm


class CiTests:
    def __init__(self, x, y, cond_list=None):
        super(CiTests, self).__init__()
        self.x = x
        self.y = y
        if cond_list is None:
            self.cond_list = []
        else:
            self.cond_list = cond_list

    def get_dependence(self, df):
        print("To be implemented")

    def get_pvalue(self, df):
        print("To be implemented")


class FisherZ(CiTests):
    def __init__(self, x, y, cond_list=None):
        CiTests.__init__(self, x, y, cond_list)

    def get_dependence(self, df):
        list_nodes = [self.x, self.y] + self.cond_list
        df = df[list_nodes]
        a = df.values.T

        if len(self.cond_list) > 0:
            cond_list_int = [i + 2 for i in range(len(self.cond_list))]
        else:
            cond_list_int = []

        correlation_matrix = np.corrcoef(a)
        var = list((0, 1) + tuple(cond_list_int))
        sub_corr_matrix = correlation_matrix[np.ix_(var, var)]
        if np.linalg.det(sub_corr_matrix) == 0:
            r = 1
        else:
            inv = np.linalg.inv(sub_corr_matrix)
            r = -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])
        return r

    def get_pvalue(self, df):
        r = self.get_dependence(df)
        if r == 1:
            r = r - 0.0000000001
        z = 0.5 * np.log((1 + r) / (1 - r))
        pval = np.sqrt(df.shape[0] - len(self.cond_list) - 3) * abs(z)
        pval = 2 * (1 - norm.cdf(abs(pval)))

        return pval, r

    def diff_two_fisherz(self, df1, df2):
        r1 = self.get_dependence(df1)
        if r1 == 1:
            r1 = r1 - 0.0000000001
        r2 = self.get_dependence(df2)
        if r2 == 1:
            r2 = r2 - 0.0000000001
        z1 = 0.5 * np.log((1 + r1) / (1 - r1))
        z2 = 0.5 * np.log((1 + r2) / (1 - r2))
        n1 = df1.shape[0]
        n2 = df2.shape[0]
        sezdiff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
        ztest = (z1 - z2) / sezdiff
        pval = 2 * (1 - norm.cdf(abs(ztest), 0, 1))
        return pval


class LinearRegression:
    def __init__(self, x, y, cond_list=[]):
        self.x = x
        self.y = y
        self.list_nodes = [x] + cond_list

    def get_coeff(self, df):
        X_data = df[self.list_nodes].values
        Y_data = df[self.y].values
        reg = lr().fit(X_data, Y_data)

        return reg.coef_[0]

    def test_zeo_coef(self, df):
        X_data = df[self.list_nodes].values
        Y_data = df[self.y].values
        pval = fr(X_data, Y_data)[1][0]

        return pval


def grubb_test(quatification_list, confidence_level=0.05):
    n = len(quatification_list)

    stat_value = max(abs(np.array(quatification_list) - np.mean(quatification_list)))/ np.std(quatification_list)

    t_dist = scp.stats.t.ppf(1 - confidence_level / (2 * n), n - 2)
    numerator = (n - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(n) * np.sqrt(n - 2 + np.square(t_dist))
    critical_value = numerator / denominator

    quatification_list_sorted_idx = sorted(range(len(quatification_list)), key=quatification_list.__getitem__)
    quatification_list_sorted = sorted(quatification_list)
    # locate potential outlier
    l1 = quatification_list_sorted[1] - quatification_list_sorted[0]
    l2 = quatification_list_sorted[-1] - quatification_list_sorted[-2]

    anomaly_position = None
    if stat_value > critical_value:
        if l1 > l2:
            anomaly_position = quatification_list_sorted_idx[0]
        else:
            anomaly_position = quatification_list_sorted_idx[-1]

    return {"anomaly_position": anomaly_position, "stat_value": stat_value, "critical_value": critical_value}


if __name__ == '__main__':
    x = np.random.normal(size=100)
    y = 2*x + 0.1 * np.random.normal(size=100)
    z = 5*x + 0.2* np.random.normal(size=100)
    data = pd.DataFrame(np.array([x, y, z]).T, columns=["x", "y", "z"])

    ci = FisherZ("z", "y", ["x"])
    p, d = ci.get_pvalue(data)
    print(p, d)

    ci = FisherZ("z", "y")
    p, d = ci.get_pvalue(data)
    print(p, d)
