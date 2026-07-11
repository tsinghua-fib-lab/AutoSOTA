import math
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols

# vif test base on OLS
# test collinearity or internal correlation
"""
def vifTest(x, col_i):
    x = x_train
    y = y_train
    model = LinearRegression()
    r2 = model.fit(x, y).rsquared
    return 1. / (1. - r2)
"""


def vif_test(data, col_i):
    cols = list(data.columns)
    cols.remove(col_i)
    colsNoI = cols
    formula = col_i + '~' + '+'.join(colsNoI)
    r2 = ols(formula, data).fit().rsquared
    return 1. / (1. - r2)
