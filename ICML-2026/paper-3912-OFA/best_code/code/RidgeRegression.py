# When result shows collinearity, we using Ridge Regression to replace normal OLS.
import numpy as np
from sklearn.linear_model import Ridge


def ridge_regression(x, y):
    n_alphas = 200
    alphas = np.logspace(-10, -2, num=n_alphas)
    coefficients = []
    for i in alphas:
        ridge = Ridge(alpha=i, fit_intercept=False)
        ridge.fit(x, y)
        coefficients.append(ridge.coef_[0])
    return coefficients
