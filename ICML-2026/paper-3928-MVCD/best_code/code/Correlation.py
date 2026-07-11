import numpy as np
from sklearn.linear_model import LinearRegression
import PrincipalComponentsRegression
import RidgeRegression
from scipy.optimize import minimize

def correlation_for_whole(correlations):
    cfw = np.sum(correlations)
    correlations = [0 for _ in range(cfw.shape[0])]

    for i in range(cfw.shape[0]):
        if cfw[i] > 0:
            correlations[i] = 1
        elif cfw[i] < 0:
            correlations[i] = -1
        else:
            correlations[i] = 0

    return correlations


def attribution_correlation(model_num, attribute, state):
    cor = regression_models(model_num, attribute, state)
    correlation = [0 for _ in range(cor.shape[0])]

    for i in range(cor.shape[0]):
        if cor[i] > 0:
            correlation[i] = 1
        elif cor[i] < 0:
            correlation[i] = -1
        else:
            correlation[i] = 0

    return correlation


def attribution_correlate_coe(model_num, attribute, state):
    cor = regression_models(model_num, attribute, state)


    return cor

# def regression_models(model_num, x_train, y_train):
#     if model_num == 1:
#         # Using Principal Components Regression to fit model
#         coefficient_set = PrincipalComponentsRegression.PCR(x_train, y_train)
#
#     elif model_num == 2:
#         # Using Ridge Regression to fit model
#         coefficient_set = RidgeRegression.ridge_regression(x_train, y_train)
#
#     elif model_num == 3:
#
#         # try to build the linear regression
#         model_OLS = LinearRegression()
#         # when we need to remove intercept
#         # model = LinearRegression(fit_intercept=False)
#
#         model_OLS.fit(x_train, y_train)
#         # obtained the coefficients
#         coefficient_set = model_OLS.coef_
#
#
#     else:
#         coefficient_set = 0
#         print("error!")
#
#     return coefficient_set


def regression_models(model_num, x_train, y_train):
    if model_num == 1:
        # Using Principal Components Regression to fit model
        coefficient_set = PrincipalComponentsRegression.PCR(x_train, y_train)

    elif model_num == 2:
        # Using Ridge Regression to fit model
        coefficient_set = RidgeRegression.ridge_regression(x_train, y_train)

    elif model_num == 3:

        n, k = x_train.shape
        model = LinearRegression()
        model.fit(x_train, y_train)
        beta_init = model.coef_

        def neg_log_likelihood(beta):
            residuals = y_train - x_train @ beta
            # print(residuals)  # suppressed for evaluation
            sigma2 = np.sum(residuals ** 2) / n
            return 0.5 * n * np.log(2 * np.pi * sigma2) + np.sum(residuals ** 2) / (2 * sigma2)


        res = minimize(neg_log_likelihood, beta_init, method='BFGS')

        coefficient_set = res.x
        return coefficient_set



    else:
        coefficient_set = 0
        print("error!")

    return coefficient_set


