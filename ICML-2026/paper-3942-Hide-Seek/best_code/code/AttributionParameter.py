"""
This program estimates different attributions of elements in part to get
the coefficients set, which will be the basic factor of part. We can use
these to distinguish between different parts.
"""

from sklearn.linear_model import LinearRegression
import PrincipalComponentsRegression
import RidgeRegression


# If the results show collinearity or ?internal correlation, it means that
# the dependent variables are not independent of each other, we can not estimate
# model by using OLS

def attribution_parameter(model_choice, x_train, y_train):
    if model_choice == 1:
        # Using Principal Components Regression to fit model
        coefficient_set = PrincipalComponentsRegression.PCR(x_train, y_train)

    elif model_choice == 2:
        # Using Ridge Regression to fit model
        coefficient_set = RidgeRegression.ridge_regression(x_train, y_train)

    elif model_choice == 3:
        # normal OLS is ok
        # try to build the linear regression
        model_OLS = LinearRegression()
        # when we need to remove intercept
        # model = LinearRegression(fit_intercept=False)
        model_OLS.fit(x_train, y_train)
        # obtained the coefficients
        coefficient_set = model_OLS.coef_

    else:
        coefficient_set = 0
        print("error!")

    return coefficient_set
