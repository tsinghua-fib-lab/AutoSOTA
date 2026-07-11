import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from itertools import product
from sklearn.linear_model import LinearRegression

def rs_estimator_cate(data, test_size=0.2, random_state=41):
    """ 
    Calculate CATE using response surface modeling. 
    """
    
    X = data[['X']].values
    Y = data['Y'].values
    A = data['A'].values

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(
        X, Y, A, test_size=test_size, random_state=random_state
    )

    # Create masks for treatment and control groups
    train_treatment_mask = A_train == 1
    train_control_mask = A_train == 0
    test_treatment_mask = A_test == 1
    test_control_mask = A_test == 0

    # Hyperparameter grid
    hp_grid = {
        'n_estimators': [100, 250, 350],
        'max_depth': [100],
        'min_samples_split': [10],
        'max_features': ['sqrt']
    }

    # Function to perform grid search
    def grid_search(X_train, Y_train, X_test, Y_test):
        best_mae = float('inf')
        best_model = None
        best_params = None

        for params in product(*hp_grid.values()):
            print(f'[trying params: {params}]')
            current_params = dict(zip(hp_grid.keys(), params))
            model = RandomForestRegressor(random_state=random_state, **current_params)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(Y_test, y_pred)

            if mae < best_mae:
                best_mae = mae
                best_model = model
                best_params = current_params

        return best_model, best_params, best_mae

    # Perform grid search for treatment and control models
    model_treatment, params_treatment, mae_treatment = grid_search(
        X_train[train_treatment_mask], Y_train[train_treatment_mask],
        X_test[test_treatment_mask], Y_test[test_treatment_mask]
    )

    model_control, params_control, mae_control = grid_search(
        X_train[train_control_mask], Y_train[train_control_mask],
        X_test[test_control_mask], Y_test[test_control_mask]
    )

    # model_treatment = LinearRegression()
    # model_control = LinearRegression()

    # model_treatment.fit(X[A==1], Y[A==1])
    # model_control.fit(X[A==0], Y[A==0])

    print(f"Best params for treatment model: {params_treatment}")
    print(f"Best params for control model: {params_control}")
    # mae_treatment = mean_absolute_error(Y[A==1], model_treatment.predict(X[A==1]))
    # mae_control = mean_absolute_error(Y[A==0], model_control.predict(X[A==0]))
    print(f"MAE for treatment model on test data: {mae_treatment:.4f}")
    print(f"MAE for control model on test data: {mae_control:.4f}")

    # Compute CATE for each sample on full data (I will cross-fit later)
    Y_treatment = model_treatment.predict(X)
    Y_control = model_control.predict(X)

    cate = np.where(A == 1, Y - Y_control, Y_treatment - Y)

    return cate

def ipw_estimator_cate(data):
    """
    Calculate the inverse probability weighting (IPW) estimator.

    Parameters:
    - data: DataFrame, observational study data

    Returns:
    - CATE over values of X 
    """

    # fit propensity score model with logistic regression 
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Prepare the data
    X = data['X'].values.reshape(-1, 1)  # Reshape for sklearn
    A = data['A'].values
    Y = data['Y'].values

    # Standardize the features (don't need to do this for X, since it's bounded 
    # between -2 and 2)
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    # Fit logistic regression model with L2 regularization
    propensity_model = LogisticRegression(random_state=41, penalty='l2', C=0.1)
    propensity_model.fit(X, A)

    # Calculate propensity scores and then compute what treatment assignment 
    # would be if we let A = 1 if propensity score is greater than 0.5 and 
    # A = 0 otherwise. Then compute the accuracy of this assignment by comparing 
    # to ground truth assignment. 
    propensity_scores = propensity_model.predict_proba(X)[:, 1]
    A_pred = np.where(propensity_scores > 0.5, 1, 0)
    accuracy = np.mean(A_pred == A)
    print(f"Accuracy of propensity score model: {accuracy:.4f}")

    # Calculate the IPW estimator 
    data['cate'] = (A / propensity_scores - (1 - A) / (1 - propensity_scores)) * Y 

    return data 

