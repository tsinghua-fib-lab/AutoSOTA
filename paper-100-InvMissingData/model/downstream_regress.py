import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error

def svm_regress(X, y, gamma='scale'):
    """
    Evaluates SVM regression performance using 5-fold cross-validation on a dataset.
    Repeats the process 10 times with different random seeds to assess the stability of the results.
    Returns both mean squared error (MSE) and R² metrics.

    Parameters:
    - X: DataFrame, input features.
    - y: Series, target variable.
    - gamma: str, the gamma setting for the SVM ('auto' or 'scale').

    Returns:
    - mean_mse: float, the mean of the mean squared errors across all runs.
    - std_mse: float, the standard deviation of the mean squared errors across all runs.
    - mean_r2: float, the mean of the R² values across all runs.
    - std_r2: float, the standard deviation of the R² values across all runs.
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize the SVM with RBF kernel
    svm_model = SVR(kernel='rbf', gamma=gamma)

    # List to store results of each run
    mae_results_mean, mae_results_max, mae_results_min = [], [], []
    r2_results_mean, r2_results_max, r2_results_min = [], [], []
    mse_results_mean, mse_results_max, mse_results_min = [], [], []
    # Repeat the cross-validation process 10 times with different random seeds
    for seed in range(10):
        # Configure the random state for reproducibility
        np.random.seed(seed)

        # Perform 5-fold cross-validation
        scores = cross_validate(svm_model, X_scaled, y, cv=5, scoring={'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
                                                                       'MSE': make_scorer(mean_squared_error, greater_is_better=False),
                                                                       'R2': 'r2'}, return_train_score=False)

        # Store the mean MSE and R2 of the 5-fold cross-validation
        mae_results_mean.append(-1.0 * np.mean(scores['test_MAE']))
        mae_results_max.append(-1.0 * np.min(scores['test_MAE']))
        mae_results_min.append(-1.0 * np.max(scores['test_MAE']))

        mse_results_mean.append(-1.0 * np.mean(scores['test_MSE']))
        mse_results_max.append(-1.0 * np.min(scores['test_MSE']))
        mse_results_min.append(-1.0 * np.max(scores['test_MSE']))

        r2_results_mean.append(np.mean(scores['test_R2']))
        r2_results_max.append(np.max(scores['test_R2']))
        r2_results_min.append(np.min(scores['test_R2']))

        r2_results_mean.append(np.mean(scores['test_R2']))

    # Calculate and report the mean and standard deviation of MSEs and R²
    mean_mae = np.mean(mae_results_mean)
    min_mae = np.min(mae_results_min)
    max_mae = np.max(mae_results_max)
    std_mae = np.std(mae_results_mean)

    mean_mse = np.mean(mse_results_mean)
    min_mse = np.min(mse_results_min)
    max_mse = np.max(mse_results_max)
    std_mse = np.std(mse_results_mean)

    mean_r2 = np.mean(r2_results_mean)
    min_r2 = np.min(r2_results_min)
    max_r2 = np.max(r2_results_max)
    std_r2 = np.std(r2_results_mean)



    return mean_mae, min_mae, max_mae, std_mae, mean_r2, min_r2, max_r2, std_r2, mean_mse, min_mse, max_mse, std_mse

# Example usage:
# X, y = load_your_data_somehow()  # Load or define your feature matrix X and target variable y
# mean_mse, std_mse, mean_r2, std_r2 = svm_regression_metrics(X, y, gamma='scale')
# print(f'Mean MSE: {mean_mse:.4f}, Standard Deviation of MSE: {std_mse:.4f}')
# print(f'Mean R²: {mean_r2:.4f}, Standard Deviation of R²: {std_r2:.4f}')
