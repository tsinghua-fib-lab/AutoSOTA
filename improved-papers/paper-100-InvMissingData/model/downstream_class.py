import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def svm_classification(X, y, gamma='scale', score_scale="mean"):
    """
    Evaluates SVM classification accuracy using 5-fold cross-validation on a dataset loaded from a CSV file.
    Repeats the process 10 times with different random seeds to assess the stability of the results.

    Parameters:
    - csv_path: str, path to the CSV file containing the dataset.
    - gamma: str, the gamma setting for the SVM ('auto' or 'scale').

    Returns:
    - mean_accuracy: float, the mean accuracy across all runs.
    - std_accuracy: float, the standard deviation of the accuracies across all runs.
    """
    # Read the dataset
    # data = pd.read_csv(csv_path)

    # Assuming the last column is the target variable
    # X = data.iloc[:, :-1]  # Features
    # y = data.iloc[:, -1]   # Target labels

    # Standardize features
    scaler = StandardScaler()
    X_scaled = X # scaler.fit_transform(X)

    # Initialize the SVM with RBF kernel
    svm_model = SVC(kernel='rbf', gamma=gamma)

    # List to store results of each run
    accuracy_results_mean = []
    accuracy_results_min = []
    accuracy_results_max = []

    # Repeat the cross-validation process 10 times with different random seeds
    for seed in range(6):
        # Configure the random state for reproducibility
        np.random.seed(seed)

        # Perform 5-fold cross-validation
        scores = cross_val_score(svm_model, X_scaled, y, cv=5, scoring='accuracy')

        # Store the mean accuracy of the 5-fold cross-validation

        accuracy_results_mean.append(scores.mean())

        accuracy_results_min.append(scores.min())

        accuracy_results_max.append(scores.max())

    # Calculate and report the mean and standard deviation of accuracies
    mean_accuracy_mean, std_accuracy_mean = np.mean(accuracy_results_mean), np.std(accuracy_results_mean)
    mean_accuracy_max, std_accuracy_max = np.mean(accuracy_results_max), np.std(accuracy_results_max)
    mean_accuracy_min, std_accuracy_min = np.mean(accuracy_results_min), np.std(accuracy_results_min)


    return mean_accuracy_mean, std_accuracy_mean, mean_accuracy_max, std_accuracy_max, mean_accuracy_min, std_accuracy_min

# # Example usage:
# file_path = 'path_to_your_csv_file.csv'  # Replace with your CSV file path
# mean_acc, std_acc = evaluate_svm_classification(file_path, gamma='scale')
# print(f'Mean Accuracy: {mean_acc:.4f}')
# print(f'Standard Deviation of Accuracy: {std_acc:.4f}')
