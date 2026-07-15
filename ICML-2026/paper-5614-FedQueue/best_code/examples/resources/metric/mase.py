import numpy as np


def accuracy(y_true, y_pred):
    """
    y_true and y_pred are both of type np.ndarray
    y_true (N, d) where N is the size of the validation set, and d is the dimension of the label
    y_pred (N, D) where N is the size of the validation set, and D is the output dimension of the ML model
    """
    y_true, y_pred = y_true.squeeze(), y_pred.squeeze()
    min_number = 1e-8  # floor for denominator to prevent inf losses

    numerator = np.mean(np.abs(y_true - y_pred))
    denominator = np.mean(np.abs(np.diff(y_pred, n=1)))
    denominator = np.maximum(
        denominator, np.multiply(np.ones_like(denominator), min_number)
    )
    return np.divide(numerator, denominator)
