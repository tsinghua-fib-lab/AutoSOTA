import numpy as np


def mae(y_true, y_pred):
    """
    y_true and y_pred are both of type np.ndarray
    y_true (N, d) where N is the size of the validation set, and d is the dimension of the label
    y_pred (N, D) where N is the size of the validation set, and D is the output dimension of the ML model
    """
    return np.mean(np.abs(y_true - y_pred))
