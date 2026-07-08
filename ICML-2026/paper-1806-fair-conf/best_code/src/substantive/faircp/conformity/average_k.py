import torch
import numpy as np

# ================================================================================================
# Note: The below code has been borrowed and adapted from the paper "Lorieul et al., 2021; Classification Under Ambiguity"

def calibrate_avg_k(avg_set_size, y_score):
    """
    Find the quantile value that gives the desired average set size
    based on the softmax scores of a calibration dataset.

    Parameters:
    avg_set_size (float): The target average set size.
    y_score (numpy.ndarray): The array of softmax outputs.

    Returns:
    float: The calibrated quantile.
    """

    n_classes = y_score.shape[-1]
    p = 1 - (avg_set_size / n_classes)
    q = np.quantile(y_score, q=p, method="lower")

    return q


def get_average_k_sets(logits_calib, logits_test, avg_set_size):
    """
    Generate average-k sets by calibrating on a calibration set for average set size k,
    then predicting the classes above a threshold.

    Parameters:
    logits_calib (torch.Tensor): Calibration logits obtained from the model.
    logits_test (torch.Tensor): Test logits obtained from the model.
    avg_set_size (float): The target average set size.

    Returns:
    List[np.ndarray]: Average-k sets for the test set.
    """

    # Temperature scaling is unnecessary
    y_calib = torch.softmax(logits_calib, dim=1).cpu().numpy()
    quantile = calibrate_avg_k(avg_set_size, y_calib)

    y_test = torch.softmax(logits_test, dim=1).cpu().numpy()
    n_samples = y_test.shape[0]

    avg_k_set = (y_test > quantile)
    ties_set = (y_test == quantile)

    labels_to_be_assigned = avg_set_size * n_samples - np.sum(avg_k_set)
    labels_to_be_assigned = int(np.ceil(labels_to_be_assigned))
    # If we have not overshot the desired set size, add in elements equal to the quantile
    if labels_to_be_assigned > 0:
        i, j = np.where(ties_set)
        # Select random eligible elements
        perm = np.random.permutation(len(i))
        i, j = i[perm], j[perm]
        i, j = i[:labels_to_be_assigned], j[:labels_to_be_assigned]
        avg_k_set[i, j] = True

    # Convert the prediction set to labels
    avg_k_set_labels = [
        np.array([idx for idx, val in enumerate(row) if val == 1])
        for row in avg_k_set
    ]

    return avg_k_set_labels
