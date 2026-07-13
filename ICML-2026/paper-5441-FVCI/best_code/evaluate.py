from sklearn import metrics
import numpy as np


def calculate_AUROC(score_matrix, ground_truth, ignore_self_links=False):
    """
    Calculates the Area Under Receiver Operating Characteristic Curve

    Args:
        score_matrix: ndarray
            ndarray containing scores for every potential causal link
        ground_truth: ndarray
            binary ndarrray containing ground truth causal links
        ignore_self_links: bool
            indicates whether we should ignore self-links (i.e. principal diagonal) in AUROC calculation
    Returns:
        aucroc: float
            Area Under Receiver Operating Characteristic Curve
    """
    score_matrix_flattened = score_matrix.flatten()
    ground_truth_flattened = ground_truth.flatten()
    if ignore_self_links:
        indeces = np.arange(0, score_matrix_flattened.shape[0], int(np.round(np.sqrt(score_matrix_flattened.shape[0]))) + 1)
        score_matrix_flattened = np.delete(score_matrix_flattened, indeces)
        ground_truth_flattened = np.delete(ground_truth_flattened, indeces)
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth_flattened, score_matrix_flattened)
    aucroc = metrics.auc(fpr, tpr)
    return aucroc


def dream_file_to_causal_matrix(file):
    """
    Transforms the ground truth text file from the DREAM3 data set to a causal matrix

    Args:
        file: string
            Path to text file containing the ground truth of a DREAM3 experiment
    Returns:
        causal_matrix: ndarray
            Causal matrix for the DREAM3 experiment, where element causal_matrix[i,j] == 1 indicates a causal link i -> j
    """
    causal_matrix = np.zeros((100, 100))
    with open(file) as f:
        for line in f:
            elements = line.split("\t")
            cause = elements[0].split("G")[1]
            effect = elements[1].split("G")[1]
            if elements[2] == '1\n':
                causal_matrix[int(cause) - 1, int(effect) - 1] = 1
    return causal_matrix
