import numpy as np


def imbalance_degree(lst):
    """
    Calculate the proportions of each class and the proportions for a balanced distribution and return the euclidean distance between the two distributions.

    Parameters:
    - lst: List of classes.

    Returns:
        Euclidean distance between the actual and balanced distributions.
    """
    # Calculate the count of each unique element in the list
    counts = {}
    for elem in lst:
        counts[elem] = counts.get(elem, 0) + 1

    # Calculate the total number of elements in the list
    total_elements = len(lst)

    # Calculate the actual class proportions
    actual_proportions = np.array(
        [counts[elem] / total_elements for elem in sorted(counts.keys())]
    )

    # Calculate the proportions for a balanced distribution
    num_classes = len(counts)
    balanced_proportions = np.array([1 / num_classes] * num_classes)

    # Calculate the Euclidean distance
    euclidean_distance = np.linalg.norm(actual_proportions - balanced_proportions)

    return euclidean_distance
