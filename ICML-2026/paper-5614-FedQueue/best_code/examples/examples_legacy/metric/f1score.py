import numpy as np


def f1score(y_true, y_pred, average="macro"):
    if len(y_pred.shape) == 1:
        y_pred = np.round(y_pred)
    else:
        y_pred = y_pred.argmax(axis=1)
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))

    f1_scores = []

    for class_label in classes:
        # Binary classification for the current class vs. the rest
        binary_y_true = (y_true == class_label).astype(int)
        binary_y_pred = (y_pred == class_label).astype(int)

        # Calculate F1 score for the current class
        tp = np.sum((binary_y_true == 1) & (binary_y_pred == 1))
        fp = np.sum((binary_y_true == 0) & (binary_y_pred == 1))
        fn = np.sum((binary_y_true == 1) & (binary_y_pred == 0))

        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)

        f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)

        f1_scores.append(f1)

    # Calculate the overall F1 score based on the averaging strategy
    if average == "micro":
        overall_f1 = np.sum(f1_scores) / max(len(f1_scores), 1e-8)
    elif average == "macro":
        overall_f1 = np.mean(f1_scores)
    elif average == "weighted":
        class_counts = [np.sum(y_true == class_label) for class_label in classes]
        overall_f1 = np.average(f1_scores, weights=class_counts)
    else:
        raise ValueError(
            "Invalid 'average' parameter. Use 'micro', 'macro', or 'weighted'."
        )

    return overall_f1
