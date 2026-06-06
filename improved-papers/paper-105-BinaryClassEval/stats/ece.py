import numpy as np

def expected_calibration_error(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE) with equal-width bins.

    Args:
        y_true:     Array of shape (n_samples,) with integer class labels.
        y_pred_probs:
                    Array of shape (n_samples, n_classes) with predicted probabilities.
        n_bins:     Number of bins to partition [0, 1] into.

    Returns:
        The scalar ECE score.
    """
    assert len(y_true.shape) == 1, y_true.shape
    assert len(y_pred_probs.shape) == 2, y_pred_probs.shape
    assert y_true.shape[0] == y_pred_probs.shape[0]
    # 1) Get per-sample confidence and correctness
    confidences = y_pred_probs.max(axis=1)
    correct = (y_pred_probs.argmax(axis=1) == y_true).astype(float)

    # 2) Assign each sample to a bin [0,1)→0 ... [(n_bins-1)/n_bins,1]→n_bins-1
    #    np.clip ensures that a confidence of exactly 1.0 lands in the last bin.
    bin_indices = np.minimum((confidences * n_bins).astype(int), n_bins - 1)

    # 3) Count how many samples per bin
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    # 4) Sum of confidences & correctness per bin
    sum_conf = np.bincount(bin_indices, weights=confidences, minlength=n_bins)
    sum_correct = np.bincount(bin_indices, weights=correct, minlength=n_bins)

    # 5) Avoid division by zero for empty bins
    nonempty = bin_counts > 0

    # 6) Compute accuracy and average confidence in each non-empty bin
    accuracy_per_bin = np.zeros(n_bins)
    avg_conf_per_bin = np.zeros(n_bins)
    accuracy_per_bin[nonempty] = sum_correct[nonempty] / bin_counts[nonempty]
    avg_conf_per_bin[nonempty] = sum_conf[nonempty] / bin_counts[nonempty]

    # 7) Bin weights = fraction of samples in each bin
    prop_per_bin = bin_counts / len(confidences)

    # 8) ECE is the weighted average absolute gap
    ece = np.sum(prop_per_bin[nonempty] * np.abs(accuracy_per_bin[nonempty] - avg_conf_per_bin[nonempty]))
    return float(ece)


# --- Example Usage ---
if __name__ == "__main__":
    np.random.seed(42)
    n_samples, n_classes = 1000, 10

    # Dummy true labels and over‑confident random softmax preds
    y = np.random.randint(0, n_classes, size=n_samples)
    p = np.random.rand(n_samples, n_classes)
    p /= p.sum(axis=1, keepdims=True)

    print("ECE (15 bins):", expected_calibration_error(y, p, n_bins=15))