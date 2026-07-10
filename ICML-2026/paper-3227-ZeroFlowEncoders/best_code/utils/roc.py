import numpy as np

def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    """
    Compute ROC curve (FPR, TPR) by sorting scores descending.
    (Deterministic given y_true and y_score.)
    """
    y_true = y_true.flatten()
    y_score = y_score.flatten()
    
    y_true = y_true.astype(np.int64)
    order = np.argsort(-y_score)  # descending
    y_sorted = y_true[order]

    P = int(y_sorted.sum())
    N = int(y_sorted.size - P)

    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]

    for yt in y_sorted:
        if yt == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / P)
        fpr.append(fp / N)

    return np.asarray(fpr, dtype=np.float64), np.asarray(tpr, dtype=np.float64)

def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))