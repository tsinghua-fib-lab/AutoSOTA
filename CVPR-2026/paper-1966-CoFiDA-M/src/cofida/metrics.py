import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score


def compute_binary_metrics(y_true, y_prob, threshold: float):
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    y_pred_np = (y_prob_np >= threshold).astype(int)
    try:
        auroc = roc_auc_score(y_true_np, y_prob_np)
    except Exception:
        auroc = float("nan")
    return {
        "acc": accuracy_score(y_true_np, y_pred_np),
        "bacc": balanced_accuracy_score(y_true_np, y_pred_np),
        "auroc": auroc,
        "y_true": y_true_np,
        "y_pred": y_pred_np,
        "y_prob": y_prob_np,
        "cm": confusion_matrix(y_true_np, y_pred_np, labels=[0, 1]),
    }


def youden_threshold(y_true, y_prob):
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_threshold = 0.5
    best_bacc = -1.0
    best_acc = -1.0
    best_cm = None
    for threshold in thresholds:
        y_pred = (y_prob_np >= threshold).astype(int)
        bacc = balanced_accuracy_score(y_true_np, y_pred)
        if bacc > best_bacc:
            best_bacc = bacc
            best_threshold = float(threshold)
            best_acc = accuracy_score(y_true_np, y_pred)
            best_cm = confusion_matrix(y_true_np, y_pred, labels=[0, 1])
    return {
        "threshold": best_threshold,
        "bacc": best_bacc,
        "acc": best_acc,
        "cm": best_cm,
    }
