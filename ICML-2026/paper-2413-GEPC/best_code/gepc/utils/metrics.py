# gepc/utils/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score


def auroc_ood_high(id_scores, ood_scores) -> float:
    """AUROC where OOD is the positive class (label=1) and higher score means more OOD."""
    id_scores = np.asarray(id_scores).reshape(-1)
    ood_scores = np.asarray(ood_scores).reshape(-1)
    y = np.concatenate([
        np.zeros_like(id_scores, dtype=np.int32),
        np.ones_like(ood_scores, dtype=np.int32),
    ])
    s = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y, s))


def fpr_at_tpr(id_scores, ood_scores, tpr_target=0.95, id_higher=True) -> float:
    """
    FPR at a target TPR (positive = ID). If id_higher=False, scores are negated first.
    """
    id_scores = np.asarray(id_scores).ravel()
    ood_scores = np.asarray(ood_scores).ravel()
    if not id_higher:
        id_scores = -id_scores
        ood_scores = -ood_scores

    thr = np.quantile(id_scores, 1.0 - float(tpr_target))
    fpr = (ood_scores >= thr).mean()
    return float(fpr)
