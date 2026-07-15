from collections import defaultdict
import numpy as np
from sklearn.metrics import auc, roc_curve


def sweep(prediction, labels):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, _ = roc_curve(labels, prediction)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    return fpr, tpr, auc(fpr, tpr), acc


def compute_metric(prediction, answers, sweep_fn=sweep, legend=""):

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))
    low = tpr[np.where(fpr < 0.05)[0][-1]]  # TPR@5%FPR
    print("Attack %s   AUC %.4f, TPR@5%%FPR of %.4f\n" % (legend, auc, low))
    return legend, auc, low


def fig_fpr_tpr(all_output):
    answers = []
    metric2predictions = defaultdict(list)
    for ex in all_output:
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            metric2predictions[metric].append(ex["pred"][metric])

    for metric, predictions in metric2predictions.items():
        legend, auc, low = compute_metric(predictions, answers, legend=metric)
    return legend, auc, low
