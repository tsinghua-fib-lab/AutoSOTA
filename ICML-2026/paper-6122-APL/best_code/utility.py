import numpy as np
from sklearn.metrics import f1_score


def get_acc(phi_ik, truthfile):
    score = (phi_ik == phi_ik.max(axis=1, keepdims=True)).astype(float)
    score /= score.sum(axis=1, keepdims=True)
    return score[truthfile.item.values, truthfile.truth.values].sum() / truthfile.shape[0]


def get_macro_f1(phi_ik, truthfile):
    y_pred = np.argmax(phi_ik, axis=1)
    y_pred_test = y_pred[truthfile.item.values]
    y_true = truthfile.truth.values
    return f1_score(y_true, y_pred_test, average='macro', zero_division=0)

