# stdlib
import copy
from typing import Any, Tuple

# third party
import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold

from catenets.experiment_utils.torch_metrics import abs_error_ATE, sqrt_PEHE


def generate_score(metric: np.ndarray) -> Tuple[float, float]:
    percentile_val = 1.96
    return (np.mean(metric), percentile_val * np.std(metric) / np.sqrt(len(metric)))


def print_score(score: Tuple[float, float]) -> str:
    return str(round(score[0], 4)) + " +/- " + str(round(score[1], 4))


def evaluate_treatments_model(
    estimator: Any,
    X: torch.Tensor,
    Y: torch.Tensor,
    Y_full: torch.Tensor,
    W: torch.Tensor,
    n_folds: int = 3,
    seed: int = 0,
) -> dict:
    metric_pehe = np.zeros(n_folds)
    metric_ate = np.zeros(n_folds)

    indx = 0
    if len(np.unique(Y)) == 2:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in skf.split(X, Y):

        X_train = X[train_index]
        Y_train = Y[train_index]
        W_train = W[train_index]

        X_test = X[test_index]
        Y_full_test = Y_full[test_index]

        model = copy.deepcopy(estimator)
        model.fit(X_train, Y_train, W_train)

        try:
            te_pred = model.predict(X_test).detach().cpu().numpy()
        except BaseException:
            te_pred = np.asarray(model.predict(X_test))

        metric_ate[indx] = abs_error_ATE(Y_full_test, te_pred)
        metric_pehe[indx] = sqrt_PEHE(Y_full_test, te_pred)
        indx += 1

    output_pehe = generate_score(metric_pehe)
    output_ate = generate_score(metric_ate)

    return {
        "raw": {
            "pehe": output_pehe,
            "ate": output_ate,
        },
        "str": {
            "pehe": print_score(output_pehe),
            "ate": print_score(output_ate),
        },
    }
