from functools import partial

import cma
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import jit
from sklearn.metrics import roc_auc_score


@jit(nopython=True)
def fast_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += 1 - y_i
        auc += y_i * nfalse
    auc /= nfalse * (n - nfalse)
    return auc


# From Abhishek's book, pg 276
class OptimizeAUC:
    """
    Class for optimizing AUC.
    This class is all you need to find best weights for
    any model and for any metric and for any types of predictions.
    With very small changes, this class can be used for optimization of
    weights in ensemble models of _any_ type of predictions
    """

    def __init__(self):
        self.coef_ = 0

    def _auc(self, coef, X, y):
        """
        This functions calulates and returns AUC.
        :param coef: coef list, of the same length as number of models
        :param X: predictions, in this case a 2d array
        :param y: targets, in our case binary 1d array
        """
        x_coef = X * coef
        predictions = np.sum(x_coef, axis=1)

        #         auc_score = roc_auc_score(y, predictions)
        auc_score = fast_auc(y.values, predictions.values)
        return -1.0 * auc_score

    def fit(self, X, y):
        loss_partial = partial(self._auc, X=X, y=y)

        # dirichlet distribution. you can use any distribution you want
        # to initialize the coefficients
        # we want the coefficients to sum to 1
        # initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1)
        initial_coef = np.ones(X.shape[1]) / X.shape[1]

        es = cma.CMAEvolutionStrategy(
            initial_coef,
            0.5,
            {"seed": 2021, "bounds": [None, None], "maxfevals": 2500, "tolfun": 1e-6},
        )
        es.optimize(loss_partial)
        self.coef_ = es.result.xbest

    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions


def scorer_cma(oofs, col_names, parallel=False):
    def score_fold(fold):
        trn_df = oofs.query(f"fold != {fold}")
        val_df = oofs.query(f"fold == {fold}")

        bnds = [(-1, 1) for _ in range(len(col_names))]

        opt = OptimizeAUC()
        opt.fit(trn_df[col_names], trn_df["target"])
        y_pred = opt.predict(val_df[col_names])
        return roc_auc_score(val_df["target"], y_pred)

    if parallel:
        fold_scores = Parallel(n_jobs=5)(delayed(score_fold)(fold) for fold in range(5))
    else:
        fold_scores = [score_fold(fold) for fold in range(5)]

    return np.mean(fold_scores)


def pruning(oofs, candidates):
    history = []
    score = scorer_cma(oofs, candidates)
    print(f"Initial score {score:0.6f}")

    while len(candidates) > 2:
        trial_candidates = [
            candidates[:i] + candidates[i + 1 :] for i, _ in enumerate(candidates)
        ]
        scores = Parallel(n_jobs=32)(
            delayed(scorer_cma)(oofs, tc) for tc in trial_candidates
        )
        removed = candidates[np.argmax(scores)]
        del candidates[np.argmax(scores)]
        score = scorer_cma(oofs, candidates, True)
        history.append(
            {
                "models": candidates.copy(),
                "score": score,
                "removed": removed,
            }
        )
        print(
            f"{len(history)} New score {score:0.6f}",
            f"Removed {removed}. {len(candidates)} models",
        )

    history = pd.DataFrame(history)
    history.to_csv("pruning_cma-es.csv", index=False)
    print(history.tail(40))
