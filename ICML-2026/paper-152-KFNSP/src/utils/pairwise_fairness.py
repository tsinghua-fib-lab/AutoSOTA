#  Implementation of pairwise fairness measure based on:
#  https://aaai.org/papers/05248-pairwise-fairness-for-ranking-and-regression/
#  https://github.com/google-research/google-research/blob/master/pairwise_fairness/regression_crime.ipynb


import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch


def pairwise_fairness(y_true, y_pred, p, use_label=True):
    # use_label : True -> EqOpp
    # use_label : False -> SP
    return np.abs(
        AG(y_true, y_pred, p, ">", use_label=use_label)
        - AG(y_pred, y_true, p, "<", use_label=use_label)
    )


def AG(y_true, y_pred, p, arg, use_label=False):
    n = len(p)

    X = torch.stack([y_true, y_pred, p], dim=0).t().detach().numpy()

    # use pairwise distance for better efficiency
    K = pairwise_distances(X, X, metric=custom_pf_dist, arg=arg, use_label=use_label)

    return np.sum(K) / n**2


def custom_pf_dist(x, y, arg, use_label=False):
    # x = (y,ŷ,p)
    if arg == "<":
        auxil = x[2] < y[2]
    elif arg == ">":
        auxil = x[2] > y[2]
    else:
        raise Exception("Unknown operator (use > or <)")
    if auxil:
        if x[0] > y[0] or not use_label:
            if x[1] > y[1]:
                return 1

    return 0
