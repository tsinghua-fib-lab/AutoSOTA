import numpy as np
import re
from typing import List

## For LTT
from scipy.stats import binom
# from confseq import betting


"""Util functions for gCRC medical QA experiments"""


def remove_specific_leading_chars(input_string):
    # Remove leading commas
    input_string = re.sub(r"^,+", "", input_string)
    # Remove numbers followed by a comma
    return re.sub(r"^\d+,+", "", input_string)


## LTT util functions from: https://github.com/aangelopoulos/ltt/blob/main/core/bounds.py
def h1(y, mu):
    return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))


def hb_p_value(r_hat, n, alpha):
    bentkus_p_value = np.e * binom.cdf(np.ceil(n * r_hat), n, alpha)

    def h1(y, mu):
        with np.errstate(divide="ignore"):
            return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))

    hoeffding_p_value = np.exp(-n * h1(min(r_hat, alpha), alpha))
    return min(bentkus_p_value, hoeffding_p_value)


def split_dataset(dataset, rng, train_frac=0.8, train_num=None):
    x, y = dataset
    ind = np.arange(len(x))
    rng.shuffle(ind)
    if train_num is None:
        ## Providing train_num as argument overrules train_frac (else train_frac determines train_num)
        train_num = int(train_frac * len(x))
    train_ind = ind[0:train_num]
    calib_ind = ind[train_num:]

    x_train = [x[i] for i in train_ind]
    y_train = [y[i] for i in train_ind]

    x_calib = [x[i] for i in calib_ind]
    y_calib = [y[i] for i in calib_ind]

    return (x_train, y_train), (x_calib, y_calib), train_ind, calib_ind


## Recall
def get_frac_true_claims_retained(claim_scores, annotations, thresholds):
    recall = []
    for cs, a, t in zip(claim_scores, annotations, thresholds):
        frac = np.sum((cs > t) & a) / np.sum(a) if np.sum(a) > 0 else 0
        recall.append(frac)
    return recall


# def get_retained_claims(claim_scores, thresholds):
#     claims_retained = []
#     for cs, t in zip(claim_scores, thresholds):
#         claims_retained.append(np.mean(cs > t))
#     return claims_retained

# def get_retained_claim_indices(claim_scores, thresholds):
#     claims_retained = []
#     for cs, t in zip(claim_scores, thresholds):
#         claims_retained.append(np.where(cs > t)[0])
#     return claims_retained


def get_taus_grid_from_data(
    claim_scores: List,  ## List of arrays of subclaim scores
):
    taus_set = []
    for i, cs in enumerate(claim_scores):
        taus_set.extend(cs)

    return np.array(taus_set)


def loss_factuality(
    claim_scores: List[np.ndarray],  ## Float point scores
    annotations: List[np.ndarray],  ## Boolean annotations
    tau: float,
    min_score: int = 0,
):
    ## Returns 1 if there is some included claim (with score >= tau) that is False
    annotations_included = annotations[claim_scores >= tau]
    return (
        int(max(~annotations_included)) if len(annotations_included) > 0 else min_score
    )
