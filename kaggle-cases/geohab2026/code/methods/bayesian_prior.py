"""
Bayesian prior adjustment for classifier probability outputs.

When a model is trained on data whose class distribution differs from the
target (test) distribution, its predict_proba outputs are calibrated to
training frequencies. This module applies a post-hoc correction using
Bayes' theorem to shift decision boundaries toward the target distribution
without retraining.

The mechanism is identical to interpreting a medical diagnostic test: the
test's sensitivity doesn't change, but the posterior probability depends on
disease prevalence (the prior). Here, the model's feature-based likelihood
is trusted; only the base-rate assumption is corrected.

Reference
---------
The general framework is standard Bayesian decision theory. For the
specific application to habitat classification with known test priors
derived from a source paper's published confusion matrix, see the
accompanying writeup.
"""

import numpy as np


def adjust_priors(proba_raw, train_prior, test_prior):
    """
    Adjust classifier probability outputs for train/test prior mismatch.

    Parameters
    ----------
    proba_raw : np.ndarray, shape (n_samples, n_classes)
        Raw predict_proba output from the trained model, calibrated
        to training class frequencies.
    train_prior : np.ndarray, shape (n_classes,)
        Class frequencies in the training set. Must sum to 1.
    test_prior : np.ndarray, shape (n_classes,)
        Estimated class frequencies in the target (test) set. Must sum to 1.

    Returns
    -------
    proba_adj : np.ndarray, shape (n_samples, n_classes)
        Adjusted probabilities, renormalized to sum to 1 per sample.
    preds_adj : np.ndarray, shape (n_samples,)
        Argmax class indices after adjustment.
    flips : np.ndarray, shape (n_samples,), dtype bool
        True where the adjustment changed the predicted class.

    Notes
    -----
    The adjustment multiplies each class probability by the ratio
    test_prior[c] / train_prior[c], then renormalizes:

        P_adj[c] = P_raw[c] * (test_prior[c] / train_prior[c])
        P_adj   /= sum(P_adj)

    Classes enriched in the test set (ratio > 1) get stretched;
    classes diluted (ratio < 1) get shrunk. The model itself is never
    retrained. Only the decision boundary moves.

    This is a no-op when train_prior == test_prior.
    """
    train_prior = np.asarray(train_prior, dtype=np.float64)
    test_prior = np.asarray(test_prior, dtype=np.float64)
    proba_raw = np.asarray(proba_raw, dtype=np.float64)

    if train_prior.shape != test_prior.shape:
        raise ValueError(
            f"Prior shapes must match: train {train_prior.shape} "
            f"vs test {test_prior.shape}"
        )
    if proba_raw.shape[1] != train_prior.shape[0]:
        raise ValueError(
            f"Number of classes in proba_raw ({proba_raw.shape[1]}) "
            f"does not match prior length ({train_prior.shape[0]})"
        )
    if np.any(train_prior <= 0):
        raise ValueError("All train_prior entries must be > 0")

    # Prior ratio: how much each class is enriched or diluted in the test set
    ratio = test_prior / train_prior

    # Rescale
    proba_adj = proba_raw * ratio[np.newaxis, :]

    # Renormalize each sample to sum to 1
    row_sums = proba_adj.sum(axis=1, keepdims=True)
    proba_adj = proba_adj / row_sums

    # Argmax before and after
    preds_raw = np.argmax(proba_raw, axis=1)
    preds_adj = np.argmax(proba_adj, axis=1)
    flips = preds_raw != preds_adj

    return proba_adj, preds_adj, flips


def summarize_flips(preds_raw, preds_adj, class_names=None):
    """
    Print a summary of which predictions changed after prior adjustment.

    Parameters
    ----------
    preds_raw : array-like, shape (n_samples,)
        Original argmax predictions (integer class indices).
    preds_adj : array-like, shape (n_samples,)
        Adjusted argmax predictions.
    class_names : list of str, optional
        Human-readable class names. If None, uses integer indices.
    """
    preds_raw = np.asarray(preds_raw)
    preds_adj = np.asarray(preds_adj)
    flips = preds_raw != preds_adj
    n_flips = flips.sum()
    n_total = len(preds_raw)

    print(f"Flips: {n_flips} / {n_total} predictions changed")

    if n_flips > 0:
        flip_indices = np.where(flips)[0]
        for idx in flip_indices:
            old = preds_raw[idx]
            new = preds_adj[idx]
            if class_names is not None:
                old = class_names[old]
                new = class_names[new]
            print(f"  Sample {idx}: {old} -> {new}")
