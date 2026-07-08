"""Classifier two-sample test (C2ST) — the SBI benchmark metric.

Copied from sbibm (Lueckmann et al. 2021): trains an sklearn MLP to distinguish
the guided samples from reference posterior samples under 5-fold CV; the mean
accuracy is the C2ST (0.5 = indistinguishable = perfect, 1.0 = fully separable).
"""

from typing import Optional

import numpy as np
import torch


def c2st(
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> torch.Tensor:
    """Classifier-based 2-sample test returning accuracy.

    Trains classifiers with N-fold cross-validation. Scikit-learn MLPClassifier
    are used, with 2 hidden layers of 10x dim each, where dim is the
    dimensionality of the samples X and Y.
    """
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neural_network import MLPClassifier

    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * torch.randn(X.shape)
        Y += noise_scale * torch.randn(Y.shape)

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    ndim = X.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=10000,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((X, Y))
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return torch.from_numpy(np.atleast_1d(scores))
