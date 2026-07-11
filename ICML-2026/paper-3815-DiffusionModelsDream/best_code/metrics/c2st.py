from typing import Optional
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
# Adapted from https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/metrics/c2st.py
from sklearn.exceptions import ConvergenceWarning
import warnings
 
def check_c2st_convergence(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = 1,
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> bool:
    """
    Performs a single test fit to check if the MLPClassifier converges.

    This function uses the *exact same* preprocessing and model
    configuration as the main c2st function, but fits on the *entire*
    dataset once to check for ConvergenceWarnings.

    Args:
        X: Sample 1, shape (n_x, d)
        Y: Sample 2, shape (n_y, d)
        seed: Seed for sklearn
        z_score: Z-scoring using X's mean/std (per feature)
        noise_scale: If passed, adds Gaussian noise

    Returns:
        True if the model converged, False otherwise.
    """
    print("--- Starting Convergence Test ---")
    
    # --- 1. Preprocessing (Copied from c2st) ---
    if z_score:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * np.random.randn(*X.shape)
        Y += noise_scale * np.random.randn(*Y.shape)

    ndim = X.shape[1]

    # --- 2. Model Definition (Copied from c2st) ---
    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=10000,  # This is the parameter being tested
        solver="adam",
        random_state=seed,
    )

    # --- 3. Data Prep (Copied from c2st) ---
    data = np.concatenate((X, Y), axis=0)
    target = np.concatenate(
        (np.zeros((X.shape[0],)), np.ones((Y.shape[0],))), axis=0
    )

    # --- 4. The Test Fit ---
    converged = True
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning) 

        try:
            clf.fit(data, target)
        except Exception as e:
            print(f"Test fit FAILED with an error: {e}")
            converged = False
            return converged # Exit early

        # Check if any ConvergenceWarnings were caught
        if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
            print(f"Test fit DID NOT converge within {clf.max_iter} iterations.")
            print("   (Received ConvergenceWarning)")
            converged = False
        else:
            print(f"Test fit converged successfully in {clf.n_iter_} iterations.")
    
    print("--- End of Convergence Test ---\n")
    return converged

def c2st(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> np.ndarray:
    """Classifier-based 2-sample test returning accuracy (as a 1D np.ndarray)

    Trains classifiers with N-fold cross-validation. Scikit-learn MLPClassifier is
    used, with 2 hidden layers of 10×dim each, where dim is the dimensionality of
    the samples X and Y.

    Args:
        X: Sample 1, shape (n_x, d)
        Y: Sample 2, shape (n_y, d)
        seed: Seed for sklearn
        n_folds: Number of folds
        scoring: Scikit-learn scoring string (e.g., "accuracy", "roc_auc")
        z_score: Z-scoring using X's mean/std (per feature)
        noise_scale: If passed, adds Gaussian noise with std=noise_scale to samples

    Returns:
        A 1D np.ndarray with a single float32 value: the mean cross-val score.
    """
    
    # import pdb; pdb.set_trace()
    if z_score:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)  # matches torch.std default behavior closely
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * np.random.randn(*X.shape)
        Y += noise_scale * np.random.randn(*Y.shape)

    ndim = X.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=10000,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((X, Y), axis=0)
    target = np.concatenate(
        (np.zeros((X.shape[0],)), np.ones((Y.shape[0],))), axis=0
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return np.atleast_1d(scores)


def c2st_auc(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = 1,
    n_folds: int = 5,
    z_score: bool = True,
    noise_scale: Optional[float] = None,
) -> np.ndarray:
    """Classifier-based 2-sample test returning ROC AUC (as a 1D np.ndarray)."""
    return c2st(
        X,
        Y,
        seed=seed,
        n_folds=n_folds,
        scoring="roc_auc",
        z_score=z_score,
        noise_scale=noise_scale,
    )
