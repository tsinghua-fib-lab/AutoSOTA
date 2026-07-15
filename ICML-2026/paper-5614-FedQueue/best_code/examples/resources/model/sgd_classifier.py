import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from appfl.algorithm.trainer.sklearn_trainer import SklearnModelWrapper


def get_sgd_classifier(
    n_features: int,
    n_classes: int,
    loss: str = "log_loss",
    alpha: float = 1e-4,
    random_state: int = 42,
    **kwargs,
):
    """
    Return a ``SklearnModelWrapper`` around a zero-initialised ``SGDClassifier``.

    A tiny dummy fit (one zero-vector per class) is performed so that ``coef_``
    and ``intercept_`` exist with the correct shape before any real training
    begins.  This gives all FL clients a common, identical starting point.

    Args:
        n_features: input feature dimension (e.g. 784 for flat MNIST).
        n_classes:  number of output classes (e.g. 10 for MNIST).
        loss:       sklearn loss name (default ``"log_loss"`` ≡ logistic regression).
        alpha:      L2 regularisation strength.
        random_state: reproducibility seed.
    """
    model = SGDClassifier(loss=loss, alpha=alpha, random_state=random_state, **kwargs)
    classes = np.arange(n_classes)
    X_dummy = np.zeros((n_classes, n_features))
    model.partial_fit(X_dummy, classes, classes=classes)
    # Zero out the weights so training starts from a neutral point.
    model.coef_[:] = 0.0
    model.intercept_[:] = 0.0
    return SklearnModelWrapper(model)


def load_sgd_classifier(checkpoint_path: str):
    """
    Load a pre-trained ``SGDClassifier`` from a joblib checkpoint and wrap it
    for use as the server's global model.

    The checkpoint must have been saved with::

        import joblib
        joblib.dump(fitted_model, "checkpoint.joblib")

    All clients will receive the checkpoint's ``coef_`` and ``intercept_`` as
    their starting point via ``get_global_model(init_model=True)``.

    Args:
        checkpoint_path: path to the ``.joblib`` file.
    """
    model = joblib.load(checkpoint_path)
    return SklearnModelWrapper(model)
