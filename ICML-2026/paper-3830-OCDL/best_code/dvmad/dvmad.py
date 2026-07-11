"""PyOD-compatible wrapper for DVM-AD.

The :class:`DVMAD` class subclasses :class:`pyod.models.base.BaseDetector`
so it plugs directly into PyOD pipelines, benchmarks, and ensembles.

If PyOD is not installed, the import of this module will raise
``ImportError``. Use :class:`dvmad.core.DVMADCore` for a NumPy-only API.
"""

from __future__ import annotations

import numpy as np

try:
    from pyod.models.base import BaseDetector
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "DVMAD requires `pyod`. Install it with `pip install pyod`, or use "
        "`dvmad.core.DVMADCore` for a PyOD-free interface."
    ) from exc

from sklearn.utils.validation import check_array, check_is_fitted

from .core import DVMADCore


class DVMAD(BaseDetector):
    """PyOD-compatible Discriminant Vector Machine for Anomaly Detection.

    DVM-AD learns a closed-form spectral projection from the training data
    augmented with a single artificial reference point, then scores test
    points by their nearest-neighbour distance in the projected space.

    Parameters
    ----------
    contamination : float, default=0.1
        Expected proportion of outliers in the data. Used to set the
        decision threshold ``threshold_`` (see PyOD ``BaseDetector``).
    mode : {"min", "max", "both"}, default="both"
        Two-tailed eigenvector selection.
    eps : float, default=0.1
        Tail threshold in (0, 0.5).
    artificial_mode : {"max", "min", "farthest"}, default="max"
        Reference point construction strategy.
    use_faiss : bool, default=True
        Use FAISS for exact NN search when available.
    faiss_float32 : bool, default=True
        Cast to float32 for FAISS.
    chunk_size : int, default=4096
        Chunk size for the NumPy NN fallback.

    Attributes
    ----------
    decision_scores_ : ndarray of shape (n_samples,)
        Anomaly scores on the training data. Higher = more anomalous.
    threshold_ : float
        Decision threshold derived from ``contamination``.
    labels_ : ndarray of shape (n_samples,)
        Binary labels for the training data (0 = inlier, 1 = outlier).
    model_ : DVMADCore
        Underlying NumPy implementation.

    Examples
    --------
    >>> from dvmad import DVMAD
    >>> from pyod.utils.data import generate_data
    >>> X_train, X_test, y_train, y_test = generate_data(
    ...     n_train=200, n_test=100, n_features=5, contamination=0.1
    ... )
    >>> clf = DVMAD(contamination=0.1).fit(X_train)
    >>> scores = clf.decision_function(X_test)
    >>> labels = clf.predict(X_test)
    """

    def __init__(
        self,
        contamination: float = 0.1,
        mode: str = "both",
        eps: float = 0.1,
        artificial_mode: str = "max",
        use_faiss: bool = True,
        faiss_float32: bool = True,
        chunk_size: int = 4096,
    ):
        super().__init__(contamination=contamination)
        self.mode = mode
        self.eps = eps
        self.artificial_mode = artificial_mode
        self.use_faiss = use_faiss
        self.faiss_float32 = faiss_float32
        self.chunk_size = chunk_size

    # ------------------------------------------------------------------
    # PyOD API
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """Fit the detector. Labels ``y`` are ignored (one-class)."""
        X = check_array(X)
        self._set_n_classes(y)

        self.model_ = DVMADCore(
            mode=self.mode,
            eps=self.eps,
            artificial_mode=self.artificial_mode,
            use_faiss=self.use_faiss,
            faiss_float32=self.faiss_float32,
            chunk_size=self.chunk_size,
        ).fit(X)

        # PyOD convention: decision_scores_ is on the training data.
        self.decision_scores_ = np.asarray(self.model_.predict(X), dtype=np.float64)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Raw anomaly scores. Higher = more anomalous."""
        check_is_fitted(self, ["model_", "decision_scores_"])
        X = check_array(X)
        return np.asarray(self.model_.predict(X), dtype=np.float64)

    # ------------------------------------------------------------------
    # Convenience pass-throughs
    # ------------------------------------------------------------------
    def transform(self, X):
        """Project ``X`` onto the learned discriminant subspace."""
        check_is_fitted(self, ["model_"])
        X = check_array(X)
        return self.model_.transform(X)

    @property
    def projection_(self):
        """Learned discriminant projection matrix of shape (n_features, m)."""
        check_is_fitted(self, ["model_"])
        return self.model_.npd_

    @property
    def filtered_eigvals_(self):
        """Eigenvalues of the discriminant directions retained at fit time."""
        check_is_fitted(self, ["model_"])
        return self.model_.filtered_eigvals_
