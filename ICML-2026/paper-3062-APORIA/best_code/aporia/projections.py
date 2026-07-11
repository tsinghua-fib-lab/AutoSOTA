"""
Projection objects used by the label propagator.

Each projector follows a minimal scikit-learn-like API
(``fit(X, y) -> self`` plus ``transform(X) -> Z``).  ``FisherProjection``
is the main one used in the paper; the others are baselines studied in
Appendix F.

The standalone helper :func:`fisher_direction` is exposed for
``analyse_prompt`` in the structural module, which needs the raw
direction vector ``v`` in addition to the projected coordinates.
"""

from __future__ import annotations

import numpy as np

from sklearn.utils.extmath import randomized_svd

from .data import split_by_label


# ============================================================
# ===================== FISHER DIRECTION =====================
# ============================================================

class DegenerateFisherInputError(np.linalg.LinAlgError):
    """Raised when within-class scatter is rank-zero (all in-class embeddings identical)."""
    pass

def fisher_direction(
    X_G: np.ndarray,
    X_H: np.ndarray,
    lambda_reg: float = 1e-3,
    normalise: bool = True,
    normalise_by_trace: bool = True,
) -> np.ndarray:
    """Compute the regularised Fisher discriminant direction.

    Parameters
    ----------
    X_G, X_H : np.ndarray
        Embeddings of shape (n_G, d) and (n_H, d) for the two classes.
    lambda_reg : float
        Dimensionless regularisation strength.
    normalise_by_trace : bool
        If True, scale lambda by ``trace(S_W) / d`` so the parameter is
        comparable across embedding spaces of different dimensionality.
    normalise : bool
        L2-normalise the returned direction.

    Returns
    -------
    v : np.ndarray, shape (d,)
    """
    mu_G = X_G.mean(axis=0)
    mu_H = X_H.mean(axis=0)

    # within-class scatter (biased = MLE)
    S_G = np.cov(X_G, rowvar=False, bias=True)
    S_H = np.cov(X_H, rowvar=False, bias=True)
    S_W = S_G + S_H

    d = S_W.shape[0]

    if normalise_by_trace:
        trace_per_dim = np.trace(S_W) / d
        if trace_per_dim < 1e-12:
            raise DegenerateFisherInputError(
                "trace(S_W) ≈ 0 — all in-class embeddings appear identical, "
                "Fisher direction is undefined."
            )
        lambda_eff = lambda_reg * trace_per_dim
    else:
        lambda_eff = lambda_reg

    S_W_reg = S_W + lambda_eff * np.eye(d)
    v = np.linalg.solve(S_W_reg, mu_H - mu_G)

    if normalise:
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm

    return v


# ============================================================
# ======================== PROJECTORS ========================
# ============================================================

class ProjectionBase:
    """Minimal interface for a projector ``X in R^d -> Z in R^k``."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ProjectionBase":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


class FisherProjection(ProjectionBase):
    """Supervised 1D projection along the regularised Fisher direction."""

    def __init__(
        self,
        lambda_reg: float = 1e-3,
        normalise: bool = True,
        normalise_by_trace: bool = True,
        l2_normalize_input: bool = False,
    ):
        self.lambda_reg = lambda_reg
        self.normalise = normalise
        self.normalise_by_trace = normalise_by_trace
        self.l2_normalize_input = l2_normalize_input
        self.v: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FisherProjection":
        if self.l2_normalize_input:
            from sklearn.preprocessing import normalize
            X = normalize(X, norm="l2")
        X_G, X_H = split_by_label(X, y)
        self.v = fisher_direction(
            X_G, X_H,
            lambda_reg=self.lambda_reg,
            normalise=self.normalise,
            normalise_by_trace=self.normalise_by_trace,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.v is None:
            raise RuntimeError("FisherProjection.fit must be called before transform")
        return (X @ self.v).reshape(-1, 1)


class WhitenedPCAProjection(ProjectionBase):
    """Top-``k`` PCA components, whitened by their singular values."""

    def __init__(
        self,
        n_components: int = 1,
        n_iter: int = 3,
        random_state: int = 0,
    ):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "WhitenedPCAProjection":
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        _, S, Vt = randomized_svd(
            Xc,
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        self.components_ = Vt
        self.singular_values_ = S
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = X - self.mean_
        Z = Xc @ self.components_.T
        Z = Z / (self.singular_values_ + 1e-12)
        return Z


class RandomProjection(ProjectionBase):
    """Gaussian random projection into ``k`` dimensions."""

    def __init__(self, n_components: int = 1, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state
        self.R: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RandomProjection":
        rng = np.random.default_rng(self.random_state)
        d = X.shape[1]
        self.R = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(self.n_components),
            size=(d, self.n_components),
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.R


class SupervisedUMAPProjection(ProjectionBase):
    """Supervised UMAP embedding (optional dependency: ``umap-learn``)."""

    def __init__(
        self,
        n_components: int = 1,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ):
        try:
            import umap                                       # noqa: F401
        except ImportError as exc:                            # pragma: no cover
            raise ImportError(
                "SupervisedUMAPProjection requires umap-learn. "
                "Install with `pip install aporia[umap]`."
            ) from exc

        import umap as _umap

        self.reducer = _umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupervisedUMAPProjection":
        self.reducer.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.reducer.transform(X)


class IdentityProjection(ProjectionBase):
    """No-op projector: returns ``X`` unchanged.

    Used in Appendix E to study baselines (centroid / LR / SVM / kNN) in
    the full SBERT 384-D embedding space, contrasted with their
    Fisher 1-D counterparts.
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "IdentityProjection":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class CentroidFeaturesProjection(ProjectionBase):
    """Three-dimensional feature map for a logistic-regression baseline.

    For each input ``x``, returns ``(d_G, d_H, v^T x)``, where ``d_G``
    and ``d_H`` are the Euclidean distances to the genuine and
    hallucinated class centroids, and ``v`` is the regularised Fisher
    direction.

    This is the *Centroid-feat 3D* row in Table 7 of the paper.
    """

    def __init__(
        self,
        lambda_reg: float = 1.2,
        metric: str = "euclidean",
        normalise: bool = True,
        normalise_by_trace: bool = True,
    ):
        self.lambda_reg         = lambda_reg
        self.metric             = metric
        self.normalise          = normalise
        self.normalise_by_trace = normalise_by_trace
        self.mu_G: np.ndarray | None = None
        self.mu_H: np.ndarray | None = None
        self.v:    np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CentroidFeaturesProjection":
        X_G, X_H = split_by_label(X, y)
        self.mu_G = X_G.mean(axis=0, keepdims=True)
        self.mu_H = X_H.mean(axis=0, keepdims=True)
        self.v = fisher_direction(
            X_G, X_H,
            lambda_reg=self.lambda_reg,
            normalise=self.normalise,
            normalise_by_trace=self.normalise_by_trace,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        from scipy.spatial.distance import cdist
        if self.mu_G is None or self.mu_H is None or self.v is None:
            raise RuntimeError("CentroidFeaturesProjection.fit must be called before transform")
        dG = cdist(X, self.mu_G, metric=self.metric).ravel()
        dH = cdist(X, self.mu_H, metric=self.metric).ravel()
        z  = X @ self.v
        return np.stack([dG, dH, z], axis=1)



# ============================================================
# ============== FISHER-DIRECTION POST-PROCESSING ============
# ============================================================

def extract_fisher_directions(geometry_store: dict, results_df) -> dict:
    """Unit-normalised Fisher directions for every valid (model, prompt) pair.

    Returns a dict ``{(model_id, prompt_id): v}`` where ``v`` is the
    unit-norm Fisher direction.  Pairs lacking a Fisher direction or
    failing the ``valid_geom`` check are omitted.
    """
    valid_keys = set(zip(
        results_df[results_df["valid_geom"]]["model_id"],
        results_df[results_df["valid_geom"]]["prompt_id"],
    ))
    vecs: dict = {}
    for (m, p), gs in geometry_store.items():
        if (m, p) not in valid_keys:
            continue
        v = gs.get("v_fisher")
        if v is None:
            continue
        norm = np.linalg.norm(v)
        if norm > 0:
            vecs[(m, p)] = v / norm
    return vecs
