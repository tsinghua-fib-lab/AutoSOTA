"""
Label propagation in projected space, using Wasserstein-style scoring.

A new point ``x`` is projected with the fitted projector, and assigned to
the class whose intra-class distance distribution best matches the
point-to-set distance distribution induced by ``x``.
"""

from __future__ import annotations

from typing import Type

import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from tqdm.auto import tqdm

from .caching import ensure_dir, per_pair_path, write_json
from .config import Config
from .data import (
    extract_prompt_data,
    generate_fixed_test_sets,
    split_by_label,
    subsample_training_set,
)
from .evaluation import LabelPropagationEvaluator
from .projections import FisherProjection, ProjectionBase


# ============================================================
# ====================== PROPAGATOR ==========================
# ============================================================

class WassersteinLabelPropagator:
    """Fit on labelled embeddings; classify by distributional consistency."""

    def __init__(self, projection: ProjectionBase, metric: str = "euclidean"):
        self.projection = projection
        self.metric = metric
        self.Z_G: np.ndarray | None = None
        self.Z_H: np.ndarray | None = None
        self.ref_G: np.ndarray | None = None
        self.ref_H: np.ndarray | None = None

    # ----- training -----

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "WassersteinLabelPropagator":
        self.projection.fit(X_train, y_train)
        Z = self.projection.transform(X_train)
        self.Z_G, self.Z_H = split_by_label(Z, y_train)
        self.ref_G = pdist(self.Z_G)
        self.ref_H = pdist(self.Z_H)
        return self

    # ----- scoring -----

    def score_point(self, x: np.ndarray) -> tuple[float, float]:
        z  = self.projection.transform(x[None, :])
        dG = cdist(z, self.Z_G, metric=self.metric).ravel()
        dH = cdist(z, self.Z_H, metric=self.metric).ravel()
        return (
            wasserstein_distance(dG, self.ref_G),
            wasserstein_distance(dH, self.ref_H),
        )

    def predict_point(self, x: np.ndarray) -> int:
        W_G, W_H = self.score_point(x)
        return int(W_H < W_G)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_point(x) for x in X])

    # ----- margins -----

    def margins(self, X: np.ndarray) -> np.ndarray:
        """Signed margin: ``W_H - W_G`` per sample."""
        return np.array([
            (lambda s: s[1] - s[0])(self.score_point(x))
            for x in X
        ])

    def abs_margins(self, X: np.ndarray) -> np.ndarray:
        return np.abs(self.margins(X))

    def margins_by_class(self, X: np.ndarray, y: np.ndarray) -> dict:
        signed = self.margins(X)
        absval = np.abs(signed)
        return {
            int(c): {"signed": signed[y == c], "abs": absval[y == c]}
            for c in np.unique(y)
        }


# ============================================================
# ================== BASELINE PROPAGATORS ====================
# ============================================================
#
# These mirror :class:`WassersteinLabelPropagator` so that
# :func:`run_full_label_propagation_study` can swap one for another
# without further changes.  All follow the convention:
#
#   * ``__init__(projection, **kwargs)``
#   * ``fit(X_train, y_train)`` — fits the projection internally
#   * ``predict(X) -> 0/1``
#   * ``margins(X)``      — signed score (negative → label 1 / H)
#   * ``abs_margins(X)``  — magnitude of the above
#
# The margin sign convention matches that of WassersteinLabelPropagator
# so that the evaluator's per-class margin statistics are comparable
# across propagator types.


class CentroidPropagator:
    """Nearest-centroid classifier in projected space.

    ``metric`` is forwarded to :func:`scipy.spatial.distance.cdist`;
    ``"euclidean"`` and ``"cosine"`` are the two reported in Table 7.
    """

    def __init__(self, projection: ProjectionBase, metric: str = "euclidean"):
        self.projection = projection
        self.metric     = metric
        self.mu_G: np.ndarray | None = None
        self.mu_H: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CentroidPropagator":
        self.projection.fit(X_train, y_train)
        Z = self.projection.transform(X_train)
        Z_G, Z_H = split_by_label(Z, y_train)
        self.mu_G = Z_G.mean(axis=0, keepdims=True)
        self.mu_H = Z_H.mean(axis=0, keepdims=True)
        return self

    def _distances(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = self.projection.transform(X)
        dG = cdist(Z, self.mu_G, metric=self.metric).ravel()
        dH = cdist(Z, self.mu_H, metric=self.metric).ravel()
        return dG, dH

    def predict(self, X: np.ndarray) -> np.ndarray:
        dG, dH = self._distances(X)
        return (dH < dG).astype(int)

    def margins(self, X: np.ndarray) -> np.ndarray:
        dG, dH = self._distances(X)
        return dH - dG                        # negative → predict H

    def abs_margins(self, X: np.ndarray) -> np.ndarray:
        return np.abs(self.margins(X))


class SKLearnPropagator:
    """Wraps any scikit-learn estimator as a propagator.

    Suitable for the LR / SVM / kNN baselines of Appendix E.  The
    ``estimator`` is expected to expose ``fit`` and ``predict``; if it
    also exposes ``decision_function`` or ``predict_proba``, the
    propagator will surface a margin signal.
    """

    def __init__(self, projection: ProjectionBase, estimator):
        self.projection = projection
        self.estimator  = estimator

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SKLearnPropagator":
        self.projection.fit(X_train, y_train)
        Z = self.projection.transform(X_train)
        self.estimator.fit(Z, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = self.projection.transform(X)
        return self.estimator.predict(Z).astype(int)

    def margins(self, X: np.ndarray) -> np.ndarray:
        """Signed score; sign convention matches the other propagators."""
        Z = self.projection.transform(X)
        if hasattr(self.estimator, "decision_function"):
            # decision_function > 0 → predict class 1 (H).  Flip the sign.
            return -np.asarray(self.estimator.decision_function(Z)).ravel()
        if hasattr(self.estimator, "predict_proba"):
            p = self.estimator.predict_proba(Z)
            return (p[:, 0] - p[:, 1]).ravel()      # positive → class 0
        # last-resort: ±0.5 around the predicted label
        return 0.5 - self.estimator.predict(Z).astype(float)

    def abs_margins(self, X: np.ndarray) -> np.ndarray:
        return np.abs(self.margins(X))


# ============================================================
# ======================= EXPERIMENT =========================
# ============================================================

def run_label_propagation_experiment(
    df: pd.DataFrame,
    model_id: int,
    prompt_id: int,
    cfg: Config,
    projector_class: Type[ProjectionBase] = FisherProjection,
    projector_kwargs: dict | None = None,
    propagator_class: type = WassersteinLabelPropagator,
    propagator_kwargs: dict | None = None,
    train_fractions: list[float] | None = None,
    n_iter: int = 10,
    test_fraction: float = 0.2,
    n_splits: int = 5,
    ref_lambda_reg: float | None = None,
    random_state: int | None = None,
    logskip: bool = False,
) -> pd.DataFrame | None:
    """Label propagation for one (model, prompt) pair with fixed test sets
    and multiple random subsamples of the training set per fraction.

    Parameters
    ----------
    projector_class, projector_kwargs
        Determine how the embedding is mapped to a (possibly low-dim)
        space; e.g. :class:`FisherProjection` for the main paper.
    propagator_class, propagator_kwargs
        Choose the scorer that runs on top of the projection; defaults
        to :class:`WassersteinLabelPropagator`.  Use
        :class:`CentroidPropagator` or :class:`SKLearnPropagator` for
        the Appendix E ablation.
    ref_lambda_reg
        If non-None, also fits a Fisher-Wasserstein reference detector
        and surfaces ``agreement_fisher`` metrics.
    """
    projector_kwargs  = {} if projector_kwargs  is None else projector_kwargs
    propagator_kwargs = {} if propagator_kwargs is None else propagator_kwargs
    random_state      = cfg.experiment.random_state if random_state is None else random_state

    if train_fractions is None:
        train_fractions = [1.0]
        n_iter = 1

    # ---- extract & validate ----
    X, y = extract_prompt_data(df, model_id, prompt_id, cfg)
    n_H = int(y.sum())
    n_G = len(y) - n_H
    if n_H < cfg.experiment.min_per_class or n_G < cfg.experiment.min_per_class:
        if not logskip:
            print(
                f"Skipping model {model_id}, prompt {prompt_id} "
                f"(n_G={n_G}, n_H={n_H})"
            )
        return None

    # ---- fixed stratified splits ----
    trn_sets, tst_sets = generate_fixed_test_sets(
        X, y,
        n_splits=n_splits,
        test_fraction=test_fraction,
        random_state=random_state,
    )

    results: list[dict] = []

    for test_id, ((X_train_full, y_train_full), (X_test, y_test)) in enumerate(
        zip(trn_sets, tst_sets)
    ):
        for tf in train_fractions:
            for iter_id in range(n_iter):
                X_sub, y_sub = subsample_training_set(X_train_full, y_train_full, tf, random_state=random_state + iter_id)

                X_G_sub, X_H_sub = split_by_label(X_sub, y_sub)
                if len(X_G_sub) < 2 or len(X_H_sub) < 2:
                    continue

                # optional Fisher reference (for agreement metrics)
                fisher_detector = None
                if ref_lambda_reg is not None:
                    fisher_detector = WassersteinLabelPropagator(
                        FisherProjection(lambda_reg=ref_lambda_reg)
                    )
                    fisher_detector.fit(X_sub, y_sub)

                # fit main detector
                projector = projector_class(**projector_kwargs)
                detector  = propagator_class(projector, **propagator_kwargs)
                try:
                    detector.fit(X_sub, y_sub)
                except np.linalg.LinAlgError:
                    # Rank-deficient within-class scatter — happens on CoQA when the model
                    # repeats the same response and the sampled subset collides.  Skip this
                    # iteration; other iterations of the same (model, prompt) still contribute.
                    if not logskip:
                        print(
                            f"Skipping iter (singular S_W): model {model_id}, prompt {prompt_id}, "
                            f"tf={tf}, iter={iter_id}"
                        )
                    continue

                # evaluate
                evaluator = LabelPropagationEvaluator(
                    detector, X_test, y_test, fisher_ref=fisher_detector,
                )
                metrics = evaluator.evaluate()

                metrics.update({
                    "train_fraction": tf,
                    "iter_id": iter_id,
                    "test_set_id": test_id,
                    "model_id": model_id,
                    "prompt_id": prompt_id,
                    "n_train": len(X_sub),
                })
                results.append(metrics)

    return pd.DataFrame(results) if results else None


# ============================================================
# ========================== STUDY ===========================
# ============================================================

def run_full_label_propagation_study(
    df: pd.DataFrame,
    cfg: Config,
    projector_class: Type[ProjectionBase] = FisherProjection,
    projector_kwargs: dict | None = None,
    propagator_class: type = WassersteinLabelPropagator,
    propagator_kwargs: dict | None = None,
    train_fractions: list[float] | None = None,
    ref_lambda_reg: float | None = None,
    n_iter: int = 10,
    test_fraction: float = 0.2,
    n_splits: int = 5,
    random_state: int | None = None,
    use_cache: bool = False,
    cache_dir: str | None = None,
    overwrite_cache: bool = False,
    logskip: bool = False,
    model_ids: list[int] | None = None,
    prompt_ids_by_model: dict[int, list[int]] | None = None,
) -> pd.DataFrame:
    """Run :func:`run_label_propagation_experiment` over every (model, prompt) pair."""
    projector_kwargs  = {} if projector_kwargs  is None else projector_kwargs
    propagator_kwargs = {} if propagator_kwargs is None else propagator_kwargs
    random_state      = cfg.experiment.random_state if random_state is None else random_state

    if train_fractions is None:
        train_fractions = [1.0]
        n_iter = 1

    if cache_dir is None:
        cache_dir = f"{cfg.cache.root}/label_propagation"

    if use_cache:
        ensure_dir(cache_dir)

    # ---- pair enumeration ----
    if prompt_ids_by_model is None:
        from .data import prompt_ids_by_model as _ids
        prompt_ids_by_model = _ids(df, cfg)
    if model_ids is None:
        # use the models actually present in the dataframe, not cfg.model_ids,
        # so a cfg declaring more models than the data won't trigger KeyError
        model_ids = list(prompt_ids_by_model.keys())

    all_results: list[pd.DataFrame] = []

    for mid in tqdm(model_ids, desc="Model"):
        for pid in tqdm(prompt_ids_by_model[mid], desc="Prompt", leave=False):

            cache_path = None
            if use_cache:
                cache_path = per_pair_path(
                    cache_dir,
                    model=mid, prompt=pid,
                    detector=projector_class.__name__,
                    propagator=propagator_class.__name__,
                )
                if cache_path.exists() and not overwrite_cache:
                    all_results.append(pd.read_parquet(cache_path))
                    continue

            res_df = run_label_propagation_experiment(
                df=df,
                model_id=mid,
                prompt_id=pid,
                cfg=cfg,
                projector_class=projector_class,
                projector_kwargs=projector_kwargs,
                propagator_class=propagator_class,
                propagator_kwargs=propagator_kwargs,
                train_fractions=train_fractions,
                n_iter=n_iter,
                test_fraction=test_fraction,
                n_splits=n_splits,
                ref_lambda_reg=ref_lambda_reg,
                random_state=random_state,
                logskip=logskip,
            )

            if res_df is None or len(res_df) == 0:
                continue

            all_results.append(res_df)

            if use_cache and cache_path is not None:
                res_df.to_parquet(cache_path, index=False)

    if not all_results:
        return pd.DataFrame()

    results_lp = pd.concat(all_results, ignore_index=True)

    if use_cache:
        write_json(
            f"{cache_dir}/meta.json",
            {
                "model_ids":         list(model_ids),
                "train_fractions":   list(train_fractions),
                "n_iter":            n_iter,
                "test_fraction":     test_fraction,
                "n_splits":          n_splits,
                "random_state":      random_state,
                "n_cached_pairs":    len(all_results),
                "projector_class":   projector_class.__name__,
                "projector_kwargs":  projector_kwargs,
                "propagator_class":  propagator_class.__name__,
                "propagator_kwargs": {k: str(v) for k, v in propagator_kwargs.items()},
                "ref_lambda_reg":    ref_lambda_reg,
                "dataset_name":      cfg.dataset.name,
            },
        )

    return results_lp
