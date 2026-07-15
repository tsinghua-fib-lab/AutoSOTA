import numpy as np

from score_ot_utils import (
    get_rsacp_quantile,
    prepare_rsacp_state,
    rsacp_decision_from_state,
    spi_quantile_scores,
    standard_conformal_quantile,
)


try:
    from calibration.arc.methods import SplitConformal  # type: ignore
except Exception:
    class SplitConformal:  # minimal fallback used in lean environments
        def compute_quantiles(self, alpha_list, s_sorted, y=None, class_conditional=False, is_aps_score=False):
            return {
                float(a): standard_conformal_quantile(s_sorted, float(a), is_aps=is_aps_score)
                for a in _alpha_list(alpha_list)
            }


IMAGENET_SUBSET_CLASSES = np.array(
    [16, 207, 250, 626, 852, 862, 444, 17, 676, 217,
     880, 337, 336, 208, 222, 18, 13, 270, 20, 15,
     321, 392, 157, 326, 993, 991, 994, 389, 395, 0],
    dtype=int,
)


_APS_SCORE_CACHE = {}
_RSACP_SCORE_CACHE = {}


def _alpha_list(alpha):
    if isinstance(alpha, (list, tuple, np.ndarray)):
        return [float(a) for a in alpha]
    return [float(alpha)]


def _as_alpha_scalar(alpha):
    alphas = _alpha_list(alpha)
    if len(alphas) != 1:
        raise ValueError("This operation expects a scalar alpha.")
    return float(alphas[0])


def _cheap_arr_key(a):
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.size == 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    return (int(a.size), float(a.mean()), float(a.var()), float(a.min()), float(a.max()))


def _is_score_input(X):
    X = np.asarray(X)
    return X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1)


def _is_cqr_input(X):
    X = np.asarray(X)
    return X.ndim == 2 and X.shape[1] == 2


def cqr_scores(intervals, y):
    """CQR nonconformity score S=max(lower-y, y-upper)."""
    intervals = np.asarray(intervals, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if _is_score_input(intervals):
        return intervals.reshape(-1).astype(float)
    if not _is_cqr_input(intervals):
        raise ValueError(f"CQR input must have two columns, got shape {intervals.shape}.")
    return np.maximum(intervals[:, 0] - y, y - intervals[:, 1])


def _classes_for_dataset(dataset, X_shape=None, y=None):
    if str(dataset).startswith("ImageNet"):
        if "subset" in str(dataset):
            return IMAGENET_SUBSET_CLASSES.copy()
        if y is not None:
            return np.unique(y).astype(int)
        if X_shape is not None and len(X_shape) > 1:
            return np.arange(int(X_shape[1]))
    if y is not None and str(dataset).startswith("ImageNet"):
        return np.unique(y).astype(int)
    if X_shape is not None and len(X_shape) > 1 and not _is_cqr_input(np.empty((1, X_shape[1]))):
        return np.arange(int(X_shape[1]))
    return np.array([0], dtype=int)


def _aps_components_for_labels(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if X.ndim != 2:
        raise ValueError("APS scores require a 2D probability/logit matrix.")
    order = np.argsort(-X, axis=1)
    sorted_probs = np.take_along_axis(X, order, axis=1)
    csum = np.cumsum(sorted_probs, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    row_ids = np.arange(X.shape[0])[:, None]
    ranks[row_ids, order] = np.arange(X.shape[1], dtype=np.int32)

    if labels.ndim == 1 and labels.size == X.shape[0]:
        pos = ranks[np.arange(X.shape[0]), labels]
        return csum[np.arange(X.shape[0]), pos], X[np.arange(X.shape[0]), labels]

    pos = ranks[:, labels]
    return np.take_along_axis(csum, pos, axis=1), X[:, labels]


def get_aps_score_local(X, y, epsilon=None):
    y = np.asarray(y, dtype=int).reshape(-1)
    cdf, prob = _aps_components_for_labels(X, y)
    if epsilon is None:
        epsilon = np.ones(len(y), dtype=float)
    epsilon = np.asarray(epsilon, dtype=float).reshape(-1)
    return np.maximum(cdf - epsilon * prob, 0.0)


def _aps_scores_for_candidate_labels(X, labels, epsilon=None):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    cdf, prob = _aps_components_for_labels(X, labels)
    if epsilon is None:
        epsilon = np.ones(X.shape[0], dtype=float)
    epsilon = np.asarray(epsilon, dtype=float)
    if epsilon.ndim == 1:
        epsilon = epsilon.reshape(-1, 1)
    return np.maximum(cdf - epsilon * prob, 0.0)


def _compute_scores(X, y, *, epsilon=None, random_state=2020, cache_tag=None):
    X = np.asarray(X)
    y = np.asarray(y)
    if _is_score_input(X):
        return X.reshape(-1).astype(float), "score"
    if _is_cqr_input(X):
        return cqr_scores(X, y), "cqr"
    if epsilon is None:
        epsilon = np.random.default_rng(int(random_state)).uniform(0.0, 1.0, size=len(y))
    key = None
    if cache_tag is not None:
        key = (id(X), id(y), int(random_state), cache_tag)
        if key in _APS_SCORE_CACHE:
            return _APS_SCORE_CACHE[key].copy(), "aps"
    scores = get_aps_score_local(X, y, epsilon=epsilon).reshape(-1).astype(float)
    if key is not None:
        _APS_SCORE_CACHE[key] = scores.copy()
    return scores, "aps"


def _score_type_from_X(X):
    if _is_score_input(X):
        return "score"
    if _is_cqr_input(X):
        return "cqr"
    return "aps"


def _interval_predictions(X, quantiles):
    X = np.asarray(X, dtype=float)
    out = {a: [] for a in quantiles}
    for row in X:
        for a, qhat in quantiles.items():
            if _is_cqr_input(X):
                out[a].append((float(row[0] - qhat), float(row[1] + qhat)))
            else:
                center = float(np.asarray(row).reshape(-1)[0])
                out[a].append((center - float(qhat), center + float(qhat)))
    return out


def _set_predictions_from_quantiles(X, alpha, quantiles, classes, epsilon=None):
    classes = np.asarray(classes, dtype=int)
    scores = _aps_scores_for_candidate_labels(X, classes, epsilon=epsilon)
    out = {a: [] for a in _alpha_list(alpha)}
    for a in out:
        qhat = quantiles[float(a)]
        mask = scores <= qhat
        for i in range(scores.shape[0]):
            out[float(a)].append(classes[mask[i]])
    return out


class _ScoreOnlyBase(SplitConformal):
    def _finish_standard(self, *, method_name, dataset, alpha, scores, score_type, X_calib, y_calib):
        self.method_name = method_name
        self.dataset = dataset
        self.alpha = _alpha_list(alpha)
        self.score_type = score_type
        self.is_aps_score = score_type == "aps"
        self.s_calib = np.sort(np.asarray(scores, dtype=float).reshape(-1))
        self.y_calib = np.asarray(y_calib)
        self.class_conditional = False
        self.classes = _classes_for_dataset(dataset, np.asarray(X_calib).shape, y_calib)
        self.quantiles = {
            float(a): standard_conformal_quantile(self.s_calib, float(a), is_aps=self.is_aps_score)
            for a in self.alpha
        }

    def predict(self, X, random_state=2020, epsilon=None):
        if self.score_type in ("score", "cqr"):
            return _interval_predictions(X, self.quantiles)
        return _set_predictions_from_quantiles(X, self.alpha, self.quantiles, self.classes, epsilon=epsilon)


class SplitConformalRealPlusSynth(_ScoreOnlyBase):
    """Pooled CP: use real/minority and reference/majority scores together."""
    def __init__(
        self,
        method_name,
        dataset,
        X_calib,
        y_calib,
        X_maj_calib,
        y_maj_calib,
        alpha,
        random_state=2020,
        epsilon=None,
        epsilon_maj=None,
        **kwargs,
    ):
        s_real, score_type = _compute_scores(X_calib, y_calib, epsilon=epsilon, random_state=random_state, cache_tag="pooled_real")
        s_ref, ref_type = _compute_scores(
            X_maj_calib, y_maj_calib, epsilon=epsilon_maj, random_state=random_state + 1, cache_tag="pooled_ref"
        )
        if ref_type != score_type:
            raise ValueError(f"Mismatched score types: {score_type} vs {ref_type}.")
        self._finish_standard(
            method_name=method_name,
            dataset=dataset,
            alpha=alpha,
            scores=np.concatenate([s_real, s_ref]),
            score_type=score_type,
            X_calib=X_calib,
            y_calib=y_calib,
        )


class SplitConformalSynthOnly(_ScoreOnlyBase):
    """Synthetic-only CP: calibrate using reference/majority scores only."""
    def __init__(
        self,
        method_name,
        dataset,
        X_calib,
        y_calib,
        X_maj_calib,
        y_maj_calib,
        alpha,
        random_state=2020,
        epsilon=None,
        epsilon_maj=None,
        **kwargs,
    ):
        _, score_type = _compute_scores(X_calib, y_calib, epsilon=epsilon, random_state=random_state, cache_tag="synth_type")
        s_ref, ref_type = _compute_scores(
            X_maj_calib, y_maj_calib, epsilon=epsilon_maj, random_state=random_state + 1, cache_tag="synth_ref"
        )
        if ref_type != score_type:
            raise ValueError(f"Mismatched score types: {score_type} vs {ref_type}.")
        self._finish_standard(
            method_name=method_name,
            dataset=dataset,
            alpha=alpha,
            scores=s_ref,
            score_type=score_type,
            X_calib=X_calib,
            y_calib=y_calib,
        )


class SplitConformalSPI(_ScoreOnlyBase):
    """Score-level SPI fast-form benchmark."""
    def __init__(
        self,
        method_name,
        dataset,
        X_calib,
        y_calib,
        X_maj_calib,
        y_maj_calib,
        alpha,
        random_state=2020,
        epsilon=None,
        epsilon_maj=None,
        beta=0.4,
        **kwargs,
    ):
        s_real, score_type = _compute_scores(X_calib, y_calib, epsilon=epsilon, random_state=random_state, cache_tag="spi_real")
        s_ref, ref_type = _compute_scores(
            X_maj_calib, y_maj_calib, epsilon=epsilon_maj, random_state=random_state + 1, cache_tag="spi_ref"
        )
        if ref_type != score_type:
            raise ValueError(f"Mismatched score types: {score_type} vs {ref_type}.")
        self.method_name = method_name
        self.dataset = dataset
        self.alpha = _alpha_list(alpha)
        self.score_type = score_type
        self.is_aps_score = score_type == "aps"
        self.s_real = np.asarray(s_real, dtype=float)
        self.s_ref = np.asarray(s_ref, dtype=float)
        self.s_calib = np.sort(self.s_real)
        self.y_calib = np.asarray(y_calib)
        self.class_conditional = False
        self.classes = _classes_for_dataset(dataset, np.asarray(X_calib).shape, y_calib)
        self.beta = float(beta)
        self.quantiles = {
            float(a): spi_quantile_scores(self.s_real, self.s_ref, float(a), self.beta, is_aps=self.is_aps_score)
            for a in self.alpha
        }


class SplitConformalRealPlusOTScore(SplitConformal):
    """
    RSA-CP (OT), score-level implementation.

    The transport direction is real/minority scores -> reference/majority score
    scale. Prediction uses candidate-specific Beta-Binomial rank-window
    decisions, not augmented standard conformal quantiles.
    """
    def __init__(
        self,
        method_name,
        dataset,
        X_calib,
        y_calib,
        X_maj_calib,
        y_maj_calib,
        alpha,
        random_state=2020,
        epsilon=None,
        epsilon_maj=None,
        n_score_synth=1000,
        beta=0.4,
        rsacp_grid_size=5000,
        rsacp_max_expand=6,
        use_ot=True,
        **kwargs,
    ):
        self.method_name = method_name
        self.dataset = dataset
        self.alpha = _alpha_list(alpha)
        self.beta = float(beta)
        self.random_state = int(random_state)
        self.use_ot = bool(use_ot)
        self.rsacp_grid_size = int(rsacp_grid_size)
        self.rsacp_max_expand = int(rsacp_max_expand)
        self.class_conditional = kwargs.get("class_conditional", False)
        if self.class_conditional:
            raise NotImplementedError("Score-level RSA-CP currently supports marginal coverage.")

        s_real, score_type = _compute_scores(X_calib, y_calib, epsilon=epsilon, random_state=random_state, cache_tag="rsa_real")
        s_ref, ref_type = _compute_scores(
            X_maj_calib, y_maj_calib, epsilon=epsilon_maj, random_state=random_state + 1, cache_tag="rsa_ref"
        )
        if ref_type != score_type:
            raise ValueError(f"Mismatched score types: {score_type} vs {ref_type}.")
        self.score_type = score_type
        self.is_aps_score = score_type == "aps"
        self.s_real = np.asarray(s_real, dtype=float).reshape(-1)
        self.s_ref = np.asarray(s_ref, dtype=float).reshape(-1)
        if self.s_real.size == 0 or self.s_ref.size == 0:
            raise ValueError("RSA-CP requires nonempty real and reference score arrays.")

        self.classes = _classes_for_dataset(dataset, np.asarray(X_calib).shape, y_calib)
        self.states = {
            float(a): prepare_rsacp_state(
                self.s_real, self.s_ref, alpha=float(a), beta=self.beta, use_ot=self.use_ot
            )
            for a in self.alpha
        }
        self.quantiles = {}
        if self.score_type in ("score", "cqr"):
            for a in self.alpha:
                cache_key = (
                    _cheap_arr_key(self.s_real),
                    _cheap_arr_key(self.s_ref),
                    float(a),
                    self.beta,
                    self.use_ot,
                    self.rsacp_grid_size,
                    self.rsacp_max_expand,
                )
                if cache_key not in _RSACP_SCORE_CACHE:
                    _RSACP_SCORE_CACHE[cache_key] = get_rsacp_quantile(
                        self.s_real,
                        self.s_ref,
                        alpha=float(a),
                        beta=self.beta,
                        use_ot=self.use_ot,
                        grid_size=self.rsacp_grid_size,
                        max_expand=self.rsacp_max_expand,
                    )
                self.quantiles[float(a)] = _RSACP_SCORE_CACHE[cache_key]
        self.s_calib = np.sort(self.s_real)
        self.y_calib = np.asarray(y_calib)

    def predict(self, X, random_state=2020, epsilon=None):
        if self.score_type in ("score", "cqr"):
            return _interval_predictions(X, self.quantiles)

        candidate_scores = _aps_scores_for_candidate_labels(X, self.classes, epsilon=epsilon)
        flat = candidate_scores.reshape(-1)
        out = {}
        for a, state in self.states.items():
            include = rsacp_decision_from_state(flat, state)["include"].reshape(candidate_scores.shape)
            out[float(a)] = [self.classes[include[i]] for i in range(include.shape[0])]
        return out


class SplitConformalScoreRandOnly(_ScoreOnlyBase):
    """Synthetic-score-only toy baseline with random Uniform(0, 1) scores."""
    def __init__(
        self,
        method_name,
        dataset,
        X_calib,
        y_calib,
        X_maj_calib,
        y_maj_calib,
        alpha,
        random_state=2020,
        epsilon=None,
        epsilon_maj=None,
        n_score_synth=1000,
        **kwargs,
    ):
        score_type = _score_type_from_X(X_calib)
        rng = np.random.default_rng(int(random_state))
        scores = rng.uniform(0.0, 1.0, size=int(n_score_synth))
        self._finish_standard(
            method_name=method_name,
            dataset=dataset,
            alpha=alpha,
            scores=scores,
            score_type=score_type,
            X_calib=X_calib,
            y_calib=y_calib,
        )
