import numpy as np

from score_only_methods import _ScoreOnlyBase, _compute_scores


_SYN_SCORE_CACHE = {}


def _cheap_arr_key(a):
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.size == 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    return (int(a.size), float(a.mean()), float(a.var()), float(a.min()), float(a.max()))


def gen_synth_scores(dist, n, seed=0, target_scores=None):
    """Generate score-only synthetic calibration scores."""
    ref_key = _cheap_arr_key(target_scores) if target_scores is not None else None
    key = (str(dist), int(n), int(seed), ref_key)
    if key in _SYN_SCORE_CACHE:
        return _SYN_SCORE_CACHE[key].copy()

    rng = np.random.default_rng(int(seed))
    n = int(n)
    if dist == "uniform01":
        out = rng.uniform(0.0, 1.0, size=n)
    elif dist in ("normal01_sigmoid", "logit_normal_sigmoid"):
        z = rng.normal(0.0, 1.0 if dist == "normal01_sigmoid" else 2.0, size=n)
        out = 1.0 / (1.0 + np.exp(-z))
    elif dist == "beta_2_5":
        out = rng.beta(2.0, 5.0, size=n)
    elif dist == "beta_5_2":
        out = rng.beta(5.0, 2.0, size=n)
    elif dist == "bootstrap_target":
        ref = np.asarray(target_scores, dtype=float).reshape(-1)
        if ref.size == 0:
            raise ValueError("bootstrap_target requires nonempty target_scores.")
        out = rng.choice(ref, size=n, replace=True)
    elif dist == "empirical_like":
        ref = np.sort(np.asarray(target_scores, dtype=float).reshape(-1))
        if ref.size == 0:
            raise ValueError("empirical_like requires nonempty target_scores.")
        if ref.size == 1:
            out = np.full(n, ref[0], dtype=float)
        else:
            grid = (np.arange(ref.size) + 0.5) / ref.size
            out = np.interp(rng.uniform(0.0, 1.0, size=n), grid, ref, left=ref[0], right=ref[-1])
    else:
        raise ValueError(f"Unknown score_dist: {dist!r}")

    _SYN_SCORE_CACHE[key] = np.asarray(out, dtype=float)
    return _SYN_SCORE_CACHE[key].copy()


class SplitConformalScoreOnlyDist(_ScoreOnlyBase):
    """
    Synthetic-score-only baseline with selectable score distributions.

    For empirical/bootstrap distributions the reference scores are either
    majority/reference scores or real scores, depending on score_ref.
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
        score_dist="bootstrap_target",
        score_ref="majority",
        **kwargs,
    ):
        s_real, score_type = _compute_scores(X_calib, y_calib, epsilon=epsilon, random_state=random_state, cache_tag="dist_real")
        s_ref, ref_type = _compute_scores(
            X_maj_calib, y_maj_calib, epsilon=epsilon_maj, random_state=random_state + 1, cache_tag="dist_ref"
        )
        if ref_type != score_type:
            raise ValueError(f"Mismatched score types: {score_type} vs {ref_type}.")
        target = s_ref if score_ref == "majority" else s_real
        scores = gen_synth_scores(score_dist, int(n_score_synth), seed=random_state, target_scores=target)
        self._finish_standard(
            method_name=method_name,
            dataset=dataset,
            alpha=alpha,
            scores=scores,
            score_type=score_type,
            X_calib=X_calib,
            y_calib=y_calib,
        )
