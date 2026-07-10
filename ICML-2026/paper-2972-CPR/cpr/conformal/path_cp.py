"""Query-level path conformal prediction (CPR paper Eq. 9-11)."""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def conformal_quantile(scores: List[float], alpha: float) -> float:
    """Split conformal (1-alpha) quantile with finite-sample correction."""
    if not scores:
        return 0.0
    s = np.array([v if np.isfinite(v) else 1e9 for v in scores], dtype=np.float64)
    n = len(s)
    q = ((n + 1) * (1.0 - float(alpha))) / n
    q = float(min(max(q, 0.0), 1.0))
    return float(np.quantile(s, q))


def nonconformity_score(path_conf: Dict, gold_entities: Set[str]) -> float:
    """Eq. 9: minimum path value among correct-terminating paths (lower is better)."""
    gold_entities = {str(e).lower() for e in gold_entities}
    correct_vals = []
    for info in path_conf.values():
        tail = info.get("tail")
        if tail is None:
            continue
        if str(tail).lower() not in gold_entities:
            continue
        try:
            correct_vals.append(float(info.get("scores", [0.0])[0]))
        except (TypeError, ValueError, IndexError):
            continue
    if not correct_vals:
        return float("inf")
    return float(min(correct_vals))


def fit_path_threshold(
    cal_scores: List[float],
    alpha: float,
) -> Tuple[float, Dict]:
    """Eq. 10: calibrate global path-score threshold tau_hat."""
    tau_hat = conformal_quantile(cal_scores, alpha)
    n = len(cal_scores)
    miss = sum(1 for v in cal_scores if not np.isfinite(v) or v >= 1e8)
    q = ((n + 1) * (1.0 - float(alpha))) / max(1, n) if n else 0.0
    return tau_hat, {
        "tau_hat": tau_hat,
        "used": n,
        "miss": miss,
        "quantile": float(min(max(q, 0.0), 1.0)),
        "alpha": float(alpha),
    }


def predict_answer_set(path_conf: Dict, tau_hat: float) -> Set[str]:
    """Eq. 11: entities from paths with value <= tau_hat."""
    selected = set()
    for info in path_conf.values():
        tail = info.get("tail")
        if tail is None:
            continue
        try:
            val = float(info.get("scores", [0.0])[0])
        except (TypeError, ValueError, IndexError):
            continue
        if val <= float(tau_hat):
            selected.add(str(tail).lower())
    return selected


def filter_path_conf(
    path_conf: Dict,
    q_entities: List[str],
    skip_mid: bool = True,
    is_mid_fn=None,
) -> Dict:
    """Pre-filter paths (exclude topic entities; optionally skip MIDs)."""
    topic = {str(e).lower() for e in (q_entities or [])}
    valid = {}
    for p_str, info in path_conf.items():
        tail = info.get("tail")
        if tail is None:
            continue
        if skip_mid and is_mid_fn is not None and is_mid_fn(tail):
            continue
        if str(tail).lower() in topic:
            continue
        valid[p_str] = info
    return valid


def path_post_process(
    path_conf: Dict,
    tau_hat: Optional[float] = None,
    post_alpha: Optional[float] = None,
) -> Tuple[Set[str], Dict[str, float]]:
    """Path-mode prediction set and display confidences."""
    if not path_conf:
        return set(), {}

    if tau_hat is None:
        vals = [float(info["scores"][0]) for info in path_conf.values()]
        if post_alpha is not None and vals:
            tau_hat = conformal_quantile(vals, post_alpha)
        else:
            tau_hat = max(vals) if vals else 0.0

    selected = predict_answer_set(path_conf, tau_hat)
    per_conf = {}
    for tail in selected:
        best = min(
            float(info["scores"][0])
            for info in path_conf.values()
            if str(info.get("tail", "")).lower() == tail
        )
        per_conf[tail] = 1.0 / (1.0 + np.exp(best))

    if not selected and path_conf:
        best_path = min(path_conf.items(), key=lambda x: float(x[1]["scores"][0]))
        tail = str(best_path[1]["tail"]).lower()
        selected.add(tail)
        per_conf[tail] = 1.0 / (1.0 + np.exp(float(best_path[1]["scores"][0])))

    return selected, per_conf