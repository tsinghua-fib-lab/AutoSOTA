import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union

def make_absdelta_weights(
    df: pd.DataFrame,
    delta_col: str = "delta_signed",
    zero_eps: Optional[float] = 1e-6,
    normalize: bool = False,
) -> np.ndarray:
    """
    Build per-example training weights proportional to |Δ|.

    Notes:
      • Use these for TRAINING loss weighting.
      • For EVALUATION metrics (e.g., weighted accuracy), recompute weights from
        the eval split as raw |Δ| (set zero_eps=0, normalize=False).
    """
    if delta_col not in df.columns:
        raise KeyError(f"Column {delta_col!r} not found in DataFrame")
    if len(df) == 0:
        raise ValueError("Empty DataFrame yields no weights")

    s = pd.to_numeric(df[delta_col], errors="coerce")
    if s.isna().any():
        bad = int(s.isna().sum())
        raise ValueError(f"{delta_col!r} contains {bad} non-numeric/NaN values")

    w = s.abs().to_numpy(dtype=np.float32).reshape(-1)

    if zero_eps is not None:
        w = np.clip(w, float(zero_eps), None, out=w)

    if normalize:
        m = float(np.mean(w, dtype=np.float64))
        if m == 0.0 or not np.isfinite(m):
            raise ValueError("Cannot normalize: mean(|Δ|) is zero or non-finite")
        w = w / m

    return np.ascontiguousarray(w, dtype=np.float32)


def make_alpha_balanced_weights(
    df: pd.DataFrame,
    delta_col: str = "delta_signed",
    zero_eps: float = 1e-6,
    normalize: bool = False,
    return_alpha: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Compute |Δ|-based weights and scale POSITIVE examples by α so that
    total positive weight equals total negative weight.

        α = (sum_{Δ<0} |Δ|) / (sum_{Δ>0} |Δ|)

    Notes:
      • Intended for TRAINING loss weighting.
      • For EVALUATION metrics, still use raw |Δ| via make_absdelta_weights(..., zero_eps=0, normalize=False).
    """
    if delta_col not in df.columns:
        raise KeyError(f"Column {delta_col!r} not found in DataFrame")
    if len(df) == 0:
        raise ValueError("Empty DataFrame yields no weights")

    s = pd.to_numeric(df[delta_col], errors="coerce")
    if s.isna().any():
        bad = int(s.isna().sum())
        raise ValueError(f"{delta_col!r} contains {bad} non-numeric/NaN values")

    s_np = s.to_numpy()  # keep native dtype for sign tests
    # Base weights = |Δ| (float32 for training), with optional floor
    w = np.abs(s_np).astype(np.float32, copy=False)
    if zero_eps is not None:
        w = np.clip(w, float(zero_eps), None, out=w)

    # Masks
    pos_mask = s_np > 0
    neg_mask = s_np < 0

    # Totals in float64 for stability
    pos_sum = float(w[pos_mask].astype(np.float64).sum())
    neg_sum = float(w[neg_mask].astype(np.float64).sum())

    # Compute alpha; handle corner cases
    if pos_sum > 0.0 and neg_sum > 0.0:
        alpha = neg_sum / pos_sum
    else:
        alpha = 1.0  # no balancing possible

    # Scale positives by alpha so totals match
    w_bal = w.copy()
    if alpha != 1.0:
        w_bal[pos_mask] *= np.float32(alpha)

    if normalize:
        m = float(np.mean(w_bal, dtype=np.float64))
        if m == 0.0 or not np.isfinite(m):
            raise ValueError("Cannot normalize: mean(weight) is zero or non-finite")
        w_bal = w_bal / m

    w_bal = np.ascontiguousarray(w_bal, dtype=np.float32)
    return (w_bal, float(alpha)) if return_alpha else w_bal
