import numpy as np
import pandas as pd

from .paths import SCORED


def mcrmse(y_true, y_pred):
    """Kaggle MCRMSE over pooled positions; y_* are [n_positions, n_cols], NaN targets ignored."""
    errs = []
    for c in range(y_true.shape[1]):
        m = ~np.isnan(y_true[:, c])
        errs.append(np.sqrt(np.mean((y_true[m, c] - y_pred[m, c]) ** 2)))
    return float(np.mean(errs)), errs


def score_test(pred, labels):
    """pred: DataFrame with id + list columns (same layout as labels); labels from data/test_labels.json.

    Returns {split: (mcrmse, per-column)} for public (629 x 68 nt) and private (1172 x 91 nt)."""
    p = pred.set_index("id")
    out = {}
    for split in ("public", "private"):
        lab = labels[labels.split == split]
        yt, yp = [], []
        for r in lab.itertuples():
            s = r.seq_scored
            yt.append(np.stack([np.asarray(getattr(r, c), float)[:s] for c in SCORED], 1))
            yp.append(np.stack([np.asarray(p.at[r.id, c], float)[:s] for c in SCORED], 1))
        out[split] = mcrmse(np.concatenate(yt), np.concatenate(yp))
    return out


def load_labels(path):
    return pd.read_json(path, lines=True)
