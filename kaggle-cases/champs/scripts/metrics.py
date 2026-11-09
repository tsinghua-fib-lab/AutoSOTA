"""CHAMPS metric and strict submission validation."""
import numpy as np
import pandas as pd

TYPES = ["1JHC", "1JHN", "2JHC", "2JHH", "2JHN", "3JHC", "3JHH", "3JHN"]


def group_log_mae(target, prediction, coupling_type):
    target, prediction, coupling_type = map(np.asarray, (target, prediction, coupling_type))
    if target.shape != prediction.shape or target.shape != coupling_type.shape:
        raise ValueError("Mismatched target, prediction or type shape")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Targets and predictions must be finite")
    metrics = {}
    for t in TYPES:
        selected = coupling_type == t
        if not selected.any():
            raise ValueError("Missing coupling type %s" % t)
        mae = np.abs(target[selected] - prediction[selected]).mean()
        metrics[t] = {"count": int(selected.sum()), "mae": float(mae),
                      "log_mae": float(np.log(max(mae, 1e-9)))}
    return {"score": float(np.mean([v["log_mae"] for v in metrics.values()])), "types": metrics}


def validate_submission(submission, test):
    if list(submission.columns) != ["id", "scalar_coupling_constant"]:
        raise ValueError("Submission requires exactly id,scalar_coupling_constant")
    if len(submission) != len(test) or submission.id.duplicated().any():
        raise ValueError("Wrong row count or duplicate IDs")
    if not np.array_equal(submission.id.to_numpy(), test.id.to_numpy()):
        raise ValueError("Submission IDs or order do not match official test")
    if not np.isfinite(submission.scalar_coupling_constant.to_numpy()).all():
        raise ValueError("Nonfinite prediction")
    return {"rows": len(submission), "unique_ids": int(submission.id.nunique()), "finite": True}
