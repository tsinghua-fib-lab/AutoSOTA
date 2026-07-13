"""Metric helpers for RLIE evaluation."""
from typing import Iterable, Optional, Sequence

from sklearn.metrics import accuracy_score, f1_score


def _resolve_label_list(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Iterable[int]] = None,
) -> Sequence[int]:
    if labels is not None:
        resolved = list(dict.fromkeys(labels))
        if not resolved:
            raise ValueError("labels argument must include at least one class")
        return resolved

    observed = set(y_true)
    observed.update(y_pred)
    if not observed:
        raise ValueError("Cannot infer labels from empty predictions and targets")
    return sorted(observed)


def compute_accuracy_and_macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Optional[Iterable[int]] = None,
) -> tuple[float, float]:
    resolved_labels = _resolve_label_list(y_true, y_pred, labels)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=resolved_labels,
        zero_division=0,
    )
    return accuracy, f1


def compute_macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Optional[Iterable[int]] = None,
) -> float:
    _, f1 = compute_accuracy_and_macro_f1(y_true, y_pred, labels=labels)
    return f1
